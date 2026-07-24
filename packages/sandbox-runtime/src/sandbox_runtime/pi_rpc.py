"""JSONL RPC transport client for the pi coding agent (pi.dev).

Pi runs as a subprocess in RPC mode (``pi --mode rpc``): commands are JSON
objects written to stdin, responses and agent events stream back on stdout as
JSON lines. This module owns every raw wire concern — process lifecycle,
strict-JSONL framing, command/response correlation, the shared event queue,
session resume via ``switch_session`` — and mirrors the bridge-facing surface
of ``OpenCodeClient`` (``create_session`` / ``session_exists`` /
``request_stop`` / ``aclose``) so the bridge can hold either client.

Prompt-lifecycle policy (timeouts, event translation) lives in
``PiPromptStream`` (pi_prompt_stream.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from .log_config import StructuredLogger

PI_COMMAND_TIMEOUT_SECONDS: Final = 30.0
PI_TERMINATE_GRACE_SECONDS: Final = 5.0
# Guards against a runaway agent flooding memory while no prompt is draining.
MAX_QUEUED_EVENTS: Final = 10_000

# Extension UI dialog methods that block the agent until answered. The bridge
# has no channel to relay dialogs to a user, so they are auto-cancelled —
# mirroring how the supervisor disables OpenCode's question tool in headless
# mode (OPENCODE_CLIENT=serve).
_DIALOG_UI_METHODS: Final = frozenset({"select", "confirm", "input", "editor"})

# Synthetic event injected into the event queue when the pi process exits so a
# stream blocked on next_event() fails fast instead of waiting for a timeout.
PROCESS_EXITED_EVENT: Final = "pi.process_exited"


class PiAgentError(Exception):
    """Raised when the pi RPC transport fails (spawn, write, or crash)."""


class PiRpcClient:
    """Transport for one long-lived ``pi --mode rpc`` subprocess.

    The subprocess is spawned lazily on first use and restarted (resuming the
    persisted session) by the next ``ensure_started`` call if it died. Only
    one prompt streams at a time — the bridge serializes prompts — so a single
    shared event queue is sufficient.
    """

    def __init__(
        self,
        *,
        log: StructuredLogger,
        model: str | None = None,
        workdir: str | Path | None = None,
        pi_executable: str = "pi",
        command_timeout_seconds: float = PI_COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        self._log = log
        self._model = model
        self._workdir = Path(workdir) if workdir else None
        self._pi_executable = pi_executable
        self._command_timeout_seconds = command_timeout_seconds

        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._next_command_id = 0
        self._start_lock = asyncio.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def ensure_started(self, *, resume_session_file: str | None = None) -> None:
        """Spawn the pi RPC process if it is not already running.

        When ``resume_session_file`` names an existing session file, the fresh
        process is switched onto it so conversation history survives process
        (and sandbox) restarts. A failed switch degrades to a new session.
        """
        async with self._start_lock:
            if self.is_running():
                return
            if self._closed:
                raise PiAgentError("Pi client is closed")

            await self._cleanup_process_state()

            args = [self._pi_executable, "--mode", "rpc"]
            if self._model:
                args.extend(["--model", self._model])

            self._log.info(
                "pi.start",
                model=self._model,
                workdir=str(self._workdir) if self._workdir else None,
                resume=bool(resume_session_file),
            )
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *args,
                    cwd=self._workdir,
                    env=os.environ,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except (OSError, FileNotFoundError) as e:
                raise PiAgentError(f"Failed to start pi agent: {e}") from e

            self._reader_task = asyncio.create_task(self._read_stdout())
            self._stderr_task = asyncio.create_task(self._read_stderr())

            if resume_session_file and Path(resume_session_file).exists():
                try:
                    response = await self._command_locked(
                        {"type": "switch_session", "sessionPath": resume_session_file}
                    )
                    self._log.info(
                        "pi.session.resume",
                        session_file=resume_session_file,
                        success=bool(response.get("success")),
                    )
                except Exception as e:
                    self._log.warn("pi.session.resume_failed", exc=e)

    async def aclose(self) -> None:
        """Terminate the pi process and release transport resources."""
        self._closed = True
        process = self._process
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=PI_TERMINATE_GRACE_SECONDS)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
        await self._cleanup_process_state()

    async def _cleanup_process_state(self) -> None:
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._reader_task = None
        self._stderr_task = None
        self._process = None
        self._fail_pending("pi agent process is not running")

    def _fail_pending(self, message: str) -> None:
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(PiAgentError(message))

    # ------------------------------------------------------------------
    # Wire protocol
    # ------------------------------------------------------------------

    async def command(
        self, cmd: dict[str, Any], *, timeout_seconds: float | None = None
    ) -> dict[str, Any]:
        """Send one RPC command and await its correlated response."""
        if not self.is_running():
            raise PiAgentError("pi agent process is not running")
        return await self._command_locked(cmd, timeout_seconds=timeout_seconds)

    async def _command_locked(
        self, cmd: dict[str, Any], *, timeout_seconds: float | None = None
    ) -> dict[str, Any]:
        self._next_command_id += 1
        command_id = f"cmd-{self._next_command_id}"
        payload = {**cmd, "id": command_id}

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[command_id] = future
        try:
            await self._write_line(payload)
            return await asyncio.wait_for(
                future, timeout=timeout_seconds or self._command_timeout_seconds
            )
        except TimeoutError:
            raise PiAgentError(
                f"pi command '{cmd.get('type')}' timed out after "
                f"{timeout_seconds or self._command_timeout_seconds:.0f}s"
            ) from None
        finally:
            self._pending.pop(command_id, None)

    async def _write_line(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise PiAgentError("pi agent process is not running")
        try:
            process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise PiAgentError(f"Failed to write to pi agent: {e}") from e

    async def next_event(self, *, timeout_seconds: float) -> dict[str, Any]:
        """Wait for the next agent event; raises TimeoutError on inactivity."""
        return await asyncio.wait_for(self._events.get(), timeout=timeout_seconds)

    def drain_events(self) -> None:
        """Discard stale queued events (called before starting a prompt)."""
        while True:
            try:
                self._events.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _read_stdout(self) -> None:
        """Route stdout JSON lines to pending futures or the event queue.

        RPC framing is strict JSONL with LF delimiters; a trailing CR is
        stripped for tolerance. Unparseable lines are logged and skipped.
        """
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            while True:
                raw = await process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as e:
                    self._log.debug("pi.stdout_parse_error", exc=e)
                    continue
                if not isinstance(message, dict):
                    continue
                await self._dispatch_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._log.error("pi.reader_error", exc=e)
        finally:
            returncode = process.returncode
            self._fail_pending(f"pi agent process exited (code {returncode})")
            self._put_event({"type": PROCESS_EXITED_EVENT, "returncode": returncode})

    async def _dispatch_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")

        if message_type == "response":
            command_id = message.get("id")
            future = self._pending.get(command_id) if isinstance(command_id, str) else None
            if future is not None and not future.done():
                future.set_result(message)
            else:
                # Un-correlated response (e.g. a parse error) — surface it to
                # the stream so failures are not silently dropped.
                self._put_event(message)
            return

        if message_type == "extension_ui_request":
            await self._auto_answer_ui_request(message)
            return

        self._put_event(message)

    def _put_event(self, event: dict[str, Any]) -> None:
        if self._events.qsize() >= MAX_QUEUED_EVENTS:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._events.get_nowait()
        self._events.put_nowait(event)

    async def _auto_answer_ui_request(self, request: dict[str, Any]) -> None:
        """Cancel dialog UI requests; drop fire-and-forget ones.

        Headless sandboxes have no user to answer extension dialogs, and an
        unanswered dialog without a timeout blocks the agent forever.
        """
        method = request.get("method")
        request_id = request.get("id")
        self._log.info("pi.extension_ui_request", method=method)
        if method not in _DIALOG_UI_METHODS or not request_id:
            return
        with contextlib.suppress(PiAgentError):
            await self._write_line(
                {"type": "extension_ui_response", "id": request_id, "cancelled": True}
            )

    async def _read_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                raw = await process.stderr.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._log.info("pi.stderr", detail=line)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._log.debug("pi.stderr_read_error", exc=e)

    # ------------------------------------------------------------------
    # Bridge-facing session surface (mirrors OpenCodeClient)
    # ------------------------------------------------------------------

    async def create_session(self) -> str | None:
        """Start the agent and return its persisted session identifier.

        Pi persists sessions as JSONL files; the absolute session file path is
        the durable identifier the bridge saves and reports upstream. Falls
        back to the in-memory session id when persistence is disabled.
        """
        await self.ensure_started()
        response = await self.command({"type": "get_state"})
        data = response.get("data") if isinstance(response.get("data"), dict) else {}
        session_file = data.get("sessionFile") or data.get("sessionId")
        return str(session_file) if session_file else None

    async def session_exists(self, session_id: str) -> bool:
        """Whether the persisted session can still be resumed.

        The identifier is a session file path (see ``create_session``); the
        session survives restarts iff the file survived.
        """
        return bool(session_id) and Path(session_id).exists()

    async def request_stop(self, session_id: str | None, *, reason: str) -> bool:
        """Best-effort abort of the active pi prompt (saves LLM compute)."""
        if not self.is_running():
            return False
        try:
            await self.command({"type": "abort"})
            self._log.info("bridge.stop_requested", reason=reason)
            return True
        except Exception as e:
            self._log.warn("bridge.stop_request_error", exc=e, reason=reason)
            return False
