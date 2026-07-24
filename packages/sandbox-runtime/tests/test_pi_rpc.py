"""
Unit tests for PiRpcClient, the JSONL RPC transport for the pi coding agent.

A fake `pi` executable (a small Python script speaking the RPC protocol on
stdio) exercises the real subprocess path: spawn arguments, command/response
correlation, event routing, extension-UI auto-cancellation, session resume via
switch_session, and crash propagation.
"""

import asyncio
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sandbox_runtime.pi_rpc import PROCESS_EXITED_EVENT, PiAgentError, PiRpcClient

FAKE_PI_SCRIPT = r"""
import json, os, sys


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


out({"type": "fake_started", "argv": sys.argv[1:]})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    cmd = json.loads(line)
    ctype = cmd.get("type")
    out({"type": "fake_echo", "cmd": cmd})
    if ctype == "extension_ui_response":
        continue
    if ctype == "get_state":
        out(
            {
                "type": "response",
                "id": cmd.get("id"),
                "command": "get_state",
                "success": True,
                "data": {
                    "sessionFile": os.environ.get("FAKE_PI_SESSION_FILE") or None,
                    "sessionId": "fake-session-id",
                },
            }
        )
    elif ctype == "prompt":
        out({"type": "response", "id": cmd.get("id"), "command": "prompt", "success": True})
        out(
            {
                "type": "extension_ui_request",
                "id": "ui-1",
                "method": "select",
                "title": "pick",
                "options": ["a", "b"],
            }
        )
        out({"type": "extension_ui_request", "id": "ui-2", "method": "notify", "message": "hi"})
        out({"type": "agent_start"})
    elif ctype == "fake_crash":
        sys.exit(3)
    else:
        out({"type": "response", "id": cmd.get("id"), "command": ctype, "success": True})
"""


@pytest.fixture
def fake_pi(tmp_path: Path) -> str:
    script = tmp_path / "fake_pi.py"
    script.write_text(FAKE_PI_SCRIPT)
    launcher = tmp_path / "pi"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return str(launcher)


def make_client(fake_pi: str, **kwargs) -> PiRpcClient:
    return PiRpcClient(log=MagicMock(), pi_executable=fake_pi, **kwargs)


async def wait_for_event(client: PiRpcClient, event_type: str, *, timeout: float = 5.0) -> dict:
    """Drain events until one of the requested type arrives."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        event = await client.next_event(timeout_seconds=max(remaining, 0.01))
        if event.get("type") == event_type:
            return event


class TestSpawn:
    async def test_passes_rpc_mode_and_model_argument(self, fake_pi):
        client = make_client(fake_pi, model="anthropic/claude-sonnet-4-6")
        try:
            await client.ensure_started()
            started = await wait_for_event(client, "fake_started")
            assert started["argv"] == [
                "--mode",
                "rpc",
                "--model",
                "anthropic/claude-sonnet-4-6",
            ]
        finally:
            await client.aclose()

    async def test_omits_model_argument_when_unset(self, fake_pi):
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            started = await wait_for_event(client, "fake_started")
            assert started["argv"] == ["--mode", "rpc"]
        finally:
            await client.aclose()

    async def test_runs_in_configured_workdir(self, fake_pi, tmp_path):
        workdir = tmp_path / "repo"
        workdir.mkdir()
        client = make_client(fake_pi, workdir=workdir)
        try:
            await client.ensure_started()
            assert client.is_running()
        finally:
            await client.aclose()

    async def test_missing_executable_raises_pi_agent_error(self, tmp_path):
        client = PiRpcClient(log=MagicMock(), pi_executable=str(tmp_path / "does-not-exist"))
        with pytest.raises(PiAgentError, match="Failed to start pi agent"):
            await client.ensure_started()

    async def test_ensure_started_is_idempotent(self, fake_pi):
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            process = client._process
            await client.ensure_started()
            assert client._process is process
        finally:
            await client.aclose()


class TestCommands:
    async def test_correlates_response_by_id(self, fake_pi):
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            response = await client.command({"type": "abort"})
            assert response["command"] == "abort"
            assert response["success"] is True
        finally:
            await client.aclose()

    async def test_command_timeout_raises(self, fake_pi):
        client = make_client(fake_pi, command_timeout_seconds=0.1)
        try:
            await client.ensure_started()
            # fake_echo events are emitted, but no response for this type.
            with pytest.raises(PiAgentError, match="timed out"):
                await client.command({"type": "extension_ui_response", "id": "x"})
        finally:
            await client.aclose()

    async def test_command_without_process_raises(self, fake_pi):
        client = make_client(fake_pi)
        with pytest.raises(PiAgentError, match="not running"):
            await client.command({"type": "abort"})


class TestEvents:
    async def test_events_routed_to_queue(self, fake_pi):
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            await client.command({"type": "prompt", "message": "hi"})
            event = await wait_for_event(client, "agent_start")
            assert event == {"type": "agent_start"}
        finally:
            await client.aclose()

    async def test_drain_events_discards_queued_events(self, fake_pi):
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            await wait_for_event(client, "fake_started")
            await client.command({"type": "prompt", "message": "hi"})
            # Wait for the last in-flight exchange (the echoed auto-cancel of
            # the select dialog) so the wire is quiet before draining.
            echo = await wait_for_event(client, "fake_echo")
            while echo["cmd"].get("type") != "extension_ui_response":
                echo = await wait_for_event(client, "fake_echo")
            client.drain_events()
            with pytest.raises(TimeoutError):
                await client.next_event(timeout_seconds=0.05)
        finally:
            await client.aclose()

    async def test_dialog_ui_requests_are_auto_cancelled(self, fake_pi):
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            await client.command({"type": "prompt", "message": "hi"})
            # The fake echoes everything it receives; the cancellation for the
            # select dialog must come back, and only for the dialog method.
            echo = await wait_for_event(client, "fake_echo")
            while echo["cmd"].get("type") != "extension_ui_response":
                echo = await wait_for_event(client, "fake_echo")
            assert echo["cmd"] == {
                "type": "extension_ui_response",
                "id": "ui-1",
                "cancelled": True,
            }
        finally:
            await client.aclose()


class TestCrash:
    async def test_process_exit_fails_pending_and_queues_sentinel(self, fake_pi):
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            with pytest.raises(PiAgentError, match="exited"):
                await client.command({"type": "fake_crash"})
            sentinel = await wait_for_event(client, PROCESS_EXITED_EVENT)
            assert sentinel["returncode"] == 3
        finally:
            await client.aclose()

    async def test_restart_after_crash_resumes_session(self, fake_pi, tmp_path):
        session_file = tmp_path / "session.jsonl"
        session_file.write_text("{}\n")
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            with pytest.raises(PiAgentError):
                await client.command({"type": "fake_crash"})
            await wait_for_event(client, PROCESS_EXITED_EVENT)

            await client.ensure_started(resume_session_file=str(session_file))
            assert client.is_running()
            echo = await wait_for_event(client, "fake_echo")
            assert echo["cmd"]["type"] == "switch_session"
            assert echo["cmd"]["sessionPath"] == str(session_file)
        finally:
            await client.aclose()

    async def test_missing_resume_file_skips_switch_session(self, fake_pi, tmp_path):
        client = make_client(fake_pi)
        try:
            await client.ensure_started(resume_session_file=str(tmp_path / "gone.jsonl"))
            await wait_for_event(client, "fake_started")
            state = await client.command({"type": "get_state"})
            assert state["success"] is True
            # The first command the fake saw must be get_state — no
            # switch_session was attempted for the missing file.
            echo = await wait_for_event(client, "fake_echo")
            assert echo["cmd"]["type"] == "get_state"
        finally:
            await client.aclose()


class TestBridgeSurface:
    async def test_create_session_returns_session_file(self, fake_pi, tmp_path, monkeypatch):
        session_file = tmp_path / "abc.jsonl"
        monkeypatch.setenv("FAKE_PI_SESSION_FILE", str(session_file))
        client = make_client(fake_pi)
        try:
            assert await client.create_session() == str(session_file)
        finally:
            await client.aclose()

    async def test_create_session_falls_back_to_session_id(self, fake_pi, monkeypatch):
        monkeypatch.setenv("FAKE_PI_SESSION_FILE", "")
        client = make_client(fake_pi)
        try:
            assert await client.create_session() == "fake-session-id"
        finally:
            await client.aclose()

    async def test_session_exists_checks_the_filesystem(self, fake_pi, tmp_path):
        client = make_client(fake_pi)
        existing = tmp_path / "session.jsonl"
        existing.write_text("{}\n")
        assert await client.session_exists(str(existing)) is True
        assert await client.session_exists(str(tmp_path / "missing.jsonl")) is False
        assert await client.session_exists("") is False

    async def test_request_stop_sends_abort(self, fake_pi):
        client = make_client(fake_pi)
        try:
            await client.ensure_started()
            assert await client.request_stop("whatever", reason="test") is True
            echo = await wait_for_event(client, "fake_echo")
            assert echo["cmd"]["type"] == "abort"
        finally:
            await client.aclose()

    async def test_request_stop_without_process_returns_false(self, fake_pi):
        client = make_client(fake_pi)
        assert await client.request_stop("whatever", reason="test") is False

    async def test_aclose_terminates_process(self, fake_pi):
        client = make_client(fake_pi)
        await client.ensure_started()
        process = client._process
        await client.aclose()
        assert process is not None and process.returncode is not None
        assert not client.is_running()
        with pytest.raises(PiAgentError, match="closed"):
            await client.ensure_started()


class TestFraming:
    async def test_tolerates_crlf_and_garbage_lines(self, fake_pi, tmp_path):
        script = tmp_path / "crlf_pi.py"
        script.write_text(
            "import sys\n"
            'sys.stdout.write(\'{"type": "agent_start"}\\r\\n\')\n'
            "sys.stdout.write('not json\\n')\n"
            'sys.stdout.write(\'{"type": "agent_settled"}\\n\')\n'
            "sys.stdout.flush()\n"
            "sys.stdin.read()\n"
        )
        launcher = tmp_path / "pi"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)

        client = make_client(str(launcher))
        try:
            await client.ensure_started()
            assert (await wait_for_event(client, "agent_start"))["type"] == "agent_start"
            assert (await wait_for_event(client, "agent_settled"))["type"] == "agent_settled"
        finally:
            await client.aclose()
