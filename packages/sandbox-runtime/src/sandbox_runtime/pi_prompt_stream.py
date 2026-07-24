"""Pi prompt streaming: translates pi RPC events to bridge events.

Counterpart of ``OpenCodePromptStream`` for the pi coding agent (pi.dev).
Sends one ``prompt`` command per bridge prompt, then consumes the pi event
stream until the run settles (``agent_settled``), translating events into the
bridge's provider-agnostic event vocabulary:

- ``message_update`` text deltas   -> ``token`` (cumulative per text block)
- ``tool_execution_start/update/end`` -> ``tool_call`` (running/completed/error)
- ``turn_start`` / ``turn_end``    -> ``step_start`` / ``step_finish``
- assistant ``stopReason: error``  -> ``error``

Pi has no server-side heartbeat, so the SSE-style inactivity deadline is only
armed while the model is streaming; during tool execution (which can legally
stay silent for minutes) the wait is bounded by the prompt max duration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from .pi_rpc import PROCESS_EXITED_EVENT, PiAgentError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .attachment_processor import HydratedSessionAttachment
    from .log_config import StructuredLogger
    from .pi_rpc import PiRpcClient

# Model ids arrive as "pi/<provider>/<model>" (catalog form) or
# "<provider>/<model>"; this prefix selects the pi harness and is not part of
# the provider/model pair pi itself understands.
PI_MODEL_PREFIX: Final = "pi/"

# ReasoningEffort -> pi thinking level. Pi levels are a superset of the shared
# efforts except "none", which pi spells "off".
_THINKING_LEVELS: Final[dict[str, str]] = {
    "none": "off",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def strip_pi_model_prefix(model: str) -> str:
    """Drop the harness-selecting "pi/" prefix from a catalog model id."""
    if model.startswith(PI_MODEL_PREFIX):
        return model[len(PI_MODEL_PREFIX) :]
    return model


def split_provider_model(model: str) -> tuple[str, str]:
    """Split "<provider>/<model>" (default provider: anthropic)."""
    stripped = strip_pi_model_prefix(model)
    if "/" in stripped:
        provider, model_id = stripped.split("/", 1)
        return provider, model_id
    return "anthropic", stripped


@dataclass
class _PromptState:
    """Mutable translation state for one ``stream_prompt`` call."""

    message_id: str
    start_time: float
    # Cumulative text per assistant text block, keyed by
    # "<assistant message ordinal>:<content index>".
    cumulative_text: dict[str, str] = field(default_factory=dict)
    assistant_message_ordinal: int = 0
    # Accumulated output and remembered args per in-flight tool call
    # (tool_execution_end does not repeat the args).
    tool_output: dict[str, str] = field(default_factory=dict)
    tool_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    in_tool_execution: bool = False
    emitted_error_messages: set[str] = field(default_factory=set)


class PiPromptStream:
    """Streams one prompt through pi and translates its RPC events.

    The instance is long-lived (one per bridge). The pi session identifier is
    a per-call parameter — it is the persisted session file the client resumes
    after a process or sandbox restart.
    """

    def __init__(
        self,
        *,
        client: PiRpcClient,
        log: StructuredLogger,
        inactivity_timeout_seconds: float,
        prompt_max_duration_seconds: float,
    ) -> None:
        self._client = client
        self._log = log
        self._inactivity_timeout_seconds = inactivity_timeout_seconds
        self._prompt_max_duration_seconds = prompt_max_duration_seconds
        # Best-effort model configuration is skipped when unchanged.
        self._configured_model: str | None = None
        self._configured_thinking: str | None = None

    async def stream_prompt(
        self,
        *,
        opencode_session_id: str,
        message_id: str,
        content: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        attachments: list[HydratedSessionAttachment] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream one prompt's bridge events from the pi agent.

        ``opencode_session_id`` keeps the OpenCode-era parameter name the
        bridge uses for the persisted agent session identifier; for pi it is
        the session file path used to resume after restarts.
        """
        await self._client.ensure_started(resume_session_file=opencode_session_id)
        await self._configure_model(model, reasoning_effort)

        state = _PromptState(message_id=message_id, start_time=time.time())
        deadline = state.start_time + self._prompt_max_duration_seconds

        self._client.drain_events()
        await self._post_prompt(content, attachments)

        while True:
            event = await self._next_event(state, deadline, message_id)
            for bridge_event in self._apply_event(state, event):
                yield bridge_event
            if event.get("type") == "agent_settled":
                self._log.debug(
                    "bridge.pi_agent_settled",
                    elapsed_s=round(time.time() - state.start_time, 1),
                )
                return

    async def _next_event(
        self, state: _PromptState, deadline: float, message_id: str
    ) -> dict[str, Any]:
        """Fetch the next pi event under the inactivity/max-duration policy."""
        remaining = deadline - time.time()
        if remaining <= 0:
            await self._stop_for_timeout("prompt_max_duration_timeout", state)
            raise RuntimeError(
                f"Prompt exceeded max duration of {self._prompt_max_duration_seconds:.0f}s."
            )

        # Silent tool executions are legal; model streaming should not be.
        timeout = (
            remaining
            if state.in_tool_execution
            else min(self._inactivity_timeout_seconds, remaining)
        )
        try:
            event = await self._client.next_event(timeout_seconds=timeout)
        except TimeoutError:
            if time.time() >= deadline:
                await self._stop_for_timeout("prompt_max_duration_timeout", state)
                raise RuntimeError(
                    f"Prompt exceeded max duration of {self._prompt_max_duration_seconds:.0f}s."
                ) from None
            elapsed = time.time() - state.start_time
            self._log.error(
                "bridge.pi_inactivity_timeout",
                timeout_name="pi_inactivity",
                timeout_ms=int(self._inactivity_timeout_seconds * 1000),
                elapsed_ms=int(elapsed * 1000),
                message_id=message_id,
            )
            await self._stop_for_timeout("inactivity_timeout", state)
            raise RuntimeError(
                f"Pi agent event stream inactive for "
                f"{self._inactivity_timeout_seconds:.0f}s (no data received)."
            ) from None

        if event.get("type") == PROCESS_EXITED_EVENT:
            raise RuntimeError(
                f"Pi agent process exited unexpectedly (code {event.get('returncode')})."
            )
        return event

    async def _stop_for_timeout(self, reason: str, state: _PromptState) -> None:
        try:
            await self._client.request_stop(state.message_id, reason=reason)
        except Exception as e:
            self._log.warn("bridge.stop_request_error", exc=e, reason=reason)

    async def _post_prompt(
        self, content: str, attachments: list[HydratedSessionAttachment] | None
    ) -> None:
        prompt_command: dict[str, Any] = {"type": "prompt", "message": content}
        images = [
            {
                "type": "image",
                "data": attachment["content"],
                "mimeType": attachment["mimeType"],
            }
            for attachment in attachments or []
        ]
        if images:
            prompt_command["images"] = images

        response = await self._client.command(prompt_command)
        if response.get("success"):
            return

        error = str(response.get("error") or "")
        if "streaming" in error.lower():
            # The agent is mid-run (e.g. resumed after a reconnect while a
            # previous prompt is settling): queue as a follow-up instead.
            self._log.warn("bridge.pi_prompt_queued_follow_up", detail=error)
            retry = await self._client.command({**prompt_command, "streamingBehavior": "followUp"})
            if retry.get("success"):
                return
            error = str(retry.get("error") or error)
        raise RuntimeError(f"Pi agent rejected prompt: {error or 'unknown error'}")

    async def _configure_model(self, model: str | None, reasoning_effort: str | None) -> None:
        """Best-effort per-prompt model/thinking configuration.

        Failures are logged, not raised — the process-level ``--model``
        default keeps the prompt runnable.
        """
        if model and model != self._configured_model:
            provider, model_id = split_provider_model(model)
            try:
                response = await self._client.command(
                    {"type": "set_model", "provider": provider, "modelId": model_id}
                )
                if response.get("success"):
                    self._configured_model = model
                    # Model switches reset pi's thinking level.
                    self._configured_thinking = None
                else:
                    self._log.warn(
                        "bridge.pi_set_model_failed",
                        model=model,
                        detail=str(response.get("error") or ""),
                    )
            except PiAgentError as e:
                self._log.warn("bridge.pi_set_model_failed", model=model, exc=e)

        level = _THINKING_LEVELS.get(reasoning_effort or "")
        if level and level != self._configured_thinking:
            try:
                response = await self._client.command(
                    {"type": "set_thinking_level", "level": level}
                )
                if response.get("success"):
                    self._configured_thinking = level
                else:
                    self._log.warn(
                        "bridge.pi_set_thinking_failed",
                        detail=str(response.get("error") or ""),
                        reasoning_effort=reasoning_effort,
                    )
            except PiAgentError as e:
                self._log.warn(
                    "bridge.pi_set_thinking_failed", exc=e, reasoning_effort=reasoning_effort
                )

    # ------------------------------------------------------------------
    # Event translation
    # ------------------------------------------------------------------

    def _apply_event(self, state: _PromptState, event: dict[str, Any]) -> list[dict[str, Any]]:
        event_type = event.get("type")

        if event_type == "message_start":
            message = event.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                state.assistant_message_ordinal += 1
            return []

        if event_type == "message_update":
            return self._on_message_update(state, event)

        if event_type == "message_end":
            return self._on_message_end(state, event)

        if event_type == "tool_execution_start":
            state.in_tool_execution = True
            call_id = str(event.get("toolCallId") or "")
            state.tool_output[call_id] = ""
            args = event.get("args")
            if isinstance(args, dict):
                state.tool_args[call_id] = args
            return [self._tool_call_event(state, event, status="running", output="")]

        if event_type == "tool_execution_update":
            output = self._extract_content_text(
                event.get("partialResult") if isinstance(event.get("partialResult"), dict) else {}
            )
            state.tool_output[str(event.get("toolCallId") or "")] = output
            return [self._tool_call_event(state, event, status="running", output=output)]

        if event_type == "tool_execution_end":
            state.in_tool_execution = False
            call_id = str(event.get("toolCallId") or "")
            output = self._extract_content_text(
                event.get("result") if isinstance(event.get("result"), dict) else {}
            )
            state.tool_output.pop(call_id, None)
            status = "error" if event.get("isError") else "completed"
            bridge_event = self._tool_call_event(state, event, status=status, output=output)
            state.tool_args.pop(call_id, None)
            return [bridge_event]

        if event_type == "turn_start":
            return [{"type": "step_start", "messageId": state.message_id}]

        if event_type == "turn_end":
            return self._on_turn_end(state, event)

        if event_type == "agent_end":
            # A retry or queued continuation may follow; agent_settled decides.
            return []

        if event_type in ("compaction_start", "compaction_end"):
            self._log.info(
                "bridge.pi_compaction",
                phase=event_type,
                detail=str(event.get("reason") or ""),
            )
            state.in_tool_execution = False
            return []

        if event_type == "auto_retry_start":
            self._log.warn(
                "bridge.pi_auto_retry",
                attempt=event.get("attempt"),
                detail=str(event.get("errorMessage") or ""),
            )
            return []

        if event_type == "auto_retry_end" and not event.get("success"):
            return self._error_event_once(state, str(event.get("finalError") or "Pi agent error"))

        if event_type == "extension_error":
            self._log.warn(
                "bridge.pi_extension_error",
                detail=str(event.get("error") or ""),
                extension=str(event.get("extensionPath") or ""),
            )
            return []

        if event_type == "response" and event.get("success") is False:
            # Un-correlated command failure surfaced by the transport.
            return self._error_event_once(state, str(event.get("error") or "Pi agent error"))

        return []

    def _on_message_update(
        self, state: _PromptState, event: dict[str, Any]
    ) -> list[dict[str, Any]]:
        delta_event = event.get("assistantMessageEvent")
        if not isinstance(delta_event, dict):
            return []

        delta_type = delta_event.get("type")
        block_key = f"{state.assistant_message_ordinal}:{delta_event.get('contentIndex', 0)}"

        if delta_type == "text_delta":
            delta = delta_event.get("delta")
            if not isinstance(delta, str) or not delta:
                return []
            state.cumulative_text[block_key] = state.cumulative_text.get(block_key, "") + delta
            return [
                {
                    "type": "token",
                    "content": state.cumulative_text[block_key],
                    "messageId": state.message_id,
                }
            ]

        if delta_type == "text_end":
            # Authoritative full block content — correct any missed deltas.
            content = delta_event.get("content")
            if (
                isinstance(content, str)
                and content
                and content != state.cumulative_text.get(block_key, "")
            ):
                state.cumulative_text[block_key] = content
                return [
                    {
                        "type": "token",
                        "content": content,
                        "messageId": state.message_id,
                    }
                ]
            return []

        if delta_type == "error":
            reason = str(delta_event.get("reason") or "")
            if reason == "aborted":
                # Stop/abort is reported through the bridge's cancel path.
                return []
            error_message = self._assistant_error_message(delta_event.get("partial")) or (
                "Pi agent error"
            )
            return self._error_event_once(state, error_message)

        return []

    def _on_message_end(self, state: _PromptState, event: dict[str, Any]) -> list[dict[str, Any]]:
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return []
        if message.get("stopReason") != "error":
            return []
        error_message = self._assistant_error_message(message) or "Pi agent error"
        return self._error_event_once(state, error_message)

    def _on_turn_end(self, state: _PromptState, event: dict[str, Any]) -> list[dict[str, Any]]:
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        cost = usage.get("cost") if isinstance(usage.get("cost"), dict) else {}

        tokens: dict[str, Any] | None = None
        if usage:
            tokens = {
                "input": usage.get("input", 0),
                "output": usage.get("output", 0),
                "reasoning": 0,
                "cache": {
                    "read": usage.get("cacheRead", 0),
                    "write": usage.get("cacheWrite", 0),
                },
            }

        return [
            {
                "type": "step_finish",
                "cost": cost.get("total"),
                "tokens": tokens,
                "reason": message.get("stopReason"),
                "messageId": state.message_id,
            }
        ]

    def _tool_call_event(
        self,
        state: _PromptState,
        event: dict[str, Any],
        *,
        status: str,
        output: str,
    ) -> dict[str, Any]:
        call_id = str(event.get("toolCallId") or "")
        args = event.get("args")
        if not isinstance(args, dict):
            args = state.tool_args.get(call_id, {})
        return {
            "type": "tool_call",
            "tool": str(event.get("toolName") or ""),
            "args": args,
            "callId": call_id,
            "status": status,
            "output": output,
            "messageId": state.message_id,
        }

    def _error_event_once(self, state: _PromptState, error_message: str) -> list[dict[str, Any]]:
        if error_message in state.emitted_error_messages:
            return []
        state.emitted_error_messages.add(error_message)
        self._log.error("bridge.pi_agent_error", error_msg=error_message)
        return [
            {
                "type": "error",
                "error": error_message,
                "messageId": state.message_id,
            }
        ]

    @staticmethod
    def _assistant_error_message(message: object) -> str | None:
        if not isinstance(message, dict):
            return None
        error_message = message.get("errorMessage") or message.get("error")
        return str(error_message) if error_message else None

    @staticmethod
    def _extract_content_text(result: dict[str, Any]) -> str:
        content = result.get("content")
        if not isinstance(content, list):
            return ""
        return "\n".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
