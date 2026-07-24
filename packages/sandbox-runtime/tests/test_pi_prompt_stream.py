"""
Unit tests for PiPromptStream: pi RPC event -> bridge event translation.

A fake in-memory client stands in for PiRpcClient; transport behavior is
covered by test_pi_rpc.py.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from sandbox_runtime.pi_prompt_stream import (
    PiPromptStream,
    split_provider_model,
    strip_pi_model_prefix,
)
from sandbox_runtime.pi_rpc import PROCESS_EXITED_EVENT

MESSAGE_ID = "msg-1"
SESSION_FILE = "/tmp/pi-sessions/session.jsonl"


class FakePiClient:
    """In-memory stand-in matching the PiRpcClient surface the stream uses."""

    def __init__(
        self,
        events: list[Any] | None = None,
        responses: dict[str, dict[str, Any]] | None = None,
    ):
        self.events = list(events or [])
        self.responses = responses or {}
        self.commands: list[dict[str, Any]] = []
        self.ensure_started_calls: list[str | None] = []
        self.drained = 0
        self.stop_reasons: list[str] = []

    async def ensure_started(self, *, resume_session_file: str | None = None) -> None:
        self.ensure_started_calls.append(resume_session_file)

    def drain_events(self) -> None:
        self.drained += 1

    async def command(
        self, cmd: dict[str, Any], *, timeout_seconds: float | None = None
    ) -> dict[str, Any]:
        self.commands.append(cmd)
        return self.responses.get(cmd["type"], {"type": "response", "success": True})

    async def next_event(self, *, timeout_seconds: float) -> dict[str, Any]:
        if not self.events:
            raise TimeoutError
        event = self.events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    async def request_stop(self, session_id: str | None, *, reason: str) -> bool:
        self.stop_reasons.append(reason)
        return True


def make_stream(client: FakePiClient, **overrides) -> PiPromptStream:
    kwargs = {
        "client": client,
        "log": MagicMock(),
        "inactivity_timeout_seconds": 120.0,
        "prompt_max_duration_seconds": 5400.0,
    }
    kwargs.update(overrides)
    return PiPromptStream(**kwargs)


async def collect(stream: PiPromptStream, **prompt_overrides) -> list[dict[str, Any]]:
    kwargs = {
        "opencode_session_id": SESSION_FILE,
        "message_id": MESSAGE_ID,
        "content": "do the thing",
    }
    kwargs.update(prompt_overrides)
    return [event async for event in stream.stream_prompt(**kwargs)]


def text_delta(delta: str, content_index: int = 0) -> dict[str, Any]:
    return {
        "type": "message_update",
        "message": {},
        "assistantMessageEvent": {
            "type": "text_delta",
            "contentIndex": content_index,
            "delta": delta,
        },
    }


SETTLED = {"type": "agent_settled"}


class TestHelpers:
    def test_strip_pi_model_prefix(self):
        assert strip_pi_model_prefix("pi/anthropic/claude-sonnet-4-6") == (
            "anthropic/claude-sonnet-4-6"
        )
        assert strip_pi_model_prefix("anthropic/claude-sonnet-4-6") == (
            "anthropic/claude-sonnet-4-6"
        )

    def test_split_provider_model(self):
        assert split_provider_model("pi/anthropic/claude-sonnet-4-6") == (
            "anthropic",
            "claude-sonnet-4-6",
        )
        assert split_provider_model("openai/gpt-5.4") == ("openai", "gpt-5.4")
        assert split_provider_model("claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")


class TestPromptDispatch:
    async def test_resumes_session_and_sends_prompt(self):
        client = FakePiClient(events=[SETTLED])
        await collect(make_stream(client))

        assert client.ensure_started_calls == [SESSION_FILE]
        assert client.drained == 1
        assert client.commands == [{"type": "prompt", "message": "do the thing"}]

    async def test_attachments_become_pi_images(self):
        client = FakePiClient(events=[SETTLED])
        attachments = [{"mimeType": "image/png", "content": "aGVsbG8="}]
        await collect(make_stream(client), attachments=attachments)

        assert client.commands[0]["images"] == [
            {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}
        ]

    async def test_rejected_prompt_raises(self):
        client = FakePiClient(
            events=[SETTLED],
            responses={"prompt": {"type": "response", "success": False, "error": "bad state"}},
        )
        with pytest.raises(RuntimeError, match="rejected prompt: bad state"):
            await collect(make_stream(client))

    async def test_streaming_rejection_retries_as_follow_up(self):
        client = FakePiClient(events=[SETTLED])
        rejections = [
            {"type": "response", "success": False, "error": "agent is streaming"},
            {"type": "response", "success": True},
        ]

        original_command = client.command

        async def command(cmd, *, timeout_seconds=None):
            if cmd["type"] == "prompt":
                client.commands.append(cmd)
                return rejections.pop(0)
            return await original_command(cmd, timeout_seconds=timeout_seconds)

        client.command = command  # type: ignore[method-assign]
        await collect(make_stream(client))

        assert client.commands[0] == {"type": "prompt", "message": "do the thing"}
        assert client.commands[1]["streamingBehavior"] == "followUp"


class TestModelConfiguration:
    async def test_sets_model_with_pi_prefix_stripped(self):
        client = FakePiClient(events=[SETTLED])
        await collect(make_stream(client), model="pi/anthropic/claude-sonnet-4-6")

        assert {
            "type": "set_model",
            "provider": "anthropic",
            "modelId": "claude-sonnet-4-6",
        } in client.commands

    async def test_maps_reasoning_effort_to_thinking_level(self):
        client = FakePiClient(events=[SETTLED])
        await collect(make_stream(client), model="pi/openai/gpt-5.4", reasoning_effort="none")

        assert {"type": "set_thinking_level", "level": "off"} in client.commands

    async def test_unchanged_model_is_configured_once(self):
        client = FakePiClient(events=[SETTLED, SETTLED])
        stream = make_stream(client)
        await collect(stream, model="pi/anthropic/claude-sonnet-4-6", reasoning_effort="high")
        client.events = [SETTLED]
        await collect(stream, model="pi/anthropic/claude-sonnet-4-6", reasoning_effort="high")

        assert len([c for c in client.commands if c["type"] == "set_model"]) == 1
        assert len([c for c in client.commands if c["type"] == "set_thinking_level"]) == 1

    async def test_set_model_failure_is_non_fatal(self):
        client = FakePiClient(
            events=[SETTLED],
            responses={"set_model": {"type": "response", "success": False, "error": "nope"}},
        )
        events = await collect(make_stream(client), model="pi/anthropic/claude-sonnet-4-6")
        assert events == []


class TestTextTranslation:
    async def test_text_deltas_become_cumulative_tokens(self):
        client = FakePiClient(events=[text_delta("Hello"), text_delta(" world"), SETTLED])
        events = await collect(make_stream(client))

        assert events == [
            {"type": "token", "content": "Hello", "messageId": MESSAGE_ID},
            {"type": "token", "content": "Hello world", "messageId": MESSAGE_ID},
        ]

    async def test_text_end_corrects_missed_deltas(self):
        client = FakePiClient(
            events=[
                text_delta("Hel"),
                {
                    "type": "message_update",
                    "message": {},
                    "assistantMessageEvent": {
                        "type": "text_end",
                        "contentIndex": 0,
                        "content": "Hello world",
                    },
                },
                SETTLED,
            ]
        )
        events = await collect(make_stream(client))

        assert events[-1] == {"type": "token", "content": "Hello world", "messageId": MESSAGE_ID}

    async def test_matching_text_end_is_not_re_emitted(self):
        client = FakePiClient(
            events=[
                text_delta("Hi"),
                {
                    "type": "message_update",
                    "message": {},
                    "assistantMessageEvent": {
                        "type": "text_end",
                        "contentIndex": 0,
                        "content": "Hi",
                    },
                },
                SETTLED,
            ]
        )
        events = await collect(make_stream(client))
        assert len([e for e in events if e["type"] == "token"]) == 1

    async def test_new_assistant_message_starts_a_new_text_block(self):
        client = FakePiClient(
            events=[
                {"type": "message_start", "message": {"role": "assistant"}},
                text_delta("first"),
                {"type": "message_start", "message": {"role": "assistant"}},
                text_delta("second"),
                SETTLED,
            ]
        )
        events = await collect(make_stream(client))

        assert [e["content"] for e in events if e["type"] == "token"] == ["first", "second"]


class TestToolTranslation:
    async def test_tool_lifecycle_maps_to_tool_call_events(self):
        client = FakePiClient(
            events=[
                {
                    "type": "tool_execution_start",
                    "toolCallId": "call-1",
                    "toolName": "bash",
                    "args": {"command": "ls"},
                },
                {
                    "type": "tool_execution_update",
                    "toolCallId": "call-1",
                    "toolName": "bash",
                    "args": {"command": "ls"},
                    "partialResult": {"content": [{"type": "text", "text": "partial"}]},
                },
                {
                    "type": "tool_execution_end",
                    "toolCallId": "call-1",
                    "toolName": "bash",
                    "result": {"content": [{"type": "text", "text": "full output"}]},
                    "isError": False,
                },
                SETTLED,
            ]
        )
        events = await collect(make_stream(client))

        assert events == [
            {
                "type": "tool_call",
                "tool": "bash",
                "args": {"command": "ls"},
                "callId": "call-1",
                "status": "running",
                "output": "",
                "messageId": MESSAGE_ID,
            },
            {
                "type": "tool_call",
                "tool": "bash",
                "args": {"command": "ls"},
                "callId": "call-1",
                "status": "running",
                "output": "partial",
                "messageId": MESSAGE_ID,
            },
            {
                "type": "tool_call",
                "tool": "bash",
                "args": {"command": "ls"},
                "callId": "call-1",
                "status": "completed",
                "output": "full output",
                "messageId": MESSAGE_ID,
            },
        ]

    async def test_failed_tool_maps_to_error_status(self):
        client = FakePiClient(
            events=[
                {
                    "type": "tool_execution_end",
                    "toolCallId": "call-2",
                    "toolName": "read",
                    "result": {"content": [{"type": "text", "text": "no such file"}]},
                    "isError": True,
                },
                SETTLED,
            ]
        )
        events = await collect(make_stream(client))
        assert events[0]["status"] == "error"
        assert events[0]["output"] == "no such file"


class TestTurnTranslation:
    async def test_turn_events_map_to_steps_with_usage(self):
        client = FakePiClient(
            events=[
                {"type": "turn_start"},
                {
                    "type": "turn_end",
                    "message": {
                        "role": "assistant",
                        "stopReason": "stop",
                        "usage": {
                            "input": 100,
                            "output": 50,
                            "cacheRead": 10,
                            "cacheWrite": 5,
                            "cost": {"total": 0.42},
                        },
                    },
                    "toolResults": [],
                },
                SETTLED,
            ]
        )
        events = await collect(make_stream(client))

        assert events == [
            {"type": "step_start", "messageId": MESSAGE_ID},
            {
                "type": "step_finish",
                "cost": 0.42,
                "tokens": {
                    "input": 100,
                    "output": 50,
                    "reasoning": 0,
                    "cache": {"read": 10, "write": 5},
                },
                "reason": "stop",
                "messageId": MESSAGE_ID,
            },
        ]


class TestErrors:
    async def test_assistant_error_stop_reason_emits_error_once(self):
        error_message = {
            "role": "assistant",
            "stopReason": "error",
            "errorMessage": "429 rate limited",
        }
        client = FakePiClient(
            events=[
                {"type": "message_end", "message": error_message},
                {"type": "message_end", "message": error_message},
                SETTLED,
            ]
        )
        events = await collect(make_stream(client))

        assert events == [{"type": "error", "error": "429 rate limited", "messageId": MESSAGE_ID}]

    async def test_aborted_stop_is_silent(self):
        client = FakePiClient(
            events=[
                {
                    "type": "message_update",
                    "message": {},
                    "assistantMessageEvent": {"type": "error", "reason": "aborted"},
                },
                SETTLED,
            ]
        )
        assert await collect(make_stream(client)) == []

    async def test_final_auto_retry_failure_emits_error(self):
        client = FakePiClient(
            events=[
                {"type": "auto_retry_end", "success": False, "attempt": 3, "finalError": "boom"},
                SETTLED,
            ]
        )
        events = await collect(make_stream(client))
        assert events == [{"type": "error", "error": "boom", "messageId": MESSAGE_ID}]

    async def test_process_exit_raises(self):
        client = FakePiClient(events=[{"type": PROCESS_EXITED_EVENT, "returncode": 3}])
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            await collect(make_stream(client))


class TestTimeouts:
    async def test_inactivity_timeout_stops_agent_and_raises(self):
        client = FakePiClient(events=[])  # next_event raises TimeoutError
        stream = make_stream(client, inactivity_timeout_seconds=0.01)

        with pytest.raises(RuntimeError, match="inactive"):
            await collect(stream)
        assert client.stop_reasons == ["inactivity_timeout"]

    async def test_max_duration_timeout_stops_agent_and_raises(self):
        client = FakePiClient(events=[text_delta("x")] * 50)
        stream = make_stream(client, prompt_max_duration_seconds=0.0)

        with pytest.raises(RuntimeError, match="max duration"):
            await collect(stream)
        assert client.stop_reasons == ["prompt_max_duration_timeout"]
