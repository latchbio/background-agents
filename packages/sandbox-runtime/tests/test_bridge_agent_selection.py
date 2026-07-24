"""
Unit tests for the bridge's agent-harness selection (opencode vs pi).
"""

from sandbox_runtime.bridge import AgentBridge
from sandbox_runtime.opencode_client import OpenCodeClient
from sandbox_runtime.pi_prompt_stream import PiPromptStream
from sandbox_runtime.pi_rpc import PiRpcClient
from sandbox_runtime.prompt_stream import OpenCodePromptStream


def make_bridge(**kwargs) -> AgentBridge:
    return AgentBridge(
        sandbox_id="sb-1",
        session_id="sess-1",
        control_plane_url="https://cp.example.com",
        auth_token="token",
        **kwargs,
    )


class TestAgentSelection:
    def test_defaults_to_opencode(self):
        bridge = make_bridge()
        assert bridge.agent == "opencode"
        assert isinstance(bridge.opencode_client, OpenCodeClient)
        assert isinstance(bridge._ensure_prompt_stream(), OpenCodePromptStream)
        assert bridge.session_id_file.name == "opencode-session-id"

    def test_pi_agent_uses_pi_client_and_stream(self):
        bridge = make_bridge(
            agent="pi",
            agent_model="anthropic/claude-sonnet-4-6",
            agent_workdir="/workspace/repo",
        )
        assert bridge.agent == "pi"
        assert isinstance(bridge.opencode_client, PiRpcClient)
        assert isinstance(bridge._ensure_prompt_stream(), PiPromptStream)
        assert bridge.session_id_file.name == "pi-session-id"

    def test_unknown_agent_falls_back_to_opencode(self):
        bridge = make_bridge(agent="clippy")
        assert bridge.agent == "opencode"
        assert isinstance(bridge.opencode_client, OpenCodeClient)

    def test_injected_client_wins_over_agent_selection(self):
        injected = PiRpcClient(log=None)  # type: ignore[arg-type]
        bridge = make_bridge(agent="pi", opencode_client=injected)
        assert bridge.opencode_client is injected

    def test_pi_stream_inherits_bridge_timeouts(self):
        bridge = make_bridge(agent="pi")
        stream = bridge._ensure_prompt_stream()
        assert isinstance(stream, PiPromptStream)
        assert stream._inactivity_timeout_seconds == bridge.sse_inactivity_timeout
        assert stream._prompt_max_duration_seconds == bridge.PROMPT_MAX_DURATION
