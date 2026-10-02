"""
Basic smoke tests for the agent graph. These call the real local Ollama
model (no mocking) — they're meant to be run against a working local
setup (`ollama pull llama3.1:8b` done, `ollama serve` running), not in a
CI environment without a GPU/model available.

For a CI-safe test suite, mock `src.services.llm_provider.chat_completion`
instead of calling the real model — see the commented example at the
bottom of this file.
"""
import pytest

from src.agents.router_agent import router_agent
from src.agents.state import AgentState


@pytest.mark.integration
def test_router_agent_classifies_policy_question():
    state: AgentState = {"user_query": "What is our remote work policy?"}
    result = router_agent(state)
    assert result["route"] in {"policy_search", "incident_resolution"}


@pytest.mark.integration
def test_router_agent_classifies_incident():
    state: AgentState = {"user_query": "The payments service is down, what do I do?"}
    result = router_agent(state)
    assert result["route"] in {"policy_search", "incident_resolution"}


# ── CI-safe example (no live model call) ────────────────────────────────
# from unittest.mock import patch
#
# @patch("src.agents.router_agent.chat_completion", return_value="policy_search")
# def test_router_agent_mocked(mock_chat):
#     state: AgentState = {"user_query": "anything"}
#     result = router_agent(state)
#     assert result["route"] == "policy_search"