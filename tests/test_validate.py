"""Unit tests for Agent._validate_and_fix and GameMemory."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import MagicMock, patch


# We need to mock external dependencies before importing agent
with patch.dict("sys.modules", {"mistralai.client": MagicMock(), "mistralai": MagicMock()}):
    from agent import Agent, GameMemory, _dot, _parse_observation


class TestDot:
    """Tests for the _dot helper function."""

    def test_basic(self):
        """Verify dot product with non-zero values."""
        assert _dot([10, 20, 30], [1, 2, 3]) == 10 + 40 + 90

    def test_zeros(self):
        """Verify dot product with zero allocation returns zero."""
        assert _dot([10, 20, 30], [0, 0, 0]) == 0


class TestParseObservation:
    """Tests for JSON extraction from incoming messages."""

    def test_json_string(self):
        """Parse a plain JSON string."""
        obs = _parse_observation('{"action": "PROPOSE", "round_index": 1}')
        assert obs is not None
        assert obs["action"] == "PROPOSE"

    def test_json_in_code_block(self):
        """Parse JSON wrapped in a markdown code block."""
        obs = _parse_observation('```json\n{"action": "ACCEPT_OR_REJECT"}\n```')
        assert obs is not None
        assert obs["action"] == "ACCEPT_OR_REJECT"

    def test_invalid(self):
        """Return None for non-JSON input."""
        assert _parse_observation("not json at all") is None


class TestGameMemory:
    """Tests for the GameMemory class."""

    def test_initial_state(self):
        """Verify default values after construction."""
        mem = GameMemory(opp_key="test_opp")
        assert mem.opp_key == "test_opp"
        assert mem.my_offers == []
        assert mem.opp_offers == []
        assert mem.best_offer_value == 0

    def test_build_game_summary_timeout(self):
        """Verify game summary text for a timeout outcome."""
        mem = GameMemory(opp_key="tough")
        mem.valuations = [10, 20, 30]
        mem.quantities = [7, 4, 1]
        mem.batna = 50
        mem.round_index = 5
        mem.max_rounds = 5
        summary = mem.build_game_summary("timeout", 50.0)
        assert "timeout" in summary.lower()
        assert "tough" in summary
        assert "50" in summary

    def test_build_game_summary_agreement(self):
        """Verify game summary text for an agreement outcome."""
        mem = GameMemory(opp_key="soft")
        mem.valuations = [10, 20, 30]
        mem.quantities = [7, 4, 1]
        mem.batna = 50
        mem.round_index = 2
        mem.max_rounds = 5
        mem.my_offers = [([5, 2, 1], 120)]
        mem.opp_offers = [([3, 2, 0], 70)]
        summary = mem.build_game_summary("agreement", 120.0)
        assert "agreement" in summary.lower()
        assert "120" in summary
        assert "gain:" in summary.lower()


class TestValidateAndFix:
    """Tests for the M1-M5 validation and fix logic."""

    @pytest.fixture
    def agent(self):
        """Create an Agent with mocked Mistral client and preset game memory."""
        with patch.dict("sys.modules", {"mistralai.client": MagicMock(), "mistralai": MagicMock()}):
            with patch("agent.Mistral"):
                a = Agent()
                a.game_memory = GameMemory()
                a.game_memory.valuations = [10, 20, 30]
                a.game_memory.quantities = [7, 4, 1]
                a.game_memory.batna = 80
                a.game_memory.max_rounds = 5
                a.game_memory.round_index = 1
                return a

    def test_valid_proposal_passes(self, agent):
        """A valid proposal should pass through unchanged."""
        reply = json.dumps({"allocation_self": [5, 2, 1], "allocation_other": [2, 2, 0], "reason": "test"})
        result = agent._validate_and_fix(reply, {}, "PROPOSE")
        parsed = json.loads(result)
        assert parsed["allocation_self"] == [5, 2, 1]

    def test_m2_fix_below_batna(self, agent):
        """M2: proposal with value below BATNA should be fixed upward."""
        # value = 10*1 + 20*1 + 30*0 = 30 < BATNA(80)
        reply = json.dumps({"allocation_self": [1, 1, 0], "allocation_other": [6, 3, 1], "reason": "bad"})
        result = agent._validate_and_fix(reply, {}, "PROPOSE")
        parsed = json.loads(result)
        fixed_val = _dot(agent.game_memory.valuations, parsed["allocation_self"])
        assert fixed_val >= 80
        assert "M2 fix" in parsed.get("reason", "")

    def test_m1_fix_worse_than_previous(self, agent):
        """M1: proposal worse than previous best should be fixed."""
        agent.game_memory.my_offers = [([5, 3, 1], 140)]
        agent.game_memory.best_offer_value = 140
        # value = 10*5 + 20*2 + 30*0 = 90 < prev best(140)
        reply = json.dumps({"allocation_self": [5, 2, 0], "allocation_other": [2, 2, 1], "reason": "test"})
        result = agent._validate_and_fix(reply, {}, "PROPOSE")
        parsed = json.loads(result)
        fixed_val = _dot(agent.game_memory.valuations, parsed["allocation_self"])
        assert fixed_val >= 140
        assert "M1 fix" in parsed.get("reason", "")

    def test_m3_fix_all_items(self, agent):
        """M3: taking all items should be fixed to give at least one."""
        reply = json.dumps({"allocation_self": [7, 4, 1], "allocation_other": [0, 0, 0], "reason": "greedy"})
        result = agent._validate_and_fix(reply, {}, "PROPOSE")
        parsed = json.loads(result)
        assert sum(parsed["allocation_self"]) < sum(agent.game_memory.quantities)
        assert "M3 fix" in parsed.get("reason", "")

    def test_m3_fix_zero_items(self, agent):
        """M3: giving away all items (zero self) should be fixed."""
        # [0,0,0] triggers M2 first (value=0 < BATNA), which is correct — M2 catches it before M3
        # Test with a zero-items allocation that passes M2 by setting BATNA=0
        agent.game_memory.batna = 0
        reply = json.dumps({"allocation_self": [0, 0, 0], "allocation_other": [7, 4, 1], "reason": "generous"})
        result = agent._validate_and_fix(reply, {}, "PROPOSE")
        parsed = json.loads(result)
        assert sum(parsed["allocation_self"]) > 0
        assert "M3 fix" in parsed.get("reason", "")

    def test_m4_fix_accept_below_batna(self, agent):
        """M4: accepting an offer below BATNA should be overridden to reject."""
        reply = json.dumps({"accept": True, "reason": "bad deal"})
        obs = {"offer_value": 50, "batna_value": 80}
        result = agent._validate_and_fix(reply, obs, "ACCEPT_OR_REJECT")
        parsed = json.loads(result)
        assert parsed["accept"] is False
        assert "M4 fix" in parsed["reason"]

    def test_m5_fix_reject_on_last_round(self, agent):
        """M5: rejecting an above-BATNA offer on the last round should be overridden to accept."""
        agent.game_memory.round_index = 5
        agent.game_memory.max_rounds = 5
        reply = json.dumps({"accept": False, "reason": "want more"})
        obs = {"offer_value": 100, "batna_value": 80}
        result = agent._validate_and_fix(reply, obs, "ACCEPT_OR_REJECT")
        parsed = json.loads(result)
        assert parsed["accept"] is True
        assert "M5 fix" in parsed["reason"]

    def test_valid_accept_passes(self, agent):
        """Accepting an offer above BATNA should pass through unchanged."""
        reply = json.dumps({"accept": True, "reason": "good deal"})
        obs = {"offer_value": 150, "batna_value": 80}
        result = agent._validate_and_fix(reply, obs, "ACCEPT_OR_REJECT")
        parsed = json.loads(result)
        assert parsed["accept"] is True

    def test_valid_reject_passes(self, agent):
        """Rejecting an above-BATNA offer in a non-last round should pass through."""
        reply = json.dumps({"accept": False, "reason": "can do better"})
        obs = {"offer_value": 90, "batna_value": 80}
        result = agent._validate_and_fix(reply, obs, "ACCEPT_OR_REJECT")
        parsed = json.loads(result)
        assert parsed["accept"] is False

    def test_non_json_passes_through(self, agent):
        """Non-JSON replies should pass through without modification."""
        reply = "I don't know what to do"
        result = agent._validate_and_fix(reply, {}, "PROPOSE")
        assert result == reply
