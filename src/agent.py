import itertools
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mistralai.client import Mistral
from openai import OpenAI as _OpenAIClient

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text

from memory import markdown_memory

load_dotenv()

# LLM provider selection: set LLM_PROVIDER=openrouter or LLM_PROVIDER=mistral (default)
_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "mistral").lower()

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(
    os.environ.get("AGENT_MEMORY_DIR", Path(__file__).parent.parent / "memory")
)
LOGS_DIR = Path(os.environ.get("AGENT_LOGS_DIR", Path(__file__).parent.parent / "logs"))
GAME_LOG_FILE = LOGS_DIR / "games.jsonl"

# Conversation history trimming thresholds
MAX_HISTORY_LENGTH = 20  # trim when history exceeds this
KEEP_HISTORY_LENGTH = 16  # keep this many recent messages after trim

SYSTEM_PROMPT = """You are a negotiator in the AgentBeats bargaining meta-game.
You negotiate over 3 item types with given quantities. You know your own valuations and BATNA, but NOT the opponent's.

## HARD RULES — violating these loses the game:
- M2: Never propose a deal worth less than your BATNA
- M3: Never offer ALL items or ZERO items to yourself
- M4: Never ACCEPT an offer worth less than your BATNA
- M5: Never WALK AWAY from an offer worth MORE than your BATNA on the last round

## GOAL: MAXIMIZE NASH WELFARE
Nash Welfare = sqrt(your_payoff × opponent_payoff). This is what the leaderboard measures.
A deal where you get 80 and opponent gets 80 beats one where you get 95 and opponent gets 30.
**KEY INSIGHT**: Items you value LEAST may be worth a LOT to the opponent.
Always give the opponent items you value least — this is free generosity that boosts Nash Welfare.

## CONCESSION STRATEGY
- You may make concessions (lower your own share) across rounds if it leads to a deal.
- Do NOT lock yourself into greed — a deal at 75% of max is better than no deal at BATNA.
- If opponent is generous, reciprocate to build mutual gain.
- If opponent is tough, hold your ground but stay above BATNA.

## GAME MECHANICS
- Each round, value decays by the discount factor. Waiting costs you.
- If no agreement by the last round, both sides get only their BATNA.
- You don't know the opponent's valuations — items you value least may be very valuable to them.

## MEMORY
You may receive past lessons from games against this opponent in the [SITUATION] block.
Study them carefully — they contain what worked and what failed against this specific opponent.
Adapt your strategy based on these lessons rather than guessing.

## YOUR TASK
You will receive a [SITUATION] block with numbers, opponent behavior, and past lessons.
Think step-by-step, then decide.

## RESPONSE FORMAT
**Part 1 — THINKING (inside <think> tags):**
Reason about:
1. What do I know about this opponent? (from current game behavior + past lessons)
2. My position: value of items, BATNA, rounds left, discount pressure
3. Which items are cheapest for me? Give those to the opponent first.
4. Nash Welfare tradeoff: what split maximizes sqrt(my_payoff × their_payoff)?
5. My decision and why

**Part 2 — DECISION (valid JSON after </think>):**
For PROPOSE: {"allocation_self": [x, y, z], "allocation_other": [a, b, c], "reason": "one-line"}
For ACCEPT_OR_REJECT: {"accept": true/false, "reason": "one-line"}

The JSON must come AFTER </think>. No markdown code blocks."""


class GameLogger:
    """
    Log each turn and game outcome to a JSONL file.

    Format: one JSON line per event. Event types:
    - "turn"  -- each turn (observation + response + analysis)
    - "game"  -- game outcome (payoff, rounds, opponent type, errors)

    File: logs/games.jsonl
    """

    def __init__(self, path: Path = GAME_LOG_FILE):
        """Initialize the logger with a file path and create parent directories."""
        self._path = path
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict[str, Any]) -> None:
        """Write a timestamped JSON record to the log file under a lock."""
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_turn(
        self,
        *,
        game_key: str,
        opp_key: str,
        round_index: int,
        action: str,
        observation: dict[str, Any],
        llm_reply: str,
        final_reply: str,
        was_fixed: bool,
        fix_reason: str = "",
        decision: dict[str, Any] | None = None,
        my_value: int | None = None,
        batna: int = 0,
        opp_type: str = "unknown",
        elapsed_ms: int = 0,
        model: str = "",
        thinking: str = "",
    ) -> None:
        """Log a single negotiation turn with observation, LLM response, and validation details."""
        self._write(
            {
                "event": "turn",
                "game_key": game_key,
                "opp_key": opp_key,
                "round": round_index,
                "action": action,
                "model": model,
                "thinking": thinking,
                "observation": {
                    "valuations_self": observation.get("valuations_self"),
                    "batna_self": observation.get("batna_self"),
                    "quantities": observation.get("quantities"),
                    "discount": observation.get("discount"),
                    "round_index": observation.get("round_index"),
                    "offer_value": observation.get("offer_value"),
                    "batna_value": observation.get("batna_value"),
                    "counter_value": observation.get("counter_value"),
                },
                "llm_reply": llm_reply,
                "final_reply": final_reply,
                "was_fixed": was_fixed,
                "fix_reason": fix_reason,
                "decision": decision,
                "my_value": my_value,
                "batna": batna,
                "opp_type": opp_type,
                "elapsed_ms": elapsed_ms,
            }
        )

    def log_game_end(
        self,
        *,
        game_key: str,
        opp_key: str,
        outcome: str,
        my_payoff: float,
        rounds_played: int,
        opp_type: str,
        my_offers: list[tuple[list[int], int]],
        opp_offers: list[tuple[list[int], int]],
        valuations: list[int],
        batna: int,
        quantities: list[int],
    ) -> None:
        """Log the final outcome of a game including payoff, offers, and opponent info."""
        self._write(
            {
                "event": "game",
                "game_key": game_key,
                "opp_key": opp_key,
                "outcome": outcome,
                "my_payoff": my_payoff,
                "rounds_played": rounds_played,
                "opp_type": opp_type,
                "my_offers": [{"alloc": a, "value": v} for a, v in my_offers],
                "opp_offers": [{"alloc": a, "value": v} for a, v in opp_offers],
                "valuations": valuations,
                "batna": batna,
                "quantities": quantities,
            }
        )


game_logger = GameLogger()


def _check_ef1(v: list[int], my_alloc: list[int], opp_alloc: list[int]) -> bool:
    """Check EF1: agent does not envy opponent up to removal of one item.

    Returns True if my_val >= opp_val - max_single_item_value_in_opp_alloc.
    """
    my_val = _dot(v, my_alloc)
    opp_val = _dot(v, opp_alloc)
    if my_val >= opp_val:
        return True
    max_removable = max(
        (v[i] for i in range(len(v)) if opp_alloc[i] > 0),
        default=0,
    )
    return my_val >= opp_val - max_removable


def _dot(v: list[int], a: list[int]) -> int:
    """Compute dot product of two vectors (valuations x allocation)."""
    return sum(vi * ai for vi, ai in zip(v, a, strict=True))


def _parse_observation(text: str) -> dict[str, Any] | None:
    """Extract JSON observation from an incoming message."""
    blocks = re.findall(
        r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    candidates = list(blocks) + [text]
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


class GameMemory:
    """In-game memory for a single negotiation: offer history and opponent analysis."""

    def __init__(self, opp_key: str = ""):
        """Initialize game memory with default values for a new negotiation."""
        self.opp_key: str = opp_key
        self.my_offers: list[tuple[list[int], int]] = []  # (allocation_self, value)
        self.opp_offers: list[tuple[list[int], int]] = (
            []
        )  # (allocation_to_me, value_for_me)
        self.round_index: int = 0
        self.opp_accept_count: int = 0
        self.opp_reject_count: int = 0
        self.valuations: list[int] = []
        self.batna: int = 0
        self.quantities: list[int] = []
        self.discount: float = 0.98
        self.max_rounds: int = 5
        self.best_offer_value: int = 0  # best offer value we've made
        self.fallback_rounds: list[int] = (
            []
        )  # rounds where LLM failed and code fallback was used

    def build_game_summary(self, outcome: str, my_payoff: float) -> str:
        """Builds a text summary of the game for the LLM to generate a lesson from."""
        total_value = (
            _dot(self.valuations, self.quantities)
            if self.valuations and self.quantities
            else 0
        )
        lines = [
            f"Opponent: {self.opp_key}",
            f"Outcome: {outcome}",
            f"My payoff: {my_payoff}, BATNA was: {self.batna}, total possible: {total_value}",
            f"Rounds played: {self.round_index} / {self.max_rounds}",
            f"Discount: {self.discount}",
        ]
        if self.my_offers:
            lines.append("My offers:")
            for i, (alloc, val) in enumerate(self.my_offers):
                lines.append(f"  Round {i+1}: {alloc} (value={val})")
        if self.opp_offers:
            lines.append("Opponent offers to me:")
            for i, (alloc, val) in enumerate(self.opp_offers):
                lines.append(f"  Round {i+1}: {alloc} (value for me={val})")
        lines.append(
            f"Opponent accepted {self.opp_accept_count}x, rejected {self.opp_reject_count}x"
        )
        if self.fallback_rounds:
            lines.append(
                f"WARNING: Rounds {self.fallback_rounds} used automatic code fallback (LLM failed to produce valid JSON). "
                f"Decisions in those rounds were NOT made by the agent — they were safe defaults based on arithmetic only. "
                f"Do not treat them as intentional strategy."
            )
        if outcome == "timeout":
            lines.append(
                "Game ended in timeout — no agreement reached, both got BATNA."
            )
        elif outcome == "agreement":
            lines.append(
                f"Deal was made. My payoff: {my_payoff} vs BATNA: {self.batna} (gain: {my_payoff - self.batna})."
            )
        return "\n".join(lines)

    def log_game_end(self, game_key: str, outcome: str, my_payoff: float) -> None:
        """Log game end to JSONL file."""
        if not self.opp_key:
            return
        game_logger.log_game_end(
            game_key=game_key,
            opp_key=self.opp_key,
            outcome=outcome,
            my_payoff=my_payoff,
            rounds_played=self.round_index,
            opp_type="llm",
            my_offers=list(self.my_offers),
            opp_offers=list(self.opp_offers),
            valuations=list(self.valuations),
            batna=self.batna,
            quantities=list(self.quantities),
        )


class Agent:
    """Bargaining agent that uses Mistral LLM with code-level M1-M5 safety validation."""

    def __init__(self):
        """Initialize the agent with LLM client, conversation state, and game memory."""
        if _LLM_PROVIDER == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable is required")
            self.client = _OpenAIClient(
                api_key=api_key, base_url="https://openrouter.ai/api/v1"
            )
            self.model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        else:
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable is required")
            self.client = Mistral(api_key=api_key)
            self.model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
        self.conversation_history: list[dict[str, str]] = []
        self.game_memory = GameMemory()
        self._last_game_key: str = ""
        self._last_thinking: str = ""
        self._game_saved: bool = False  # prevent double-saving game results

    def _extract_opp_key(self, obs: dict) -> str:
        """Extract opponent key from observation for persistent memory."""
        pair = obs.get("pair", "")
        # pair is typically "agent_a__vs__agent_b" -- extract opponent name
        if "__vs__" in pair:
            parts = pair.split("__vs__")
            role = obs.get("role", "row")
            # If we are row, opponent is second; if col, opponent is first
            opp_name = parts[1] if role == "row" else parts[0]
            return opp_name.strip()
        return pair

    def _detect_new_game(self, obs: dict) -> bool:
        """Detect whether a new game has started."""
        game_key = f"{obs.get('pair', '')}_{obs.get('game_index', '')}"
        if game_key != self._last_game_key:
            # Save previous game result if not already saved
            if (
                self._last_game_key
                and self.game_memory.valuations
                and not self._game_saved
            ):
                mem = self.game_memory
                if mem.opp_accept_count > 0:
                    outcome = "agreement"
                    payoff = float(mem.my_offers[-1][1]) if mem.my_offers else 0.0
                else:
                    outcome = "timeout"
                    payoff = float(mem.batna)
                self._save_lesson_to_memory(outcome, payoff)
                mem.log_game_end(self._last_game_key, outcome, payoff)

            self._last_game_key = game_key
            self._game_saved = False
            return True
        return False

    def _update_memory(self, obs: dict) -> None:
        """Update game memory from the observation."""
        if self._detect_new_game(obs):
            opp_key = self._extract_opp_key(obs)
            self.game_memory = GameMemory(opp_key=opp_key)
            self.conversation_history = []

        mem = self.game_memory
        mem.valuations = obs.get("valuations_self", mem.valuations)
        mem.batna = obs.get("batna_self", obs.get("batna_value", mem.batna))
        mem.quantities = obs.get("quantities", mem.quantities)
        mem.discount = obs.get("discount", mem.discount)
        mem.max_rounds = obs.get("max_rounds", mem.max_rounds)
        mem.round_index = obs.get("round_index", mem.round_index)

        # Record opponent's offer if present
        pending = obs.get("pending_offer", {})
        if pending and mem.valuations:
            opp_alloc_to_me = pending.get("offer_allocation_self")
            if opp_alloc_to_me and isinstance(opp_alloc_to_me, list):
                value_for_me = _dot(mem.valuations, opp_alloc_to_me)
                mem.opp_offers.append((opp_alloc_to_me, value_for_me))

    def _build_situation(self, obs: dict, action: str) -> str:
        """Build a [SITUATION] block with data for LLM chain-of-thought reasoning."""
        mem = self.game_memory
        lines = ["\n[SITUATION — use these facts for your reasoning]"]

        v = mem.valuations
        q = mem.quantities
        round_idx = mem.round_index
        max_rounds = mem.max_rounds
        remaining = max_rounds - round_idx

        # ── Position ──
        if v and q:
            total = _dot(v, q)
            lines.append("\n## My Position")
            lines.append(f"  Valuations per item type: {v}")
            lines.append(f"  Quantities: {q}")
            lines.append(f"  Total value if I keep everything: {total}")
            lines.append(f"  BATNA (fallback if no deal): {mem.batna}")
            lines.append(f"  Discount factor: {mem.discount}")
            lines.append(f"  Round: {round_idx} / {max_rounds} ({remaining} remaining)")
            per_item = ", ".join(f"type{i}={v[i]}" for i in range(len(v)))
            lines.append(f"  Value per item: {per_item}")
            # Nash Welfare hint: cheapest items to give away
            sorted_by_value = sorted(range(len(v)), key=lambda i: v[i])
            cheapest = [f"type{i}(val={v[i]})" for i in sorted_by_value]
            lines.append(
                f"  Items cheapest for me (offer these first): {', '.join(cheapest)}"
            )
            # Discount pressure
            if remaining <= 2:
                lines.append(
                    f"  !! TIME PRESSURE: only {remaining} rounds left, value decays to {mem.discount**remaining:.2f}x"
                )

        # ── Opponent: current game ──
        lines.append("\n## Opponent Behavior (this game)")
        if mem.opp_offers:
            lines.append(f"  Offers received from opponent: {len(mem.opp_offers)}")
            for i, (alloc, val) in enumerate(mem.opp_offers):
                lines.append(f"    Round {i+1}: gave me {alloc} (value for me = {val})")
            avg_gen = sum(val for _, val in mem.opp_offers) / len(mem.opp_offers)
            total_val = _dot(v, q) if v and q else 1
            lines.append(
                f"  Average generosity: {avg_gen:.0f} / {total_val} = {avg_gen/max(1,total_val):.0%}"
            )
        else:
            lines.append("  No offers from opponent yet.")
        lines.append(
            f"  Opponent accepted {mem.opp_accept_count}x, rejected {mem.opp_reject_count}x"
        )

        # ── Opponent: long-term memory (from Markdown file) ──
        if mem.opp_key:
            memory_text = markdown_memory.read_for_prompt(mem.opp_key)
            lines.append(f"\n{memory_text}")

        # ── My previous actions ──
        if mem.my_offers:
            lines.append("\n## My Previous Offers (this game)")
            for i, (alloc, val) in enumerate(mem.my_offers[-5:]):
                lines.append(f"  Offer {i+1}: {alloc} (value for me = {val})")
            lines.append(
                f"  Best offer value so far: {mem.best_offer_value} (M1: next must be >= this)"
            )

        # ── Action-specific context ──
        if action == "PROPOSE" and v and q:
            lines.append("\n## Your Task: PROPOSE an allocation")
            lines.append(
                f"  You must split {q} items into allocation_self + allocation_other = {q}"
            )
            lines.append(
                f"  Hard constraint: value >= BATNA ({mem.batna}), not all/nothing"
            )
            lines.append(
                f"  Soft constraint: value >= {int(mem.best_offer_value * 0.85)} (85% of prev best {mem.best_offer_value}) — concessions allowed for Nash Welfare"
            )
            lines.append(
                "  STRATEGY: give opponent items you value LEAST. Nash Welfare = sqrt(your_val * opp_val) — balance matters!"
            )
            lines.append(
                "  EF1 GOAL: ensure your_val >= opponent's_val_by_your_measure - max_single_item_you_value_in_their_share"
            )

        elif action == "ACCEPT_OR_REJECT":
            offer_value = obs.get("offer_value", 0)
            batna_value = obs.get("batna_value", mem.batna)
            counter_value = obs.get("counter_value", 0)
            lines.append("\n## Your Task: ACCEPT or REJECT the offer")
            lines.append(f"  Offer value for me: {offer_value}")
            lines.append(f"  My BATNA: {batna_value}")
            lines.append(
                f"  Ratio: offer/BATNA = {offer_value/max(1, batna_value):.2f}x"
            )
            lines.append(
                f"  Counter value (MY ideal, NOT guaranteed): ~{counter_value}"
            )
            lines.append(
                "  WARNING: counter_value is what I WANT, not what opponent will accept!"
            )
            lines.append(
                "  Realistic: opponent will likely offer much less than my counter."
            )
            lines.append(f"  Rounds remaining after this: {remaining}")
            if offer_value >= batna_value:
                margin = offer_value - batna_value
                ratio = offer_value / max(1, batna_value)
                lines.append(
                    f"  Offer is {margin} ABOVE BATNA ({ratio:.1f}x) — accepting is SAFE"
                )
                if ratio >= 1.2:
                    lines.append(
                        f"  STRONG OFFER: {ratio:.1f}x BATNA — seriously consider accepting"
                    )
                if remaining <= 2:
                    lines.append(
                        f"  !! URGENT: only {remaining} rounds left, accepting avoids timeout risk"
                    )
            else:
                deficit = batna_value - offer_value
                lines.append(
                    f"  Offer is {deficit} BELOW BATNA — M4 says you MUST reject"
                )
            if remaining <= 1:
                lines.append(
                    f"  !! LAST CHANCE: if you reject, game ends with BATNA={batna_value}"
                )
            # EF1 check on opponent's offer
            if mem.opp_offers and v and q:
                opp_alloc_to_me = mem.opp_offers[-1][0]
                opp_alloc_to_opp = [q[i] - opp_alloc_to_me[i] for i in range(len(q))]
                ef1_ok = _check_ef1(v, opp_alloc_to_me, opp_alloc_to_opp)
                lines.append(
                    f"  EF1 check on this offer: {'SATISFIED ✓' if ef1_ok else 'VIOLATED — you envy opponent too much'}"
                )

        if mem.fallback_rounds:
            lines.append(
                f"\n## WARNING: Automatic fallback was used in rounds {mem.fallback_rounds}"
            )
            lines.append(
                "  Those decisions were NOT yours — they were safe arithmetic defaults."
            )
            lines.append(
                "  You must produce valid JSON this time to regain control of the negotiation."
            )

        lines.append(
            "\n## Now think step-by-step in <think> tags, then give your JSON decision."
        )
        return "\n".join(lines)

    def _extract_json_from_cot(self, reply: str) -> str:
        """Extract JSON part from chain-of-thought response (after </think>).

        Handles cases where </think> is missing (truncated output) or JSON
        appears inside the think block.
        """
        self._last_thinking = ""

        # Case 1: <think>...</think> present — look for JSON after it
        think_match = re.search(r"<think>(.*?)</think>(.*)", reply, re.DOTALL)
        if think_match:
            self._last_thinking = think_match.group(1).strip()
            after_think = think_match.group(2).strip()
            parsed = _parse_observation(after_think)
            if parsed:
                return json.dumps(parsed)
            try:
                data = json.loads(after_think)
                if isinstance(data, dict):
                    return after_think
            except json.JSONDecodeError:
                pass
            # JSON might be inside the think block itself
            parsed = _parse_observation(self._last_thinking)
            if parsed:
                return json.dumps(parsed)

        # Case 2: <think> opened but never closed (truncated LLM output)
        open_match = re.search(r"<think>(.*)", reply, re.DOTALL)
        if open_match and not think_match:
            self._last_thinking = open_match.group(1).strip()
            # Search for JSON inside the truncated thinking
            parsed = _parse_observation(self._last_thinking)
            if parsed:
                return json.dumps(parsed)

        # Case 3: no think tags — search anywhere in the response
        parsed = _parse_observation(reply)
        if parsed:
            return json.dumps(parsed)

        try:
            data = json.loads(reply.strip())
            if isinstance(data, dict):
                return reply.strip()
        except json.JSONDecodeError:
            pass

        return reply

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        """Check if text is valid JSON dict."""
        try:
            data = json.loads(text.strip())
            return isinstance(data, dict)
        except (json.JSONDecodeError, ValueError):
            return False

    def _generate_fallback(self, obs: dict, action: str) -> str:
        """Generate a safe fallback response when LLM fails to produce valid JSON."""
        mem = self.game_memory
        v = mem.valuations
        q = mem.quantities

        if action == "PROPOSE" and v and q:
            # Use _fix_proposal logic to find a valid allocation
            target = max(mem.batna, mem.best_offer_value)
            dummy = {"reason": "fallback: LLM returned non-JSON"}
            return self._fix_proposal(dummy, v, q, mem.batna, target)

        elif action == "ACCEPT_OR_REJECT":
            offer_value = obs.get("offer_value", 0)
            batna_value = obs.get("batna_value", mem.batna)
            accept = offer_value >= batna_value
            return json.dumps(
                {
                    "accept": accept,
                    "reason": f"fallback: offer={offer_value} vs BATNA={batna_value}",
                }
            )

        return json.dumps({"accept": False, "reason": "fallback: unknown action"})

    def _chat_complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Call the configured LLM provider and return the response text."""
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if _LLM_PROVIDER in ("openai", "openrouter"):
            response = self.client.chat.completions.create(**kwargs)
        else:
            response = self.client.chat.complete(**kwargs)
        return response.choices[0].message.content or ""

    def _call_llm_with_retry(self, max_retries: int = 3) -> str:
        """Call LLM with retry and exponential backoff on 429 rate limit."""
        for attempt in range(max_retries):
            try:
                return self._chat_complete(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *self.conversation_history,
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate" in error_str.lower():
                    wait = 2**attempt  # 1s, 2s, 4s
                    time.sleep(wait)
                    continue
                raise
        # All retries exhausted
        raise RuntimeError(f"LLM rate limit exceeded after {max_retries} retries")

    def _save_lesson_to_memory(self, outcome: str, my_payoff: float) -> None:
        """Ask LLM to generate a lesson from the game and save it to markdown memory."""
        mem = self.game_memory
        if not mem.opp_key or not mem.valuations:
            return

        game_summary = mem.build_game_summary(outcome, my_payoff)
        lesson_prompt = markdown_memory.build_lesson_prompt(mem.opp_key, game_summary)

        try:
            lesson = self._chat_complete(
                messages=[
                    {
                        "role": "system",
                        "content": "You are analyzing a bargaining game to extract a lesson. Be concise and actionable.",
                    },
                    {"role": "user", "content": lesson_prompt},
                ],
                temperature=0.3,
            ).strip()
        except Exception:
            # If LLM fails, save a factual fallback lesson
            lesson = (
                f"{outcome.capitalize()}: payoff={my_payoff}, BATNA={mem.batna}, "
                f"rounds={mem.round_index}/{mem.max_rounds}, opponent={mem.opp_key}."
            )

        is_agreement = outcome == "agreement"
        needs_consolidation = markdown_memory.add_lesson_and_increment(
            mem.opp_key, lesson, is_agreement=is_agreement
        )

        if needs_consolidation:
            self._consolidate_memory(mem.opp_key)

    def _consolidate_memory(self, opp_key: str) -> None:
        """Ask LLM to consolidate lessons into a summary."""
        consolidation_prompt = markdown_memory.build_consolidation_prompt(opp_key)
        if not consolidation_prompt:
            return

        try:
            result = self._chat_complete(
                messages=[
                    {
                        "role": "system",
                        "content": "You are consolidating lessons from past bargaining games. Be concise.",
                    },
                    {"role": "user", "content": consolidation_prompt},
                ],
                temperature=0.2,
            ).strip()
            new_summary, kept_lessons = markdown_memory.parse_consolidation_response(
                result
            )
            if new_summary and kept_lessons:
                markdown_memory.apply_consolidation(opp_key, new_summary, kept_lessons)
        except Exception as e:
            logger.warning("Memory consolidation failed for %s: %s", opp_key, e)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process an incoming A2A message: parse observation, call LLM, validate, and respond."""
        input_text = get_message_text(message)
        t0 = time.monotonic()

        await updater.update_status(TaskState.working)

        # Parse observation
        obs = _parse_observation(input_text)
        action = ""
        if obs:
            self._update_memory(obs)
            action = obs.get("action", "")

        # Build situation context for chain-of-thought reasoning
        situation = self._build_situation(obs or {}, action) if obs else ""

        # Append situation context to the message for LLM
        enriched_input = input_text
        if situation:
            enriched_input = input_text + "\n" + situation

        self.conversation_history.append({"role": "user", "content": enriched_input})

        llm_reply = self._call_llm_with_retry()

        # Extract JSON from response (may be after </think>)
        json_reply = self._extract_json_from_cot(llm_reply)

        # If extraction failed (no valid JSON found), generate a safe fallback
        used_fallback = False
        if obs and not self._is_valid_json(json_reply):
            logger.warning(
                "LLM returned non-JSON, using code fallback for action=%s", action
            )
            json_reply = self._generate_fallback(obs, action)
            used_fallback = True
            self.game_memory.fallback_rounds.append(self.game_memory.round_index)

        # Validate response: check M1-M5 at code level
        final_reply = self._validate_and_fix(json_reply, obs or {}, action)
        was_fixed = final_reply != json_reply

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Log the turn
        if obs:
            mem = self.game_memory
            # Determine decision and my_value from final_reply
            parsed_reply = None
            try:
                parsed_reply = json.loads(final_reply)
            except Exception:
                pass

            my_value = None
            if parsed_reply and mem.valuations:
                alloc = parsed_reply.get("allocation_self")
                if alloc and isinstance(alloc, list):
                    try:
                        alloc_int = [int(x) for x in alloc]
                        my_value = _dot(mem.valuations, alloc_int)
                        # Track offers for memory and M1 validation
                        mem.my_offers.append((alloc_int, my_value))
                        mem.best_offer_value = max(mem.best_offer_value, my_value)
                    except (ValueError, TypeError):
                        pass

            game_logger.log_turn(
                game_key=self._last_game_key,
                opp_key=mem.opp_key,
                round_index=mem.round_index,
                action=action,
                observation=obs,
                llm_reply=llm_reply,
                final_reply=final_reply,
                was_fixed=was_fixed or used_fallback,
                fix_reason=(
                    "code fallback: LLM returned non-JSON"
                    if used_fallback
                    else (
                        parsed_reply.get("reason", "")
                        if was_fixed and parsed_reply
                        else ""
                    )
                ),
                decision=parsed_reply,
                my_value=my_value,
                batna=mem.batna,
                opp_type="llm",
                elapsed_ms=elapsed_ms,
                model=self.model,
                thinking=getattr(self, "_last_thinking", ""),
            )

        # Save game result on accept or last round
        if obs and self.game_memory.valuations:
            mem = self.game_memory
            is_accept = False
            if parsed_reply and isinstance(parsed_reply, dict):
                is_accept = parsed_reply.get("accept") is True
            is_last_round = mem.round_index >= mem.max_rounds
            if is_accept or is_last_round:
                outcome = (
                    "agreement" if is_accept or mem.opp_accept_count > 0 else "timeout"
                )
                if is_accept:
                    # We accepted opponent's offer — payoff is the offer_value
                    payoff = float(obs.get("offer_value", 0))
                elif mem.opp_accept_count > 0:
                    # Opponent accepted our offer — payoff is our last offer's value
                    payoff = float(mem.my_offers[-1][1]) if mem.my_offers else 0.0
                else:
                    # Timeout — both get BATNA
                    payoff = float(mem.batna)
                self._save_lesson_to_memory(outcome, payoff)
                mem.log_game_end(self._last_game_key, outcome, payoff)
                self._game_saved = True

        self.conversation_history.append({"role": "assistant", "content": final_reply})

        # Keep conversation history bounded
        if len(self.conversation_history) > MAX_HISTORY_LENGTH:
            self.conversation_history = self.conversation_history[-KEEP_HISTORY_LENGTH:]

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=final_reply))],
            name="Response",
        )
        # Log turn summary to stdout for debugging (not as artifact to avoid
        # polluting the A2A response that the green agent parses).
        summary = self._build_turn_summary(obs or {}, action, final_reply)
        if summary:
            print(summary)

    def _build_turn_summary(self, obs: dict, action: str, final_reply: str) -> str:
        """Build a human-readable summary of the turn: opponent offer, agent thinking, decision, and verdict."""
        if not obs:
            return ""

        mem = self.game_memory
        lines = []
        lines.append(f"=== Round {mem.round_index} / {mem.max_rounds} ===")

        # Opponent's offer
        pending = obs.get("pending_offer", {})
        if pending and action == "ACCEPT_OR_REJECT":
            opp_alloc = pending.get("offer_allocation_self")
            offer_value = obs.get("offer_value", "?")
            lines.append(
                f"\n[Opponent's offer] Allocation to me: {opp_alloc}, value for me: {offer_value}"
            )

        # Agent's thinking
        thinking = self._last_thinking
        if thinking:
            truncated = thinking[:500] + "..." if len(thinking) > 500 else thinking
            lines.append(f"\n[Agent thinking]\n{truncated}")

        # Agent's decision
        try:
            decision = json.loads(final_reply)
        except Exception:
            decision = None

        if decision and isinstance(decision, dict):
            if action == "PROPOSE":
                alloc_self = decision.get("allocation_self", "?")
                alloc_other = decision.get("allocation_other", "?")
                my_val = (
                    _dot(mem.valuations, alloc_self)
                    if mem.valuations and isinstance(alloc_self, list)
                    else "?"
                )
                reason = decision.get("reason", "")
                lines.append(
                    f"\n[Agent's proposal] Take: {alloc_self}, give: {alloc_other} (value for me: {my_val})"
                )
                if reason:
                    lines.append(f"  Reason: {reason}")
                lines.append(
                    f"\n>>> VERDICT: PROPOSE — offered {alloc_other} to opponent, keeping {alloc_self}"
                )

            elif action == "ACCEPT_OR_REJECT":
                accepted = decision.get("accept")
                reason = decision.get("reason", "")
                if accepted:
                    lines.append("\n[Agent's decision] ACCEPT")
                    lines.append(
                        f"\n>>> VERDICT: DEAL ACCEPTED — agent takes the offer (value: {obs.get('offer_value', '?')})"
                    )
                else:
                    lines.append("\n[Agent's decision] REJECT")
                    if reason:
                        lines.append(f"  Reason: {reason}")
                    lines.append(
                        "\n>>> VERDICT: OFFER REJECTED — negotiation continues"
                    )
        else:
            lines.append(f"\n[Agent reply] {final_reply[:200]}")

        return "\n".join(lines)

    def _validate_and_fix(self, reply: str, obs: dict, action: str) -> str:
        """Safety net: checks LLM response for M1-M5 violations and fixes them."""
        parsed = _parse_observation(reply)
        if not parsed:
            try:
                parsed = json.loads(reply.strip())
            except Exception:
                pass

        if not parsed or not isinstance(parsed, dict):
            return reply

        mem = self.game_memory
        v = mem.valuations
        q = mem.quantities

        if action == "PROPOSE" and v and q:
            alloc = parsed.get("allocation_self")
            if alloc and isinstance(alloc, list) and len(alloc) == len(v):
                try:
                    alloc_int = [int(x) for x in alloc]
                    my_val = _dot(v, alloc_int)
                except (ValueError, TypeError):
                    return reply

                # M2: value < BATNA
                if my_val < mem.batna:
                    parsed["reason"] = (
                        f"M2 fix: value {my_val} < BATNA {mem.batna}. "
                        + parsed.get("reason", "")
                    )
                    # Clamp: give ourselves more of the most valuable items
                    return self._fix_proposal(
                        parsed, v, q, mem.batna, mem.best_offer_value
                    )

                # M1 (relaxed): allow concessions up to 15% below best offer,
                # but never below BATNA. This enables Nash Welfare improvement
                # by letting the agent give more to the opponent strategically.
                if mem.my_offers and my_val < mem.best_offer_value:
                    concession_floor = max(
                        mem.batna,
                        int(mem.best_offer_value * 0.85),
                    )
                    if my_val < concession_floor:
                        parsed["reason"] = (
                            f"M1 fix: value {my_val} < floor {concession_floor}. "
                            + parsed.get("reason", "")
                        )
                        return self._fix_proposal(
                            parsed, v, q, mem.batna, concession_floor
                        )

                # M3: all or nothing
                if sum(alloc_int) == 0 or sum(alloc_int) == sum(q):
                    parsed["reason"] = "M3 fix: all or nothing. " + parsed.get(
                        "reason", ""
                    )
                    return self._fix_proposal(
                        parsed, v, q, mem.batna, mem.best_offer_value
                    )

        elif action == "ACCEPT_OR_REJECT":
            accept_field = parsed.get("accept")
            offer_value = obs.get("offer_value", 0)
            batna_value = obs.get("batna_value", mem.batna)

            # M4: accepting below BATNA
            if accept_field is True and offer_value < batna_value:
                return json.dumps(
                    {
                        "accept": False,
                        "reason": f"M4 fix: offer {offer_value} < BATNA {batna_value}",
                    }
                )

            # M5: rejecting above BATNA on last round
            if accept_field is False and offer_value > batna_value:
                remaining = mem.max_rounds - mem.round_index
                if remaining <= 0:
                    return json.dumps(
                        {
                            "accept": True,
                            "reason": f"M5 fix: last round, offer {offer_value} > BATNA {batna_value}",
                        }
                    )

        return reply

    def _fix_proposal(
        self, parsed: dict, v: list[int], q: list[int], batna: int, min_value: int
    ) -> str:
        """Fix an invalid proposal by finding a valid allocation that maximizes Nash Welfare.

        Among all allocations satisfying constraints (value >= target, not all/nothing),
        picks the one that maximizes sqrt(my_val * opponent_val) to boost Nash Welfare
        on the leaderboard, rather than simply minimizing distance to target.
        """
        target = max(batna, min_value)
        best_alloc = None
        best_nw = -1.0
        total_items = sum(q)

        ranges = [range(qi + 1) for qi in q]
        for combo in itertools.product(*ranges):
            a = list(combo)
            combo_sum = sum(a)
            if combo_sum == 0 or combo_sum == total_items:
                continue
            my_val = sum(vi * ai for vi, ai in zip(v, a))
            if my_val < target:
                continue
            opp_items = total_items - combo_sum
            # Nash Welfare proxy: balance my value against items given to opponent
            nw_proxy = (my_val**0.5) * (opp_items + 1) ** 0.5
            if nw_proxy > best_nw:
                best_nw = nw_proxy
                best_alloc = a

        if best_alloc is None:
            # Fallback: no allocation meets target without violating M3.
            # Lower the target to BATNA and try again; if still nothing,
            # give one unit of the least valuable item to the opponent.
            best_nw = -1.0
            for combo in itertools.product(*ranges):
                a = list(combo)
                combo_sum = sum(a)
                if combo_sum == 0 or combo_sum == total_items:
                    continue
                my_val = sum(vi * ai for vi, ai in zip(v, a))
                if my_val < batna:
                    continue
                opp_items = total_items - combo_sum
                nw_proxy = (my_val**0.5) * (opp_items + 1) ** 0.5
                if nw_proxy > best_nw:
                    best_nw = nw_proxy
                    best_alloc = a

        if best_alloc is None:
            # Last resort: take all but give 1 unit of the least valuable item
            min_idx = min(
                range(len(v)), key=lambda i: v[i] if q[i] > 0 else float("inf")
            )
            best_alloc = list(q)
            best_alloc[min_idx] = max(0, q[min_idx] - 1)

        a_other = [q[i] - best_alloc[i] for i in range(len(q))]
        parsed["allocation_self"] = best_alloc
        parsed["allocation_other"] = a_other
        return json.dumps(parsed)
