# Bargaining Agent (Purple)

**AgentBeats Competition Submission: Purple Agent for Multi-Agent Negotiation**

A purple agent that negotiates item allocations in the AgentBeats bargaining meta-game. Uses LLM with chain-of-thought reasoning, code-level safety validation (M1-M5), and persistent per-opponent memory.

## Competition Context

This agent competes in the **AgentBeats x AgentX Competition 2025** bargaining scenario. A green evaluator agent (Meta-Game Bargaining Evaluator) pits purple agents against a pool of baseline strategies and RL-trained policies, then computes Maximum Entropy Nash Equilibrium to rank them.

- **Agent Type**: Purple (Challenger)
- **Domain**: Multi-agent negotiation / bargaining
- **LLM**: Mistral or OpenRouter (configurable via `LLM_PROVIDER`)
- **Protocol**: A2A (Agent-to-Agent)

### Game Rules

- 3 item types with quantities (7, 4, 1)
- Each player has private valuations and a private BATNA (fallback payoff)
- Players take turns proposing allocations or accepting/rejecting offers
- Value decays each round by a discount factor
- If no agreement is reached, both sides get only their BATNA

## How It Works

```
Green agent sends observation (JSON via A2A)
        |
        v
  _parse_observation()          -- extract JSON from message
        |
        v
  _update_memory()              -- update GameMemory + detect new game
        |
        v
  _build_situation()            -- build [SITUATION] block with numbers,
        |                          opponent history, and past lessons
        v
  Mistral LLM (chain-of-thought) -- <think>reasoning</think> then JSON
        |
        v
  _validate_and_fix()           -- enforce M1-M5 rules at code level
        |
        v
  JSON response to green agent
```

### Decision Flow

1. **Context enrichment** -- the agent builds a detailed `[SITUATION]` block with computed values: item worth, BATNA comparison, discount pressure, cheapest items to offer (for Nash Welfare), EF1 status of opponent's offer, opponent behavior patterns, and lessons from past games against this opponent.

2. **LLM reasoning** -- Mistral receives the observation plus the situation block and reasons step-by-step inside `<think>` tags before producing a JSON decision. The agent is explicitly instructed to maximize Nash Welfare and aim for EF1-compatible allocations.

3. **Safety validation** -- code checks the LLM output for M1-M5 violations and fixes them automatically. This is the last line of defense against arithmetic mistakes.

### M1-M5 Safety Rules

| Rule | Violation | Protection |
|------|-----------|------------|
| M1 (relaxed) | Offer more than 15% below previous best | Concessions up to 15% allowed to enable Nash Welfare improvement; hard floor is BATNA |
| M2 | Offer worth less than BATNA | Filter `my_val < batna`, fix via `_fix_proposal()` |
| M3 | Offer all items or zero items | Filter `sum == 0 or sum == total`, fix via `_fix_proposal()` |
| M4 | Accept offer worth less than BATNA | Override to reject |
| M5 | Walk away from offer above BATNA on last round | Override to accept |

The `_fix_proposal()` method finds the best valid allocation by brute-force search over all possible item splits, optimizing for Nash Welfare proxy: `sqrt(my_val) * sqrt(opponent_items + 1)`.

## Nash Welfare Strategy

The agent is optimized for the leaderboard metrics: Nash Welfare (NW), Nash Welfare above BATNA (NWA), and Envy-Freeness up to one item (EF1).

**Key principles:**

- **Give cheapest items first** -- items the agent values least may be very valuable to the opponent. The situation block always shows items sorted by value so the LLM can make informed offers.
- **Concessions are allowed** -- M1 is relaxed to allow up to 15% concessions below the previous best offer. This lets the agent move toward mutually beneficial deals rather than getting locked into greed.
- **EF1 awareness** -- the agent checks and reports EF1 status (envy-freeness up to one item) for both its proposals and incoming offers, guiding the LLM toward fairer splits.
- **Fallback optimization** -- when the LLM fails to produce valid JSON, `_fix_proposal()` selects the allocation that maximizes `sqrt(my_val) * sqrt(opponent_items + 1)` rather than simply minimizing distance to the previous offer.

## Memory System

### In-Game Memory (GameMemory)

Lives for one negotiation, resets on new game:

- `my_offers` / `opp_offers` -- full offer history with computed values
- `best_offer_value` -- highest value we've offered (for M1 enforcement)
- Round tracking, discount, BATNA

### Persistent Memory (Markdown files)

Stored in `memory/opponents/`, survives restarts. Each opponent gets a `.md` file with:

- **Summary** -- consolidated strategic insight (2-3 sentences)
- **Lessons** -- dated one-line takeaways from past games

After each game, the agent asks the LLM to extract one actionable lesson. When lessons exceed 5, the LLM consolidates them into an updated summary and keeps only the 2 most valuable.

This memory is injected into the `[SITUATION]` block so the agent adapts its strategy from the first move against a known opponent.

## Project Structure

```
src/
  agent.py       -- Core logic: GameMemory, Agent, chain-of-thought, M1-M5 validation
  memory.py      -- MarkdownMemory: per-opponent persistent memory
  executor.py    -- A2A request routing
  server.py      -- HTTP server, AgentCard configuration
  messenger.py   -- A2A messaging utilities
memory/
  opponents/     -- Markdown memory files (created automatically)
tests/
  test_agent.py      -- A2A integration tests
  test_validate.py   -- M1-M5 validation tests
  test_memory.py     -- Memory system tests
```

## Quick Start

### Run Locally

```bash
# Install dependencies
uv sync

# Configure API key
cp sample.env .env
# Edit .env: set LLM_PROVIDER and the corresponding API key
# For Mistral: MISTRAL_API_KEY=your_key_here
# For OpenRouter: LLM_PROVIDER=openrouter, OPENROUTER_API_KEY=your_key_here

# Start the agent server
uv run src/server.py --host 127.0.0.1 --port 9009

# Verify it's running
curl http://127.0.0.1:9009/.well-known/agent.json
```

### Run Tests

```bash
# Start the server first, then in another terminal:
uv run pytest -v --agent-url http://localhost:9009
```

### Docker

```bash
# Build
docker build --platform linux/amd64 -t ghcr.io/yuliaov22/meta-game-bargaining-agent-purple:v1.1 .

# Run
docker run -p 9009:9009 --env-file .env ghcr.io/yuliaov22/meta-game-bargaining-agent-purple:v1.0

# Push
docker push ghcr.io/yuliaov22/meta-game-bargaining-agent-purple:v1.1
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mistral` | LLM provider: `mistral` or `openrouter` |
| `MISTRAL_API_KEY` | -- | Mistral API key (required when `LLM_PROVIDER=mistral`) |
| `MISTRAL_MODEL` | `mistral-large-latest` | Mistral model to use |
| `OPENROUTER_API_KEY` | -- | OpenRouter API key (required when `LLM_PROVIDER=openrouter`) |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | OpenRouter model to use |
| `AGENT_MEMORY_DIR` | `./memory` | Directory for persistent memory files |
| `AGENT_LOGS_DIR` | `./logs` | Directory for game log files |

## Response Format

The agent communicates with the green evaluator via A2A protocol using JSON messages.

**Propose an allocation:**
```json
{"allocation_self": [5, 2, 1], "allocation_other": [2, 2, 0], "reason": "..."}
```

**Accept or reject an offer:**
```json
{"accept": true, "reason": "..."}
```

## License

Apache 2.0
