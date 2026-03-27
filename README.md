# Bargaining Agent (Purple)

**AgentBeats Competition Submission: Purple Agent for Multi-Agent Negotiation**

A purple agent that negotiates item allocations in the AgentBeats bargaining meta-game. Uses LLM with chain-of-thought reasoning, code-level safety validation (M1-M5), and persistent per-opponent memory.

## Competition Context

This agent competes in the **AgentBeats x AgentX Competition 2025** bargaining scenario. A green evaluator agent (Meta-Game Bargaining Evaluator) pits purple agents against a pool of baseline strategies and RL-trained policies, then computes Maximum Entropy Nash Equilibrium to rank them.

- **Agent Type**: Purple (Challenger)
- **Domain**: Multi-agent negotiation / bargaining
- **LLM**: Mistral Large (configurable)
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

1. **Context enrichment** -- the agent builds a detailed `[SITUATION]` block with computed values: item worth, BATNA comparison, discount pressure, opponent behavior patterns, and lessons from past games against this opponent.

2. **LLM reasoning** -- Mistral receives the observation plus the situation block and reasons step-by-step inside `<think>` tags before producing a JSON decision.

3. **Safety validation** -- code checks the LLM output for M1-M5 violations and fixes them automatically. This is the last line of defense against arithmetic mistakes.

### M1-M5 Safety Rules

| Rule | Violation | Protection |
|------|-----------|------------|
| M1 | Offer worse than your previous offer | Track `best_offer_value`, fix via `_fix_proposal()` |
| M2 | Offer worth less than BATNA | Filter `my_val < batna`, fix via `_fix_proposal()` |
| M3 | Offer all items or zero items | Filter `sum == 0 or sum == total`, fix via `_fix_proposal()` |
| M4 | Accept offer worth less than BATNA | Override to reject |
| M5 | Walk away from offer above BATNA on last round | Override to accept |

The `_fix_proposal()` method finds the closest valid allocation by brute-force search over all possible item splits.

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
# Edit .env: set MISTRAL_API_KEY=your_key_here

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
docker build --platform linux/amd64 -t ghcr.io/yuliaov22/meta-game-bargaining-agent-purple:v1.0 .

# Run
docker run -p 9009:9009 --env-file .env ghcr.io/yuliaov22/meta-game-bargaining-agent-purple:v1.0

# Push
docker push ghcr.io/yuliaov22/meta-game-bargaining-agent-purple:v1.0
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | -- | Mistral API key (required) |
| `MISTRAL_MODEL` | `mistral-large-latest` | Mistral model to use |
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
