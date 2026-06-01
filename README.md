# PolimiBot — LLM Pokémon Agent with Bayesian Opponent Modeling

A thesis project introducing **PolimiBot**: a competitive Gen 9 OU agent that combines a structured multi-step LLM decision loop with a Bayesian predictor for real-time opponent modeling.

---

## Architecture

```
pokemon_thesis/
├── polimi/
│   ├── polimi_bot.py          # PolimiBot agent (main contribution)
│   ├── teams/                 # Team files in Showdown format (team1–5.txt)
│   ├── teams_json/            # Team metadata: items, abilities, EVs, tera (team1–5.json)
│   └── strategies/            # Per-Pokémon strategic notes for the system prompt
├── bayesian/
│   ├── pokemon_predictor.py   # Singleton wrapper used by PolimiBot
│   ├── team_predictor.py      # BayesianTeamPredictor (core model)
│   ├── component_model.py     # Factorized per-species frequency tables
│   ├── smogon_prior.py        # Gen9OU Smogon usage-stats prior
│   ├── archetype_model.py     # LDA team-composition model (20 archetypes)
│   ├── evidence_updater.py    # Bayesian evidence conditioning
│   ├── damage_inference.py    # Backward EV inference from observed damage
│   └── train_full_predictor.py
├── pokechamp/                 # Upstream LLM player backends (GPT, Gemini, DeepSeek, OpenRouter)
├── poke_env/                  # Battle engine (forked from poke-env)
├── js_damage/
│   └── calc_turns.js          # @smogon/calc Node.js subprocess for KO-turn calc
├── local_1v1.py               # Main battle runner
└── tests/
```

---

## PolimiBot

`PolimiBot` (`polimi/polimi_bot.py`) extends `poke_env`'s `Player` and drives every battle decision through a three-stage LLM pipeline.

### Decision Loop

Each turn:

1. **Update damage evidence** — track opponent HP changes to feed EV-spread backward inference.
2. **Build context prompts** — assemble rich natural-language state descriptions.
3. **Run the 3-prompt flow** — Move sub-agent → Switch sub-agent → Merger judge.
4. **Parse and return** a `BattleOrder` via exact match + fuzzy `difflib` fallback.

### Context Prompts

Every LLM call receives a detailed state description covering:

| Block | Content |
|---|---|
| System prompt | Role + per-Pokémon strategy from `polimi/strategies/` |
| Side conditions | Hazards (Stealth Rock, Spikes, Toxic Spikes, Sticky Web) with HP costs on switch-in |
| Active Pokémon | HP %, type, ability, item, status, stat boosts, all moves with `[Type / Category / Power]` + KO-turn count |
| Opponent active | Same as above — unknown fields (ability, item, tera type, unrevealed moves) filled by the Bayesian predictor |
| Bench Pokémon | All non-active, non-fainted Pokémon on each side, full move lists, entry hazard costs |
| Terastallization | Availability, active tera type, predicted opponent tera |
| Speed order | Effective speed for both active Pokémon accounting for boosts, Choice Scarf, Iron Ball, paralysis, Swift Swim / Chlorophyll / Sand Rush / Slush Rush / Surge Surfer, Tailwind, Trick Room, Booster Energy / Protosynthesis / Quark Drive |

### Damage Calculator

`turns_to_ko(battle, move_id)` calls `js_damage/calc_turns.js` as a Node.js subprocess (`@smogon/calc`). It serialises both Pokémon with their EVs, nature, item, ability, boosts, HP fraction, and status, plus the full field state (weather, terrain, screens, rooms). The KO-turn count is embedded inline in every move description and drives the Merger judge's decision.

### 3-Prompt Decision Flow

```
Move sub-agent   →  { "move": "...", "terastallize": bool }
Switch sub-agent →  { "switch": "..." | "Nothing" }

if switch == "Nothing":
    execute move directly

else:
    Merger judge receives:
      - both sub-agent raw reasonings
      - pre-computed KO-race summary table (all move/switch combos)
    outputs: { "choice": "move" | "switch" }
```

The Merger prompt defaults to moving (switching grants the opponent a free attack) and requires a clear numerical KO-race advantage to choose a switch.

Special cases that bypass the full flow:
- **Fainted active Pokémon** → Switch sub-agent only (forced switch).
- **Pivot move** (U-turn / Volt Switch / Flip Turn) → Switch sub-agent only.
- **No switches available** → Move sub-agent only.

### Majority Voting

`--voting_n N` runs each sub-agent call N times in parallel. The most-voted `BattleOrder` (by action message string) is used. A majority of `"Nothing"` switch votes short-circuits to the move immediately, without calling the Merger.

---

## Bayesian Predictor

The predictor (`bayesian/`) infers unknown opponent fields — moves, item, ability, tera type, EV spread — from Gen 9 OU replay data and Smogon usage statistics. Its output is injected directly into the LLM prompt every turn: predicted moves fill the opponent's move list up to 4 slots (each with type, power, and KO-turn estimate), and the top-predicted item, ability, and tera type replace the corresponding "Unknown" fields in the opponent's description. This gives the LLM a complete picture of the opponent even before anything has been revealed in battle.

### Pipeline

```
predict_component_probabilities(species, teammates, observed_moves, ...)

  1. _fuse(species, component)
       replay data (ComponentModel) × Smogon prior (SmogonPrior)
       adaptive alpha: rare species shrink harder toward Smogon prior

  2. apply_all_evidence(...)
       + observed moves → re-weight via move co-occurrence (log-space)
       − negative evidence → attenuate infrequent co-movers (≤80%)
       confirmed item/ability/tera → zero out all other candidates

  3. backward_update(...)  [optional]
       observed HP drop + attacker stats → narrow EV-spread distribution

  4. return sorted (name, probability) lists for each component
```

### Components

**ComponentModel** (`component_model.py`) — factorized per-species `Counter` tables trained on ~140k Gen9OU replay teams:

| Table | What it stores |
|---|---|
| `move_counts[species]` | move usage frequency |
| `item_counts[species]` | item usage frequency |
| `ability_counts[species]` | ability usage frequency |
| `spread_counts[species]` | `"Nature:HP/Atk/Def/SpA/SpD/Spe"` frequency |
| `tera_counts[species]` | tera type frequency |
| `move_pair_counts[(species, move_a)][move_b]` | move co-occurrence |

All distributions use Dirichlet smoothing (α = 0.5) at query time.

**SmogonPrior** (`smogon_prior.py`) — loads `polimi/smogon_stats.json` and exposes normalized per-species distributions over moves, items, abilities, spreads, and tera types. The `"Other"` bucket is discarded before normalization.

**Adaptive Smogon Fusion** — for each `(species, component)`:

```
alpha = max(0.1, 1 − sqrt(replay_count / (replay_count + smogon_count × 0.05)))
fused[k] = alpha × smogon[k] + (1 − alpha) × replay[k]
```

**ArchetypeModel** (`archetype_model.py`) — `sklearn` LDA with 20 topics over ~140k teams (species = words, teams = documents). Given revealed Pokémon, infers archetype weights and marginalizes to predict unrevealed teammates without the double-counting problem of naive Bayes:

```
blended[sp] = 0.7 × archetype_score[sp] + 0.3 × base_usage[sp]
```

**EvidenceUpdater** (`evidence_updater.py`) — Bayesian conditioning given battle observations:
- Positive move evidence re-weights remaining moves via co-occurrence.
- Negative evidence attenuates moves that rarely appear alongside the observed set.
- Confirmed item / ability / tera zeroes out all other candidates.

**DamageInference** (`damage_inference.py`) — given an observed HP drop, attacker stats, and move name, re-weights the EV-spread distribution so spreads incompatible with the observed damage receive lower posterior weight.

### Training

```sh
uv run python bayesian/train_full_predictor.py
```

Processes all replay team files, builds the component model and archetype model, and caches to disk. Subsequent runs load from cache (version-stamped pickle).

---

## Quick Start

### Requirements

```sh
uv sync
cd js_damage && npm install   # for @smogon/calc damage calculator
```

### Run a Battle

Start the local Pokémon Showdown server first:

```sh
# In the pokemon-showdown repo:
node pokemon-showdown start --no-security
```

Then:

```sh
# PolimiBot (DeepSeek backend) vs Abyssal baseline
uv run python local_1v1.py --player_name polimi --opponent_name abyssal

# Different LLM backend
uv run python local_1v1.py --player_name polimi --player_backend anthropic/claude-3-5-sonnet --opponent_name abyssal

# With majority voting (3 parallel LLM calls per sub-agent)
uv run python local_1v1.py --player_name polimi --voting_n 3 --opponent_name abyssal
```

### LLM Backend Selection

Set `--player_backend` to any of:
- **DeepSeek**: `deepseek-v4-pro` (default), `deepseek-r1`
- **Gemini**: `gemini-2.5-flash`, `gemini-2.5-pro` → requires `GEMINI_API_KEY`
- **OpenRouter-routed** (`anthropic/`, `openai/`, `google/`, `meta/`, `mistralai/`, …) → requires `OPENROUTER_API_KEY`

### Ladder Battles

```sh
uv run python showdown_ladder.py --USERNAME $USERNAME --PASSWORD $PASSWORD
```

### Tests

```sh
uv run pytest tests/
uv run pytest tests/ -m bayesian      # Bayesian predictor tests
uv run pytest tests/ -m moves         # Move normalization tests
uv run pytest tests/ -m teamloader    # Team loading tests
```

---

## Other Bots

The battle runner discovers bots automatically from `bots/`. Built-in baselines:

| Name | Description |
|---|---|
| `abyssal` | Abyssal Bot (strong heuristic baseline) |
| `max_power` | Always uses highest base-power move |
| `random` | Random move selection |

---

## Acknowledgments

- [poke-env](https://github.com/hsahovic/poke-env)
- [PokéChamp](https://github.com/sethkarten/pokechamp)
