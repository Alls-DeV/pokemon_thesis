# Project Context: PolimiBot Pokemon Showdown Agent

This document outlines the architecture and workflow of the Pokemon Showdown agent implemented in this repository, specifically focusing on `PolimiBot` and its integration with `GeminiPlayer`. 

**Note: LLM calls within `PolimiBot.choose_move()` are currently commented out to facilitate step-by-step debugging of prompt generation.**

## 1. GeminiPlayer (`@pokechamp/gemini_player.py`)

`GeminiPlayer` acts as the interface to the Google Gemini API.

- **Initialization**: Requires a Gemini API key (passed directly or via `GEMINI_API_KEY` env var). Automatically maps model names (like `gemini-2.0-flash`) to their official API string identifiers.
- **Core Methods**:
  - `get_LLM_action(system_prompt, user_prompt, ...)`: Submits combined prompts to the model. Features built-in JSON extraction logic: if `json_format=True`, it parses the response looking for curly braces to isolate and validate JSON output. It also tracks estimated prompt and completion tokens. Includes a retry mechanism (2-second sleep) on API failure.
  - `get_LLM_query(...)`: A simpler variant for querying that also supports naive JSON parsing.

## 2. PolimiBot (`@polimi/polimi_bot.py`)

`PolimiBot` extends the `poke_env` `Player` class and orchestrates the battle logic, state extraction, and LLM prompting.

### 2.1. Initialization and Resources
- Instantiates a `GeminiPlayer` when the `backend` argument contains `"gemini"`.
- Loads static game data (moves, abilities, items) from JSON files in `./poke_env/data/static/`.
- Initializes `PokemonPredictor` (a Bayesian predictor) to guess unknown opponent abilities, items, and moves.
- Loads team-specific strategy files from `polimi/strategies/team{idx}.json`.
- Maintains extensive hardcoded dictionaries to "denormalize" internal showdown IDs into readable names for the LLM (e.g., `"ironvaliant"` -> `"Iron Valiant"`, `"chillyreception"` -> `"Chilly Reception"`).

### 2.2. State Extraction & Prompt Generation
`PolimiBot` contains several methods to translate the `poke_env` battle state into natural language prompts:

- **`get_system_prompt()`**: Defines the agent's persona ("competitive Pokemon battler") and injects specific strategic advice for the active Pokemon based on the loaded strategy file.
- **`get_active_pokemon_prompt()`**: Details HP, abilities, items, moves, status, and Tera type. For the opponent's active Pokemon, if `enhanced=True`, it uses the `bayesian_predictor` to fill in unrevealed information with probabilities. It also calculates "turns to KO".
- **`turns_to_ko()`**: An external calculator integration. It spawns a Node.js subprocess (`js_damage/calc_turns.js`) utilizing `@smogon/calc` to accurately estimate how many turns a specific move will take to KO the opponent, factoring in stats, items, EVs, IVs, field effects, and statuses.
- **`get_side_conditions_prompt()`**: Explains active field hazards (Spikes, Stealth Rock, Sticky Web, Toxic Spikes).
- **`get_player_team_prompt()`**: Summarizes the remaining team, their moves, speed comparisons, and the opponent's fastest KO time against them.
- **`get_terastallization_prompt()`**: Explains Gen 9 Tera mechanics and the current Tera state of both active Pokemon.

### 2.3. Decision Making Flow (`choose_move`)

The `choose_move` method defines how the agent decides its next action. The workflow is designed as a multi-step LLM evaluation:

1. **Forced Switch**: If the player's active Pokemon is fainted, it builds a "Switch Prompt", asking the LLM to choose a replacement. (Returns a fallback random move if no switches are available or parsing fails).
2. **Move Only**: If the active Pokemon is alive but no switches are available, it builds a "Move Prompt" and asks the LLM to pick a move (and whether to Terastallize).
3. **Move vs. Switch (Merger Evaluation)**: If both moves and switches are available, the agent prepares *three* prompts:
   - A **Move Prompt** asking the LLM to explain and choose the best move.
   - A **Switch Prompt** asking the LLM to explain and choose the best switch.
   - A **Merger Prompt** that presents the LLM with the outputs of both the move and switch evaluations, asking it to act as a judge to decide which of the two actions is superior for winning the battle.
4. **Action Parsing**: The agent expects structured JSON responses from the LLM. It extracts the `"choice"`, `"move"`, or `"switch"` fields, then uses fuzzy string matching (`difflib.get_close_matches`) to map the LLM's text back to actual valid `BattleOrder` objects.

*Currently, the calls to `self.llm.get_LLM_action(...)` within `choose_move` are bypassed (commented out), and variables like `switch_response_raw` are hardcoded to empty strings, culminating in a fallback to `choose_random_move(battle)`.*