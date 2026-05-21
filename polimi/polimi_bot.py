import os
import re
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.local_simulation import LocalSim
from poke_env.player.player import Player
from pokechamp.gpt_player import GPTPlayer
from pokechamp.openrouter_player import OpenRouterPlayer
from pokechamp.gemini_player import GeminiPlayer
from pokechamp.deepseek_player import DeepSeekPlayer
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from bayesian.pokemon_predictor import PokemonPredictor
import string
import json
import subprocess
import shutil
import tempfile
import concurrent.futures
from pathlib import Path
from difflib import get_close_matches


class PolimiBot(Player):
    def __init__(
        self,
        battle_format,
        api_key="",
        backend="gpt-4-1106-preview",
        temperature=1.0,
        device=0,
        team=None,
        team_idx=1,
        account_configuration=None,
        server_configuration=None,
    ):
        super().__init__(
            battle_format=battle_format,
            team=team,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
        )
        self.api_key = api_key
        self.temperature = temperature
        self.team_idx = team_idx
        self.backend = backend
        if "gpt" in backend and not backend.startswith("openai/"):
            self.llm = GPTPlayer(api_key)
        elif "gemini" in backend:
            self.llm = GeminiPlayer(api_key)
        elif "deepseek" in backend and not backend.startswith("deepseek-ai/"):
            self.llm = DeepSeekPlayer(api_key)
        elif backend.startswith(
            (
                "openai/",
                "anthropic/",
                "google/",
                "meta/",
                "mistral/",
                "cohere/",
                "perplexity/",
                "deepseek/",
                "microsoft/",
                "nvidia/",
                "huggingface/",
                "together/",
                "replicate/",
                "fireworks/",
                "localai/",
                "vllm/",
                "sagemaker/",
                "vertex/",
                "bedrock/",
                "azure/",
                "custom/",
            )
        ):
            # OpenRouter supports hundreds of models from various providers
            self.llm = OpenRouterPlayer(api_key)
        else:
            # raise NotImplementedError('LLM type not implemented:', backend)
            self.llm = None
            print("No LLM will be used.")
        # Move effects and Pokemon move mappings
        try:
            with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
                self.move_effect = json.load(f)
        except FileNotFoundError:
            print("[WARNING]: moves_effect.json not found, using empty dict")
            self.move_effect = {}

        # Ability effects and Pokemon ability mappings
        try:
            with open("./poke_env/data/static/abilities/ability_effect.json", "r") as f:
                self.ability_effect = json.load(f)
        except FileNotFoundError:
            print("[WARNING]: ability_effect.json not found, using empty dict")
            self.ability_effect = {}

        # Item effects
        try:
            with open("./poke_env/data/static/items/item_effect.json", "r") as f:
                self.item_effect = json.load(f)
        except FileNotFoundError:
            print("[WARNING]: item_effect.json not found, using empty dict")
            self.item_effect = {}

        # Initialize Bayesian predictor
        try:
            self.bayesian_predictor = PokemonPredictor()
            print("✅ Bayesian predictor initialized successfully!")
        except Exception as e:
            print(f"[WARNING]: Failed to initialize Bayesian predictor: {e}")
            self.bayesian_predictor = None

        self.set_team_idx(self.team_idx)

        # Create name mapping for common Pokemon
        self.name_mapping = {
            "slowkinggalar": "Slowking-Galar",
            "slowbrogalar": "Slowbro-Galar",
            "tinglu": "Ting-Lu",
            "chiyu": "Chi-Yu",
            "wochien": "Wo-Chien",
            "chienpao": "Chien-Pao",
            "ironmoth": "Iron Moth",
            "ironvaliant": "Iron Valiant",
            "irontreads": "Iron Treads",
            "ironbundle": "Iron Bundle",
            "ironhands": "Iron Hands",
            "ironjugulis": "Iron Jugulis",
            "ironthorns": "Iron Thorns",
            "ironboulder": "Iron Boulder",
            "ironcrown": "Iron Crown",
            "greattusk": "Great Tusk",
            "screamtail": "Scream Tail",
            "brutebonnet": "Brute Bonnet",
            "fluttermane": "Flutter Mane",
            "slitherwing": "Slither Wing",
            "sandyshocks": "Sandy Shocks",
            "roaringmoon": "Roaring Moon",
            "walkingwake": "Walking Wake",
            "ragingbolt": "Raging Bolt",
            "gougingfire": "Gouging Fire",
            "ogerponwellspring": "Ogerpon-Wellspring",
            "ogerponhearthflame": "Ogerpon-Hearthflame",
            "ogerponcornerstone": "Ogerpon-Cornerstone",
            "ogerponteal": "Ogerpon",
            "ogerpontealtera": "Ogerpon",
            "ursalunabloodmoon": "Ursaluna",
            "ninetalesalola": "Ninetales-Alola",
            "sandslashalola": "Sandslash-Alola",
            "tapukoko": "Zapdos",
            "tapulele": "Clefable",
            "tapubulu": "Zapdos",
            "tapufini": "Primarina",
            "hydrapple": "Hydrapple",
            "zapdos": "Zapdos",
            "zamazenta": "Zamazenta",
            "tinkaton": "Tinkaton",
            "hoopaunbound": "Hoopa-Unbound",
            "mausholdfour": "Maushold-Four",
            "polteageistantique": "Polteageist-Antique",
            "deoxysspeed": "Deoxys-Speed",
            "deoxysdefense": "Deoxys-Defense",
            "deoxysattack": "Deoxys-Attack",
            "goodrahisui": "Goodra-Hisui",
            "kommoo": "Kommo-o",
            "landorustherian": "Landorus-Therian",
            "moltresgalar": "Moltres-Galar",
            "porygonz": "Porygon-Z",
            "rotomwash": "Rotom-Wash",
            "rotomheat": "Rotom-Heat",
            "rotomfrost": "Rotom-Frost",
            "rotomfan": "Rotom-Fan",
            "rotommow": "Rotom-Mow",
            "samurotthisui": "Samurott-Hisui",
            "thundurustherian": "Thundurus-Therian",
            "tornadustherian": "Tornadus-Therian",
            "weezinggalar": "Weezing-Galar",
            "zapdosgalar": "Zapdos-Galar",
            "arcaninehisui": "Arcanine-Hisui",
            "braviaryhisui": "Braviary-Hisui",
            "enamorustherian": "Enamorus-Therian",
            "lilliganthisui": "Lilligant-Hisui",
            "sneaselhisui": "Sneasel-Hisui",
            "taurospaldeablaze": "Tauros-Paldea-Blaze",
            "zarudedada": "Zarude-Dada",
            "zoroarkhisui": "Zoroark-Hisui",
            "decidueyehisui": "Decidueye-Hisui",
            "mimikyubusted": "Mimikyu",
            "miniormeteor": "Minior",
            "morpekohangry": "Morpeko",
            "eiscuenoice": "Eiscue",
            "cramorantgulping": "Cramorant",
            "cramorantgorging": "Cramorant",
            "sawsbucksummer": "Sawsbuck",
            "sawsbuckautumn": "Sawsbuck",
            "sawsbuckwinter": "Sawsbuck",
            # Additional gen9ou Pokemon normalizations
            "basculegionf": "Basculegion-F",
            "basculegionm": "Basculegion",
            "mukalola": "Muk-Alola",
            "raichualola": "Raichu-Alola",
            "golemalaola": "Golem-Alola",
            "magnetonalola": "Magnezone",
            "dugtrioalola": "Dugtrio-Alola",
            "grimerala": "Grimer-Alola",
            "marrowaka": "Marowak-Alola",
            "exeggutoralola": "Exeggutor-Alola",
            "vulpixalola": "Vulpix-Alola",
            "persianalola": "Persian-Alola",
            "meowthala": "Meowth-Alola",
            "rattataalola": "Rattata-Alola",
            "raticatealola": "Raticate-Alola",
            "geodudealola": "Geodude-Alola",
            "graveleralola": "Graveler-Alola",
            "magnemitealola": "Magnemite",
            "magnetonalola": "Magneton",
            # More gender/form variants (map to common gen9ou Pokemon)
            "indeedf": "Clefable",
            "indeedeem": "Indeedee",  # Female form maps to common psychic type
            "indeedeef": "Clefable",  # Internal Pokemon constructor name for Indeedee-F
            "meowsticf": "Meowstic",
            "meowsticm": "Meowstic",  # Female form maps to base Meowstic
            "unfezantf": "Staraptor",
            "unfezantm": "Staraptor",  # Map to common Normal/Flying type
            # Paldean forms (map to common gen9ou Pokemon)
            "taurospaldeaaqua": "Tauros-Paldea-Aqua",
            "taurospaldeacombat": "Tauros-Paldea-Aqua",
            "wooperpalda": "Wooper-Paldea",
            "clodsirepalda": "Clodsire",
            # Hisuian forms that might be missed (map to known Hisuian forms or common Pokemon)
            "voltorbhisui": "Electrode-Hisui",
            "electrodehisui": "Electrode-Hisui",  # Map to known Hisui form
            "typhloshionhisui": "Typhlosion-Hisui",
            "typhlosionhisui": "Typhlosion-Hisui",  # Both variants
            "qwilfishhisui": "Overqwil",  # Map to known evolution
            "growlithehisui": "Arcanine-Hisui",
            "sneaslerhisui": "Sneasler",
            "overqwilhisui": "Overqwil",
            "kleavohisui": "Kleavor",
            "basculinhisui": "Basculegion",
            "basculinwhitestriped": "Basculegion",  # Map to known evolution
            # Additional common variants (map to common rock types)
            "lycanrocmidday": "Lycanroc",
            "lycanrocmidnight": "Lycanroc-Dusk",
            "lycanrocdusk": "Lycanroc-Dusk",
            "oricoriomeadow": "Kilowattrel",
            "oricoriopompom": "Kilowattrel",  # Map to common Electric/Flying
            "oricoriopau": "Kilowattrel",
            "oricoriosensu": "Kilowattrel",  # All Oricorio forms to common similar type
            "toxapexgmax": "Toxapex",
            "corviknightgmax": "Corviknight",
            "grimmsnarrgmax": "Grimmsnarl",
        }

    def set_team_idx(self, team_idx: int):
        self.team_idx = team_idx
        strategy_file = f"polimi/strategies/team{self.team_idx}.json"
        try:
            with open(strategy_file, "r") as f:
                self.strategy_data = json.load(f)
        except FileNotFoundError:
            print(f"[WARNING]: Strategy file {strategy_file} not found, using empty dict")
            self.strategy_data = {}

        team_json_file = f"polimi/teams_json/team{self.team_idx}.json"
        try:
            with open(team_json_file, "r") as f:
                self.team_json_data = json.load(f)
        except FileNotFoundError:
            print(f"[WARNING]: Team JSON file {team_json_file} not found, using empty dict")
            self.team_json_data = {}

    def check_status(self, status):
        if status:
            if status.value == 1:
                return "burnt"
            elif status.value == 2:
                return "fainted"
            elif status.value == 3:
                return "frozen"
            elif status.value == 4:
                return "paralyzed"
            elif status.value == 5:
                return "poisoned"
            elif status.value == 7:
                return "toxic"
            elif status.value == 6:
                return "sleeping"
        else:
            return ""

    def denormalize_pokemon_name(self, name: str) -> str:
        """Denormalize Pokemon name from battle format to training data format."""
        # First check if it's in our mapping
        lower_name = name.lower()
        if lower_name in self.name_mapping:
            return self.name_mapping[lower_name]

        # Otherwise, capitalize first letter of each word part
        # Handle special cases like "mr-mime" -> "Mr. Mime"
        if lower_name == "mrmime":
            return "Mr. Mime"
        elif lower_name == "mimejr":
            return "Mime Jr."
        elif lower_name == "typenull":
            return "Type: Null"
        elif lower_name == "hooh":
            return "Ho-Oh"
        elif lower_name == "porygonz":
            return "Porygon-Z"
        elif lower_name == "porygon2":
            return "Porygon2"

        # Default: capitalize first letter
        return name.capitalize()

    def denormalize_move_name(self, move_id: str) -> str:
        """Denormalize move name from battle format to training data format."""
        # Common move name transformations
        move_mapping = {
            "chillyreception": "Chilly Reception",
            "thunderwave": "Thunder Wave",
            "stealthrock": "Stealth Rock",
            "earthquake": "Earthquake",
            "ruination": "Ruination",
            "whirlwind": "Whirlwind",
            "spikes": "Spikes",
            "rest": "Rest",
            "closecombat": "Close Combat",
            "crunch": "Crunch",
            "gigadrain": "Giga Drain",
            "earthpower": "Earth Power",
            "nastyplot": "Nasty Plot",
            "ficklebeam": "Fickle Beam",
            "leafstorm": "Leaf Storm",
            "dracometeor": "Draco Meteor",
            "futuresight": "Future Sight",
            "sludgebomb": "Sludge Bomb",
            "psychicnoise": "Psychic Noise",
            "flamethrower": "Flamethrower",
            "gigatonhammer": "Gigaton Hammer",
            "encore": "Encore",
            "knockoff": "Knock Off",
            "playrough": "Play Rough",
        }

        # Check if we have a direct mapping
        lower_move = move_id.lower()
        if lower_move in move_mapping:
            return move_mapping[lower_move]

        # Default: capitalize first letter and add spaces before capital letters
        # Convert "iceBeam" to "Ice Beam"
        # Add space before capital letters that follow lowercase letters
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", move_id)
        return spaced.title()

    def get_active_pokemon_prompt(
        self, battle: AbstractBattle, opponent: bool, enhanced: bool
    ) -> str:
        """Create a detailed prompt describing the active Pokemon."""
        # Get the active Pokemon
        if opponent:
            active_mon = battle.opponent_active_pokemon
            prefix = "opponent"
        else:
            active_mon = battle.active_pokemon
            prefix = "your"

        if not active_mon:
            return (
                f"Information about {prefix} active pokemon: No active Pokemon found."
            )

        # Basic information
        species = active_mon.species
        hp_percentage = round(active_mon.current_hp_fraction * 100, 1)

        # Get ability information
        ability_name = active_mon.ability if active_mon.ability else "Unknown"
        ability_effect = ""
        if active_mon.ability:
            try:
                ability_name = self.ability_effect[active_mon.ability]["name"]
                ability_effect = (
                    f" ({self.ability_effect[active_mon.ability]['effect']})"
                )
            except:
                ability_name = active_mon.ability
                ability_effect = ""

        # Get item information
        item_name = "unknown_item"
        item_effect = ""
        if active_mon.item:
            try:
                item_name = self.item_effect[active_mon.item]["name"]
                item_effect = f" ({self.item_effect[active_mon.item]['effect']})"
            except:
                item_name = active_mon.item

        # Get status
        status = self.check_status(active_mon.status)

        # Get moves information
        moves_info = []
        known_moves = []

        # Collect known moves
        # For player's pokemon, use battle.available_moves (current turn available moves)
        # For opponent's pokemon, use active_mon.moves (revealed moves)
        if not opponent and battle.available_moves:
            # Player's active pokemon - use available moves for this turn
            for move in battle.available_moves:
                if move:
                    move_name = self.denormalize_move_name(move.id)
                    try:
                        move_explanation = self.move_effect[move.id]
                    except:
                        move_explanation = ""
                    # Compute turns to KO using this move
                    ko_turns = None
                    try:
                        ko_result = self.turns_to_ko(
                            battle, move.id, attacker_is_opponent=opponent
                        )
                        if isinstance(ko_result, int):
                            ko_turns = ko_result
                    except Exception:
                        ko_turns = None
                    ko_str = (
                        f" ({ko_turns} turns to KO opponent's pokemon)"
                        if ko_turns is not None
                        else ""
                    )
                    moves_info.append(f"  * {move_name}: {move_explanation}{ko_str}")
                    known_moves.append(move.id)
        elif hasattr(active_mon, "moves") and active_mon.moves:
            # Opponent's pokemon - use revealed moves
            for move in active_mon.moves.values():
                if move:
                    move_name = self.denormalize_move_name(move.id)
                    try:
                        move_explanation = self.move_effect[move.id]
                    except:
                        move_explanation = ""
                    # Compute turns to KO using this move
                    ko_turns = None
                    try:
                        ko_result = self.turns_to_ko(
                            battle, move.id, attacker_is_opponent=opponent
                        )
                        if isinstance(ko_result, int):
                            ko_turns = ko_result
                    except Exception:
                        ko_turns = None
                    ko_str = (
                        f" ({ko_turns} turns to KO your active pokemon)"
                        if ko_turns is not None
                        else ""
                    )
                    moves_info.append(f"  * {move_name}: {move_explanation}{ko_str}")
                    known_moves.append(move.id)

        # Get tera type
        tera_type = "Unknown"
        if (
            hasattr(active_mon, "_terastallized_type")
            and active_mon._terastallized_type
        ):
            tera_type = active_mon._terastallized_type.name.capitalize() if hasattr(active_mon._terastallized_type, 'name') else str(active_mon._terastallized_type)
        elif hasattr(active_mon, "terastallized") and active_mon.terastallized:
            tera_type = f"Active (unknown type)"
            if not opponent and self.team_json_data and self.denormalize_pokemon_name(species) in self.team_json_data:
                tera_type = self.team_json_data[self.denormalize_pokemon_name(species)].get("tera", tera_type)
        elif not opponent and self.team_json_data and self.denormalize_pokemon_name(species) in self.team_json_data:
            tera_type = self.team_json_data[self.denormalize_pokemon_name(species)].get("tera", "Unknown")

        # Enhanced predictions if requested and available
        if enhanced and self.bayesian_predictor and opponent:
            # TODO: maybe we should leave the try to avoid crashes
            # try:
            # Denormalize names for predictor
            # TODO: change from normalize to denormalize in the functions
            normalized_species = self.denormalize_pokemon_name(species)

            # Get teammate names
            revealed_opponents = []
            for pokemon in battle.opponent_team.values():
                if pokemon.species:
                    normalized_name = self.denormalize_pokemon_name(pokemon.species)
                    revealed_opponents.append(normalized_name)

            # Denormalize known moves
            observed_moves = []
            for move_id in known_moves:
                normalized_move = self.denormalize_move_name(move_id)
                observed_moves.append(normalized_move)

            # Get predictions
            predictions = self.bayesian_predictor.predict_component_probabilities(
                species=normalized_species,
                teammates=revealed_opponents,
                observed_moves=observed_moves,
            )

            # Use highest probability predictions to fill missing information
            if (
                "abilities" in predictions
                and predictions["abilities"]
                and ability_name == "Unknown"
            ):
                best_ability = predictions["abilities"][0][0]  # (name, probability)
                normalized_best_ability = best_ability.lower().replace(" ", "")
                if normalized_best_ability not in self.ability_effect:
                    print(
                        f"[WARNING]: Predicted ability '{normalized_best_ability}' not found in ability_effect.json"
                    )
                ability_effect = (
                    ""
                    if normalized_best_ability not in self.ability_effect
                    else f" ({self.ability_effect[normalized_best_ability]['effect']})"
                )
                ability_name = f"{best_ability}"

            if (
                "items" in predictions
                and predictions["items"]
                and item_name == "unknown_item"
            ):
                best_item = predictions["items"][0][0]  # (name, probability)
                normalized_best_item = best_item.lower().replace(" ", "")
                if normalized_best_item not in self.item_effect:
                    print(
                        f"[WARNING]: Predicted item '{normalized_best_item}' not found in item_effect.json"
                    )
                item_effect = (
                    ""
                    if normalized_best_item not in self.item_effect
                    else f" ({self.item_effect[normalized_best_item]['effect']})"
                )
                item_name = f"{best_item}"

            if (
                "tera_types" in predictions
                and predictions["tera_types"]
                and tera_type == "Unknown"
            ):
                best_tera_type = predictions["tera_types"][0][0]
                tera_type = f"{best_tera_type}"

            # Add predicted moves if we have fewer than 4 known moves
            if "moves" in predictions and predictions["moves"] and len(moves_info) < 4:
                predicted_moves = predictions["moves"]
                moves_to_add = 4 - len(moves_info)

                for move_name, prob in predicted_moves[:moves_to_add]:
                    # Skip if we already know this move
                    normalized_move_name = move_name.lower().replace(" ", "")
                    normalized_move_name = normalized_move_name.translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    if normalized_move_name not in known_moves:
                        if normalized_move_name not in self.move_effect:
                            print(
                                f"[WARNING]: Predicted move '{normalized_move_name}' not found in move_effect.json"
                            )
                        move_effect = (
                            ""
                            if normalized_move_name not in self.move_effect
                            else f": {self.move_effect[normalized_move_name]}"
                        )
                        # Estimate KO turns for predicted move as well
                        ko_turns = None
                        try:
                            # Build a pseudo move id for calc (normalize like in turns_to_ko)
                            pseudo_id = normalized_move_name
                            ko_result = self.turns_to_ko(
                                battle, pseudo_id, attacker_is_opponent=opponent
                            )
                            if isinstance(ko_result, int):
                                ko_turns = ko_result
                        except Exception:
                            ko_turns = None
                        if opponent:
                            ko_str = (
                                f" ({ko_turns} turns to KO your active pokemon)"
                                if ko_turns is not None
                                else ""
                            )
                        else:
                            ko_str = (
                                f" ({ko_turns} turns to KO opponent's pokemon)"
                                if ko_turns is not None
                                else ""
                            )
                        moves_info.append(f"  * {move_name}{move_effect}{ko_str}")

            # except Exception as e:
            #     print(f"[WARNING]: Error in enhanced prediction: {e}")

        # Fill remaining move slots if we have fewer than 4
        while opponent and len(moves_info) < 4:
            # TODO: if this appears with your active pokemon there are some problems
            moves_info.append(f"  * Unknown move: Move not yet revealed")

        if status:
            status_line = f"* Status: {status}\n"
        else:
            status_line = ""

        # Add speed comparison for player's pokemon
        speed_line = ""
        if not opponent and battle.opponent_active_pokemon:
            speed_line = self.get_speed_prompt(
                active_mon, battle.opponent_active_pokemon
            )
            if speed_line:
                speed_line = f"* Speed comparison: {speed_line}"

        # Build the final prompt
        prompt = f"""Information about {prefix} active pokemon:
* Species: {self.denormalize_pokemon_name(species)}
* HP percentage: {hp_percentage}%
* Ability: {ability_name}{ability_effect}
* Item: {item_name}{item_effect}
{status_line}* Moves:
{chr(10).join(moves_info)}
* Tera type: {tera_type}
{speed_line}"""

        return prompt

    def turns_to_ko(
        self,
        battle: AbstractBattle,
        move_id: str,
        gen: int = 9,
        attacker_is_opponent: bool = False,
        attacker_pokemon: Pokemon | None = None,
        defender_pokemon: Pokemon | None = None,
    ) -> int | dict:
        """Estimate average number of turns to KO using @smogon/calc via Node helper.

        Returns an integer (average turns to KO). On failure returns {'error': ...}.

        Args:
            battle: The current battle
            move_id: The move to calculate damage for
            gen: Generation (default 9)
            attacker_is_opponent: Whether the attacker is the opponent (default False)
            attacker_pokemon: Optional custom attacker Pokemon (if None, uses battle.active_pokemon or battle.opponent_active_pokemon)
            defender_pokemon: Optional custom defender Pokemon (if None, uses battle.opponent_active_pokemon or battle.active_pokemon)
        """
        node_path = shutil.which("node")
        if node_path is None:
            return {"error": "Node.js not found in PATH."}

        repo_root = Path(__file__).resolve().parents[1]
        js_dir = repo_root / "js_damage"
        calc_script = js_dir / "calc_turns.js"
        if not calc_script.exists():
            return {"error": f"Missing calc script {calc_script}"}

        # Ensure dependency installed
        if not (js_dir / "node_modules" / "@smogon" / "calc" / "package.json").exists():
            try:
                subprocess.run(
                    ["npm", "install", "--no-audit", "--no-fund"],
                    cwd=str(js_dir),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception as e:
                return {"error": f"npm install failed: {e}"}

        status_map = {
            "burnt": "brn",
            "paralyzed": "par",
            "poisoned": "psn",
            "toxic": "tox",
            "frozen": "frz",
            "sleeping": "slp",
        }

        def normalize_item(item_val):
            if not item_val or not isinstance(item_val, str):
                return None
            low = item_val.lower()
            if low in ("unknown", "unknown_item", "none"):
                return None
            return item_val

        def build_side(battle: AbstractBattle, mon: Pokemon):
            try:
                observed = list(mon.moves.values()) if mon.moves else []
                ev_array, nature = mon.guess_stats(
                    battle=battle, observed_moves=observed
                )
            except Exception:
                ev_array, nature = ([0, 0, 0, 0, 0, 0], "Hardy")
            if not isinstance(ev_array, (list, tuple)) or len(ev_array) != 6:
                ev_array = [0, 0, 0, 0, 0, 0]
            evs = {
                "hp": int(ev_array[0]),
                "atk": int(ev_array[1]),
                "def": int(ev_array[2]),
                "spa": int(ev_array[3]),
                "spd": int(ev_array[4]),
                "spe": int(ev_array[5]),
            }
            ivs = {k: 31 for k in ["hp", "atk", "def", "spa", "spd", "spe"]}
            boosts = {
                k: v
                for k, v in mon.boosts.items()
                if k in ["atk", "def", "spa", "spd", "spe"] and v
            }
            raw_status = self.check_status(mon.status) or None
            if raw_status == "fainted":
                raw_status = None
            calc_status = status_map.get(raw_status) if raw_status else None
            species_name = self.denormalize_pokemon_name(mon.species)
            return {
                "species": species_name,
                "level": mon.level,
                "item": normalize_item(mon.item),
                "ability": mon.ability if mon.ability else None,
                "nature": nature if nature else "Hardy",
                "evs": evs,
                "ivs": ivs,
                "boosts": boosts if boosts else None,
                "status": calc_status,
                "hp": mon.current_hp if mon.current_hp else mon.max_hp,
            }

        # ---- Build field (weather / terrain / screens / rooms) ----
        def build_field(battle: AbstractBattle):
            try:
                from poke_env.environment.weather import Weather
                from poke_env.environment.field import Field
                from poke_env.environment.side_condition import SideCondition
            except ImportError:
                return None

            weather_map = {
                Weather.SUNNYDAY: "Sun",
                Weather.DESOLATELAND: "Harsh Sunshine",
                Weather.RAINDANCE: "Rain",
                Weather.PRIMORDIALSEA: "Heavy Rain",
                Weather.SANDSTORM: "Sand",
                Weather.HAIL: "Hail",
                Weather.SNOW: "Snow",
                Weather.SNOWSCAPE: "Snow",
                Weather.DELTASTREAM: "Strong Winds",
            }
            weather_name = None
            if battle.weather:
                try:
                    w = next(iter(battle.weather.keys()))
                    weather_name = weather_map.get(w)
                except Exception:
                    weather_name = None

            terrain_map = {
                Field.ELECTRIC_TERRAIN: "Electric",
                Field.GRASSY_TERRAIN: "Grassy",
                Field.MISTY_TERRAIN: "Misty",
                Field.PSYCHIC_TERRAIN: "Psychic",
            }
            terrain_name = None
            if battle.fields:
                for f in battle.fields:
                    if f.is_terrain:
                        terrain_name = terrain_map.get(f)
                        break

            field_names = {f.name for f in battle.fields}
            is_trick_room = "TRICK_ROOM" in field_names
            is_gravity = "GRAVITY" in field_names
            is_magic_room = "MAGIC_ROOM" in field_names
            is_wonder_room = "WONDER_ROOM" in field_names

            atk_side = battle.side_conditions
            def_side = battle.opponent_side_conditions

            def side_flags(side_dict):
                return {
                    "isReflect": SideCondition.REFLECT in side_dict,
                    "isLightScreen": SideCondition.LIGHT_SCREEN in side_dict,
                    "isAuroraVeil": SideCondition.AURORA_VEIL in side_dict,
                    "isTailwind": SideCondition.TAILWIND in side_dict,
                }

            field_obj = {
                "gameType": "Singles",
                "attackerSide": side_flags(atk_side),
                "defenderSide": side_flags(def_side),
            }
            if weather_name:
                field_obj["weather"] = weather_name
            if terrain_name:
                field_obj["terrain"] = terrain_name
            if is_trick_room:
                field_obj["isTrickRoom"] = True
            if is_gravity:
                field_obj["isGravity"] = True
            if is_magic_room:
                field_obj["isMagicRoom"] = True
            if is_wonder_room:
                field_obj["isWonderRoom"] = True

            any_side = any(field_obj["attackerSide"].values()) or any(
                field_obj["defenderSide"].values()
            )
            if (
                weather_name
                or terrain_name
                or any_side
                or is_trick_room
                or is_gravity
                or is_magic_room
                or is_wonder_room
            ):
                return field_obj
            return None

        # Decide which side is attacker based on flag
        if attacker_is_opponent:
            attacker_mon = (
                attacker_pokemon if attacker_pokemon else battle.opponent_active_pokemon
            )
            defender_mon = (
                defender_pokemon if defender_pokemon else battle.active_pokemon
            )
            attacker_side = build_side(battle, attacker_mon)
            defender_side = build_side(battle, defender_mon)
        else:
            attacker_mon = (
                attacker_pokemon if attacker_pokemon else battle.active_pokemon
            )
            defender_mon = (
                defender_pokemon if defender_pokemon else battle.opponent_active_pokemon
            )
            attacker_side = build_side(battle, attacker_mon)
            defender_side = build_side(battle, defender_mon)
        move_name_for_calc = move_id.lower().replace(" ", "").replace("-", "")
        payload = {
            "gen": gen,
            "attacker": attacker_side,
            "defender": defender_side,
            "move": {"name": move_name_for_calc},
        }
        field_data = build_field(battle)
        if field_data:
            payload["field"] = field_data

        try:
            proc = subprocess.run(
                [node_path, str(calc_script)],
                input=json.dumps(payload).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                cwd=str(js_dir),
            )
        except subprocess.CalledProcessError as e:
            return {
                "error": f"Node calc failed: {e.stderr.decode('utf-8', errors='ignore')}",
                "payload": payload,
            }
        except Exception as e:
            return {"error": f"Unexpected node error: {e}", "payload": payload}

        try:
            data = json.loads(proc.stdout.decode("utf-8"))
            if isinstance(data, dict) and "turns_avg" in data:
                return int(data["turns_avg"])
            return {"error": "Unexpected response format", "raw": data}
        except Exception as e:
            return {
                "error": f"Parse error: {e}",
                "raw": proc.stdout.decode("utf-8", errors="ignore"),
                "payload": payload,
            }

    def get_side_conditions_prompt(self, battle: AbstractBattle) -> str:
        player_side_condition_prompt = ""
        opponent_side_condition_prompt = ""

        # Player side conditions
        player_side_condition_list = []
        for side_condition in battle.side_conditions:
            side_condition_name = " ".join(side_condition.name.lower().split("_"))
            if side_condition == SideCondition.SPIKES:
                effect = (
                    " (cause damage to your pokémon when switch in except flying type)"
                )
            elif side_condition == SideCondition.STEALTH_ROCK:
                effect = " (cause rock-type damage to your pokémon when switch in)"
            elif side_condition == SideCondition.STICKY_WEB:
                effect = " (reduce the speed stat of your pokémon when switch in)"
            elif side_condition == SideCondition.TOXIC_SPIKES:
                effect = " (cause your pokémon toxic when switch in)"
            else:
                effect = ""

            side_condition_name = side_condition_name + effect
            player_side_condition_list.append(f"  * {side_condition_name}")

        if player_side_condition_list:
            player_side_condition_prompt = (
                "Your team's side conditions:\n"
                + "\n".join(player_side_condition_list)
                + "\n"
            )

        # Opponent side conditions
        opponent_side_condition_list = []
        for side_condition in battle.opponent_side_conditions:
            side_condition_name = " ".join(side_condition.name.lower().split("_"))
            if side_condition == SideCondition.SPIKES:
                effect = " (cause damage to opponent pokémon when switch in except flying type)"
            elif side_condition == SideCondition.STEALTH_ROCK:
                effect = " (cause rock-type damage to opponent pokémon when switch in)"
            elif side_condition == SideCondition.STICKY_WEB:
                effect = " (reduce the speed stat of opponent pokémon when switch in)"
            elif side_condition == SideCondition.TOXIC_SPIKES:
                effect = " (cause opponent pokémon toxic when switch in)"
            else:
                effect = ""

            side_condition_name = side_condition_name + effect
            opponent_side_condition_list.append(f"  * {side_condition_name}")

        if opponent_side_condition_list:
            opponent_side_condition_prompt = (
                "Opponent team's side conditions:\n"
                + "\n".join(opponent_side_condition_list)
                + "\n"
            )

        return player_side_condition_prompt + opponent_side_condition_prompt

    def get_player_team_prompt(self, battle: AbstractBattle) -> str:
        """Create a prompt with each pokemon of the player that is alive"""
        team_info = []

        for pokemon in battle.team.values():
            if pokemon and pokemon.species:
                # Basic pokemon info
                species = self.denormalize_pokemon_name(pokemon.species)
                hp_percentage = round(pokemon.current_hp_fraction * 100, 1)
                status = self.check_status(pokemon.status)
                if status == "fainted" or pokemon.active:
                    continue

                # Get ability information
                ability_name = pokemon.ability if pokemon.ability else "Unknown"
                ability_effect = ""
                if pokemon.ability:
                    try:
                        ability_name = self.ability_effect[pokemon.ability]["name"]
                        ability_effect = (
                            f" ({self.ability_effect[pokemon.ability]['effect']})"
                        )
                    except:
                        ability_name = pokemon.ability
                        ability_effect = ""

                # Get item information
                item_name = "No item"
                item_effect = ""
                if pokemon.item:
                    try:
                        item_name = self.item_effect[pokemon.item]["name"]
                        item_effect = f" ({self.item_effect[pokemon.item]['effect']})"
                    except:
                        item_name = pokemon.item

                # Get moves
                moves_list = []
                # Use team_json_data to get the full moveset for the inactive team member
                if self.team_json_data and species in self.team_json_data:
                    team_member_data = self.team_json_data[species]
                    if "moves" in team_member_data:
                        for move_name in team_member_data["moves"]:
                            move_id = move_name.lower().replace(" ", "").replace("-", "")
                            
                            try:
                                move_explanation = self.move_effect[move_id]
                                if move_explanation:
                                    move_explanation = ": " + move_explanation
                            except:
                                print(
                                    f"[WARNING]: Move '{move_id}' not found in move_effect.json"
                                )
                                move_explanation = ": Effect unknown"

                            # Calculate turns to KO if opponent active pokemon exists
                            ko_turns = None
                            if battle.opponent_active_pokemon:
                                try:
                                    # Pass the team pokemon as the attacker to calculate damage from this pokemon
                                    ko_result = self.turns_to_ko(
                                        battle,
                                        move_id,
                                        attacker_is_opponent=False,
                                        attacker_pokemon=pokemon,
                                    )
                                    if isinstance(ko_result, int):
                                        ko_turns = ko_result
                                except Exception:
                                    ko_turns = None

                            ko_str = (
                                f" ({ko_turns} turns to KO opponent's pokemon)"
                                if ko_turns is not None
                                else ""
                            )
                            moves_list.append(
                                f"    * {move_name}{move_explanation}{ko_str}"
                            )

                # Build status line conditionally
                if status:
                    status_line = f"  * Status: {status}\n"
                else:
                    status_line = ""

                # Add speed comparison against opponent's active pokemon
                speed_line = ""
                if battle.opponent_active_pokemon:
                    speed_comparison = self.get_speed_prompt(
                        pokemon, battle.opponent_active_pokemon
                    )
                    if speed_comparison:
                        speed_line = f"  * Speed comparison: {speed_comparison}"

                # Calculate opponent's fastest KO time against this Pokemon (worst case for player)
                opponent_ko_line = ""
                if battle.opponent_active_pokemon:
                    min_opponent_ko_turns = None
                    fastest_opponent_move = None

                    # Check all opponent's known moves
                    if (
                        hasattr(battle.opponent_active_pokemon, "moves")
                        and battle.opponent_active_pokemon.moves
                    ):
                        for opp_move in battle.opponent_active_pokemon.moves.values():
                            if opp_move:
                                try:
                                    # Calculate how many turns this opponent move takes to KO this team Pokemon
                                    # Attacker is opponent, defender is this team Pokemon
                                    ko_result = self.turns_to_ko(
                                        battle,
                                        opp_move.id,
                                        attacker_is_opponent=True,
                                        attacker_pokemon=battle.opponent_active_pokemon,
                                        defender_pokemon=pokemon,
                                    )
                                    if isinstance(ko_result, int):
                                        if (
                                            min_opponent_ko_turns is None
                                            or ko_result < min_opponent_ko_turns
                                        ):
                                            min_opponent_ko_turns = ko_result
                                            fastest_opponent_move = (
                                                self.denormalize_move_name(opp_move.id)
                                            )
                                except Exception:
                                    pass

                    if min_opponent_ko_turns is not None and fastest_opponent_move:
                        opponent_ko_line = f"  * Opponent's fastest KO: {min_opponent_ko_turns} turns (using {fastest_opponent_move})\n"

                # Build pokemon entry using the same style as get_active_pokemon_prompt
                pokemon_info = f"""* Species: {species}
  * HP percentage: {hp_percentage}%
  * Ability: {ability_name}{ability_effect}
  * Item: {item_name}{item_effect}
{status_line}  * Moves:
{chr(10).join(moves_list) if moves_list else "    * No moves revealed"}
{speed_line}
{opponent_ko_line}"""

                team_info.append(pokemon_info)

        if team_info:
            return "Your other pokemons in the team:\n" + "\n\n".join(team_info) + "\n"
        else:
            return "Your team: No Pokemon information available.\n"

    def get_speed_prompt(self, mon: Pokemon, mon_opp: Pokemon) -> str:
        mon_stats = mon.calculate_stats(battle_format=self.format)
        mon_opp_stats = mon_opp.calculate_stats(battle_format=self.format)
        if mon_stats["spe"] > mon_opp_stats["spe"]:
            return f"{self.denormalize_pokemon_name(mon.species)} outspeeds opponent {self.denormalize_pokemon_name(mon_opp.species)}\n"
        else:
            return f"{self.denormalize_pokemon_name(mon_opp.species)} outspeeds opponent {self.denormalize_pokemon_name(mon.species)}\n"

    def get_terastallization_prompt(self, battle: AbstractBattle) -> str:
        """Generate terastallization information for the prompt."""
        # TODO: seems to assign to each pokemon "Unknown"
        tera_prompt = ""

        # Check if this is generation 9 and terastallization is available
        if battle._data.gen != 9:
            return tera_prompt

        # Check if opponent can tera (use hasattr for safety)
        opponent_can_tera = getattr(battle, "opponent_can_tera", False)

        # Information about terastallization mechanic
        if battle.can_tera or opponent_can_tera:
            tera_prompt += "\nTerastallization Information:\n"
            tera_prompt += "* 'terastallize' changes a Pokemon's defensive typing to solely their tera type, changing their resistances and weaknesses.\n"
            tera_prompt += "* It also gives them a 1.5x boost to moves of their tera type (2x if the move already matches their original type).\n"
            tera_prompt += "* You can only 'terastallize' one Pokemon per battle, and it will last until they are KO'd or the battle ends.\n"
            tera_prompt += "* You can choose to 'terastallize' and use a move in the same turn.\n\n"

        # Player's terastallization status
        if battle.can_tera:
            active_tera_type = "Unknown"
            if (
                hasattr(battle.active_pokemon, "_terastallized_type")
                and battle.active_pokemon._terastallized_type
            ):
                active_tera_type = (
                    battle.active_pokemon._terastallized_type.name.capitalize()
                )
            elif self.team_json_data and self.denormalize_pokemon_name(battle.active_pokemon.species) in self.team_json_data:
                active_tera_type = self.team_json_data[self.denormalize_pokemon_name(battle.active_pokemon.species)].get("tera", "Unknown")
            
            tera_prompt += f"* You CAN terastallize this turn! Your {self.denormalize_pokemon_name(battle.active_pokemon.species)}'s tera type: {active_tera_type}\n"
        elif battle.active_pokemon and battle.active_pokemon.terastallized:
            tera_type = (
                battle.active_pokemon._terastallized_type.name.capitalize()
                if battle.active_pokemon._terastallized_type
                else "Unknown"
            )
            if tera_type == "Unknown" and self.team_json_data and self.denormalize_pokemon_name(battle.active_pokemon.species) in self.team_json_data:
                tera_type = self.team_json_data[self.denormalize_pokemon_name(battle.active_pokemon.species)].get("tera", "Unknown")
            tera_prompt += f"* Your {self.denormalize_pokemon_name(battle.active_pokemon.species)} is currently terastallized (Type: {tera_type})\n"
        elif not battle.can_tera and battle._data.gen == 9:
            tera_prompt += "* You have already used your terastallization this battle\n"

        # Opponent's terastallization status
        if battle.opponent_active_pokemon:
            if battle.opponent_active_pokemon.terastallized:
                if hasattr(battle.opponent_active_pokemon, "_terastallized_type") and battle.opponent_active_pokemon._terastallized_type:
                    opp_tera_type = battle.opponent_active_pokemon._terastallized_type.name.capitalize() if hasattr(battle.opponent_active_pokemon._terastallized_type, 'name') else str(battle.opponent_active_pokemon._terastallized_type).capitalize()
                else:
                    opp_tera_type = "Unknown"
                tera_prompt += f"* Opponent's {self.denormalize_pokemon_name(battle.opponent_active_pokemon.species)} is currently terastallized (Type: {opp_tera_type})\n"
            elif opponent_can_tera:
                tera_prompt += (
                    f"* WARNING: Opponent can still terastallize this battle\n"
                )

        return tera_prompt

    def get_system_prompt(self, battle: AbstractBattle) -> str:
        active_pokemon = battle.active_pokemon.species
        denormalized_name = self.denormalize_pokemon_name(active_pokemon)

        system_prompt = "You are a competitive Pokemon battler. Your goal is to win the battle by making optimal decisions based on the current state, type matchups, and predictions.\n"

        if active_pokemon in self.strategy_data:
            system_prompt += f"Your active Pokémon is {denormalized_name}, and you should consider this strategy:\n{self.strategy_data[active_pokemon]}\n"
        elif denormalized_name in self.strategy_data:
            system_prompt += f"Your active Pokémon is {denormalized_name}, and you should consider this strategy:\n{self.strategy_data[denormalized_name]}\n"
        else:
            print(f"Active Pokémon {active_pokemon} not found in strategy file.")

        return system_prompt

    def _extract_json(self, response_text: str) -> dict:
        """Extracts JSON from LLM response, handling potential markdown formatting."""
        if not response_text:
            raise ValueError("Empty response from LLM")
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not extract valid JSON from response text: {response_text}")

    def _get_available_switches_list(self, battle: AbstractBattle) -> str:
        available_switches_list = []
        for pokemon in battle.available_switches:
            pokemon_name = self.denormalize_pokemon_name(pokemon.species)
            hp_pct = round(pokemon.current_hp_fraction * 100, 1)
            available_switches_list.append(f"  - {pokemon_name} (HP: {hp_pct}%)")
        return (
            "\n".join(available_switches_list)
            if available_switches_list
            else "  - No switches available"
        )

    def _get_available_moves_list(self, battle: AbstractBattle) -> str:
        available_moves_list = []
        for move in battle.available_moves:
            move_name = self.denormalize_move_name(move.id)
            available_moves_list.append(f"  - {move_name}")
        return (
            "\n".join(available_moves_list)
            if available_moves_list
            else "  - No moves available"
        )

    def _build_forced_switch_prompt(
        self,
        side_conditions_prompt: str,
        opponent_active_pokemon_prompt: str,
        player_team_prompt: str,
        switches_options: str,
    ) -> str:
        switch_prompt = ""
        if side_conditions_prompt:
            switch_prompt += side_conditions_prompt + "\n"
        switch_prompt += (
            opponent_active_pokemon_prompt + "\n" + player_team_prompt + "\n"
        )
        switch_prompt += f"\nAvailable switches:\n{switches_options}\n"
        switch_prompt += """\nYour active Pokemon is fainted. You must choose a Pokemon to switch in.
Provide your response in JSON format with the following structure:
{
  "explanation": "A detailed explanation of why you chose to switch to this pokemon, considering the opponent's pokemon, current battle state, and your strategy",
  "switch": "The name of the pokemon you want to switch to (must be one from the available switches list)"
}"""
        return switch_prompt

    def _build_move_prompt(
        self,
        battle: AbstractBattle,
        side_conditions_prompt: str,
        player_active_pokemon_prompt: str,
        opponent_active_pokemon_prompt: str,
        terastallization_prompt: str,
        moves_options: str,
    ) -> str:
        move_prompt = ""
        if side_conditions_prompt:
            move_prompt += side_conditions_prompt + "\n"
        move_prompt += (
            player_active_pokemon_prompt + "\n" + opponent_active_pokemon_prompt + "\n"
        )
        if terastallization_prompt:
            move_prompt += terastallization_prompt + "\n"
        move_prompt += f"\nAvailable moves:\n{moves_options}\n"

        if battle.can_tera:
            move_prompt += """\nProvide your response in JSON format with the following structure:
{
  "explanation": "A detailed explanation of why you chose this move, considering the opponent's pokemon, current battle state, terastallization options, and your strategy",
  "move": "The name of the move you want to use (must be one from the available moves list)",
  "terastallize": true or false (whether to terastallize this turn while using the move)
}"""
        else:
            move_prompt += """\nProvide your response in JSON format with the following structure:
{
  "explanation": "A detailed explanation of why you chose this move, considering the opponent's pokemon, current battle state, and your strategy",
  "move": "The name of the move you want to use (must be one from the available moves list)"
}"""
        return move_prompt

    def _build_switch_prompt(
        self,
        side_conditions_prompt: str,
        opponent_active_pokemon_prompt: str,
        player_team_prompt: str,
        switches_options: str,
    ) -> str:
        switch_prompt = ""
        if side_conditions_prompt:
            switch_prompt += side_conditions_prompt + "\n"
        switch_prompt += (
            opponent_active_pokemon_prompt + "\n" + player_team_prompt + "\n"
        )
        switch_prompt += f"\nAvailable switches:\n{switches_options}\n"
        switch_prompt += """\nIf you believe that using a move is absolutely better and there is no valid reason to switch, you can set "switch" to "Nothing".
Provide your response in JSON format with the following structure:
{
  "explanation": "A detailed explanation of why you chose to switch to this pokemon, considering the opponent's pokemon, current battle state, and your strategy",
  "switch": "The name of the pokemon you want to switch to (must be one from the available switches list, or 'Nothing' if you strongly prefer moving)"
}"""
        return switch_prompt

    def _build_merger_prompt(
        self,
        battle: AbstractBattle,
        terastallization_prompt: str,
        moves_options: str,
        switches_options: str,
        move_response_raw: str,
        switch_response_raw: str,
    ) -> str:
        opponent_species = (
            self.denormalize_pokemon_name(battle.opponent_active_pokemon.species)
            if battle.opponent_active_pokemon
            else "Unknown"
        )
        opponent_hp = (
            round(battle.opponent_active_pokemon.current_hp_fraction * 100, 1)
            if battle.opponent_active_pokemon
            else 0
        )

        tera_context = ""
        if terastallization_prompt:
            tera_context = f"\n{terastallization_prompt}\n"

        return f"""You are a pokemon battler that targets to win the pokemon battle. 

Current Opponent's Active Pokemon: {opponent_species} (HP: {opponent_hp}%)
{tera_context}
You have two possible actions to choose from:

1) MOVE ACTION:
Available moves:
{moves_options}
Response: {move_response_raw}

2) SWITCH ACTION:
Available switches:
{switches_options}
Response: {switch_response_raw}

Analyze both options considering:
- The opponent's pokemon type, HP, and predicted moves
- Type advantages/disadvantages
- Terastallization opportunities and threats
- Current battle momentum and strategy
- Which action gives you the best chance to win

Choose the action with the better explanation that would lead you to win the battle.

Provide your response in JSON format:
{{
  "explanation": "Detailed reasoning comparing both options and why one is superior",
  "choice": "move" or "switch"
}}"""

    def _parse_move_choice(
        self, battle: AbstractBattle, move_response_raw: str, move_prompt: str = ""
    ) -> BattleOrder | None:
        try:
            move_data = self._extract_json(move_response_raw)
            chosen_move_name = move_data.get("move", "").strip()
            should_terastallize = move_data.get("terastallize", False)

            for move in battle.available_moves:
                if move.id.lower().replace(" ", "").replace(
                    "-", ""
                ) == chosen_move_name.lower().replace(" ", "").replace("-", ""):
                    return BattleOrder(
                        move,
                        terastallize=(
                            should_terastallize if battle.can_tera else False
                        ),
                    )

            available_move_ids = [m.id for m in battle.available_moves]
            matches = get_close_matches(
                chosen_move_name.lower().replace(" ", "").replace("-", ""),
                [
                    m.lower().replace(" ", "").replace("-", "")
                    for m in available_move_ids
                ],
                n=1,
                cutoff=0.6,
            )
            if matches:
                for move in battle.available_moves:
                    if move.id.lower().replace(" ", "").replace("-", "") == matches[0]:
                        return BattleOrder(
                            move,
                            terastallize=(
                                should_terastallize if battle.can_tera else False
                            ),
                        )
        except Exception as e:
            print(f"Error parsing move response: {e}\n--- PROMPT SENT TO LLM ---\n{move_prompt}\n--------------------------")
        return None

    def _parse_switch_choice(
        self, battle: AbstractBattle, switch_response_raw: str, switch_prompt: str = ""
    ) -> BattleOrder | None:
        try:
            switch_data = self._extract_json(switch_response_raw)
            chosen_pokemon_name = switch_data.get("switch", "").strip()

            for pokemon in battle.available_switches:
                if pokemon.species.lower().replace(" ", "").replace(
                    "-", ""
                ) == chosen_pokemon_name.lower().replace(" ", "").replace("-", ""):
                    return BattleOrder(pokemon)

            available_switch_species = [p.species for p in battle.available_switches]
            matches = get_close_matches(
                chosen_pokemon_name.lower().replace(" ", "").replace("-", ""),
                [
                    s.lower().replace(" ", "").replace("-", "")
                    for s in available_switch_species
                ],
                n=1,
                cutoff=0.6,
            )
            if matches:
                for pokemon in battle.available_switches:
                    if (
                        pokemon.species.lower().replace(" ", "").replace("-", "")
                        == matches[0]
                    ):
                        return BattleOrder(pokemon)
        except Exception as e:
            print(f"Error parsing switch response: {e}\n--- PROMPT SENT TO LLM ---\n{switch_prompt}\n--------------------------")
        return None

    def choose_max_damage_move(self, battle: AbstractBattle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        system_prompt = self.get_system_prompt(battle)
        side_conditions_prompt = self.get_side_conditions_prompt(battle)
        opponent_active_pokemon_prompt = self.get_active_pokemon_prompt(
            battle, opponent=True, enhanced=True
        )

        active_pokemon_fainted = (
            not battle.active_pokemon
            or battle.active_pokemon.fainted
            or self.check_status(battle.active_pokemon.status) == "fainted"
        )
        has_available_switches = bool(battle.available_switches)

        if active_pokemon_fainted:
            if not has_available_switches:
                print("[WARNING]: Active Pokemon fainted and no switches available!")
                return self.choose_random_move(battle)

            if len(battle.available_switches) == 1:
                return self.create_order(battle.available_switches[0])

            player_team_prompt = self.get_player_team_prompt(battle)
            switches_options = self._get_available_switches_list(battle)
            switch_prompt = self._build_forced_switch_prompt(
                side_conditions_prompt,
                opponent_active_pokemon_prompt,
                player_team_prompt,
                switches_options,
            )

            # print("----- Switch Prompt (Forced) -----")
            # print(switch_prompt)
            # print("----------------------------------")

            switch_response_raw = self.llm.get_LLM_action(system_prompt, switch_prompt, model=self.backend, json_format=True, battle=battle)[0]

            parsed_order = self._parse_switch_choice(battle, switch_response_raw, switch_prompt)
            if parsed_order:
                return parsed_order
            return self.choose_random_move(battle)

        player_active_pokemon_prompt = self.get_active_pokemon_prompt(
            battle, opponent=False, enhanced=False
        )
        player_team_prompt = self.get_player_team_prompt(battle)
        terastallization_prompt = self.get_terastallization_prompt(battle)

        moves_options = self._get_available_moves_list(battle)
        move_prompt = self._build_move_prompt(
            battle,
            side_conditions_prompt,
            player_active_pokemon_prompt,
            opponent_active_pokemon_prompt,
            terastallization_prompt,
            moves_options,
        )

        if not has_available_switches:
            # print("[INFO]: No switches available, only considering moves")
            # print("----- Move Prompt (Only Option) -----")
            # print(move_prompt)
            # print("-------------------------------------")

            move_response_raw = self.llm.get_LLM_action(system_prompt, move_prompt, model=self.backend, json_format=True, battle=battle)[0]

            parsed_order = self._parse_move_choice(battle, move_response_raw, move_prompt)
            if parsed_order:
                return parsed_order
            return self.choose_random_move(battle)

        switches_options = self._get_available_switches_list(battle)
        switch_prompt = self._build_switch_prompt(
            side_conditions_prompt,
            opponent_active_pokemon_prompt,
            player_team_prompt,
            switches_options,
        )

        # print("----- Move Prompt -----")
        # print(move_prompt)
        # print("----- Switch Prompt -----")
        # print(switch_prompt)
        # print("-----------------------")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_move = executor.submit(
                self.llm.get_LLM_action, 
                system_prompt=system_prompt, 
                user_prompt=move_prompt, 
                model=self.backend,
                json_format=True,
                battle=battle
            )
            future_switch = executor.submit(
                self.llm.get_LLM_action, 
                system_prompt=system_prompt, 
                user_prompt=switch_prompt, 
                model=self.backend,
                json_format=True,
                battle=battle
            )
            
            move_response_raw = future_move.result()[0]
            switch_response_raw = future_switch.result()[0]

        switch_is_nothing = False
        try:
            s_data = self._extract_json(switch_response_raw)
            if s_data.get("switch", "").strip().lower() == "nothing":
                switch_is_nothing = True
        except:
            pass
        
        if switch_is_nothing:
            parsed_order = self._parse_move_choice(battle, move_response_raw, move_prompt)
            if parsed_order:
                return parsed_order
            return self.choose_max_damage_move(battle)

        merger_prompt = self._build_merger_prompt(
            battle,
            terastallization_prompt,
            moves_options,
            switches_options,
            move_response_raw,
            switch_response_raw,
        )

        # print("----- Merger Prompt -----")
        # print(merger_prompt)
        merger_response_json = self.llm.get_LLM_action(system_prompt, merger_prompt, model=self.backend, json_format=True, battle=battle)[0]

        try:
            merger_data = self._extract_json(merger_response_json)
            choice = merger_data.get("choice", "").strip().lower()
        except:
            choice = "move" if "move" in merger_response_json.lower() else "switch"

        if "move" in choice:
            parsed_order = self._parse_move_choice(battle, move_response_raw, move_prompt)
            if parsed_order:
                return parsed_order
        elif "switch" in choice:
            parsed_order = self._parse_switch_choice(battle, switch_response_raw, switch_prompt)
            if parsed_order:
                return parsed_order

        return self.choose_random_move(battle)
