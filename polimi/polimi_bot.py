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
from poke_env.environment.move import Move
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
        save_replays=False,
    ):
        super().__init__(
            battle_format=battle_format,
            team=team,
            account_configuration=account_configuration,
            server_configuration=server_configuration,
            save_replays=save_replays,
        )
        self.api_key = api_key
        self.temperature = temperature
        self.team_idx = team_idx
        self.backend = backend
        self._last_active_pokemon = {}
        if "gpt" in backend and not backend.startswith("openai/"):
            self.llm = GPTPlayer(api_key)
            self.llm.is_polimi = True
        elif "gemini" in backend:
            self.llm = GeminiPlayer(api_key)
            self.llm.is_polimi = True
        elif "deepseek" in backend and not backend.startswith("deepseek-ai/"):
            self.llm = DeepSeekPlayer(api_key)
            self.llm.is_polimi = True
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

    async def _handle_battle_message(self, split_messages: list[list[str]]):
        await super()._handle_battle_message(split_messages)
        
        if not split_messages or not split_messages[0]:
            return
            
        battle_tag = split_messages[0][0]
        if battle_tag.startswith('>'):
            battle_tag = battle_tag[1:]
            
        if not hasattr(self, 'battles') or battle_tag not in self.battles:
            return
            
        battle = self.battles[battle_tag]
        
        if not hasattr(self, '_opponent_last_move'):
            self._opponent_last_move = {}
        if not hasattr(self, '_opponent_last_switch_turn'):
            self._opponent_last_switch_turn = {}
            
        opponent_role = "p2" if battle.player_role == "p1" else "p1"
        
        for msg in split_messages:
            if len(msg) > 2:
                event = msg[1]
                if event == "move":
                    actor = msg[2]
                    if actor.startswith(f"{opponent_role}a:"):
                        move_name = msg[3]
                        self._opponent_last_move[battle_tag] = move_name
                elif event in ["switch", "drag"]:
                    actor = msg[2]
                    if actor.startswith(f"{opponent_role}a:"):
                        self._opponent_last_move[battle_tag] = None
                        self._opponent_last_switch_turn[battle_tag] = battle.turn

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

    def get_known_opponent_team(self, battle: AbstractBattle):
        if not hasattr(self, 'known_opponent_teams'):
            self.known_opponent_teams = []
            teams_dir = Path(__file__).parent / "teams_json"
            if teams_dir.exists():
                for team_file in teams_dir.glob("*.json"):
                    try:
                        with open(team_file, "r") as f:
                            self.known_opponent_teams.append(json.load(f))
                    except Exception as e:
                        print(f"Error loading {team_file}: {e}")

        valid_teams = self.known_opponent_teams.copy()
        
        for pokemon in battle.opponent_team.values():
            if not pokemon.species:
                continue
            species_name = self.denormalize_pokemon_name(pokemon.species)
            
            next_valid = []
            for team in valid_teams:
                if species_name not in team:
                    continue
                
                team_mon_data = team[species_name]
                moves_match = True
                if hasattr(pokemon, "moves") and pokemon.moves:
                    team_moves_normalized = [m.lower().replace(" ", "").replace("-", "") for m in team_mon_data.get("moves", [])]
                    for move_id in pokemon.moves.keys():
                        norm_move = move_id.lower().replace(" ", "").replace("-", "")
                        if norm_move not in team_moves_normalized:
                            moves_match = False
                            break
                if moves_match:
                    next_valid.append(team)
            valid_teams = next_valid
            
        if valid_teams:
            return valid_teams[0]
        return None

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

        types_list = [t.name.capitalize() for t in active_mon.types if t] if hasattr(active_mon, "types") and active_mon.types else ["Unknown"]
        types_str = "/".join(types_list)

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

        # Get status and volatile statuses
        status = self.check_status(active_mon.status)
        volatile_statuses = []
        if hasattr(active_mon, "effects"):
            for effect in active_mon.effects:
                try:
                    if effect.is_volatile_status:
                        volatile_statuses.append(effect.name.replace("_", " ").lower())
                except:
                    pass

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
                    move_type = move.type.name.capitalize() if hasattr(move, 'type') and move.type else "Unknown"
                    move_cat = move.category.name.capitalize() if hasattr(move, 'category') and move.category else "Unknown"
                    move_pow = move.base_power if hasattr(move, 'base_power') else 0
                    
                    try:
                        move_explanation = f"[Type: {move_type}, Category: {move_cat}, Power: {move_pow}] " + self.move_effect[move.id]
                    except:
                        move_explanation = f"[Type: {move_type}, Category: {move_cat}, Power: {move_pow}]"
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
                    move_type = move.type.name.capitalize() if hasattr(move, 'type') and move.type else "Unknown"
                    move_cat = move.category.name.capitalize() if hasattr(move, 'category') and move.category else "Unknown"
                    move_pow = move.base_power if hasattr(move, 'base_power') else 0
                    
                    try:
                        move_explanation = f"[Type: {move_type}, Category: {move_cat}, Power: {move_pow}] " + self.move_effect[move.id]
                    except:
                        move_explanation = f"[Type: {move_type}, Category: {move_cat}, Power: {move_pow}]"
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
        matched_team = None
        if enhanced and opponent:
            matched_team = self.get_known_opponent_team(battle)

        if enhanced and opponent and matched_team:
            normalized_species = self.denormalize_pokemon_name(species)
            if normalized_species in matched_team:
                team_data = matched_team[normalized_species]
                
                if "tera" in team_data and tera_type == "Unknown":
                    tera_type = team_data["tera"]
                    
                if "item" in team_data and item_name == "unknown_item":
                    best_item = team_data["item"]
                    normalized_best_item = best_item.lower().replace(" ", "")
                    item_effect = "" if normalized_best_item not in self.item_effect else f" ({self.item_effect[normalized_best_item]['effect']})"
                    item_name = f"{best_item}"
                    
                if "ability" in team_data and ability_name == "Unknown":
                    best_ability = team_data["ability"]
                    normalized_best_ability = best_ability.lower().replace(" ", "")
                    ability_effect = "" if normalized_best_ability not in self.ability_effect else f" ({self.ability_effect[normalized_best_ability]['effect']})"
                    ability_name = f"{best_ability}"
                    
                if "moves" in team_data and len(moves_info) < 4:
                    for move_name in team_data["moves"]:
                        if len(moves_info) >= 4:
                            break
                        normalized_move_name = move_name.lower().replace(" ", "").translate(str.maketrans("", "", string.punctuation))
                        if normalized_move_name not in known_moves:
                            move_type = "Unknown"
                            move_cat = "Unknown"
                            move_pow = 0
                            try:
                                m = Move(normalized_move_name, gen=9)
                                move_type = m.type.name.capitalize() if hasattr(m, 'type') and m.type else "Unknown"
                                move_cat = m.category.name.capitalize() if hasattr(m, 'category') and m.category else "Unknown"
                                move_pow = m.base_power if hasattr(m, 'base_power') else 0
                            except Exception:
                                pass
                                
                            if normalized_move_name not in self.move_effect:
                                move_effect_str = f": [Type: {move_type}, Category: {move_cat}, Power: {move_pow}] Effect unknown"
                            else:
                                move_effect_str = f": [Type: {move_type}, Category: {move_cat}, Power: {move_pow}] {self.move_effect[normalized_move_name]}"
                                
                            ko_turns = None
                            try:
                                pseudo_id = normalized_move_name
                                ko_result = self.turns_to_ko(battle, pseudo_id, attacker_is_opponent=True)
                                if isinstance(ko_result, int):
                                    ko_turns = ko_result
                            except Exception:
                                ko_turns = None
                                
                            ko_str = f" ({ko_turns} turns to KO your active pokemon)" if ko_turns is not None else ""
                            moves_info.append(f"  * {move_name}{move_effect_str}{ko_str}")
                            known_moves.append(normalized_move_name)

        elif enhanced and self.bayesian_predictor and opponent:
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
                        move_type = "Unknown"
                        move_cat = "Unknown"
                        move_pow = 0
                        try:
                            m = Move(normalized_move_name, gen=9)
                            move_type = m.type.name.capitalize() if hasattr(m, 'type') and m.type else "Unknown"
                            move_cat = m.category.name.capitalize() if hasattr(m, 'category') and m.category else "Unknown"
                            move_pow = m.base_power if hasattr(m, 'base_power') else 0
                        except Exception:
                            pass
                            
                        if normalized_move_name not in self.move_effect:
                            print(
                                f"[WARNING]: Predicted move '{normalized_move_name}' not found in move_effect.json"
                            )
                            move_effect_str = f": [Type: {move_type}, Category: {move_cat}, Power: {move_pow}] Effect unknown"
                        else:
                            move_effect_str = f": [Type: {move_type}, Category: {move_cat}, Power: {move_pow}] {self.move_effect[normalized_move_name]}"
                            
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
                        moves_info.append(f"  * {move_name}{move_effect_str}{ko_str}")

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
            
        if volatile_statuses:
            status_line += f"* Volatile Statuses: {', '.join(volatile_statuses)}\n"

        # Get stat boosts
        boosts_info = []
        if hasattr(active_mon, "boosts"):
            stat_mapping = {
                "atk": "Atk", "def": "Def", "spa": "SpA", 
                "spd": "SpD", "spe": "Spe", "accuracy": "Accuracy", "evasion": "Evasion"
            }
            for stat, value in active_mon.boosts.items():
                if value != 0:
                    sign = "+" if value > 0 else ""
                    stat_name = stat_mapping.get(stat, stat.capitalize())
                    boosts_info.append(f"{sign}{value} {stat_name}")
        
        if boosts_info:
            status_line += f"* Stat Changes: {', '.join(boosts_info)}\n"

        # Add protect usage info
        if hasattr(active_mon, "protect_counter") and active_mon.protect_counter > 0:
            status_line += f"* Protect: Used successfully last turn (Consecutive uses: {active_mon.protect_counter}). Using a Protect-like move again this turn will likely fail.\n"

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
* Type: {types_str}
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

            # Override with exact team data if available
            is_opponent = mon in battle.opponent_team.values()
            team_source = None
            if is_opponent:
                team_source = self.get_known_opponent_team(battle)
            else:
                team_source = getattr(self, "team_json_data", None)

            mon_item = normalize_item(mon.item)
            mon_ability = mon.ability if mon.ability else None

            if team_source and species_name in team_source:
                mon_data = team_source[species_name]
                if "nature" in mon_data:
                    nature = mon_data["nature"]
                if "evs" in mon_data:
                    team_evs = mon_data["evs"]
                    evs = {
                        "hp": team_evs.get("hp", 0),
                        "atk": team_evs.get("atk", 0),
                        "def": team_evs.get("def", 0),
                        "spa": team_evs.get("spa", 0),
                        "spd": team_evs.get("spd", 0),
                        "spe": team_evs.get("spe", 0),
                    }
                if "item" in mon_data and not mon_item:
                    mon_item = mon_data["item"]
                if "ability" in mon_data and not mon_ability:
                    mon_ability = mon_data["ability"].lower().replace(" ", "")

            return {
                "species": species_name,
                "level": mon.level,
                "item": mon_item,
                "ability": mon_ability,
                "nature": nature if nature else "Hardy",
                "evs": evs,
                "ivs": ivs,
                "boosts": boosts if boosts else None,
                "status": calc_status,
                "hp_fraction": mon.current_hp_fraction,
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

            if attacker_is_opponent:
                atk_side = battle.opponent_side_conditions
                def_side = battle.side_conditions
            else:
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
                effect = " (will damage YOUR Pokémon when they switch in, unless Flying-type/Levitate)"
            elif side_condition == SideCondition.STEALTH_ROCK:
                effect = " (will deal Rock-type damage to YOUR Pokémon when they switch in)"
            elif side_condition == SideCondition.STICKY_WEB:
                effect = " (will lower the Speed stat of YOUR Pokémon when they switch in)"
            elif side_condition == SideCondition.TOXIC_SPIKES:
                effect = " (will poison YOUR Pokémon when they switch in)"
            else:
                effect = ""

            side_condition_name = side_condition_name + effect
            player_side_condition_list.append(f"  * {side_condition_name}")

        if player_side_condition_list:
            player_side_condition_prompt = (
                "Hazards/Conditions on YOUR side of the field (These affect YOUR Pokémon):\n"
                + "\n".join(player_side_condition_list)
                + "\n"
            )

        # Opponent side conditions
        opponent_side_condition_list = []
        for side_condition in battle.opponent_side_conditions:
            side_condition_name = " ".join(side_condition.name.lower().split("_"))
            if side_condition == SideCondition.SPIKES:
                effect = " (will damage the OPPONENT'S Pokémon when they switch in, unless Flying-type/Levitate)"
            elif side_condition == SideCondition.STEALTH_ROCK:
                effect = " (will deal Rock-type damage to the OPPONENT'S Pokémon when they switch in)"
            elif side_condition == SideCondition.STICKY_WEB:
                effect = " (will lower the Speed stat of the OPPONENT'S Pokémon when they switch in)"
            elif side_condition == SideCondition.TOXIC_SPIKES:
                effect = " (will poison the OPPONENT'S Pokémon when they switch in)"
            else:
                effect = ""

            side_condition_name = side_condition_name + effect
            opponent_side_condition_list.append(f"  * {side_condition_name}")

        if opponent_side_condition_list:
            opponent_side_condition_prompt = (
                "Hazards/Conditions on the OPPONENT'S side of the field (These affect THEIR Pokémon):\n"
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

                types_list = [t.name.capitalize() for t in pokemon.types if t] if hasattr(pokemon, "types") and pokemon.types else ["Unknown"]
                types_str = "/".join(types_list)

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

                # Build pokemon entry using the same style as get_active_pokemon_prompt
                pokemon_info = f"""* Species: {species}
  * Type: {types_str}
  * HP percentage: {hp_percentage}%
  * Ability: {ability_name}{ability_effect}
  * Item: {item_name}{item_effect}
{status_line}  * Moves:
{chr(10).join(moves_list) if moves_list else "    * No moves revealed"}
{speed_line}"""

                team_info.append(pokemon_info)

        if team_info:
            return "Your other pokemons in the team:\n" + "\n\n".join(team_info) + "\n"
        else:
            return "Your team: No Pokemon information available.\n"

    def get_opponent_team_prompt(self, battle: AbstractBattle, enhanced: bool) -> str:
        """Create a prompt with each revealed opponent pokemon that is not currently active"""
        # number of fainted pokemon
        opponent_fainted_num = 0
        for _, opponent_pokemon in battle.opponent_team.items():
            if opponent_pokemon.fainted:
                opponent_fainted_num += 1

        opponent_unfainted_num = 6 - opponent_fainted_num

        team_info = []

        for pokemon in battle.opponent_team.values():
            if pokemon.active or pokemon == battle.opponent_active_pokemon:
                continue
                
            if not pokemon.species:
                continue

            status = self.check_status(pokemon.status)
            if status == "fainted" or pokemon.fainted or pokemon.current_hp_fraction == 0:
                continue

            species = pokemon.species
            hp_percentage = round(pokemon.current_hp_fraction * 100, 1)

            types_list = [t.name.capitalize() for t in pokemon.types if t] if hasattr(pokemon, "types") and pokemon.types else ["Unknown"]
            types_str = "/".join(types_list)

            # Get ability information
            ability_name = pokemon.ability if pokemon.ability else "Unknown"
            ability_effect = ""
            if pokemon.ability:
                try:
                    ability_name = self.ability_effect[pokemon.ability]["name"]
                    ability_effect = f" ({self.ability_effect[pokemon.ability]['effect']})"
                except:
                    ability_name = pokemon.ability
                    ability_effect = ""

            # Get item information
            item_name = "unknown_item"
            item_effect = ""
            if pokemon.item:
                try:
                    item_name = self.item_effect[pokemon.item]["name"]
                    item_effect = f" ({self.item_effect[pokemon.item]['effect']})"
                except:
                    item_name = pokemon.item

            # Get status and volatile statuses
            status = self.check_status(pokemon.status)
            volatile_statuses = []
            if hasattr(pokemon, "effects"):
                for effect in pokemon.effects:
                    try:
                        if effect.is_volatile_status:
                            volatile_statuses.append(effect.name.replace("_", " ").lower())
                    except:
                        pass

            # Get moves information
            moves_info = []
            known_moves = []
            
            if hasattr(pokemon, "moves") and pokemon.moves:
                for move in pokemon.moves.values():
                    if move:
                        move_name = self.denormalize_move_name(move.id)
                        move_type = move.type.name.capitalize() if hasattr(move, 'type') and move.type else "Unknown"
                        move_cat = move.category.name.capitalize() if hasattr(move, 'category') and move.category else "Unknown"
                        move_pow = move.base_power if hasattr(move, 'base_power') else 0
                        
                        try:
                            move_explanation = f"[Type: {move_type}, Category: {move_cat}, Power: {move_pow}] " + self.move_effect[move.id]
                        except:
                            move_explanation = f"[Type: {move_type}, Category: {move_cat}, Power: {move_pow}]"
                        
                        # Calculate KO turns against player's active
                        ko_turns = None
                        if battle.active_pokemon and status != "fainted":
                            try:
                                ko_result = self.turns_to_ko(
                                    battle, move.id, attacker_is_opponent=True, attacker_pokemon=pokemon, defender_pokemon=battle.active_pokemon
                                )
                                if isinstance(ko_result, int):
                                    ko_turns = ko_result
                            except Exception:
                                ko_turns = None
                        ko_str = f" ({ko_turns} turns to KO your active pokemon)" if ko_turns is not None else ""
                        
                        moves_info.append(f"    * {move_name}: {move_explanation}{ko_str}")
                        known_moves.append(move.id)

            # Get tera type
            tera_type = "Unknown"
            if hasattr(pokemon, "_terastallized_type") and pokemon._terastallized_type:
                tera_type = pokemon._terastallized_type.name.capitalize() if hasattr(pokemon._terastallized_type, 'name') else str(pokemon._terastallized_type)
            elif hasattr(pokemon, "terastallized") and pokemon.terastallized:
                tera_type = "Active (unknown type)"

            # Enhanced predictions
            matched_team = None
            if enhanced:
                matched_team = self.get_known_opponent_team(battle)

            if enhanced and matched_team:
                normalized_species = self.denormalize_pokemon_name(species)
                if normalized_species in matched_team:
                    team_data = matched_team[normalized_species]
                    
                    if "tera" in team_data and tera_type == "Unknown":
                        tera_type = team_data["tera"]
                        
                    if "item" in team_data and item_name == "unknown_item":
                        best_item = team_data["item"]
                        normalized_best_item = best_item.lower().replace(" ", "")
                        item_effect = "" if normalized_best_item not in self.item_effect else f" ({self.item_effect[normalized_best_item]['effect']})"
                        item_name = f"{best_item}"
                        
                    if "ability" in team_data and ability_name == "Unknown":
                        best_ability = team_data["ability"]
                        normalized_best_ability = best_ability.lower().replace(" ", "")
                        ability_effect = "" if normalized_best_ability not in self.ability_effect else f" ({self.ability_effect[normalized_best_ability]['effect']})"
                        ability_name = f"{best_ability}"
                        
                    if "moves" in team_data and len(moves_info) < 4:
                        for move_name in team_data["moves"]:
                            if len(moves_info) >= 4:
                                break
                            normalized_move_name = move_name.lower().replace(" ", "").translate(str.maketrans("", "", string.punctuation))
                            if normalized_move_name not in known_moves:
                                move_type = "Unknown"
                                move_cat = "Unknown"
                                move_pow = 0
                                try:
                                    m = Move(normalized_move_name, gen=9)
                                    move_type = m.type.name.capitalize() if hasattr(m, 'type') and m.type else "Unknown"
                                    move_cat = m.category.name.capitalize() if hasattr(m, 'category') and m.category else "Unknown"
                                    move_pow = m.base_power if hasattr(m, 'base_power') else 0
                                except Exception:
                                    pass
                                    
                                if normalized_move_name not in self.move_effect:
                                    move_effect_str = f": [Type: {move_type}, Category: {move_cat}, Power: {move_pow}] Effect unknown"
                                else:
                                    move_effect_str = f": [Type: {move_type}, Category: {move_cat}, Power: {move_pow}] {self.move_effect[normalized_move_name]}"
                                
                                ko_turns = None
                                if battle.active_pokemon and status != "fainted":
                                    try:
                                        ko_result = self.turns_to_ko(
                                            battle, normalized_move_name, attacker_is_opponent=True, attacker_pokemon=pokemon, defender_pokemon=battle.active_pokemon
                                        )
                                        if isinstance(ko_result, int):
                                            ko_turns = ko_result
                                    except Exception:
                                        ko_turns = None
                                ko_str = f" ({ko_turns} turns to KO your active pokemon)" if ko_turns is not None else ""
                                moves_info.append(f"    * {move_name}{move_effect_str}{ko_str}")
                                known_moves.append(normalized_move_name)
                                
            elif enhanced and self.bayesian_predictor:
                normalized_species = self.denormalize_pokemon_name(species)
                revealed_opponents = []
                for p in battle.opponent_team.values():
                    if p.species:
                        revealed_opponents.append(self.denormalize_pokemon_name(p.species))
                
                observed_moves = [self.denormalize_move_name(m) for m in known_moves]
                
                predictions = self.bayesian_predictor.predict_component_probabilities(
                    species=normalized_species,
                    teammates=revealed_opponents,
                    observed_moves=observed_moves,
                )
                
                if "abilities" in predictions and predictions["abilities"] and ability_name == "Unknown":
                    best_ability = predictions["abilities"][0][0]
                    normalized_best_ability = best_ability.lower().replace(" ", "")
                    ability_effect = "" if normalized_best_ability not in self.ability_effect else f" ({self.ability_effect[normalized_best_ability]['effect']})"
                    ability_name = f"{best_ability}"
                    
                if "items" in predictions and predictions["items"] and item_name == "unknown_item":
                    best_item = predictions["items"][0][0]
                    normalized_best_item = best_item.lower().replace(" ", "")
                    item_effect = "" if normalized_best_item not in self.item_effect else f" ({self.item_effect[normalized_best_item]['effect']})"
                    item_name = f"{best_item}"
                    
                if "tera_types" in predictions and predictions["tera_types"] and tera_type == "Unknown":
                    tera_type = f"{predictions['tera_types'][0][0]}"
                    
                if "moves" in predictions and predictions["moves"] and len(moves_info) < 4:
                    predicted_moves = predictions["moves"]
                    moves_to_add = 4 - len(moves_info)
                    for move_name, prob in predicted_moves[:moves_to_add]:
                        normalized_move_name = move_name.lower().replace(" ", "").translate(str.maketrans("", "", string.punctuation))
                        if normalized_move_name not in known_moves:
                            move_type = "Unknown"
                            move_cat = "Unknown"
                            move_pow = 0
                            try:
                                m = Move(normalized_move_name, gen=9)
                                move_type = m.type.name.capitalize() if hasattr(m, 'type') and m.type else "Unknown"
                                move_cat = m.category.name.capitalize() if hasattr(m, 'category') and m.category else "Unknown"
                                move_pow = m.base_power if hasattr(m, 'base_power') else 0
                            except Exception:
                                pass
                                
                            if normalized_move_name not in self.move_effect:
                                move_effect_str = f": [Type: {move_type}, Category: {move_cat}, Power: {move_pow}] Effect unknown"
                            else:
                                move_effect_str = f": [Type: {move_type}, Category: {move_cat}, Power: {move_pow}] {self.move_effect[normalized_move_name]}"
                            
                            ko_turns = None
                            if battle.active_pokemon and status != "fainted":
                                try:
                                    ko_result = self.turns_to_ko(
                                        battle, normalized_move_name, attacker_is_opponent=True, attacker_pokemon=pokemon, defender_pokemon=battle.active_pokemon
                                    )
                                    if isinstance(ko_result, int):
                                        ko_turns = ko_result
                                except Exception:
                                    ko_turns = None
                            ko_str = f" ({ko_turns} turns to KO your active pokemon)" if ko_turns is not None else ""
                            moves_info.append(f"    * {move_name}{move_effect_str}{ko_str}")

            while len(moves_info) < 4:
                moves_info.append("    * Unknown move: Move not yet revealed")

            status_line = ""
            if status:
                status_line = f"  * Status: {status}\n"
            if volatile_statuses:
                status_line += f"  * Volatile Statuses: {', '.join(volatile_statuses)}\n"
                
            boosts_info = []
            if hasattr(pokemon, "boosts"):
                stat_mapping = {"atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe", "accuracy": "Accuracy", "evasion": "Evasion"}
                for stat, value in pokemon.boosts.items():
                    if value != 0:
                        sign = "+" if value > 0 else ""
                        stat_name = stat_mapping.get(stat, stat.capitalize())
                        boosts_info.append(f"{sign}{value} {stat_name}")
            if boosts_info:
                status_line += f"  * Stat Changes: {', '.join(boosts_info)}\n"
                
            if hasattr(pokemon, "protect_counter") and pokemon.protect_counter > 0:
                status_line += f"  * Protect: Used successfully last turn (Consecutive uses: {pokemon.protect_counter}).\n"

            speed_line = ""
            if battle.active_pokemon and status != "fainted":
                speed_comparison = self.get_speed_prompt(battle.active_pokemon, pokemon)
                if speed_comparison:
                    speed_line = f"  * Speed comparison: {speed_comparison}"

            pokemon_info = f"""* Species: {self.denormalize_pokemon_name(species)}
  * Type: {types_str}
  * HP percentage: {hp_percentage}%
  * Ability: {ability_name}{ability_effect}
  * Item: {item_name}{item_effect}
{status_line}  * Moves:
{chr(10).join(moves_info)}
  * Tera type: {tera_type}
{speed_line}"""
            team_info.append(pokemon_info)

        if team_info:
            return f"Opponent's Remaining Alive Pokemon: {opponent_unfainted_num}\nRevealed Opponent's Team (Bench):\n" + "\n\n".join(team_info) + "\n"
        else:
            return f"Opponent's Remaining Alive Pokemon: {opponent_unfainted_num}\nRevealed Opponent's Team (Bench): No other Pokemon revealed yet.\n"

    def _compute_effective_speed(
        self,
        battle: AbstractBattle,
        mon: Pokemon,
        is_player_side: bool,
    ) -> tuple[int, list[str]]:
        """Return (effective_speed, [list of modifier notes]) for mon in the current battle state.

        Accounts for: stat stage boosts, Choice Scarf, Iron Ball, paralysis,
        weather abilities (Swift Swim / Chlorophyll / Sand Rush / Slush Rush),
        Surge Surfer, Tailwind, Booster Energy / Protosynthesis / Quark Drive.
        Trick Room is NOT applied here — the caller decides turn order from the value.
        """
        from poke_env.environment.weather import Weather

        effective_spe = mon.calculate_stats(battle_format=self.format).get("spe", 1)
        notes: list[str] = []

        # Stat stage boosts (+1..+6 or −1..−6)
        spe_boost = mon.boosts.get("spe", 0)
        if spe_boost > 0:
            mult = (2 + spe_boost) / 2
            effective_spe = int(effective_spe * mult)
            notes.append(f"+{spe_boost} stage (×{mult:.2f})")
        elif spe_boost < 0:
            mult = 2 / (2 + abs(spe_boost))
            effective_spe = int(effective_spe * mult)
            notes.append(f"{spe_boost} stage (×{mult:.2f})")

        # Resolve ability / item — fall back to known opponent team data
        ability = (mon.ability or "").lower().replace(" ", "")
        item = (mon.item or "").lower().replace(" ", "")
        if not is_player_side:
            known = self.get_known_opponent_team(battle)
            if known:
                species = self.denormalize_pokemon_name(mon.species)
                data = known.get(species, {})
                if not ability and "ability" in data:
                    ability = data["ability"].lower().replace(" ", "")
                if item in ("", "unknown", "unknown_item") and "item" in data:
                    item = data["item"].lower().replace(" ", "")

        # Item modifiers
        if item == "choicescarf":
            effective_spe = int(effective_spe * 1.5)
            notes.append("Choice Scarf (×1.5)")
        elif item == "ironball":
            effective_spe = int(effective_spe * 0.5)
            notes.append("Iron Ball (×0.5)")

        # Paralysis (×0.5 in Gen 6+)
        status_str = self.check_status(mon.status) if mon.status else None
        if status_str == "paralyzed":
            effective_spe = int(effective_spe * 0.5)
            notes.append("Paralysis (×0.5)")

        # Current weather
        current_weather = None
        if battle.weather:
            try:
                current_weather = next(iter(battle.weather.keys()))
            except Exception:
                pass

        # Weather-doubling abilities
        weather_ability_map: dict[str, list] = {
            "swiftswim":   [Weather.RAINDANCE, Weather.PRIMORDIALSEA],
            "chlorophyll": [Weather.SUNNYDAY, Weather.DESOLATELAND],
            "sandrush":    [Weather.SANDSTORM],
            "slushrush":   [Weather.HAIL, Weather.SNOW, Weather.SNOWSCAPE],
        }
        for wa, weathers in weather_ability_map.items():
            if ability == wa and current_weather in weathers:
                effective_spe *= 2
                notes.append(f"{wa} in weather (×2)")
                break

        # Surge Surfer on Electric Terrain
        field_names = {f.name for f in battle.fields} if battle.fields else set()
        if ability == "surgesurfer" and "ELECTRIC_TERRAIN" in field_names:
            effective_spe *= 2
            notes.append("Surge Surfer on Electric Terrain (×2)")

        # Tailwind (player side vs opponent side)
        side_conds = battle.side_conditions if is_player_side else battle.opponent_side_conditions
        if SideCondition.TAILWIND in side_conds:
            effective_spe *= 2
            notes.append("Tailwind (×2)")

        # Booster Energy / Protosynthesis / Quark Drive — boost Speed ×1.5 when it is the highest non-HP stat
        boost_active = (
            item == "boosterenergy"
            or (ability == "protosynthesis" and current_weather in [Weather.SUNNYDAY, Weather.DESOLATELAND])
            or (ability == "quarkdrive" and "ELECTRIC_TERRAIN" in field_names)
        )
        if boost_active:
            base_stats = mon.calculate_stats(battle_format=self.format)
            non_hp = {k: v for k, v in base_stats.items() if k != "hp"}
            if non_hp and max(non_hp, key=non_hp.get) == "spe":
                effective_spe = int(effective_spe * 1.5)
                notes.append(f"{ability or item} boosting Speed (×1.5)")

        return int(effective_spe), notes

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
                opp_tera_type = "Unknown"
                matched_team = self.get_known_opponent_team(battle)
                if matched_team:
                    species = self.denormalize_pokemon_name(battle.opponent_active_pokemon.species)
                    if species in matched_team and "tera" in matched_team[species]:
                        opp_tera_type = matched_team[species]["tera"]
                
                if opp_tera_type != "Unknown":
                    tera_prompt += f"* WARNING: Opponent can still terastallize this battle. Their {self.denormalize_pokemon_name(battle.opponent_active_pokemon.species)}'s tera type is likely {opp_tera_type}\n"
                else:
                    tera_prompt += (
                        f"* WARNING: Opponent can still terastallize this battle\n"
                    )

        return tera_prompt

    def get_system_prompt(self, battle: AbstractBattle) -> str:
        active_pokemon = battle.active_pokemon.species
        denormalized_name = self.denormalize_pokemon_name(active_pokemon)

        system_prompt = "You are a pokemon battler in generation 9 OU format Pokemon Showdown that targets to win the pokemon battle.\n"
        system_prompt += "CRITICAL DAMAGE CALCULATOR INFO: When you see '(X turns to KO...)' next to a move, this is the output of a highly accurate, built-in damage calculator. It accounts for actual stats, EVs, typing. You MUST trust these calculations as the definitive source of truth for damage output, unless there are items or abilities that could change the outcome.\n\n"

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
            
        def fix_unescaped_quotes(text: str) -> str:
            # Try to fix unescaped quotes in the "explanation" value
            match = re.search(r'"explanation"\s*:\s*"(.*)",\s*"(?:move|switch|choice)"', text, re.DOTALL)
            if match:
                explanation = match.group(1)
                fixed_explanation = explanation.replace('"', '\\"')
                return text[:match.start(1)] + fixed_explanation + text[match.end(1):]
            return text

        try:
            return json.loads(fix_unescaped_quotes(response_text))
        except json.JSONDecodeError:
            match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if match:
                try:
                    return json.loads(fix_unescaped_quotes(match.group(1)))
                except json.JSONDecodeError:
                    pass
            match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(fix_unescaped_quotes(match.group(1)))
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not extract valid JSON from response text: {response_text}")

    def _compute_hazard_entry_damage(
        self, battle: AbstractBattle, pokemon: Pokemon
    ) -> tuple[float, str]:
        """Return (hp_fraction_lost, human-readable description) for hazards on the player's side."""
        side_conds = battle.side_conditions
        if not side_conds:
            return 0.0, ""

        types = [t for t in (pokemon.types if hasattr(pokemon, "types") and pokemon.types else []) if t]
        type_names = {t.name.upper() for t in types}
        ability = (pokemon.ability or "").lower().replace(" ", "")
        item = (pokemon.item or "").lower().replace(" ", "")

        is_flying = "FLYING" in type_names
        has_levitate = ability == "levitate"
        has_air_balloon = item == "airballoon"
        has_hdb = item == "heavydutyboots"
        has_magic_guard = ability == "magicguard"

        # Not grounded = immune to Spikes and Toxic Spikes
        grounded = not (is_flying or has_levitate or has_air_balloon)

        damage_fraction = 0.0
        parts: list[str] = []

        # Stealth Rock — hits all Pokemon; base 12.5% scaled by Rock type effectiveness
        if SideCondition.STEALTH_ROCK in side_conds and not has_hdb and not has_magic_guard:
            rock_chart: dict[str, float] = {
                "FLYING": 2.0, "FIRE": 2.0, "ICE": 2.0, "BUG": 2.0,
                "FIGHTING": 0.5, "GROUND": 0.5, "STEEL": 0.5,
            }
            multiplier = 1.0
            for t in type_names:
                multiplier *= rock_chart.get(t, 1.0)
            sr_dmg = 0.125 * multiplier
            damage_fraction += sr_dmg
            parts.append(f"Stealth Rock: -{sr_dmg * 100:.4g}% HP")

        # Spikes — grounded only
        if grounded and not has_hdb and not has_magic_guard:
            layers = side_conds.get(SideCondition.SPIKES, 0)
            if layers:
                spikes_dmg = {1: 1 / 8, 2: 1 / 6, 3: 1 / 4}.get(min(layers, 3), 1 / 4)
                damage_fraction += spikes_dmg
                parts.append(f"Spikes ({layers}L): -{spikes_dmg * 100:.4g}% HP")

        # Toxic Spikes — grounded only; Poison-type absorbs, Steel-type immune
        if grounded and not has_hdb:
            layers = side_conds.get(SideCondition.TOXIC_SPIKES, 0)
            if layers:
                if "POISON" in type_names:
                    parts.append("Toxic Spikes: absorbed (Poison-type safe switch-in)")
                elif "STEEL" in type_names or ability in ("immunity", "poisonheal"):
                    parts.append("Toxic Spikes: immune")
                else:
                    status = "badly poisoned" if layers >= 2 else "poisoned"
                    parts.append(f"Toxic Spikes ({layers}L): will be {status} on entry")

        return damage_fraction, ", ".join(parts)

    def _get_available_switches_list(self, battle: AbstractBattle) -> str:
        available_switches_list = []
        for pokemon in battle.available_switches:
            pokemon_name = self.denormalize_pokemon_name(pokemon.species)
            hp_pct = round(pokemon.current_hp_fraction * 100, 1)
            hz_frac, hz_desc = self._compute_hazard_entry_damage(battle, pokemon)
            entry = f"  - {pokemon_name} (HP: {hp_pct}%"
            if hz_desc:
                hp_after = max(0.0, pokemon.current_hp_fraction - hz_frac) * 100
                entry += f", Entry hazard cost: {hz_desc} → effective HP after entry: {hp_after:.1f}%"
            entry += ")"
            available_switches_list.append(entry)
        return (
            "\n".join(available_switches_list)
            if available_switches_list
            else "  - No switches available"
        )

    def _get_available_moves_list(self, battle: AbstractBattle) -> str:
        available_moves_list = []
        for move in battle.available_moves:
            move_name = self.denormalize_move_name(move.id)
            move_type = move.type.name.capitalize() if hasattr(move, 'type') and move.type else "Unknown"
            move_cat = move.category.name.capitalize() if hasattr(move, 'category') and move.category else "Unknown"
            move_pow = move.base_power if hasattr(move, 'base_power') else 0
            
            try:
                move_explanation = f"[Type: {move_type}, Category: {move_cat}, Power: {move_pow}] " + self.move_effect.get(move.id, "")
            except:
                move_explanation = f"[Type: {move_type}, Category: {move_cat}, Power: {move_pow}]"
                
            if move.id == "encore":
                last_move = getattr(self, "_opponent_last_move", {}).get(battle.battle_tag)
                if not last_move:
                    move_explanation += " NOTE: Encore will FAIL because the opponent hasn't used a move yet."
                else:
                    move_explanation += f" NOTE: If successful, it will force the opponent to repeat '{self.denormalize_move_name(last_move)}'."

            available_moves_list.append(f"  - {move_name}: {move_explanation.strip()}")
            
        return (
            "\n".join(available_moves_list)
            if available_moves_list
            else "  - No moves available"
        )

    def _build_forced_switch_prompt(
        self,
        side_conditions_prompt: str,
        opponent_active_pokemon_prompt: str,
        opponent_team_prompt: str,
        player_team_prompt: str,
        switches_options: str,
        entry_message: str = "Your active Pokemon is fainted. You must choose a Pokemon to switch in.",
        outcome_summary: str = "",
    ) -> str:
        switch_prompt = ""
        if side_conditions_prompt:
            switch_prompt += side_conditions_prompt + "\n"
        switch_prompt += (
            opponent_active_pokemon_prompt + "\n" + opponent_team_prompt + "\n" + player_team_prompt + "\n"
        )
        if outcome_summary:
            switch_prompt += f"\nDAMAGE CALCULATOR RESULTS (use these numbers to evaluate each switch-in):\n{outcome_summary}\n"
        switch_prompt += f"\nAvailable switches:\n{switches_options}\n"
        switch_prompt += f"""\n{entry_message}
Provide your response in VALID JSON format with the following structure. IMPORTANT: Do not use double quotes inside the explanation string to ensure valid JSON!
{{
  "explanation": "A detailed explanation of why you chose to switch to this pokemon, considering the opponent's pokemon, current battle state, and your strategy",
  "switch": "The name of the pokemon you want to switch to (must be one from the available switches list)"
}}"""
        return switch_prompt

    def _build_move_prompt(
        self,
        battle: AbstractBattle,
        side_conditions_prompt: str,
        player_active_pokemon_prompt: str,
        opponent_active_pokemon_prompt: str,
        opponent_team_prompt: str,
        terastallization_prompt: str,
        moves_options: str,
    ) -> str:
        move_prompt = ""
        if side_conditions_prompt:
            move_prompt += side_conditions_prompt + "\n"
        move_prompt += (
            player_active_pokemon_prompt + "\n" + opponent_active_pokemon_prompt + "\n" + opponent_team_prompt + "\n"
        )
        if terastallization_prompt:
            move_prompt += terastallization_prompt + "\n"

        # Speed order for this turn (accounts for all in-battle modifiers)
        if battle.active_pokemon and battle.opponent_active_pokemon:
            try:
                active_spe, active_notes = self._compute_effective_speed(
                    battle, battle.active_pokemon, is_player_side=True
                )
                opp_spe, opp_notes = self._compute_effective_speed(
                    battle, battle.opponent_active_pokemon, is_player_side=False
                )
                active_name = self.denormalize_pokemon_name(battle.active_pokemon.species)
                opp_name = self.denormalize_pokemon_name(battle.opponent_active_pokemon.species)
                active_note_str = f" [{', '.join(active_notes)}]" if active_notes else ""
                opp_note_str    = f" [{', '.join(opp_notes)}]" if opp_notes else ""

                field_names = {f.name for f in battle.fields} if battle.fields else set()
                trick_room = "TRICK_ROOM" in field_names

                if trick_room:
                    if active_spe < opp_spe:
                        order = (
                            f"YOU MOVE FIRST under Trick Room "
                            f"({active_name}: {active_spe}{active_note_str} slower than "
                            f"{opp_name}: {opp_spe}{opp_note_str})"
                        )
                    elif active_spe > opp_spe:
                        order = (
                            f"OPPONENT MOVES FIRST under Trick Room "
                            f"({opp_name}: {opp_spe}{opp_note_str} slower than "
                            f"{active_name}: {active_spe}{active_note_str})"
                        )
                    else:
                        order = f"SPEED TIE under Trick Room (both at {active_spe}) — coin flip"
                else:
                    if active_spe > opp_spe:
                        order = (
                            f"YOU MOVE FIRST "
                            f"({active_name}: {active_spe}{active_note_str} > "
                            f"{opp_name}: {opp_spe}{opp_note_str}). "
                            "Your attack resolves before you take damage."
                        )
                    elif active_spe < opp_spe:
                        order = (
                            f"OPPONENT MOVES FIRST "
                            f"({opp_name}: {opp_spe}{opp_note_str} > "
                            f"{active_name}: {active_spe}{active_note_str}). "
                            "You take damage before your move resolves."
                        )
                    else:
                        order = (
                            f"SPEED TIE — coin flip (both at {active_spe})"
                        )

                priority_note = " Priority moves (Quick Attack, Sucker Punch, Fake Out, etc.) ignore speed and always go first." if not trick_room else ""
                move_prompt += f"\n*** SPEED ORDER THIS TURN: {order}.{priority_note} ***\n"
            except Exception:
                pass

        move_prompt += f"\nAvailable moves:\n{moves_options}\n"

        if battle.can_tera:
            move_prompt += """\nProvide your response in VALID JSON format with the following structure. IMPORTANT: Do not use double quotes inside the explanation string to ensure valid JSON!
{
  "explanation": "A detailed explanation of why you chose this move, considering the opponent's pokemon, current battle state, terastallization options, and your strategy",
  "move": "The name of the move you want to use (must be one from the available moves list)",
  "terastallize": true or false (whether to terastallize this turn while using the move)
}"""
        else:
            move_prompt += """\nProvide your response in VALID JSON format with the following structure. IMPORTANT: Do not use double quotes inside the explanation string to ensure valid JSON!
{
  "explanation": "A detailed explanation of why you chose this move, considering the opponent's pokemon, current battle state, and your strategy",
  "move": "The name of the move you want to use (must be one from the available moves list)"
}"""
        return move_prompt

    def _build_switch_prompt(
        self,
        battle: AbstractBattle,
        side_conditions_prompt: str,
        player_active_pokemon_prompt: str,
        opponent_active_pokemon_prompt: str,
        opponent_team_prompt: str,
        player_team_prompt: str,
        switches_options: str,
        just_switched_in: bool = False,
        outcome_summary: str = "",
    ) -> str:
        switch_prompt = ""
        if side_conditions_prompt:
            switch_prompt += side_conditions_prompt + "\n"
        switch_prompt += (
            player_active_pokemon_prompt + "\n" + opponent_active_pokemon_prompt + "\n" + opponent_team_prompt + "\n" + player_team_prompt + "\n"
        )

        if outcome_summary:
            switch_prompt += f"\nDAMAGE CALCULATOR RESULTS (use these numbers to evaluate each option):\n{outcome_summary}\n"

        switch_prompt += f"\nAvailable switches:\n{switches_options}\n"

        if just_switched_in:
            switch_prompt += "\nWARNING: Your active Pokemon JUST switched in! Switching it out now wastes a turn and gives the opponent free momentum. You should ALMOST NEVER switch out immediately, unless staying in guarantees a KO without any benefit. Strongly consider using a MOVE (by returning 'Nothing') instead of switching.\n"

        switch_prompt += """\nIf you believe that using a move is absolutely better and there is no valid reason to switch, you can set "switch" to "Nothing".
Provide your response in VALID JSON format with the following structure. IMPORTANT: Do not use double quotes inside the explanation string to ensure valid JSON!
{
  "explanation": "A detailed explanation of why you chose to switch to this pokemon, considering the opponent's pokemon, current battle state, and your strategy",
  "switch": "The name of the pokemon you want to switch to (must be one from the available switches list, or 'Nothing' if you strongly prefer moving)"
}"""
        return switch_prompt

    def _compute_merger_outcome_summary(self, battle: AbstractBattle) -> str:
        """Run damage-calculator calls for all move/switch candidates and return a compact outcome table."""
        active = battle.active_pokemon
        opponent_mon = battle.opponent_active_pokemon
        if not active or not opponent_mon:
            return ""

        futures_map: dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for move in battle.available_moves:
                key = ("move_atk", move.id)
                futures_map[pool.submit(self.turns_to_ko, battle, move.id, 9, False)] = key

            for move in (opponent_mon.moves or {}).values():
                key = ("opp_vs_active", move.id)
                futures_map[pool.submit(self.turns_to_ko, battle, move.id, 9, True)] = key

            for sw_mon in battle.available_switches:
                for move in (opponent_mon.moves or {}).values():
                    key = ("opp_vs_switch", sw_mon.species, move.id)
                    futures_map[pool.submit(
                        self.turns_to_ko, battle, move.id, 9, True, None, sw_mon
                    )] = key
                for move in (sw_mon.moves or {}).values():
                    key = ("switch_atk", sw_mon.species, move.id)
                    futures_map[pool.submit(
                        self.turns_to_ko, battle, move.id, 9, False, sw_mon, None
                    )] = key

            raw: dict = {}
            for fut in concurrent.futures.as_completed(futures_map):
                key = futures_map[fut]
                try:
                    raw[key] = fut.result()
                except Exception:
                    raw[key] = None

        lines: list[str] = []
        active_name = self.denormalize_pokemon_name(active.species)
        opp_name = self.denormalize_pokemon_name(opponent_mon.species)

        # MOVE outcome: best attacker turn-count and opponent's fastest KO threat
        best_atk_turns: int | None = None
        best_atk_name: str | None = None
        for move in battle.available_moves:
            val = raw.get(("move_atk", move.id))
            if isinstance(val, int) and (best_atk_turns is None or val < best_atk_turns):
                best_atk_turns = val
                best_atk_name = self.denormalize_move_name(move.id)

        opp_threat_turns: int | None = None
        opp_threat_name: str | None = None
        for move in (opponent_mon.moves or {}).values():
            val = raw.get(("opp_vs_active", move.id))
            if isinstance(val, int) and (opp_threat_turns is None or val < opp_threat_turns):
                opp_threat_turns = val
                opp_threat_name = self.denormalize_move_name(move.id)

        lines.append(f"MOVE (stay with {active_name} and attack):")
        if best_atk_turns is not None:
            lines.append(f"  Your best attack — {best_atk_name}: KOs {opp_name} in {best_atk_turns} turn(s)")
        else:
            lines.append("  Your best attack: damage calc unavailable")
        if opp_threat_turns is not None:
            lines.append(f"  Opponent's biggest threat — {opp_threat_name}: KOs {active_name} in {opp_threat_turns} turn(s)")
        elif not opponent_mon.moves:
            lines.append("  Opponent's threat: no moves revealed yet")
        else:
            lines.append("  Opponent's threat: damage calc unavailable")

        # SWITCH outcomes: per candidate
        if battle.available_switches:
            lines.append(f"\nSWITCH (replace {active_name}):")
            for sw_mon in battle.available_switches:
                sw_name = self.denormalize_pokemon_name(sw_mon.species)

                sw_opp_threat: int | None = None
                sw_opp_move: str | None = None
                for move in (opponent_mon.moves or {}).values():
                    val = raw.get(("opp_vs_switch", sw_mon.species, move.id))
                    if isinstance(val, int) and (sw_opp_threat is None or val < sw_opp_threat):
                        sw_opp_threat = val
                        sw_opp_move = self.denormalize_move_name(move.id)

                sw_atk_turns: int | None = None
                sw_atk_name: str | None = None
                for move in (sw_mon.moves or {}).values():
                    val = raw.get(("switch_atk", sw_mon.species, move.id))
                    if isinstance(val, int) and (sw_atk_turns is None or val < sw_atk_turns):
                        sw_atk_turns = val
                        sw_atk_name = self.denormalize_move_name(move.id)

                lines.append(f"  {sw_name}:")
                hz_frac, hz_desc = self._compute_hazard_entry_damage(battle, sw_mon)
                if hz_desc:
                    hp_after = max(0.0, sw_mon.current_hp_fraction - hz_frac) * 100
                    lines.append(f"    Entry hazard cost: {hz_desc} → effective HP on arrival: {hp_after:.1f}%")
                if sw_opp_threat is not None:
                    lines.append(f"    Durability: KO'd by {opp_name}'s {sw_opp_move} in {sw_opp_threat} turn(s)")
                elif not opponent_mon.moves:
                    lines.append("    Durability: no opponent moves revealed yet")
                else:
                    lines.append("    Durability: damage calc unavailable")
                if sw_atk_turns is not None:
                    lines.append(f"    Offense: KOs {opp_name} in {sw_atk_turns} turn(s) using {sw_atk_name}")
                elif not sw_mon.moves:
                    lines.append("    Offense: no moves revealed yet")
                else:
                    lines.append("    Offense: damage calc unavailable")

        return "\n".join(lines)

    def _build_merger_prompt(
        self,
        battle: AbstractBattle,
        side_conditions_prompt: str,
        player_active_pokemon_prompt: str,
        opponent_active_pokemon_prompt: str,
        opponent_team_prompt: str,
        terastallization_prompt: str,
        moves_options: str,
        switches_options: str,
        move_response_raw: str,
        switch_response_raw: str,
        just_switched_in: bool = False,
        outcome_summary: str = "",
    ) -> str:
        tera_context = ""
        if terastallization_prompt:
            tera_context = f"\n{terastallization_prompt}\n"

        merger_prompt = "You are a competitive Pokemon battler deciding between two proposed actions.\n\n"

        if side_conditions_prompt:
            merger_prompt += side_conditions_prompt + "\n"

        merger_prompt += f"""{player_active_pokemon_prompt}

{opponent_active_pokemon_prompt}

{opponent_team_prompt}
{tera_context}"""

        if outcome_summary:
            merger_prompt += f"""DAMAGE CALCULATOR RESULTS (primary evidence — trust these numbers over narrative reasoning):
{outcome_summary}

"""

        just_switched_note = ""
        if just_switched_in:
            just_switched_note = "\n  - Your active Pokemon JUST switched in this turn. Switching again costs 2 turns of momentum — require a clear numerical advantage to justify it."

        merger_prompt += f"""Two actions are proposed by sub-agents:

1) MOVE ACTION:
Available moves:
{moves_options}
Sub-agent reasoning: {move_response_raw}

2) SWITCH ACTION:
Available switches:
{switches_options}
Sub-agent reasoning: {switch_response_raw}

DECISION RULE:
  - Switching hands the opponent a free attack, so default to MOVING.
  - Switch only when the calculator results show a clear advantage: the switch-in survives MORE turns than the active Pokemon AND threatens a faster or equal KO, OR the active Pokemon is KO'd in 1 turn with no meaningful damage in return.
  - Consider win conditions, type immunities, and positioning for factors the calculator cannot capture.{just_switched_note}

Base your decision on the damage calculator numbers first. Use the sub-agent reasoning only to factor in anything the numbers miss.

Provide your response in VALID JSON format. IMPORTANT: Do not use double quotes inside the explanation string!
{{
  "explanation": "State the KO-race numbers for staying vs each switch option. Explain which action has the better expected outcome.",
  "choice": "move" or "switch"
}}"""
        return merger_prompt

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
        if not battle.available_moves:
            return self.choose_random_move(battle)

        def ko_turns(move) -> int:
            result = self.turns_to_ko(battle, move.id)
            if isinstance(result, int) and result > 0:
                return result
            # Non-damaging or calc error: deprioritise below damaging moves
            return 999 if move.base_power > 0 else 9999

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(ko_turns, move): move for move in battle.available_moves}
            scores: dict[str, int] = {}
            for fut in concurrent.futures.as_completed(futures):
                move = futures[fut]
                try:
                    scores[move.id] = fut.result()
                except Exception:
                    scores[move.id] = 999 if move.base_power > 0 else 9999

        best_move = min(battle.available_moves, key=lambda m: scores.get(m.id, 9999))
        return self.create_order(best_move)

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
        
        active_pokemon_name = battle.active_pokemon.species if battle.active_pokemon else None
        just_switched_in = False
        
        if active_pokemon_name and not active_pokemon_fainted:
            last_active = self._last_active_pokemon.get(battle.battle_tag)
            if last_active is not None and last_active != active_pokemon_name:
                just_switched_in = True
            
            self._last_active_pokemon[battle.battle_tag] = active_pokemon_name

        has_available_switches = bool(battle.available_switches)

        if active_pokemon_fainted:
            if not has_available_switches:
                print("[WARNING]: Active Pokemon fainted and no switches available!")
                return self.choose_random_move(battle)

            if len(battle.available_switches) == 1:
                return self.create_order(battle.available_switches[0])

            opponent_team_prompt = self.get_opponent_team_prompt(battle, enhanced=True)
            player_team_prompt = self.get_player_team_prompt(battle)
            switches_options = self._get_available_switches_list(battle)
            switch_prompt = self._build_forced_switch_prompt(
                side_conditions_prompt,
                opponent_active_pokemon_prompt,
                opponent_team_prompt,
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

        # Pivot switch: U-turn / Volt Switch / Flip Turn used — no moves left, must switch
        if not battle.available_moves and has_available_switches:
            if len(battle.available_switches) == 1:
                return self.create_order(battle.available_switches[0])

            opponent_team_prompt = self.get_opponent_team_prompt(battle, enhanced=True)
            player_team_prompt = self.get_player_team_prompt(battle)
            switches_options = self._get_available_switches_list(battle)
            outcome_summary = self._compute_merger_outcome_summary(battle)
            switch_prompt = self._build_forced_switch_prompt(
                side_conditions_prompt,
                opponent_active_pokemon_prompt,
                opponent_team_prompt,
                player_team_prompt,
                switches_options,
                entry_message="Your Pokemon used a pivot move (U-turn / Volt Switch / Flip Turn / etc.) and must now switch out. Choose the best Pokemon to bring in against the current opponent.",
                outcome_summary=outcome_summary,
            )

            switch_response_raw = self.llm.get_LLM_action(system_prompt, switch_prompt, model=self.backend, json_format=True, battle=battle)[0]
            parsed_order = self._parse_switch_choice(battle, switch_response_raw, switch_prompt)
            if parsed_order:
                return parsed_order
            return self.choose_random_move(battle)

        player_active_pokemon_prompt = self.get_active_pokemon_prompt(
            battle, opponent=False, enhanced=False
        )

        opponent_team_prompt = self.get_opponent_team_prompt(battle, enhanced=True)
        player_team_prompt = self.get_player_team_prompt(battle)
        terastallization_prompt = self.get_terastallization_prompt(battle)

        moves_options = self._get_available_moves_list(battle)
        move_prompt = self._build_move_prompt(
            battle,
            side_conditions_prompt,
            player_active_pokemon_prompt,
            opponent_active_pokemon_prompt,
            opponent_team_prompt,
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

        # Compute KO-race data before building the switch prompt so the sub-agent
        # sees the same numbers as the merger judge. The thread-pool inside the method
        # overlaps with nothing here, but it's ~150ms vs 2-5s LLM calls so no issue.
        outcome_summary = self._compute_merger_outcome_summary(battle)

        switches_options = self._get_available_switches_list(battle)
        switch_prompt = self._build_switch_prompt(
            battle,
            side_conditions_prompt,
            player_active_pokemon_prompt,
            opponent_active_pokemon_prompt,
            opponent_team_prompt,
            player_team_prompt,
            switches_options,
            just_switched_in=just_switched_in,
            outcome_summary=outcome_summary,
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
            side_conditions_prompt,
            player_active_pokemon_prompt,
            opponent_active_pokemon_prompt,
            opponent_team_prompt,
            terastallization_prompt,
            moves_options,
            switches_options,
            move_response_raw,
            switch_response_raw,
            just_switched_in=just_switched_in,
            outcome_summary=outcome_summary,
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
