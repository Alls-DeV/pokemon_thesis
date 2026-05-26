"""
damage_validator.py

Runs a local battle and compares @smogon/calc damage predictions with
actual damage observed in battle messages.

Usage:
    uv run python damage_validator.py [--team1 1] [--team2 2] [--format gen9ou]

The Pokémon Showdown local server must be running:
    node pokemon-showdown start --no-security
"""

import asyncio
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.teambuilder import Teambuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_team(path: str) -> str:
    with open(path) as f:
        raw = f.read()
    team = Teambuilder.parse_showdown_team(raw)
    for mon in team:
        if mon.species is not None:
            mon.nickname = mon.species
    return Teambuilder.join_team(team)


def _parse_hp_fraction(hp_str: str) -> float:
    """'258/310' or '88/100' or '0 fnt' → fraction 0..1."""
    part = hp_str.split(" ")[0]
    if part == "0":
        return 0.0
    pieces = part.split("/")
    if len(pieces) == 2:
        return float(pieces[0]) / float(pieces[1])
    return 1.0


def _species_key(name: str) -> str:
    """Normalize species name for fuzzy matching (lowercase, no spaces/hyphens)."""
    return name.lower().replace(" ", "").replace("-", "")


def _find_in_team(team_data: dict, species: str) -> Optional[dict]:
    """Return the Pokémon entry from a team dict regardless of formatting."""
    target = _species_key(species)
    for key, val in team_data.items():
        if _species_key(key) == target:
            return val, key  # (data, canonical_name)
    return None, None


def _status_abbrev(mon: Pokemon) -> Optional[str]:
    if not mon.status:
        return None
    return {1: "brn", 3: "frz", 4: "par", 5: "psn", 7: "tox", 6: "slp"}.get(
        mon.status.value
    )


# ---------------------------------------------------------------------------
# Validator bot
# ---------------------------------------------------------------------------

class DamageValidatorBot(Player):
    """
    Plays battles while recording how well @smogon/calc predicts actual
    in-game damage for every damaging move.
    """

    def __init__(self, *args, team_idx: int = 1, label: str = "Bot", **kwargs):
        super().__init__(*args, **kwargs)
        self.label = label
        self.team_idx = team_idx
        self.validation_log: list[dict] = []

        self._js_dir = Path(__file__).resolve().parent / "js_damage"
        self._calc_script = self._js_dir / "calc_turns.js"
        self._node = shutil.which("node")

        # Load own team JSON (for exact EVs/nature/items)
        self._own_team: dict = {}
        try:
            with open(f"polimi/teams_json/team{team_idx}.json") as f:
                self._own_team = json.load(f)
        except Exception:
            pass

        # Pre-load all known teams for opponent matching
        self._known_teams: list[dict] = []
        teams_dir = Path(__file__).parent / "polimi" / "teams_json"
        if teams_dir.exists():
            for p in sorted(teams_dir.glob("*.json")):
                try:
                    with open(p) as f:
                        self._known_teams.append(json.load(f))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Move selection: just pick the first available move
    # ------------------------------------------------------------------

    def choose_move(self, battle: AbstractBattle):
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_default_move()

    # ------------------------------------------------------------------
    # Message interception
    # ------------------------------------------------------------------

    async def _handle_battle_message(self, split_messages):
        # Get battle BEFORE super() updates it
        battle_tag = (split_messages[0][0].lstrip(">") if split_messages and split_messages[0] else None)
        battle = self.battles.get(battle_tag) if battle_tag else None

        comparisons = self._extract_damage_comparisons(battle, split_messages) if battle else []

        await super()._handle_battle_message(split_messages)

        for comp in comparisons:
            self.validation_log.append(comp)
            _print_comparison(self.label, comp)

    # ------------------------------------------------------------------
    # Core comparison logic
    # ------------------------------------------------------------------

    def _extract_damage_comparisons(self, battle: AbstractBattle, split_messages) -> list:
        player_role = battle.player_role
        if not player_role:
            return []
        opp_role = "p2" if player_role == "p1" else "p1"

        # HP tracker: slot like "p1a" → fraction 0..1
        hp_tracker: dict[str, float] = {}
        if battle.active_pokemon:
            hp_tracker[f"{player_role}a"] = battle.active_pokemon.current_hp_fraction
        if battle.opponent_active_pokemon:
            hp_tracker[f"{opp_role}a"] = battle.opponent_active_pokemon.current_hp_fraction

        comparisons = []
        pending: Optional[tuple] = None  # (attacker_slot_prefix, move_name)

        for msg in split_messages[1:]:
            if not msg or len(msg) < 2:
                continue
            event = msg[1]

            if event == "move" and len(msg) >= 4:
                actor_slot = msg[2][:3]          # "p1a"
                actor_prefix = actor_slot[:2]    # "p1"
                move_name = msg[3]
                pending = (actor_prefix, actor_slot, move_name)

            elif event == "-damage" and len(msg) >= 4:
                damaged_slot = msg[2][:3]        # "p2a"
                damaged_prefix = damaged_slot[:2] # "p2"

                # Update HP tracker regardless of pending
                new_frac = _parse_hp_fraction(msg[3])

                if pending is not None:
                    attacker_prefix, _, move_name = pending
                    is_self_damage = damaged_prefix == attacker_prefix

                    if not is_self_damage:
                        prev_frac = hp_tracker.get(damaged_slot)
                        if prev_frac is not None:
                            actual_pct = max(0.0, (prev_frac - new_frac) * 100)
                            attacker_is_opp = attacker_prefix != player_role

                            attacker_mon = (
                                battle.opponent_active_pokemon if attacker_is_opp
                                else battle.active_pokemon
                            )
                            defender_mon = (
                                battle.active_pokemon if attacker_is_opp
                                else battle.opponent_active_pokemon
                            )

                            if attacker_mon and defender_mon:
                                pred = self._calc_prediction(
                                    battle, move_name, attacker_is_opp,
                                    attacker_mon, defender_mon,
                                )
                                comparisons.append({
                                    "turn": battle.turn,
                                    "move": move_name,
                                    "attacker": attacker_mon.species or "?",
                                    "attacker_is_opp": attacker_is_opp,
                                    "defender": defender_mon.species or "?",
                                    "actual_pct": round(actual_pct, 1),
                                    "pred": pred,
                                })
                        pending = None  # consumed

                hp_tracker[damaged_slot] = new_frac

            elif event == "-heal" and len(msg) >= 4:
                hp_tracker[msg[2][:3]] = _parse_hp_fraction(msg[3])

            elif event in ("-fail", "-miss", "-immune", "-notarget"):
                # Move didn't land; drop pending so we don't mis-pair
                pending = None

            elif event == "-activate" and len(msg) >= 4 and "Substitute" in msg[3]:
                pending = None  # move absorbed by Substitute

            elif event == "-end" and len(msg) >= 4 and "Substitute" in msg[3]:
                pending = None  # Substitute broke, damage went to sub not Pokémon

            elif event in ("switch", "drag"):
                pending = None
                if len(msg) >= 5:
                    hp_tracker[msg[2][:3]] = _parse_hp_fraction(msg[4])

        return comparisons

    # ------------------------------------------------------------------
    # Calculator call
    # ------------------------------------------------------------------

    def _resolve_team_data(self, battle: AbstractBattle, mon: Pokemon, is_opp: bool) -> tuple[dict, str]:
        """Return (mon_data_dict, canonical_species_name) from known team JSON."""
        if is_opp:
            # Try to match opponent team
            for team in self._known_teams:
                # Check if the revealed opponent Pokémon are all in this team
                plausible = True
                for p in battle.opponent_team.values():
                    if not p.species:
                        continue
                    _, canon = _find_in_team(team, p.species)
                    if canon is None:
                        plausible = False
                        break
                if plausible:
                    data, canon = _find_in_team(team, mon.species)
                    if data:
                        return data, canon
        else:
            data, canon = _find_in_team(self._own_team, mon.species)
            if data:
                return data, canon

        return {}, mon.species.capitalize() if mon.species else "Unknown"

    def _build_side(self, battle: AbstractBattle, mon: Pokemon, is_opp: bool) -> dict:
        mon_data, canonical = self._resolve_team_data(battle, mon, is_opp)

        evs = {k: 0 for k in ("hp", "atk", "def", "spa", "spd", "spe")}
        nature = "Hardy"
        item = None
        ability = mon.ability or None

        if mon_data:
            if "evs" in mon_data:
                evs.update({k: mon_data["evs"].get(k, 0) for k in evs})
            if "nature" in mon_data:
                nature = mon_data["nature"]
            if "item" in mon_data:
                item = mon_data["item"]
            if "ability" in mon_data and not ability:
                ability = mon_data["ability"]

        # Prefer observed item/ability from battle state
        if mon.item and mon.item.lower() not in ("unknown", "unknown_item", "none", ""):
            item = mon.item
        if mon.ability:
            ability = mon.ability

        return {
            "species": canonical,
            "level": mon.level,
            "item": item,
            "ability": ability.lower().replace(" ", "") if ability else None,
            "nature": nature,
            "evs": evs,
            "ivs": {k: 31 for k in evs},
            "boosts": ({k: v for k, v in mon.boosts.items() if v != 0} or None),
            "status": _status_abbrev(mon),
            "hp_fraction": mon.current_hp_fraction,
        }

    def _calc_prediction(
        self,
        battle: AbstractBattle,
        move_name: str,
        attacker_is_opp: bool,
        attacker_mon: Pokemon,
        defender_mon: Pokemon,
    ) -> dict:
        if not self._node or not self._calc_script.exists():
            return {"error": "Node.js or calc script not found"}

        move_id = re.sub(r"[^a-z0-9]", "", move_name.lower())
        payload = {
            "gen": 9,
            "attacker": self._build_side(battle, attacker_mon, attacker_is_opp),
            "defender": self._build_side(battle, defender_mon, not attacker_is_opp),
            "move": {"name": move_id},
        }

        try:
            proc = subprocess.run(
                [self._node, str(self._calc_script)],
                input=json.dumps(payload).encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self._js_dir),
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return {"error": "calc timeout"}
        except Exception as e:
            return {"error": str(e)}

        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="ignore").strip()
            return {"error": err[:120]}

        try:
            return json.loads(proc.stdout.decode("utf-8"))
        except Exception as e:
            return {"error": f"parse error: {e}"}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self):
        print(f"\n{'=' * 95}")
        print(f"VALIDATION REPORT — {self.label} (team {self.team_idx})")
        print(f"{'=' * 95}")

        if not self.validation_log:
            print("  No damage events recorded.")
            return

        total = errors = within = 0
        total_dev = 0.0

        for comp in self.validation_log:
            pred = comp["pred"]
            actual = comp["actual_pct"]
            if pred and "error" not in pred and "min_pct" in pred:
                total += 1
                if pred["min_pct"] <= actual <= pred["max_pct"]:
                    within += 1
                total_dev += abs(actual - pred["avg_pct"])
            else:
                errors += 1

        print(f"  Damage events observed : {len(self.validation_log)}")
        print(f"  Calc succeeded         : {total}")
        print(f"  Calc errors/status moves: {errors}")
        if total:
            print(f"  Within predicted range : {within}/{total}  ({within/total*100:.1f}%)")
            print(f"  Mean absolute deviation: {total_dev/total:.1f}% of max HP")
        print(f"{'=' * 95}")


# ---------------------------------------------------------------------------
# Per-event console output
# ---------------------------------------------------------------------------

def _print_comparison(label: str, comp: dict):
    turn = comp["turn"]
    move = comp["move"]
    attacker = comp["attacker"] + (" (opp)" if comp["attacker_is_opp"] else "")
    defender = comp["defender"]
    actual = comp["actual_pct"]
    pred = comp["pred"]

    if pred and "error" not in pred and "min_pct" in pred:
        mn, mx, av = pred["min_pct"], pred["max_pct"], pred["avg_pct"]
        ok = "✓" if mn <= actual <= mx else "✗"
        dev = actual - av
        print(
            f"[{label}] T{turn:3d} | {attacker:22s} | {move:20s} → {defender:18s} | "
            f"Pred {mn:5.1f}%-{mx:5.1f}% (avg {av:5.1f}%) | Actual {actual:5.1f}% | {ok} ({dev:+.1f}%)"
        )
    else:
        err = (pred or {}).get("error", "unknown")[:50]
        print(
            f"[{label}] T{turn:3d} | {attacker:22s} | {move:20s} → {defender:18s} | "
            f"Pred FAILED ({err}) | Actual {actual:5.1f}%"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Validate @smogon/calc damage predictions")
    parser.add_argument("--team1", type=int, default=1, help="Team index for bot 1 (default 1)")
    parser.add_argument("--team2", type=int, default=2, help="Team index for bot 2 (default 2)")
    parser.add_argument("--format", default="gen9ou", dest="battle_format",
                        choices=["gen9ou", "gen9randombattle", "gen8ou"])
    parser.add_argument("--n", type=int, default=1, help="Number of battles (default 1)")
    args = parser.parse_args()

    import numpy as np
    uid = str(np.random.randint(0, 10000))

    team1_str = _load_team(f"polimi/teams/team{args.team1}.txt")
    team2_str = _load_team(f"polimi/teams/team{args.team2}.txt")

    bot1 = DamageValidatorBot(
        battle_format=args.battle_format,
        team=team1_str,
        team_idx=args.team1,
        label=f"Bot1-team{args.team1}",
        account_configuration=AccountConfiguration(f"DmgVal{uid}a", None),
        server_configuration=LocalhostServerConfiguration,
    )
    bot2 = DamageValidatorBot(
        battle_format=args.battle_format,
        team=team2_str,
        team_idx=args.team2,
        label=f"Bot2-team{args.team2}",
        account_configuration=AccountConfiguration(f"DmgVal{uid}b", None),
        server_configuration=LocalhostServerConfiguration,
    )

    print("=" * 95)
    print("DAMAGE VALIDATOR — comparing @smogon/calc predictions with actual battle damage")
    print(f"Team {args.team1} vs Team {args.team2} | Format: {args.battle_format} | {args.n} battle(s)")
    print("=" * 95)
    print("Format: [Bot] Turn | Attacker | Move → Defender | Predicted Range | Actual | ✓/✗")
    print("-" * 95)

    for _ in range(args.n):
        await bot1.battle_against(bot2, n_battles=1)

    bot1.print_report()
    bot2.print_report()


if __name__ == "__main__":
    asyncio.run(main())
