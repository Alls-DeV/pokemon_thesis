#!/usr/bin/env python3
"""
Bayesian Team Predictor for Pokemon Gen9OU

Predicts unrevealed team members, moves, EVs/IVs, items, and abilities based
on observed information.  Uses factorized component distributions fused with
Smogon usage-statistics priors.
"""

import math
import os
import re
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from bayesian.archetype_model import ArchetypeModel
from bayesian.component_model import CACHE_VERSION, ComponentModel
from bayesian.evidence_updater import apply_all_evidence, build_summary
from bayesian.smogon_prior import SmogonPrior
from poke_env.player.team_util import get_metamon_teams

# Dirichlet smoothing alpha for teammate-conditional probabilities
_TEAMMATE_ALPHA = 0.5


@dataclass
class PokemonConfig:
    """Complete configuration for a single Pokemon."""
    species: str
    item: str
    ability: str
    moves: List[str]  # Exactly 4 moves
    nature: str
    evs: Dict[str, int]  # HP, Atk, Def, SpA, SpD, Spe
    ivs: Dict[str, int]  # HP, Atk, Def, SpA, SpD, Spe
    tera_type: str


@dataclass
class TeamData:
    """Complete team of 6 Pokemon configurations."""
    pokemon: List[PokemonConfig]

    def get_species_list(self) -> List[str]:
        return [p.species for p in self.pokemon]


class TeamParser:
    """Parse Showdown team format into structured data."""

    def __init__(self):
        self.stat_names = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']

    def parse_team_file(self, file_path: str) -> TeamData:
        """Parse a single team file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        return self.parse_team_string(content)

    def parse_team_string(self, team_str: str) -> TeamData:
        """Parse team string into TeamData."""
        # Split into individual Pokemon sections
        pokemon_sections = re.split(r'\n\s*\n', team_str.strip())
        pokemon_configs = []

        for section in pokemon_sections:
            if not section.strip():
                continue
            config = self._parse_pokemon_section(section)
            if config:
                pokemon_configs.append(config)

        return TeamData(pokemon=pokemon_configs)

    def _parse_pokemon_section(self, section: str) -> Optional[PokemonConfig]:
        """Parse a single Pokemon section."""
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if not lines:
            return None

        # Parse first line: species @ item
        first_line = lines[0]
        species_match = re.match(r'^(.+?)(?:\s+@\s+(.+))?$', first_line)
        if not species_match:
            return None

        species = species_match.group(1).strip()
        item = species_match.group(2).strip() if species_match.group(2) else ""

        # Handle gender/nickname in species
        species = re.sub(r'\s+\([MF]\)$', '', species)  # Remove (M)/(F)

        # Initialize defaults
        ability = ""
        moves = []
        nature = "Hardy"
        evs = {stat: 0 for stat in self.stat_names}
        ivs = {stat: 31 for stat in self.stat_names}
        tera_type = ""

        for line in lines[1:]:
            line = line.strip()

            # Ability
            if line.startswith('Ability:'):
                ability = line[8:].strip()

            # Tera Type
            elif line.startswith('Tera Type:'):
                tera_type = line[10:].strip()

            # EVs
            elif line.startswith('EVs:'):
                ev_str = line[4:].strip()
                evs.update(self._parse_stat_line(ev_str))

            # IVs
            elif line.startswith('IVs:'):
                iv_str = line[4:].strip()
                ivs.update(self._parse_stat_line(iv_str))

            # Nature
            elif line.endswith('Nature'):
                nature = line.replace(' Nature', '').strip()

            # Moves
            elif line.startswith('-'):
                move = line[1:].strip()
                if move:
                    moves.append(move)

        # Ensure exactly 4 moves (pad with empty if needed)
        while len(moves) < 4:
            moves.append("")
        moves = moves[:4]

        return PokemonConfig(
            species=species,
            item=item,
            ability=ability,
            moves=moves,
            nature=nature,
            evs=evs,
            ivs=ivs,
            tera_type=tera_type
        )

    def _parse_stat_line(self, stat_line: str) -> Dict[str, int]:
        """Parse EV/IV line like '252 HP / 252 Atk / 4 Def'."""
        stats = {}
        parts = [part.strip() for part in stat_line.split('/')]

        for part in parts:
            match = re.match(r'(\d+)\s+(\w+)', part)
            if match:
                value, stat = match.groups()
                if stat in self.stat_names:
                    stats[stat] = int(value)

        return stats


class BayesianTeamPredictor:
    """
    Factorized Bayesian predictor for Pokemon team configurations.

    Uses per-species component distributions (moves, items, abilities, natures,
    EV spreads, tera types) fused with Smogon usage-statistics priors via
    adaptive Bayesian shrinkage.
    """

    def __init__(
        self,
        cache_file: str = "gen9ou_team_predictor_full.pkl",
        battle_format: str = "gen9ou",
    ):
        self.cache_file = cache_file
        self.battle_format = battle_format
        self.parser = TeamParser()

        self.cache_dir = os.getenv('METAMON_CACHE_DIR', '/tmp/metamon_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, cache_file)

        # Core probability tables
        self.species_counts: Counter = Counter()            # P(species)
        self.teammate_counts: Dict[str, Counter] = defaultdict(Counter)  # P(B | A on team)
        self.total_teams: int = 0
        self.is_trained: bool = False

        # Factorized component model (replaces monolithic config_given_species)
        self.component_model: ComponentModel = ComponentModel()

        # LDA team archetype model (replaces naive Bayes teammate multiplication)
        self.archetype_model: ArchetypeModel = ArchetypeModel()
        # Corpus collected during training; not persisted after cache is written
        self._team_corpus: List[List[str]] = []

        # Smogon prior (loaded after training or on first predict call)
        self.smogon_prior: Optional[SmogonPrior] = None

    # ------------------------------------------------------------------
    # Public training interface
    # ------------------------------------------------------------------

    def load_and_train(self, force_retrain: bool = False):
        """Load cached model or train from scratch."""
        if not force_retrain and os.path.exists(self.cache_path):
            print(f"Loading cached model from {self.cache_path}...")
            loaded = self._load_cache()
            if loaded:
                self._init_smogon_prior()
                self.is_trained = True
                return
            # Cache was stale (version mismatch) — retrain

        print("Training new model from full team dataset...")
        self._train_from_data()
        print("Training archetype model (LDA)...")
        self.archetype_model.train(self._team_corpus)
        self._team_corpus = []  # free memory after training
        self._save_cache()
        self._init_smogon_prior()
        self.is_trained = True

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _init_smogon_prior(self):
        try:
            self.smogon_prior = SmogonPrior()
        except Exception as e:
            print(f"[WARNING] Could not load Smogon prior: {e}")
            self.smogon_prior = None

    def _train_from_data(self):
        """Train the model on team data."""
        team_set = get_metamon_teams(self.battle_format, "gl_05_26", version="main")
        team_files = team_set.team_files

        print(f"Training on {len(team_files)} teams...")

        for file_path in tqdm(team_files, desc="Processing teams"):
            try:
                team_data = self.parser.parse_team_file(file_path)
                self._update_counts(team_data)
                self.total_teams += 1

                if self.total_teams % 10000 == 0:
                    print(f"Processed {self.total_teams} teams...")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        print(f"Trained on {self.total_teams} teams")
        print(f"Found {len(self.species_counts)} unique species")

    def _update_counts(self, team_data: TeamData):
        """Update probability counts from a single team."""
        species_list = team_data.get_species_list()

        for i, pokemon in enumerate(team_data.pokemon):
            species = pokemon.species

            self.species_counts[species] += 1

            teammates = [s for j, s in enumerate(species_list) if j != i]
            for teammate in teammates:
                self.teammate_counts[species][teammate] += 1

            # Factorized component counts (replaces atomic config key)
            self.component_model.update(pokemon)

        # Collect for LDA archetype training
        self._team_corpus.append([s for s in species_list if s])

    # ------------------------------------------------------------------
    # Smogon-fused distribution
    # ------------------------------------------------------------------

    def _fuse(self, species: str, component: str) -> Dict[str, float]:
        """
        Return the Smogon-prior-fused distribution for (species, component).

        Adaptive alpha: species with few replay observations shrink harder
        toward the Smogon prior.
        """
        replay_dist = self.component_model.get_distribution(species, component)
        smogon_dist = (
            self.smogon_prior.get_prior(species, component)
            if self.smogon_prior else {}
        )

        if not smogon_dist and not replay_dist:
            return {}
        if not smogon_dist:
            return replay_dist
        if not replay_dist:
            return smogon_dist

        # Compute adaptive alpha (shrinkage toward Smogon prior)
        replay_count = float(self.component_model.replay_counts.get(species, 0))
        smogon_count = (
            self.smogon_prior.get_raw_count(species) if self.smogon_prior else 0.0
        )
        denom = replay_count + smogon_count * 0.05
        if denom <= 0:
            alpha = 0.5
        else:
            alpha = max(0.1, 1.0 - math.sqrt(replay_count / denom))

        # Weighted mixture over the union of both key sets
        all_keys = set(smogon_dist) | set(replay_dist)
        fused: Dict[str, float] = {}
        for k in all_keys:
            fused[k] = alpha * smogon_dist.get(k, 0.0) + (1.0 - alpha) * replay_dist.get(k, 0.0)

        # Renormalize to guard against floating-point drift
        total = sum(fused.values())
        if total > 0:
            fused = {k: v / total for k, v in fused.items()}

        return fused

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------

    def predict_unrevealed_pokemon(
        self,
        revealed_species: List[str],
        max_predictions: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely unrevealed team members.

        Uses the LDA archetype model when available to avoid double-counting
        correlated team cores.  Falls back to Dirichlet-smoothed Naive Bayes
        when the archetype model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call load_and_train() first.")
        if self.total_teams == 0:
            return []

        candidates = [s for s in self.species_counts if s not in revealed_species]

        if self.archetype_model.is_trained:
            archetype_probs = self.archetype_model.predict_unrevealed(
                revealed_species, candidates
            )
            # Blend with base usage rate so rare-but-plausible species aren't zeroed
            total_teams = max(self.total_teams, 1)
            blended: Dict[str, float] = {}
            for sp in candidates:
                base = self.species_counts[sp] / total_teams
                arch = archetype_probs.get(sp, 0.0)
                blended[sp] = 0.7 * arch + 0.3 * base
        else:
            # Fallback: Dirichlet-smoothed Naive Bayes (same as Step 1)
            total_vocab = max(len(self.species_counts), 1)
            total_teams = max(self.total_teams, 1)
            blended = {}
            for sp in candidates:
                base = self.species_counts[sp] / total_teams
                teammate_prob = 1.0
                for revealed in revealed_species:
                    revealed_total = self.species_counts.get(revealed, 0)
                    if revealed_total > 0:
                        count = self.teammate_counts.get(revealed, Counter()).get(sp, 0)
                        vocab_size = len(self.teammate_counts.get(revealed, {})) or total_vocab
                        smoothed = (count + _TEAMMATE_ALPHA) / (
                            revealed_total + _TEAMMATE_ALPHA * vocab_size
                        )
                        teammate_prob *= smoothed
                    else:
                        teammate_prob *= base
                blended[sp] = base * teammate_prob

        ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)
        return ranked[:max_predictions]

    def predict_pokemon_config(
        self,
        species: str,
        teammates: Optional[List[str]] = None,
        revealed_moves: Optional[List[str]] = None,
    ) -> Dict:
        """Predict the most likely full configuration for a species."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call load_and_train() first.")

        revealed_moves = revealed_moves or []

        item_dist = sorted(self._fuse(species, "items").items(), key=lambda x: x[1], reverse=True)
        ability_dist = sorted(self._fuse(species, "abilities").items(), key=lambda x: x[1], reverse=True)
        tera_dist = sorted(self._fuse(species, "tera_types").items(), key=lambda x: x[1], reverse=True)
        spread_dist = sorted(self._fuse(species, "spreads").items(), key=lambda x: x[1], reverse=True)
        move_dist = sorted(self._fuse(species, "moves").items(), key=lambda x: x[1], reverse=True)

        best_item = item_dist[0][0] if item_dist else ""
        best_ability = ability_dist[0][0] if ability_dist else ""
        best_tera = tera_dist[0][0] if tera_dist else ""

        # Parse best spread for nature + EVs
        best_nature = "Hardy"
        best_evs: Dict[str, int] = {}
        if spread_dist:
            parsed = SmogonPrior.parse_spread(spread_dist[0][0])
            if parsed:
                best_nature = parsed["nature"]
                best_evs = parsed["evs"]

        # Top 4 moves: prioritize revealed moves, fill remaining slots from prediction
        known = set(m for m in revealed_moves if m)
        remaining = [m for m, _ in move_dist if m not in known]
        best_moves = list(revealed_moves) + remaining[: max(0, 4 - len(revealed_moves))]
        best_moves = best_moves[:4]

        return {
            "species": species,
            "item": best_item,
            "ability": best_ability,
            "nature": best_nature,
            "moves": best_moves,
            "tera_type": best_tera,
            "ev_spread": best_evs,
            "probability": item_dist[0][1] if item_dist else 0.0,
        }

    def predict_component_probabilities(
        self,
        species: str,
        teammates: Optional[List[str]] = None,
        observed_moves: Optional[List[str]] = None,
        not_seen_moves: Optional[List[str]] = None,
        confirmed_item: Optional[str] = None,
        confirmed_ability: Optional[str] = None,
        confirmed_tera: Optional[str] = None,
        attacker_data: Optional[Dict] = None,
        observed_damage_pct: Optional[float] = None,
    ) -> Dict:
        """
        Predict probability distributions for each configuration component.

        Required backward-compatible keys returned:
          moves, items, natures, abilities, tera_types, ev_spreads, revealed_moves
        Each is a list of (name, probability) tuples sorted descending.

        Additional keys (new): summary.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call load_and_train() first.")

        observed_moves = observed_moves or []

        if species not in self.component_model.replay_counts and (
            self.smogon_prior is None
            or species not in self.smogon_prior.known_species()
        ):
            return {"error": f"No data for species {species}"}

        # Step 1: get Smogon-fused base distributions
        raw_dists = {
            "moves": self._fuse(species, "moves"),
            "items": self._fuse(species, "items"),
            "natures": self._fuse(species, "natures"),
            "abilities": self._fuse(species, "abilities"),
            "tera_types": self._fuse(species, "tera_types"),
            "spreads": self._fuse(species, "spreads"),
        }

        # Step 3: apply Bayesian evidence conditioning
        evidence = {
            "observed_moves": observed_moves,
            "not_seen_moves": not_seen_moves or [],
            "k_revealed": len(observed_moves),
            "confirmed_item": confirmed_item,
            "confirmed_ability": confirmed_ability,
            "confirmed_tera": confirmed_tera,
        }
        conditioned = apply_all_evidence(
            species, raw_dists, evidence, self.component_model.move_pair_counts
        )

        # Step 4: optional damage-calc backward inference (wired in Step 4)
        if attacker_data is not None and observed_damage_pct is not None:
            try:
                from bayesian.damage_inference import DamageInference
                di = DamageInference()
                updated_spreads = di.backward_update(
                    species=species,
                    attacker_data=attacker_data,
                    move_name=attacker_data.get("move", ""),
                    observed_damage_pct=observed_damage_pct,
                    fused_spread_dist=conditioned.get("spreads", {}),
                )
                if updated_spreads:
                    conditioned["spreads"] = updated_spreads
            except Exception:
                pass  # damage inference is optional; never fail the main prediction

        def _sorted(dist: Dict[str, float]) -> List[Tuple[str, float]]:
            return sorted(dist.items(), key=lambda x: x[1], reverse=True)

        # Re-inject confirmed moves at probability 1.0 for the output list
        move_dist = conditioned.get("moves", {})
        final_moves = {m: 1.0 for m in observed_moves if m}
        final_moves.update({m: p for m, p in move_dist.items() if m not in final_moves})
        moves_sorted = sorted(final_moves.items(), key=lambda x: x[1], reverse=True)

        # Build summary for LLM consumption
        summary_dists = {
            "moves": {m: p for m, p in moves_sorted if m not in set(observed_moves)},
            "items": dict(conditioned.get("items", {})),
            "abilities": dict(conditioned.get("abilities", {})),
            "tera_types": dict(conditioned.get("tera_types", {})),
        }
        summary = build_summary(summary_dists, evidence)

        return {
            "species": species,
            "moves": moves_sorted,
            "items": _sorted(conditioned.get("items", {})),
            "natures": _sorted(conditioned.get("natures", {})),
            "abilities": _sorted(conditioned.get("abilities", {})),
            "tera_types": _sorted(conditioned.get("tera_types", {})),
            "ev_spreads": _sorted(conditioned.get("spreads", {})),
            "revealed_moves": observed_moves,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def _save_cache(self):
        """Persist trained model to cache (version-stamped)."""
        cm = self.component_model

        def _serialise_pair_counts(d):
            # Tuple keys are picklable; convert inner Counters to plain dicts
            return {k: dict(v) for k, v in d.items()}

        cache_data = {
            "version": CACHE_VERSION,
            "total_teams": self.total_teams,
            "species_counts": dict(self.species_counts),
            "teammate_counts": {k: dict(v) for k, v in self.teammate_counts.items()},
            "archetype_model": self.archetype_model.to_dict(),
            "component_model": {
                "move_counts": {k: dict(v) for k, v in cm.move_counts.items()},
                "item_counts": {k: dict(v) for k, v in cm.item_counts.items()},
                "ability_counts": {k: dict(v) for k, v in cm.ability_counts.items()},
                "nature_counts": {k: dict(v) for k, v in cm.nature_counts.items()},
                "spread_counts": {k: dict(v) for k, v in cm.spread_counts.items()},
                "tera_counts": {k: dict(v) for k, v in cm.tera_counts.items()},
                "move_pair_counts": _serialise_pair_counts(cm.move_pair_counts),
                "replay_counts": dict(cm.replay_counts),
            },
        }

        with open(self.cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"Model cached to {self.cache_path}")

    def _load_cache(self) -> bool:
        """
        Load trained model from cache.

        Returns True on success; False if cache is missing or has a version
        mismatch (caller should retrain).
        """
        try:
            with open(self.cache_path, "rb") as f:
                cache_data = pickle.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read cache: {e}")
            return False

        if cache_data.get("version") != CACHE_VERSION:
            print(
                f"Cache version mismatch (have {cache_data.get('version')}, "
                f"need {CACHE_VERSION}). Deleting stale cache; will retrain..."
            )
            try:
                os.remove(self.cache_path)
            except OSError:
                pass
            return False  # caller will retrain and save

        self.total_teams = cache_data["total_teams"]
        self.species_counts = Counter(cache_data["species_counts"])

        self.teammate_counts = defaultdict(Counter)
        for k, v in cache_data["teammate_counts"].items():
            self.teammate_counts[k] = Counter(v)

        # Restore ArchetypeModel
        if "archetype_model" in cache_data:
            self.archetype_model = ArchetypeModel.from_dict(cache_data["archetype_model"])

        # Restore ComponentModel
        cm_data = cache_data["component_model"]
        cm = ComponentModel()
        for k, v in cm_data["move_counts"].items():
            cm.move_counts[k] = Counter(v)
        for k, v in cm_data["item_counts"].items():
            cm.item_counts[k] = Counter(v)
        for k, v in cm_data["ability_counts"].items():
            cm.ability_counts[k] = Counter(v)
        for k, v in cm_data["nature_counts"].items():
            cm.nature_counts[k] = Counter(v)
        for k, v in cm_data["spread_counts"].items():
            cm.spread_counts[k] = Counter(v)
        for k, v in cm_data["tera_counts"].items():
            cm.tera_counts[k] = Counter(v)
        for k, v in cm_data["move_pair_counts"].items():
            cm.move_pair_counts[k] = Counter(v)
        cm.replay_counts = Counter(cm_data["replay_counts"])
        self.component_model = cm

        print(f"Loaded model trained on {self.total_teams} teams")
        return True


def main():
    """Quick smoke test for the team predictor."""
    predictor = BayesianTeamPredictor()
    predictor.load_and_train(force_retrain=False)

    revealed_species = ["Gliscor", "Latios", "Zamazenta"]
    print(f"\nGiven revealed Pokemon: {revealed_species}")

    predictions = predictor.predict_unrevealed_pokemon(revealed_species)
    print("\nMost likely unrevealed teammates:")
    for species, prob in predictions:
        print(f"  {species}: {prob:.4e}")

    if predictions:
        test_species = predictions[0][0]
        config = predictor.predict_pokemon_config(test_species, revealed_species)
        print(f"\nPredicted config for {test_species}:")
        for key, value in config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
