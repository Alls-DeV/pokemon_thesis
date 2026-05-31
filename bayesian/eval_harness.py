#!/usr/bin/env python3
"""
Held-out evaluation harness for the Bayesian team predictor.

Measures top-1 and top-3 accuracy for:
  (a) Unrevealed species prediction given k=1..5 revealed Pokemon
  (b) Move prediction given 1-3 observed moves
  (c) Item prediction given species + all moves

Usage:
  uv run python bayesian/eval_harness.py               # use loaded cache
  uv run python bayesian/eval_harness.py --retrain     # retrain on 90% split
  uv run python bayesian/eval_harness.py --max-teams 2000  # quick smoke-test
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running from project root
project_root = Path(__file__).parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bayesian.team_predictor import BayesianTeamPredictor, TeamParser

_SEED = 42
_TEST_FRACTION = 0.10


def _accuracy_at_k(predicted: List[Tuple[str, float]], truth: str, k: int) -> bool:
    """True if `truth` appears in the top-k predictions."""
    return any(name == truth for name, _ in predicted[:k])


class EvalHarness:
    """90/10 train-test evaluation for BayesianTeamPredictor."""

    def __init__(self, predictor: BayesianTeamPredictor, test_teams):
        self.predictor = predictor
        self.test_teams = test_teams  # list of TeamData
        self.parser = TeamParser()

    # ------------------------------------------------------------------
    # Individual task evaluators
    # ------------------------------------------------------------------

    def eval_species_prediction(
        self, max_k: int = 5
    ) -> Dict[int, Dict[str, float]]:
        """
        For k=1..max_k revealed Pokemon on a test team, predict the rest.

        Returns {k: {"top1": acc, "top3": acc, "n": count}}.
        """
        results: Dict[int, Dict] = defaultdict(lambda: {"top1": 0, "top3": 0, "n": 0})

        for team_data in self.test_teams:
            species_list = [p.species for p in team_data.pokemon if p.species]
            if len(species_list) < 2:
                continue

            for k in range(1, min(max_k + 1, len(species_list))):
                revealed = species_list[:k]
                ground_truth = species_list[k]  # first unrevealed species

                preds = self.predictor.predict_unrevealed_pokemon(revealed, max_predictions=20)
                r = results[k]
                r["n"] += 1
                if _accuracy_at_k(preds, ground_truth, 1):
                    r["top1"] += 1
                if _accuracy_at_k(preds, ground_truth, 3):
                    r["top3"] += 1

        return {
            k: {"top1": v["top1"] / max(v["n"], 1),
                "top3": v["top3"] / max(v["n"], 1),
                "n": v["n"]}
            for k, v in results.items()
        }

    def eval_move_prediction(
        self, max_observed: int = 3
    ) -> Dict[int, Dict[str, float]]:
        """
        For 1..max_observed observed moves, predict the next move.

        Returns {n_observed: {"top1": acc, "top3": acc, "n": count}}.
        """
        results: Dict[int, Dict] = defaultdict(lambda: {"top1": 0, "top3": 0, "n": 0})

        for team_data in self.test_teams:
            for pokemon in team_data.pokemon:
                species = pokemon.species
                moves = [m for m in pokemon.moves if m]
                if len(moves) < 2:
                    continue

                for k in range(1, min(max_observed + 1, len(moves))):
                    observed = moves[:k]
                    ground_truth = moves[k]  # first unobserved move

                    result = self.predictor.predict_component_probabilities(
                        species, observed_moves=observed
                    )
                    if "error" in result or not result.get("moves"):
                        continue

                    preds = [(m, p) for m, p in result["moves"] if m not in observed]
                    r = results[k]
                    r["n"] += 1
                    if _accuracy_at_k(preds, ground_truth, 1):
                        r["top1"] += 1
                    if _accuracy_at_k(preds, ground_truth, 3):
                        r["top3"] += 1

        return {
            k: {"top1": v["top1"] / max(v["n"], 1),
                "top3": v["top3"] / max(v["n"], 1),
                "n": v["n"]}
            for k, v in results.items()
        }

    def eval_item_prediction(self) -> Dict[str, float]:
        """
        Given all 4 moves (fully revealed moveset), predict item.

        Returns {"top1": acc, "top3": acc, "n": count}.
        """
        top1, top3, n = 0, 0, 0

        for team_data in self.test_teams:
            for pokemon in team_data.pokemon:
                if not pokemon.item:
                    continue
                species = pokemon.species
                moves = [m for m in pokemon.moves if m]

                result = self.predictor.predict_component_probabilities(
                    species, observed_moves=moves
                )
                if "error" in result or not result.get("items"):
                    continue

                n += 1
                truth = pokemon.item
                if _accuracy_at_k(result["items"], truth, 1):
                    top1 += 1
                if _accuracy_at_k(result["items"], truth, 3):
                    top3 += 1

        denom = max(n, 1)
        return {"top1": top1 / denom, "top3": top3 / denom, "n": n}


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_results(
    species_results, move_results, item_results, label: str = ""
):
    header = f"\n{'='*60}\nEvaluation results{': ' + label if label else ''}\n{'='*60}"
    print(header)

    print("\nSpecies prediction (given k revealed → predict next):")
    print(f"  {'k':>3}  {'top-1':>7}  {'top-3':>7}  {'n':>8}")
    for k in sorted(species_results):
        r = species_results[k]
        print(f"  {k:>3}  {_fmt(r['top1']):>7}  {_fmt(r['top3']):>7}  {r['n']:>8,}")

    print("\nMove prediction (given k observed → predict next):")
    print(f"  {'k':>3}  {'top-1':>7}  {'top-3':>7}  {'n':>8}")
    for k in sorted(move_results):
        r = move_results[k]
        print(f"  {k:>3}  {_fmt(r['top1']):>7}  {_fmt(r['top3']):>7}  {r['n']:>8,}")

    print("\nItem prediction (all moves given → predict item):")
    r = item_results
    print(f"  top-1: {_fmt(r['top1'])}  top-3: {_fmt(r['top3'])}  n={r['n']:,}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Bayesian team predictor")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain on 90%% of teams (unbiased but slow)")
    parser.add_argument("--max-teams", type=int, default=None,
                        help="Limit number of test teams for quick smoke-tests")
    args = parser.parse_args()

    cache_dir = os.getenv("METAMON_CACHE_DIR", "/tmp/metamon_cache")
    predictor = BayesianTeamPredictor()

    if args.retrain:
        from poke_env.player.team_util import get_metamon_teams
        team_set = get_metamon_teams("gen9ou", "gl_05_26", version="main")
        team_files = team_set.team_files

        rng = random.Random(_SEED)
        all_files = list(team_files)
        rng.shuffle(all_files)
        split = int(len(all_files) * (1.0 - _TEST_FRACTION))
        train_files = all_files[:split]
        test_files = all_files[split:]

        print(f"Training on {len(train_files):,} teams, testing on {len(test_files):,} teams")
        tp = TeamParser()
        test_teams = []
        for fp in test_files:
            try:
                test_teams.append(tp.parse_team_file(fp))
            except Exception:
                continue

        # Train only on the 90% split
        from bayesian.archetype_model import ArchetypeModel
        from bayesian.component_model import ComponentModel
        from bayesian.smogon_prior import SmogonPrior
        from collections import Counter, defaultdict
        from tqdm import tqdm

        predictor.component_model = ComponentModel()
        predictor.species_counts = Counter()
        predictor.teammate_counts = defaultdict(Counter)
        predictor.total_teams = 0
        predictor._team_corpus = []

        for fp in tqdm(train_files, desc="Training"):
            try:
                td = tp.parse_team_file(fp)
                predictor._update_counts(td)
                predictor.total_teams += 1
            except Exception:
                continue

        print("Training archetype model...")
        predictor.archetype_model = ArchetypeModel()
        predictor.archetype_model.train(predictor._team_corpus)
        predictor._team_corpus = []
        predictor._init_smogon_prior()
        predictor.is_trained = True
    else:
        predictor.load_and_train(force_retrain=False)
        from poke_env.player.team_util import get_metamon_teams
        team_set = get_metamon_teams("gen9ou", "gl_05_26", version="main")
        team_files = list(team_set.team_files)

        rng = random.Random(_SEED)
        rng.shuffle(team_files)
        split = int(len(team_files) * (1.0 - _TEST_FRACTION))
        test_files = team_files[split:]

        tp = TeamParser()
        test_teams = []
        for fp in test_files:
            try:
                test_teams.append(tp.parse_team_file(fp))
            except Exception:
                continue

    if args.max_teams:
        test_teams = test_teams[: args.max_teams]

    print(f"Evaluating on {len(test_teams):,} test teams...")
    harness = EvalHarness(predictor, test_teams)

    species_r = harness.eval_species_prediction()
    move_r = harness.eval_move_prediction()
    item_r = harness.eval_item_prediction()

    label = "90/10 retrain split" if args.retrain else "pretrained cache (biased toward training set)"
    print_results(species_r, move_r, item_r, label=label)


if __name__ == "__main__":
    main()
