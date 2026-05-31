#!/usr/bin/env python3
"""
Comparative benchmark: Old Bayesian predictor vs. Redesigned Bayesian predictor.

Measures top-1 and top-3 accuracy on a held-out 10% test split for:
  (a) Species prediction given k=1..5 revealed Pokemon
  (b) Move prediction given 1-3 observed moves
  (c) Item / ability / tera type prediction (all moves given)
  (d) Confidence calibration of the new predictor (new only)

Outputs:
  benchmark_results/fig1_main_comparison.png    — headline bar chart
  benchmark_results/fig2_species_by_k.png       — accuracy vs k revealed
  benchmark_results/fig3_move_by_k.png          — accuracy vs k observed
  benchmark_results/fig4_component_breakdown.png — per-component accuracy
  benchmark_results/fig5_confidence_calibration.png — calibration (new only)
  benchmark_results/results_table.md            — markdown table

Usage:
  .venv/bin/python bayesian/benchmark_comparison.py
  .venv/bin/python bayesian/benchmark_comparison.py --retrain        # force retrain both
  .venv/bin/python bayesian/benchmark_comparison.py --max-teams 500  # quick smoke-test
"""

import argparse
import os
import pickle
import random
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from bayesian.team_predictor import BayesianTeamPredictor as NewPredictor
from bayesian.team_predictor import TeamParser
from old_bayesian.team_predictor import BayesianTeamPredictor as OldPredictor
from old_bayesian.team_predictor import TeamParser as OldTeamParser

_SEED = 42
_TEST_FRACTION = 0.10
_OUT_DIR = Path(__file__).parent / "benchmark_results"


# ── Fast wrapper for the old predictor ───────────────────────────────────────
# The old predictor calls eval() inside predict_component_probabilities for
# every config key on every call — O(N_configs × eval) per prediction.
# With 125k training teams and ~250k eval calls this takes 10+ hours.
# This wrapper pre-parses all config keys once at startup (~30s) and uses
# a plain dict for all subsequent lookups, reducing eval time to <10 minutes.

class _OldPredictorFast:
    """
    Thin wrapper around OldPredictor that pre-caches parsed config keys so
    that predict_component_probabilities is fast during evaluation.
    Delegates predict_unrevealed_pokemon to the wrapped predictor unchanged.
    """

    def __init__(self, old_pred: OldPredictor):
        self._pred = old_pred
        self.total_teams = old_pred.total_teams
        self.is_trained = old_pred.is_trained
        self._cache: Dict = {}  # species → list of (base_prob, moves_set, item, ability, nature, tera)
        self._build_cache()

    def _build_cache(self):
        print("  Pre-parsing old predictor config keys (one-time, ~30s)...")
        pred = self._pred
        for species, configs in pred.config_given_species.items():
            total = max(sum(configs.values()), 1)
            entries = []
            for ck, count in configs.items():
                try:
                    moves_set = set(pred._extract_moves_from_config_key(ck))
                    parsed = pred._parse_config_key(ck)
                    if not parsed or parsed.get("parse_error"):
                        continue
                    entries.append((
                        count / total,
                        moves_set,
                        parsed.get("item", "") or "",
                        parsed.get("ability", "") or "",
                        parsed.get("nature", "") or "",
                        parsed.get("tera_type", "") or "",
                    ))
                except Exception:
                    continue
            if entries:
                self._cache[species] = entries
        print(f"  Cache built for {len(self._cache)} species.")

    def predict_unrevealed_pokemon(self, revealed_species: List[str],
                                   max_predictions: int = 5) -> List[Tuple[str, float]]:
        return self._pred.predict_unrevealed_pokemon(revealed_species, max_predictions)

    def predict_component_probabilities(self, species: str,
                                        revealed_moves: List[str] = None,
                                        observed_moves: List[str] = None,
                                        **_) -> Dict:
        revealed = revealed_moves or observed_moves or []
        entries = self._cache.get(species)
        if not entries:
            return {"error": f"No data for {species}"}

        revealed_set = set(revealed)
        move_p: Dict[str, float] = {}
        item_p: Dict[str, float] = {}
        ability_p: Dict[str, float] = {}
        nature_p: Dict[str, float] = {}
        tera_p: Dict[str, float] = {}

        for base_prob, moves_set, item, ability, nature, tera in entries:
            if revealed:
                bonus = (1 + len(revealed_set & moves_set)) ** 2
            else:
                bonus = 1.0
            adj = base_prob * bonus

            for m in moves_set:
                move_p[m] = move_p.get(m, 0.0) + adj
            if item:
                item_p[item] = item_p.get(item, 0.0) + adj
            if ability:
                ability_p[ability] = ability_p.get(ability, 0.0) + adj
            if nature:
                nature_p[nature] = nature_p.get(nature, 0.0) + adj
            if tera:
                tera_p[tera] = tera_p.get(tera, 0.0) + adj

        def _ns(d: Dict[str, float], confirmed=None) -> List[Tuple[str, float]]:
            confirmed = set(confirmed or [])
            total = sum(d.values()) or 1.0
            result = {k: (1.0 if k in confirmed else v / total)
                      for k, v in d.items()}
            return sorted(result.items(), key=lambda x: x[1], reverse=True)

        return {
            "species": species,
            "moves": _ns(move_p, revealed),
            "items": _ns(item_p),
            "natures": _ns(nature_p),
            "abilities": _ns(ability_p),
            "tera_types": _ns(tera_p),
            "ev_spreads": [],
            "revealed_moves": revealed,
        }

_CACHE_DIR = Path(os.getenv("METAMON_CACHE_DIR", "/tmp/metamon_cache"))
_OLD_BENCH_CACHE = _CACHE_DIR / "bench_old_90.pkl"
_NEW_BENCH_CACHE = _CACHE_DIR / "bench_new_90.pkl"

# ── Plot style ────────────────────────────────────────────────────────────────
OLD_COLOR = "#e07b54"
NEW_COLOR = "#4c7bb5"
OLD_LABEL = "Old (Naïve Bayes)"
NEW_LABEL = "New (Redesigned)"


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.35,
        "legend.framealpha": 0.9,
    })


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split() -> Tuple[List, List]:
    from poke_env.player.team_util import get_metamon_teams
    team_set = get_metamon_teams("gen9ou", "gl_05_26", version="main")
    all_files = list(team_set.team_files)
    rng = random.Random(_SEED)
    rng.shuffle(all_files)
    split = int(len(all_files) * (1.0 - _TEST_FRACTION))
    return all_files[:split], all_files[split:]


def parse_teams(files: List, parser) -> List:
    teams = []
    for fp in files:
        try:
            teams.append(parser.parse_team_file(fp))
        except Exception:
            pass
    return teams


# ── Training ──────────────────────────────────────────────────────────────────

def train_new(train_files: List) -> NewPredictor:
    from bayesian.component_model import ComponentModel
    from bayesian.archetype_model import ArchetypeModel
    from tqdm import tqdm

    p = NewPredictor(cache_file="bench_new_90.pkl")
    p.component_model = ComponentModel()
    p.species_counts = Counter()
    p.teammate_counts = defaultdict(Counter)
    p.total_teams = 0
    p._team_corpus = []

    tp = TeamParser()
    for fp in tqdm(train_files, desc="  Training NEW predictor"):
        try:
            td = tp.parse_team_file(fp)
            p._update_counts(td)
            p.total_teams += 1
        except Exception:
            pass

    print("  Training archetype model...")
    p.archetype_model = ArchetypeModel()
    p.archetype_model.train(p._team_corpus)
    p._team_corpus = []
    p._init_smogon_prior()
    p.is_trained = True
    return p


def train_old(train_files: List) -> OldPredictor:
    from tqdm import tqdm
    p = OldPredictor()
    tp = OldTeamParser()
    for fp in tqdm(train_files, desc="  Training OLD predictor"):
        try:
            td = tp.parse_team_file(fp)
            p._update_counts(td)
            p.total_teams += 1
        except Exception:
            pass
    p.is_trained = True
    return p


# ── Cache I/O ─────────────────────────────────────────────────────────────────

def save_old(p: OldPredictor):
    data = {
        "species_counts": dict(p.species_counts),
        "teammate_counts": {k: dict(v) for k, v in p.teammate_counts.items()},
        "config_given_species": {k: dict(v) for k, v in p.config_given_species.items()},
        "move_given_species": {k: dict(v) for k, v in p.move_given_species.items()},
        "move_pairs": {str(k): dict(v) for k, v in p.move_pairs.items()},
        "total_teams": p.total_teams,
    }
    with open(_OLD_BENCH_CACHE, "wb") as f:
        pickle.dump(data, f)


def load_old(p: OldPredictor):
    with open(_OLD_BENCH_CACHE, "rb") as f:
        data = pickle.load(f)
    p.species_counts = Counter(data["species_counts"])
    p.teammate_counts = defaultdict(Counter)
    for k, v in data["teammate_counts"].items():
        p.teammate_counts[k] = Counter(v)
    p.config_given_species = defaultdict(Counter)
    for k, v in data["config_given_species"].items():
        p.config_given_species[k] = Counter(v)
    p.move_given_species = defaultdict(Counter)
    for k, v in data.get("move_given_species", {}).items():
        p.move_given_species[k] = Counter(v)
    p.total_teams = data["total_teams"]
    p.is_trained = True


# ── Evaluation ────────────────────────────────────────────────────────────────

def _hit(predicted: List[Tuple[str, float]], truth: str, k: int) -> bool:
    return any(name == truth for name, _ in predicted[:k])


def _predict_components(predictor, species: str, observed_moves: List[str]) -> Dict:
    """Unified call — both the fast old wrapper and new predictor accept observed_moves=."""
    return predictor.predict_component_probabilities(
        species, observed_moves=observed_moves
    )


def _ts() -> str:
    """Current time as HH:MM:SS."""
    return time.strftime("%H:%M:%S")


def eval_species(predictor, test_teams, max_k: int = 5, label: str = "") -> Dict:
    results = defaultdict(lambda: {"top1": 0, "top3": 0, "n": 0})
    t0 = time.time()
    for td in tqdm(test_teams, desc=f"    species {label}", unit="team", leave=False):
        species_list = [p.species for p in td.pokemon if p.species]
        if len(species_list) < 2:
            continue
        for k in range(1, min(max_k + 1, len(species_list))):
            preds = predictor.predict_unrevealed_pokemon(
                species_list[:k], max_predictions=20
            )
            truth = species_list[k]
            r = results[k]
            r["n"] += 1
            if _hit(preds, truth, 1):
                r["top1"] += 1
            if _hit(preds, truth, 3):
                r["top3"] += 1
    elapsed = time.time() - t0
    out = {k: {"top1": v["top1"] / max(v["n"], 1),
               "top3": v["top3"] / max(v["n"], 1),
               "n": v["n"]} for k, v in results.items()}
    print(f"    [{_ts()}] species done in {elapsed:.0f}s")
    for k in sorted(out):
        r = out[k]
        print(f"      k={k}  top-1={r['top1']*100:.1f}%  top-3={r['top3']*100:.1f}%  n={r['n']:,}")
    return out


def eval_move(predictor, test_teams, max_observed: int = 3, label: str = "") -> Dict:
    results = defaultdict(lambda: {"top1": 0, "top3": 0, "n": 0})
    t0 = time.time()
    for td in tqdm(test_teams, desc=f"    move    {label}", unit="team", leave=False):
        for pokemon in td.pokemon:
            moves = [m for m in pokemon.moves if m]
            if len(moves) < 2:
                continue
            for k in range(1, min(max_observed + 1, len(moves))):
                observed = moves[:k]
                truth = moves[k]
                result = _predict_components(predictor, pokemon.species, observed)
                if isinstance(result, dict) and "error" in result:
                    continue
                preds = [(m, p) for m, p in result.get("moves", [])
                         if m not in observed]
                r = results[k]
                r["n"] += 1
                if _hit(preds, truth, 1):
                    r["top1"] += 1
                if _hit(preds, truth, 3):
                    r["top3"] += 1
    elapsed = time.time() - t0
    out = {k: {"top1": v["top1"] / max(v["n"], 1),
               "top3": v["top3"] / max(v["n"], 1),
               "n": v["n"]} for k, v in results.items()}
    print(f"    [{_ts()}] move done in {elapsed:.0f}s")
    for k in sorted(out):
        r = out[k]
        print(f"      k={k}  top-1={r['top1']*100:.1f}%  top-3={r['top3']*100:.1f}%  n={r['n']:,}")
    return out


def eval_component(predictor, test_teams, attr: str, label: str = "") -> Dict:
    """Predict item / ability / tera_type given all 4 moves."""
    key_map = {"item": "items", "ability": "abilities", "tera_type": "tera_types"}
    result_key = key_map[attr]
    top1, top3, n = 0, 0, 0
    t0 = time.time()
    for td in tqdm(test_teams, desc=f"    {attr:<12}{label}", unit="team", leave=False):
        for pokemon in td.pokemon:
            truth = getattr(pokemon, attr, None)
            if not truth:
                continue
            moves = [m for m in pokemon.moves if m]
            result = _predict_components(predictor, pokemon.species, moves)
            if isinstance(result, dict) and "error" in result:
                continue
            preds = result.get(result_key, [])
            n += 1
            if _hit(preds, truth, 1):
                top1 += 1
            if _hit(preds, truth, 3):
                top3 += 1
    elapsed = time.time() - t0
    d = max(n, 1)
    out = {"top1": top1 / d, "top3": top3 / d, "n": n}
    print(f"    [{_ts()}] {attr} done in {elapsed:.0f}s  "
          f"top-1={out['top1']*100:.1f}%  top-3={out['top3']*100:.1f}%  n={n:,}")
    return out


def eval_confidence_calibration(new_pred, test_teams) -> Dict:
    """Actual accuracy per confidence label (Confident/Likely/Uncertain)."""
    buckets = {label: {"correct": 0, "total": 0}
               for label in ("Confident", "Likely", "Uncertain")}
    attr_to_key = {"item": "items", "ability": "abilities", "tera_type": "tera_types"}
    t0 = time.time()
    for td in tqdm(test_teams, desc="    calibration", unit="team", leave=False):
        for pokemon in td.pokemon:
            moves = [m for m in pokemon.moves if m]
            result = _predict_components(new_pred, pokemon.species, moves)
            if isinstance(result, dict) and "error" in result:
                continue
            summary = result.get("summary", {})
            for attr, key in attr_to_key.items():
                truth = getattr(pokemon, attr, None)
                if not truth or key not in summary:
                    continue
                s = summary[key]
                lbl = s.get("confidence", "Uncertain")
                top_name = s["top"][0]
                buckets[lbl]["total"] += 1
                if top_name == truth:
                    buckets[lbl]["correct"] += 1
    elapsed = time.time() - t0
    out = {lbl: {"accuracy": v["correct"] / max(v["total"], 1), "n": v["total"]}
           for lbl, v in buckets.items()}
    print(f"    [{_ts()}] calibration done in {elapsed:.0f}s")
    for lbl in ("Confident", "Likely", "Uncertain"):
        c = out[lbl]
        print(f"      {lbl:<12}  acc={c['accuracy']*100:.1f}%  n={c['n']:,}")
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    _OUT_DIR.mkdir(exist_ok=True)
    path = _OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def _avg(d: Dict, key: str) -> float:
    vals = [v[key] for v in d.values() if v["n"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def plot_main_comparison(old_sp, new_sp, old_mv, new_mv,
                         old_item, new_item):
    """Fig 1: Headline grouped bar chart."""
    _apply_style()
    metrics = ["Species\ntop-1", "Species\ntop-3",
               "Move\ntop-1",    "Move\ntop-3",
               "Item\ntop-1",    "Item\ntop-3"]
    old_vals = [_avg(old_sp, "top1"), _avg(old_sp, "top3"),
                _avg(old_mv, "top1"), _avg(old_mv, "top3"),
                old_item["top1"],     old_item["top3"]]
    new_vals = [_avg(new_sp, "top1"), _avg(new_sp, "top3"),
                _avg(new_mv, "top1"), _avg(new_mv, "top3"),
                new_item["top1"],     new_item["top3"]]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    b_old = ax.bar(x - w / 2, [v * 100 for v in old_vals],
                   w, label=OLD_LABEL, color=OLD_COLOR, alpha=0.9)
    b_new = ax.bar(x + w / 2, [v * 100 for v in new_vals],
                   w, label=NEW_LABEL, color=NEW_COLOR, alpha=0.9)
    for bar in list(b_old) + list(b_new):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, min(105, max(new_vals) * 100 + 18))
    ax.set_title("Prediction Accuracy: Old vs. Redesigned Bayesian Predictor", pad=12)
    ax.legend()
    _save(fig, "fig1_main_comparison.png")


def plot_species_by_k(old_sp, new_sp):
    """Fig 2: Species accuracy vs k revealed, top-1 and top-3."""
    _apply_style()
    ks = sorted(set(old_sp) | set(new_sp))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(
        axes, ["top1", "top3"], ["Top-1 Accuracy", "Top-3 Accuracy"]
    ):
        old_y = [old_sp.get(k, {}).get(metric, 0) * 100 for k in ks]
        new_y = [new_sp.get(k, {}).get(metric, 0) * 100 for k in ks]
        ax.plot(ks, old_y, "o--", color=OLD_COLOR, label=OLD_LABEL, linewidth=2)
        ax.plot(ks, new_y, "s-",  color=NEW_COLOR, label=NEW_LABEL, linewidth=2)
        ax.set_xlabel("Revealed Pokémon (k)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Species Prediction — {title}")
        ax.set_xticks(ks)
        ax.legend(fontsize=9)
    fig.suptitle("Species Prediction Accuracy vs. Number of Revealed Pokémon", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_species_by_k.png")


def plot_move_by_k(old_mv, new_mv):
    """Fig 3: Move accuracy vs k observed, top-1 and top-3."""
    _apply_style()
    ks = sorted(set(old_mv) | set(new_mv))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, metric, title in zip(
        axes, ["top1", "top3"], ["Top-1 Accuracy", "Top-3 Accuracy"]
    ):
        old_y = [old_mv.get(k, {}).get(metric, 0) * 100 for k in ks]
        new_y = [new_mv.get(k, {}).get(metric, 0) * 100 for k in ks]
        ax.plot(ks, old_y, "o--", color=OLD_COLOR, label=OLD_LABEL, linewidth=2)
        ax.plot(ks, new_y, "s-",  color=NEW_COLOR, label=NEW_LABEL, linewidth=2)
        ax.set_xlabel("Observed Moves (k)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Move Prediction — {title}")
        ax.set_xticks(ks)
        ax.legend(fontsize=9)
    fig.suptitle("Move Prediction Accuracy vs. Number of Observed Moves", y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_move_by_k.png")


def plot_component_breakdown(comp: Dict):
    """Fig 4: Item / ability / tera top-1 and top-3."""
    _apply_style()
    components = ["item", "ability", "tera_type"]
    labels = ["Item", "Ability", "Tera Type"]
    x = np.arange(len(components))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(
        axes, ["top1", "top3"], ["Top-1 Accuracy", "Top-3 Accuracy"]
    ):
        old_v = [comp["old"][c][metric] * 100 for c in components]
        new_v = [comp["new"][c][metric] * 100 for c in components]
        b_o = ax.bar(x - w / 2, old_v, w, label=OLD_LABEL, color=OLD_COLOR, alpha=0.9)
        b_n = ax.bar(x + w / 2, new_v, w, label=NEW_LABEL, color=NEW_COLOR, alpha=0.9)
        for bar in list(b_o) + list(b_n):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle("Per-Component Prediction Accuracy (Given All 4 Moves)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_component_breakdown.png")


def plot_confidence_calibration(calib: Dict):
    """Fig 5: Confidence calibration bar chart for new predictor."""
    _apply_style()
    labels = ["Confident", "Likely", "Uncertain"]
    accs = [calib[l]["accuracy"] * 100 for l in labels]
    counts = [calib[l]["n"] for l in labels]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, accs, color=colors, alpha=0.85, width=0.5)
    for bar, acc, n in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Actual Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Confidence Calibration — Redesigned Predictor\n"
                 "(Item, Ability, Tera Type predictions)", pad=10)
    ax.axhline(60, color="grey", linestyle=":", linewidth=1,
               label="Confident threshold (≥60%)")
    ax.axhline(35, color="grey", linestyle="--", linewidth=1,
               label="Likely threshold (≥35%)")
    ax.legend(fontsize=8)
    _save(fig, "fig5_confidence_calibration.png")


# ── File output ───────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def write_markdown(old_sp, new_sp, old_mv, new_mv, comp, calib):
    _OUT_DIR.mkdir(exist_ok=True)

    def delta(n_val, o_val):
        d = (n_val - o_val) * 100
        return f"{d:+.1f}pp"

    lines = [
        "# Benchmark Results: Old vs. Redesigned Bayesian Predictor",
        "",
        "## Species Prediction",
        "",
        "| k | Old top-1 | New top-1 | Δ top-1 | Old top-3 | New top-3 | Δ top-3 | n |",
        "|---|-----------|-----------|---------|-----------|-----------|---------|---|",
    ]
    for k in sorted(set(old_sp) | set(new_sp)):
        o = old_sp.get(k, {"top1": 0, "top3": 0, "n": 0})
        n = new_sp.get(k, {"top1": 0, "top3": 0, "n": 0})
        lines.append(f"| {k} | {_pct(o['top1'])} | {_pct(n['top1'])} | {delta(n['top1'], o['top1'])} | "
                     f"{_pct(o['top3'])} | {_pct(n['top3'])} | {delta(n['top3'], o['top3'])} | {n['n']:,} |")

    lines += [
        "",
        "## Move Prediction",
        "",
        "| k | Old top-1 | New top-1 | Δ top-1 | Old top-3 | New top-3 | Δ top-3 | n |",
        "|---|-----------|-----------|---------|-----------|-----------|---------|---|",
    ]
    for k in sorted(set(old_mv) | set(new_mv)):
        o = old_mv.get(k, {"top1": 0, "top3": 0, "n": 0})
        n = new_mv.get(k, {"top1": 0, "top3": 0, "n": 0})
        lines.append(f"| {k} | {_pct(o['top1'])} | {_pct(n['top1'])} | {delta(n['top1'], o['top1'])} | "
                     f"{_pct(o['top3'])} | {_pct(n['top3'])} | {delta(n['top3'], o['top3'])} | {n['n']:,} |")

    lines += [
        "",
        "## Component Prediction (given all 4 moves)",
        "",
        "| Component | Old top-1 | New top-1 | Δ | Old top-3 | New top-3 | Δ | n |",
        "|-----------|-----------|-----------|---|-----------|-----------|---|---|",
    ]
    for attr in ("item", "ability", "tera_type"):
        o = comp["old"][attr]
        n = comp["new"][attr]
        label = attr.replace("_", " ").title()
        lines.append(f"| {label} | {_pct(o['top1'])} | {_pct(n['top1'])} | {delta(n['top1'], o['top1'])} | "
                     f"{_pct(o['top3'])} | {_pct(n['top3'])} | {delta(n['top3'], o['top3'])} | {n['n']:,} |")

    lines += [
        "",
        "## Confidence Calibration (New Predictor — Item, Ability, Tera Type)",
        "",
        "| Confidence Label | Accuracy | n |",
        "|------------------|----------|---|",
    ]
    for lbl in ("Confident", "Likely", "Uncertain"):
        c = calib[lbl]
        lines.append(f"| {lbl} | {_pct(c['accuracy'])} | {c['n']:,} |")

    out = _OUT_DIR / "results_table.md"
    out.write_text("\n".join(lines))
    print(f"  Saved {out}")


def write_summary(old_sp, new_sp, old_mv, new_mv, comp, calib, elapsed_min: float):
    _OUT_DIR.mkdir(exist_ok=True)
    lines = [
        "BAYESIAN PREDICTOR BENCHMARK — RESULTS SUMMARY",
        "=" * 65,
        "",
        f"  {'Metric':<30} {'Old':>8} {'New':>8} {'Δ':>8}",
        "  " + "-" * 56,
    ]
    rows = [
        ("Species top-1 (avg k=1..5)", _avg(old_sp, "top1"), _avg(new_sp, "top1")),
        ("Species top-3 (avg k=1..5)", _avg(old_sp, "top3"), _avg(new_sp, "top3")),
        ("Move top-1 (avg k=1..3)",    _avg(old_mv, "top1"), _avg(new_mv, "top1")),
        ("Move top-3 (avg k=1..3)",    _avg(old_mv, "top3"), _avg(new_mv, "top3")),
        ("Item top-1",                 comp["old"]["item"]["top1"],    comp["new"]["item"]["top1"]),
        ("Item top-3",                 comp["old"]["item"]["top3"],    comp["new"]["item"]["top3"]),
        ("Ability top-1",              comp["old"]["ability"]["top1"], comp["new"]["ability"]["top1"]),
        ("Tera Type top-1",            comp["old"]["tera_type"]["top1"], comp["new"]["tera_type"]["top1"]),
    ]
    for label, o, n in rows:
        d = (n - o) * 100
        sign = "+" if d >= 0 else ""
        lines.append(f"  {label:<30} {o * 100:>7.1f}%  {n * 100:>7.1f}%  {sign}{d:>5.1f}pp")

    lines += [
        "",
        "  Confidence calibration (new predictor only):",
    ]
    for label in ("Confident", "Likely", "Uncertain"):
        c = calib[label]
        lines.append(f"    {label:<12}  acc={c['accuracy'] * 100:.1f}%   n={c['n']:,}")

    lines += ["", f"  Total evaluation time: {elapsed_min:.1f} min", ""]

    out = _OUT_DIR / "summary.txt"
    out.write_text("\n".join(lines))
    print(f"  Saved {out}")


def write_raw_data(old_sp, new_sp, old_mv, new_mv, comp, calib):
    _OUT_DIR.mkdir(exist_ok=True)

    def _round_dict(d):
        return {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                    for kk, vv in v.items()} for k, v in d.items()}

    lines = [
        "# Raw benchmark data — paste into a Python session to recreate any plot",
        "# Generated by bayesian/benchmark_comparison.py",
        "",
        f"old_sp = {_round_dict(old_sp)}",
        f"new_sp = {_round_dict(new_sp)}",
        f"old_mv = {_round_dict(old_mv)}",
        f"new_mv = {_round_dict(new_mv)}",
        f"comp = {{{', '.join(repr(side) + ': ' + repr({attr: {kk: round(vv, 6) if isinstance(vv, float) else vv for kk, vv in v.items()} for attr, v in comp[side].items()}) for side in ('old', 'new'))}}}",
        f"calib = {{{', '.join(repr(lbl) + ': ' + repr({'accuracy': round(v['accuracy'], 6), 'n': v['n']}) for lbl, v in calib.items())}}}",
    ]

    out = _OUT_DIR / "raw_data.py"
    out.write_text("\n".join(lines))
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: old vs. new Bayesian predictor")
    parser.add_argument("--retrain", action="store_true",
                        help="Ignore caches and retrain both predictors")
    parser.add_argument("--max-teams", type=int, default=None,
                        help="Cap test teams (for quick smoke-tests)")
    args = parser.parse_args()

    print("=" * 65)
    print("BAYESIAN PREDICTOR BENCHMARK: OLD vs. NEW")
    print("=" * 65)

    print("\n[1/5] Loading 90/10 split...")
    train_files, test_files = load_split()
    print(f"  Train: {len(train_files):,}   Test: {len(test_files):,}")

    # ── OLD predictor ──────────────────────────────────────────────────
    print("\n[2/5] Preparing OLD predictor...")
    old_pred = OldPredictor()
    if not args.retrain and _OLD_BENCH_CACHE.exists():
        print(f"  Loading from {_OLD_BENCH_CACHE}")
        load_old(old_pred)
    else:
        old_pred = train_old(train_files)
        save_old(old_pred)
        print(f"  Cached to {_OLD_BENCH_CACHE}")
    print(f"  Trained on {old_pred.total_teams:,} teams")
    old_pred = _OldPredictorFast(old_pred)  # replace with fast wrapper

    # ── NEW predictor ──────────────────────────────────────────────────
    print("\n[3/5] Preparing NEW predictor...")
    new_pred = NewPredictor(cache_file="bench_new_90.pkl")
    if not args.retrain and _NEW_BENCH_CACHE.exists():
        print(f"  Loading from {_NEW_BENCH_CACHE}")
        new_pred.load_and_train(force_retrain=False)
    else:
        new_pred = train_new(train_files)
        new_pred._save_cache()
        print(f"  Cached to {_NEW_BENCH_CACHE}")
    print(f"  Trained on {new_pred.total_teams:,} teams")

    # ── Test teams ──────────────────────────────────────────────────────
    print("\n[4/5] Parsing test teams...")
    test_teams = parse_teams(test_files, TeamParser())
    if args.max_teams:
        test_teams = test_teams[:args.max_teams]
    print(f"  Using {len(test_teams):,} test teams")

    # ── Evaluation ──────────────────────────────────────────────────────
    t_eval_start = time.time()
    print(f"\n[5/5] Evaluating...  started at {_ts()}")

    print(f"\n  [{_ts()}] Species prediction (k=1..5)...")
    old_sp = eval_species(old_pred, test_teams, label="[old]")
    new_sp = eval_species(new_pred, test_teams, label="[new]")

    print(f"\n  [{_ts()}] Move prediction (k=1..3)...")
    old_mv = eval_move(old_pred, test_teams, label="[old]")
    new_mv = eval_move(new_pred, test_teams, label="[new]")

    print(f"\n  [{_ts()}] Component prediction (item / ability / tera)...")
    comp = {"old": {}, "new": {}}
    for attr in ("item", "ability", "tera_type"):
        print(f"    {attr} [old]...")
        comp["old"][attr] = eval_component(old_pred, test_teams, attr, label=" [old]")
        print(f"    {attr} [new]...")
        comp["new"][attr] = eval_component(new_pred, test_teams, attr, label=" [new]")

    print(f"\n  [{_ts()}] Confidence calibration (new only)...")
    calib = eval_confidence_calibration(new_pred, test_teams)

    elapsed_min = (time.time() - t_eval_start) / 60
    print(f"\n  Total evaluation time: {elapsed_min:.1f} min")

    # ── Save all results to files ──────────────────────────────────────
    print(f"\nSaving results → {_OUT_DIR}/")
    plot_main_comparison(old_sp, new_sp, old_mv, new_mv,
                         comp["old"]["item"], comp["new"]["item"])
    plot_species_by_k(old_sp, new_sp)
    plot_move_by_k(old_mv, new_mv)
    plot_component_breakdown(comp)
    plot_confidence_calibration(calib)
    write_markdown(old_sp, new_sp, old_mv, new_mv, comp, calib)
    write_summary(old_sp, new_sp, old_mv, new_mv, comp, calib, elapsed_min)
    write_raw_data(old_sp, new_sp, old_mv, new_mv, comp, calib)

    print("\nDone. Results in bayesian/benchmark_results/")
    print("  results_table.md — full markdown table (with calibration)")
    print("  summary.txt      — headline comparison table")
    print("  raw_data.py      — Python dicts to recreate any plot")
    print("  fig1..fig5.png   — plots")


if __name__ == "__main__":
    main()
