#!/usr/bin/env python3
"""
Damage-calc backward inference for defender EV-spread estimation.

Given an observed damage percentage (opponent's Pokemon took X% HP from our move),
enumerate the top-K plausible defender spreads from the component model, run them
all through @smogon/calc in a single Node subprocess (batch mode), and update the
spread distribution using a binary likelihood filter:
  consistent = observed_pct ∈ [min_pct − tolerance, max_pct + tolerance]

v1: binary likelihood (1 = consistent, ε = inconsistent).
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bayesian.smogon_prior import SmogonPrior

_TOLERANCE_PCT = 1.0       # ± 1% HP tolerance for binary consistency
_EPSILON = 0.01            # likelihood for inconsistent candidates
_DEFAULT_TOP_K = 20        # max number of spread candidates to evaluate

_STAT_ORDER = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
_STAT_MAP = {"HP": "hp", "Atk": "atk", "Def": "def",
             "SpA": "spa", "SpD": "spd", "Spe": "spe"}


def _parse_spread_key(spread_key: str) -> Optional[Tuple[str, Dict[str, int]]]:
    """Parse 'Nature:HP/Atk/Def/SpA/SpD/Spe' into (nature, evs_lowercase_dict)."""
    parsed = SmogonPrior.parse_spread(spread_key)
    if parsed is None:
        return None
    evs_lower = {_STAT_MAP[k]: v for k, v in parsed["evs"].items()}
    return parsed["nature"], evs_lower


class DamageInference:
    """
    Backward inference module: updates defender spread distribution using
    observed damage percentages via @smogon/calc.
    """

    def __init__(self, js_dir: Optional[Path] = None):
        if js_dir is None:
            js_dir = Path(__file__).parents[1] / "js_damage"
        self.js_dir = Path(js_dir)
        self.calc_script = self.js_dir / "calc_turns.js"

    def build_candidate_spreads(
        self,
        species: str,
        fused_spread_dist: Dict[str, float],
        top_k: int = _DEFAULT_TOP_K,
    ) -> List[Tuple[str, float, str, Dict[str, int]]]:
        """
        Return the top-K (spread_key, probability, nature, evs) candidates.
        """
        ranked = sorted(fused_spread_dist.items(), key=lambda x: x[1], reverse=True)[:top_k]
        candidates = []
        for spread_key, prob in ranked:
            parsed = _parse_spread_key(spread_key)
            if parsed is None:
                continue
            nature, evs = parsed
            candidates.append((spread_key, prob, nature, evs))
        return candidates

    def run_batch_calc(
        self,
        attacker_data: Dict[str, Any],
        move_name: str,
        defender_species: str,
        candidates: List[Tuple[str, float, str, Dict[str, int]]],
        gen: int = 9,
    ) -> List[Optional[Dict]]:
        """
        Run @smogon/calc for all candidates in a single Node subprocess.

        Returns a list of result dicts (or None on individual failures) in the
        same order as `candidates`.
        """
        node_path = shutil.which("node")
        if node_path is None or not self.calc_script.exists():
            return [None] * len(candidates)

        batch = []
        for _, _, nature, evs in candidates:
            batch.append({
                "gen": gen,
                "attacker": attacker_data,
                "defender": {
                    "species": defender_species,
                    "nature": nature,
                    "evs": evs,
                },
                "move": {"name": move_name},
            })

        payload = json.dumps({"batch": batch})
        try:
            proc = subprocess.run(
                ["node", str(self.calc_script)],
                input=payload.encode(),
                capture_output=True,
                cwd=str(self.js_dir),
                timeout=10,
            )
            if proc.returncode != 0:
                return [None] * len(candidates)
            results = json.loads(proc.stdout.decode())
            # Pad or truncate to match candidates length
            while len(results) < len(candidates):
                results.append(None)
            return results[: len(candidates)]
        except Exception:
            return [None] * len(candidates)

    def backward_update(
        self,
        species: str,
        attacker_data: Dict[str, Any],
        move_name: str,
        observed_damage_pct: float,
        fused_spread_dist: Dict[str, float],
        top_k: int = _DEFAULT_TOP_K,
        gen: int = 9,
    ) -> Dict[str, float]:
        """
        Update the defender's spread distribution given an observed damage %.

        For each top-K candidate spread:
          - Compute [min_pct, max_pct] via @smogon/calc
          - Likelihood = 1 if observed ∈ [min_pct - tol, max_pct + tol], else ε
          - Posterior ∝ Prior × Likelihood

        Returns the updated spread distribution, or the original if the calc fails.
        """
        if not fused_spread_dist or not move_name:
            return fused_spread_dist

        candidates = self.build_candidate_spreads(species, fused_spread_dist, top_k)
        if not candidates:
            return fused_spread_dist

        results = self.run_batch_calc(attacker_data, move_name, species, candidates, gen)

        updated: Dict[str, float] = {}
        for (spread_key, prior_prob, _, _), calc_result in zip(candidates, results):
            if calc_result is None or "error" in calc_result:
                # Keep prior probability for failed calcs
                updated[spread_key] = prior_prob
                continue

            min_pct = calc_result.get("min_pct", 0.0)
            max_pct = calc_result.get("max_pct", 0.0)
            consistent = (
                min_pct - _TOLERANCE_PCT
                <= observed_damage_pct
                <= max_pct + _TOLERANCE_PCT
            )
            likelihood = 1.0 if consistent else _EPSILON
            updated[spread_key] = prior_prob * likelihood

        # Fill in any candidates not in top_k (keep their prior probability)
        for spread_key, prob in fused_spread_dist.items():
            if spread_key not in updated:
                updated[spread_key] = prob * _EPSILON  # outside top_k = less likely

        # Normalize
        total = sum(updated.values())
        if total > 0:
            return {k: v / total for k, v in updated.items()}
        return fused_spread_dist
