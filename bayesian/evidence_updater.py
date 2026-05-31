#!/usr/bin/env python3
"""
Bayesian evidence conditioning for Pokemon battle observations.

Replaces the (1 + matches)^2 heuristic with proper posterior updates:
  - Positive move evidence: re-weight remaining moves using move co-occurrence.
  - Negative move evidence: attenuate moves that rarely appear alongside
    the already-revealed set (capped at 80% reduction).
  - Confirmed item/ability/tera: zero out all other candidates.

All functions are stateless; they take a distribution dict and return an
updated one.  ComponentModel.move_pair_counts provides the co-occurrence data.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

# Cap: negative evidence cannot reduce a move's probability by more than this
_MAX_NEGATIVE_ATTENUATION = 0.80

# Confidence label thresholds (tight — from plan §constraints)
_CONFIDENT_PROB = 0.60
_CONFIDENT_GAP = 0.20
_LIKELY_PROB = 0.35
_LIKELY_GAP = 0.15

_DIRICHLET_ALPHA = 0.5  # smoothing for pair lookups


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return dist
    return {k: v / total for k, v in dist.items()}


def condition_on_observed_moves(
    species: str,
    observed_moves: List[str],
    move_dist: Dict[str, float],
    move_pair_counts: Any,  # ComponentModel.move_pair_counts
) -> Dict[str, float]:
    """
    Re-weight move predictions given the set of already-observed moves.

    For each candidate move m (not yet observed):
      score(m) ∝ P(m|species) * Π_{s ∈ observed} P(s | species, m)

    P(s | species, m) is computed from move_pair_counts with Dirichlet smoothing,
    so it is never zero.  Computation is done in log-space to avoid underflow.

    Observed moves are removed from the returned distribution (they are
    confirmed; the call site should add them back at prob=1.0 if needed).
    """
    if not observed_moves or not move_dist:
        return move_dist

    observed_set = set(observed_moves)
    updated: Dict[str, float] = {}

    for move, base_prob in move_dist.items():
        if move in observed_set:
            continue  # excluded — call site handles confirmed moves separately
        if base_prob <= 0:
            continue

        log_score = math.log(base_prob)
        for seen_move in observed_moves:
            # P(seen_move | species, move) from co-occurrence
            pair_counter = move_pair_counts.get((species, move), {})
            count = pair_counter.get(seen_move, 0)
            total_pairs = sum(pair_counter.values()) if pair_counter else 0
            vocab_size = max(len(pair_counter), 1)
            # Dirichlet-smoothed conditional
            p_seen_given_move = (count + _DIRICHLET_ALPHA) / (
                total_pairs + _DIRICHLET_ALPHA * vocab_size
            )
            log_score += math.log(p_seen_given_move)

        updated[move] = math.exp(log_score)

    return _normalize(updated)


def condition_on_negative_moves(
    species: str,
    not_seen_moves: List[str],
    k_revealed: int,
    move_dist: Dict[str, float],
    move_pair_counts: Any,
) -> Dict[str, float]:
    """
    Attenuate moves that are unlikely in remaining slots given that they
    haven't appeared in the first k_revealed turns.

    Attenuation factor for move m':
      attenuation = min(pair_support, MAX_ATTENUATION)
    where pair_support is the average co-occurrence probability of m' with the
    revealed moves — if m' always co-occurs with them, it's probably in the
    moveset already (but wasn't revealed yet), so we attenuate less; if it
    rarely co-occurs, it's more likely absent.

    Reduction is capped at 80% to avoid over-confident zeroing.
    Only applied when k_revealed >= 2 (too little signal otherwise).
    """
    if k_revealed < 2 or not not_seen_moves or not move_dist:
        return move_dist

    updated = dict(move_dist)
    for move in not_seen_moves:
        if move not in updated:
            continue

        pair_counter = move_pair_counts.get((species, move), {})
        total_pairs = sum(pair_counter.values()) if pair_counter else 0
        if total_pairs == 0:
            continue  # no co-occurrence data, don't attenuate

        # Average P(any_revealed_move | species, move) — high means move
        # tends to appear with things we've already seen, so NOT seeing it
        # yet is surprising → more attenuation
        vocab_size = max(len(pair_counter), 1)
        avg_co_occurrence = sum(
            (pair_counter.get(s, 0) + _DIRICHLET_ALPHA) / (total_pairs + _DIRICHLET_ALPHA * vocab_size)
            for s in not_seen_moves
        ) / len(not_seen_moves)

        # Scale attenuation with number of revealed moves (more reveals = stronger signal)
        scale = min(1.0, (k_revealed - 1) / 3.0)
        attenuation = min(avg_co_occurrence * scale, _MAX_NEGATIVE_ATTENUATION)
        updated[move] = updated[move] * (1.0 - attenuation)

    return _normalize(updated)


def condition_on_confirmed(
    value: str,
    component_dist: Dict[str, float],
) -> Dict[str, float]:
    """
    Zero out all entries except the confirmed value and return a point mass.

    Used when item, ability, or tera type has been directly observed.
    Returns {value: 1.0} even if value was not in the original distribution.
    """
    if not value:
        return component_dist
    return {value: 1.0}


def apply_all_evidence(
    species: str,
    dists: Dict[str, Dict[str, float]],
    evidence: Dict[str, Any],
    move_pair_counts: Any,
) -> Dict[str, Dict[str, float]]:
    """
    Apply all available battle evidence to the component distributions.

    evidence keys (all optional):
      observed_moves:    List[str]  — moves confirmed in the moveset
      not_seen_moves:    List[str]  — moves that have NOT appeared after k turns
      k_revealed:        int        — number of moves revealed so far
      confirmed_item:    str|None
      confirmed_ability: str|None
      confirmed_tera:    str|None

    dists keys: "moves", "items", "abilities", "natures", "spreads", "tera_types"

    Returns updated dists dict (shallow copy — original values not mutated).
    """
    result = dict(dists)

    observed = evidence.get("observed_moves") or []
    not_seen = evidence.get("not_seen_moves") or []
    k_revealed = evidence.get("k_revealed", len(observed))

    if "moves" in result and result["moves"]:
        conditioned = condition_on_observed_moves(
            species, observed, result["moves"], move_pair_counts
        )
        if not_seen:
            conditioned = condition_on_negative_moves(
                species, not_seen, k_revealed, conditioned, move_pair_counts
            )
        # Re-inject confirmed moves at probability 1.0
        final_move_dist = {m: 1.0 for m in observed if m}
        final_move_dist.update({m: p for m, p in conditioned.items()})
        result["moves"] = final_move_dist

    if evidence.get("confirmed_item") and "items" in result:
        result["items"] = condition_on_confirmed(evidence["confirmed_item"], result["items"])

    if evidence.get("confirmed_ability") and "abilities" in result:
        result["abilities"] = condition_on_confirmed(evidence["confirmed_ability"], result["abilities"])

    if evidence.get("confirmed_tera") and "tera_types" in result:
        result["tera_types"] = condition_on_confirmed(evidence["confirmed_tera"], result["tera_types"])

    return result


# ------------------------------------------------------------------
# Summary output
# ------------------------------------------------------------------

def _confidence_label(top_prob: float, second_prob: float) -> str:
    gap = top_prob - second_prob
    if top_prob >= _CONFIDENT_PROB and gap >= _CONFIDENT_GAP:
        return "Confident"
    if top_prob >= _LIKELY_PROB or gap >= _LIKELY_GAP:
        return "Likely"
    return "Uncertain"


def build_summary(
    component_dists: Dict[str, Dict[str, float]],
    evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict]:
    """
    Build a structured summary dict for the downstream LLM.

    For each component returns:
      {
        "top": (name, prob),
        "confidence": "Confident" | "Likely" | "Uncertain",
        "alternatives": [(name, prob), ...],  # top 2-3 when not Confident
        "reason": str | None,
      }

    Thresholds (tight):
      Confident: top ≥ 0.60 AND gap-to-2nd ≥ 0.20
      Likely:    top ≥ 0.35 OR  gap-to-2nd ≥ 0.15
      Uncertain: otherwise
    """
    evidence = evidence or {}
    confirmed = {
        "items": evidence.get("confirmed_item"),
        "abilities": evidence.get("confirmed_ability"),
        "tera_types": evidence.get("confirmed_tera"),
    }

    summary: Dict[str, Dict] = {}
    for component, dist in component_dists.items():
        if not dist:
            continue

        # Sort by probability
        ranked = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top_name, top_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0

        label = _confidence_label(top_prob, second_prob)

        # Build reason string
        reason: Optional[str] = None
        if confirmed.get(component) and confirmed[component] == top_name:
            reason = "confirmed: revealed in battle"
        elif component == "moves":
            observed = evidence.get("observed_moves", [])
            if top_name in (observed or []):
                reason = "confirmed: observed move"
        elif label == "Confident" and component in ("items", "abilities"):
            reason = f"high prior probability ({top_prob:.0%})"

        # Only include alternatives when confidence is below Confident
        alternatives: List[Tuple[str, float]] = []
        if label != "Confident":
            alternatives = ranked[1:4]  # top 2-3 alternatives

        summary[component] = {
            "top": (top_name, top_prob),
            "confidence": label,
            "alternatives": alternatives,
            "reason": reason,
        }

    return summary
