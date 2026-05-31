#!/usr/bin/env python3
"""
Smogon Gen9OU usage-statistics prior.

Loads polimi/smogon_stats.json and exposes normalized prior distributions
over moves, items, abilities, EV spreads, and tera types per species.
'Other' entries are discarded and remaining percentages are renormalized to
sum to 1.0; Dirichlet smoothing for unseen values is applied downstream in
ComponentModel._fuse().
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

_STAT_ORDER = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
_DEFAULT_STATS_PATH = Path(__file__).parents[1] / "polimi" / "smogon_stats.json"


def _parse_spread(spread_str: str) -> Optional[Dict]:
    """Parse 'Nature:HP/Atk/Def/SpA/SpD/Spe' into {nature, evs} dict."""
    m = re.match(r"^([A-Za-z]+):(\d+)/(\d+)/(\d+)/(\d+)/(\d+)/(\d+)$", spread_str.strip())
    if not m:
        return None
    nature = m.group(1)
    evs = {stat: int(m.group(i + 2)) for i, stat in enumerate(_STAT_ORDER)}
    return {"nature": nature, "evs": evs}


def _normalize_dist(raw: Dict[str, float]) -> Dict[str, float]:
    """Drop the 'Other' key and renormalize the remaining entries to sum to 1."""
    filtered = {k: v for k, v in raw.items() if k != "Other" and v > 0}
    total = sum(filtered.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in filtered.items()}


class SmogonPrior:
    """
    Normalized prior distributions from Smogon Gen9OU usage statistics.

    All distributions are computed once at construction time.  Each component
    distribution sums to 1.0 over the listed values; 'Other' mass is left to
    Dirichlet smoothing in the component model.
    """

    def __init__(self, stats_path: Optional[str] = None):
        path = Path(stats_path) if stats_path else _DEFAULT_STATS_PATH
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._data: Dict[str, Dict] = {}
        for species, entry in raw.items():
            self._data[species] = {
                "metadata": entry.get("metadata", {}),
                "moves": _normalize_dist(entry.get("Moves", {})),
                "items": _normalize_dist(entry.get("Items", {})),
                "abilities": _normalize_dist(entry.get("Abilities", {})),
                "spreads": _normalize_dist(entry.get("Spreads", {})),
                "tera_types": _normalize_dist(entry.get("Tera Types", {})),
                "teammates": _normalize_dist(entry.get("Teammates", {})),
            }

    def get_prior(self, species: str, component: str) -> Dict[str, float]:
        """
        Return the normalized prior distribution for (species, component).

        component ∈ {"moves", "items", "abilities", "spreads", "tera_types", "teammates"}
        Returns an empty dict if species or component is unknown.
        """
        entry = self._data.get(species)
        if entry is None:
            return {}
        return dict(entry.get(component, {}))

    def get_raw_count(self, species: str) -> float:
        """Smogon raw usage count for a species (0.0 if not found)."""
        entry = self._data.get(species)
        if entry is None:
            return 0.0
        return float(entry["metadata"].get("Raw count", 0.0))

    def get_viability_ceiling(self, species: str) -> float:
        """Smogon viability ceiling for a species (0.0 if not found)."""
        entry = self._data.get(species)
        if entry is None:
            return 0.0
        return float(entry["metadata"].get("Viability Ceiling", 0.0))

    def known_species(self) -> List[str]:
        """All species names present in the Smogon stats."""
        return list(self._data.keys())

    @staticmethod
    def parse_spread(spread_str: str) -> Optional[Dict]:
        """Parse a Smogon spread string into {nature, evs} dict. Returns None on failure."""
        return _parse_spread(spread_str)
