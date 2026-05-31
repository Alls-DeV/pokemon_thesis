#!/usr/bin/env python3
"""
Factorized per-species component model for Pokemon team prediction.

Stores separate probability distributions for each configuration component
(moves, items, abilities, natures, EV spreads, tera types) instead of a
monolithic config key.  Dirichlet add-α smoothing is applied at query time.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

CACHE_VERSION = 2

_STAT_ORDER = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
_DEFAULT_DIRICHLET_ALPHA = 0.5


def evs_to_spread_key(nature: str, evs: Dict[str, int]) -> str:
    """Encode (nature, EVs) as 'Nature:HP/Atk/Def/SpA/SpD/Spe' — matches Smogon format."""
    ev_str = "/".join(str(evs.get(s, 0)) for s in _STAT_ORDER)
    return f"{nature}:{ev_str}"


class ComponentModel:
    """
    Factorized probability tables for each component of a Pokemon's configuration.

    All tables are defaultdict(Counter) keyed by species.  Dirichlet add-α
    smoothing is applied at query time in get_distribution().
    """

    def __init__(self):
        self.move_counts: Dict[str, Counter] = defaultdict(Counter)
        self.item_counts: Dict[str, Counter] = defaultdict(Counter)
        self.ability_counts: Dict[str, Counter] = defaultdict(Counter)
        self.nature_counts: Dict[str, Counter] = defaultdict(Counter)
        # Keys are "Nature:HP/Atk/Def/SpA/SpD/Spe" strings (Smogon-compatible)
        self.spread_counts: Dict[str, Counter] = defaultdict(Counter)
        self.tera_counts: Dict[str, Counter] = defaultdict(Counter)

        # (species, move_a) → Counter[move_b]: move co-occurrence within same Pokemon
        self.move_pair_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

        # Total Pokemon observations per species (used for Smogon fusion alpha)
        self.replay_counts: Counter = Counter()

    def update(self, pokemon: Any) -> None:
        """
        Update all counters from a PokemonConfig-like object.

        Reads: .species, .moves (list), .item, .ability, .nature, .evs (dict), .tera_type
        """
        species = getattr(pokemon, "species", None)
        if not species:
            return

        self.replay_counts[species] += 1

        valid_moves = [m for m in (getattr(pokemon, "moves", []) or []) if m]
        for move in valid_moves:
            self.move_counts[species][move] += 1
        for m_a in valid_moves:
            for m_b in valid_moves:
                if m_b != m_a:
                    self.move_pair_counts[(species, m_a)][m_b] += 1

        item = getattr(pokemon, "item", None)
        if item:
            self.item_counts[species][item] += 1

        ability = getattr(pokemon, "ability", None)
        if ability:
            self.ability_counts[species][ability] += 1

        nature = getattr(pokemon, "nature", None)
        if nature:
            self.nature_counts[species][nature] += 1
            evs = getattr(pokemon, "evs", None)
            if isinstance(evs, dict):
                self.spread_counts[species][evs_to_spread_key(nature, evs)] += 1

        tera = getattr(pokemon, "tera_type", None)
        if tera:
            self.tera_counts[species][tera] += 1

    def _counter_for(self, species: str, component: str) -> Counter:
        mapping = {
            "moves": self.move_counts,
            "items": self.item_counts,
            "abilities": self.ability_counts,
            "natures": self.nature_counts,
            "spreads": self.spread_counts,
            "tera_types": self.tera_counts,
        }
        table = mapping.get(component)
        if table is None:
            return Counter()
        return table[species]  # defaultdict returns empty Counter for unknown species

    def get_distribution(
        self,
        species: str,
        component: str,
        dirichlet_alpha: float = _DEFAULT_DIRICHLET_ALPHA,
    ) -> Dict[str, float]:
        """
        Normalized Dirichlet-smoothed distribution for (species, component).

        Returns empty dict if species has no replay observations for this component.
        """
        counter = self._counter_for(species, component)
        if not counter:
            return {}
        total = sum(counter.values()) + dirichlet_alpha * len(counter)
        return {k: (v + dirichlet_alpha) / total for k, v in counter.items()}

    def get_move_pair_distribution(
        self,
        species: str,
        move: str,
        dirichlet_alpha: float = _DEFAULT_DIRICHLET_ALPHA,
    ) -> Dict[str, float]:
        """P(co_move | species, move) with Dirichlet smoothing."""
        counter = self.move_pair_counts.get((species, move), Counter())
        if not counter:
            return {}
        total = sum(counter.values()) + dirichlet_alpha * len(counter)
        return {k: (v + dirichlet_alpha) / total for k, v in counter.items()}

    def known_species(self) -> List[str]:
        """All species with at least one replay observation."""
        return list(self.replay_counts.keys())
