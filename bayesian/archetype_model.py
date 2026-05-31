#!/usr/bin/env python3
"""
LDA-based team archetype model for Gen9OU.

Uses sklearn LatentDirichletAllocation (gensim is incompatible with Python 3.14).
Each team is a "document" with up to 6 species as "words".  After fitting, the
model infers archetype weights for a set of revealed Pokemon and uses them to
predict unrevealed teammates without the double-counting problem of naive Bayes
over individual teammate conditionals.

Serialization: the fitted model and vocabulary are picklable; the training
driver embeds them into the main cache pickle rather than a separate directory.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

_DEFAULT_N_TOPICS = 20
_DEFAULT_PASSES = 20  # sklearn calls this max_iter


class ArchetypeModel:
    """
    Topic model over Pokemon teams, exposing fast archetype inference.

    Training: each of the ~140k teams is a 6-word document; species are words.
    After fitting, `_phi[k, v]` = P(species_v | archetype_k).

    Inference: given revealed species, compute P(archetype | revealed) via
    Naive Bayes over archetypes (valid because archetypes capture inter-Pokemon
    correlations), then marginalize to get P(unrevealed | revealed).
    """

    def __init__(
        self,
        num_topics: int = _DEFAULT_N_TOPICS,
        max_iter: int = _DEFAULT_PASSES,
    ):
        self.num_topics = num_topics
        self.max_iter = max_iter

        # Set after training
        self._vocab: Dict[str, int] = {}          # species → column index
        self._id2species: Dict[int, str] = {}     # column index → species
        self._phi: Optional[np.ndarray] = None    # (K, V) normalized topic-word matrix
        self._prior: Optional[np.ndarray] = None  # (K,) uniform topic prior weights
        self._topic_labels: Dict[int, str] = {}   # {topic_id: "A / B / C"}
        self._lda: Optional[LatentDirichletAllocation] = None
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, team_corpus: List[List[str]]) -> None:
        """
        Fit LDA on a corpus of teams.

        Each team is a list of species strings (up to 6).
        """
        if not team_corpus:
            return

        # Build vocabulary
        all_species = sorted({sp for team in team_corpus for sp in team if sp})
        self._vocab = {sp: i for i, sp in enumerate(all_species)}
        self._id2species = {i: sp for sp, i in self._vocab.items()}
        V = len(self._vocab)

        # Build document-term matrix (sparse, one row per team)
        rows, cols, data = [], [], []
        for doc_idx, team in enumerate(team_corpus):
            for sp in team:
                if sp and sp in self._vocab:
                    rows.append(doc_idx)
                    cols.append(self._vocab[sp])
                    data.append(1)

        X = csr_matrix((data, (rows, cols)), shape=(len(team_corpus), V), dtype=np.float32)

        # Fit LDA
        self._lda = LatentDirichletAllocation(
            n_components=self.num_topics,
            max_iter=self.max_iter,
            learning_method="batch",
            random_state=42,
            n_jobs=1,
        )
        self._lda.fit(X)

        # Normalize components_ rows to get proper P(species | archetype) distributions
        raw = self._lda.components_  # (K, V) unnormalized
        self._phi = raw / raw.sum(axis=1, keepdims=True)  # (K, V)

        # Uniform topic prior for inference
        self._prior = np.ones(self.num_topics, dtype=np.float64) / self.num_topics

        # Build human-readable topic labels (top-3 species)
        self._build_labels()
        self.is_trained = True

    def _build_labels(self) -> None:
        for k in range(self.num_topics):
            top_ids = np.argsort(self._phi[k])[::-1][:3]
            names = [self._id2species.get(i, "?") for i in top_ids]
            self._topic_labels[k] = " / ".join(names)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer_archetypes(
        self, revealed_species: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Return archetype distribution given revealed species.

        Uses Naive Bayes over archetypes:
          P(archetype_k | revealed) ∝ P(archetype_k) * Π_i P(revealed_i | archetype_k)

        This is valid because archetypes capture inter-Pokemon correlations, so
        the independence assumption holds approximately WITHIN each archetype.
        """
        if not self.is_trained or self._phi is None:
            return []

        weights = self._archetype_weights(revealed_species)  # (K,), topic-ordered
        return sorted(
            [(self._topic_labels.get(k, f"Archetype {k}"), float(weights[k]))
             for k in range(self.num_topics)],
            key=lambda x: x[1],
            reverse=True,
        )

    def _archetype_weights(self, revealed_species: List[str]) -> np.ndarray:
        """Return P(archetype_k | revealed) as a (K,) array indexed by topic number."""
        weights = self._prior.copy()
        for sp in revealed_species:
            if sp in self._vocab:
                v = self._vocab[sp]
                weights *= self._phi[:, v]
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = self._prior.copy()
        return weights  # shape (K,), index aligns with _phi rows

    def predict_unrevealed(
        self,
        revealed_species: List[str],
        candidate_species: List[str],
    ) -> Dict[str, float]:
        """
        P(unrevealed | revealed) = Σ_k P(archetype_k | revealed) * P(unrevealed | archetype_k)
        """
        if not self.is_trained or self._phi is None:
            return {}

        weights = self._archetype_weights(revealed_species)  # (K,), topic-ordered
        revealed_set = set(revealed_species)

        result: Dict[str, float] = {}
        for sp in candidate_species:
            if sp in revealed_set:
                continue
            if sp not in self._vocab:
                continue
            v = self._vocab[sp]
            result[sp] = float(np.dot(weights, self._phi[:, v]))

        return result

    # ------------------------------------------------------------------
    # Serialization helpers (used by BayesianTeamPredictor cache)
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize to a plain dict for inclusion in the main pickle cache."""
        if not self.is_trained:
            return {"is_trained": False}
        return {
            "is_trained": True,
            "num_topics": self.num_topics,
            "vocab": self._vocab,
            "phi": self._phi.tolist(),
            "prior": self._prior.tolist(),
            "topic_labels": {str(k): v for k, v in self._topic_labels.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ArchetypeModel":
        """Deserialize from a plain dict."""
        model = cls(num_topics=data.get("num_topics", _DEFAULT_N_TOPICS))
        if not data.get("is_trained"):
            return model
        model._vocab = data["vocab"]
        model._id2species = {v: k for k, v in data["vocab"].items()}
        model._phi = np.array(data["phi"], dtype=np.float64)
        model._prior = np.array(data["prior"], dtype=np.float64)
        model._topic_labels = {int(k): v for k, v in data["topic_labels"].items()}
        model.is_trained = True
        return model
