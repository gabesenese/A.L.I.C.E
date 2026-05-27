from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class SemanticDeduplicator:
    """
    Improved semantic deduplication.

    Key improvements over the legacy 0.95-threshold pass in MemorySystem:
    - Default threshold 0.82 — catches near-duplicates the old pass missed
    - Cross-type support — can dedup episodic vs semantic in one pass
    - Merge strategy: keep the higher-scored anchor, absorb access counts,
      take the union of tags from both entries
    - Returns (deduped_list, merged_pairs) for audit / SQLite cleanup
    """

    def __init__(self, threshold: float = 0.82) -> None:
        self.threshold = threshold

    def deduplicate(
        self,
        entries: List[Any],
        scorer=None,
    ) -> Tuple[List[Any], List[Tuple[str, str]]]:
        """
        Deduplicate a list of memory entries by embedding similarity.

        Args:
            entries: list of MemoryEntry-like objects (need .id, .embedding,
                     .importance, .access_count, .tags)
            scorer:  optional MemoryScorer; if provided its batch_score() is used
                     instead of raw .importance

        Returns:
            (deduplicated_list, merged_pairs)
            merged_pairs — list of (kept_id, removed_id)
        """
        if len(entries) < 2:
            return entries, []

        # Determine score per entry for keeper election
        if scorer is not None:
            scores: Dict[str, float] = scorer.batch_score(entries)
        else:
            scores = {getattr(e, "id", str(i)): float(getattr(e, "importance", 0.5)) for i, e in enumerate(entries)}

        to_remove: Set[str] = set()
        merged_pairs: List[Tuple[str, str]] = []

        for i, ei in enumerate(entries):
            ei_id = getattr(ei, "id", str(i))
            if ei_id in to_remove:
                continue
            emb_i = getattr(ei, "embedding", None)
            if not emb_i:
                continue

            for j in range(i + 1, len(entries)):
                ej = entries[j]
                ej_id = getattr(ej, "id", str(j))
                if ej_id in to_remove:
                    continue
                emb_j = getattr(ej, "embedding", None)
                if not emb_j:
                    continue

                sim = _cosine(emb_i, emb_j)
                if sim < self.threshold:
                    continue

                # Elect keeper by score
                score_i = scores.get(ei_id, 0.5)
                score_j = scores.get(ej_id, 0.5)
                if score_i >= score_j:
                    keeper, loser, keeper_id, loser_id = ei, ej, ei_id, ej_id
                else:
                    keeper, loser, keeper_id, loser_id = ej, ei, ej_id, ei_id

                # Absorb access count from loser
                try:
                    keeper.access_count = int(getattr(keeper, "access_count", 0)) + int(
                        getattr(loser, "access_count", 0)
                    )
                except AttributeError:
                    pass

                # Tag union
                try:
                    keeper_tags = list(getattr(keeper, "tags", None) or [])
                    loser_tags = list(getattr(loser, "tags", None) or [])
                    keeper.tags = list(dict.fromkeys(keeper_tags + loser_tags))
                except AttributeError:
                    pass

                to_remove.add(loser_id)
                merged_pairs.append((keeper_id, loser_id))
                logger.debug(
                    "[SemanticDedup] merged %s → %s (sim=%.3f)",
                    loser_id,
                    keeper_id,
                    sim,
                )

        deduped = [e for e in entries if getattr(e, "id", None) not in to_remove]

        if merged_pairs:
            logger.info(
                "[SemanticDedup] %d duplicates merged, %d entries remaining",
                len(merged_pairs),
                len(deduped),
            )
        return deduped, merged_pairs


_deduplicator: Optional[SemanticDeduplicator] = None


def get_deduplicator(threshold: float = 0.82) -> SemanticDeduplicator:
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = SemanticDeduplicator(threshold=threshold)
    return _deduplicator
