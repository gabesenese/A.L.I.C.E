"""Verifier for grounding personal-memory answers in retrieved evidence."""

from __future__ import annotations

import re
from typing import Any, Dict, List


# Words that frame a recalled fact rather than state one. "Yes, you mentioned
# your sister visited" should be judged on "sister visited", not failed for
# "yes" and "mentioned" appearing nowhere in the saved row.
_FRAMING = {
    "yes",
    "you",
    "your",
    "you're",
    "told",
    "said",
    "say",
    "mentioned",
    "remember",
    "recall",
    "that",
    "this",
    "the",
    "and",
    "with",
    "for",
    "about",
    "from",
    "have",
    "has",
    "had",
    "was",
    "were",
    "are",
    "but",
    "not",
    "saved",
    "noted",
    "last",
    "time",
    "also",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(t) > 2 and t not in _FRAMING}


class MemoryAnswerVerifier:
    def verify_answer(self, *, answer_text: str, evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        evidence_tokens = set()
        for row in list(evidence_items or []):
            evidence_tokens.update(_tokens(str(row.get("content") or "")))

        unsupported_claims: List[str] = []
        lines = [ln.strip("- ").strip() for ln in str(answer_text or "").splitlines() if ln.strip()]
        for ln in lines:
            if ln.lower().startswith("here is what i have saved in memory"):
                continue
            claim_tokens = _tokens(ln)
            if not claim_tokens:
                continue
            overlap = len(claim_tokens.intersection(evidence_tokens))
            ratio = float(overlap) / float(max(1, len(claim_tokens)))
            if ratio < 0.45:
                unsupported_claims.append(ln)

        evidence_count = len(list(evidence_items or []))
        accepted = evidence_count > 0 and not unsupported_claims
        confidence = 0.9 if accepted else (0.2 if evidence_count == 0 else 0.35)
        return {
            "accepted": accepted,
            "unsupported_claims": unsupported_claims,
            "evidence_count": evidence_count,
            "confidence": confidence,
        }
