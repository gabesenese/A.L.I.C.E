from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

LEARNED_RESPONSE_EXAMPLES_PATH = Path("data/learned_response_examples.jsonl")


@dataclass
class LearnedResponseExample:
    example_id: str
    surface: str
    context_signals: List[str]
    response_text: str
    created_at: str
    energy_signal: str = "unknown"
    mood_signal: str = "unknown"
    topic: str = ""
    user_context_summary: str = ""
    accepted: bool = True
    source: str = "ollama_validated"

    @classmethod
    def create(
        cls,
        *,
        surface: str,
        context_signals: List[str],
        response_text: str,
        energy_signal: str = "unknown",
        mood_signal: str = "unknown",
        topic: str = "",
        user_context_summary: str = "",
    ) -> "LearnedResponseExample":
        return cls(
            example_id=f"lre_{uuid4().hex[:12]}",
            surface=str(surface or "").strip(),
            context_signals=[str(s).strip() for s in list(context_signals or []) if str(s).strip()],
            energy_signal=str(energy_signal or "unknown").strip() or "unknown",
            mood_signal=str(mood_signal or "unknown").strip() or "unknown",
            topic=str(topic or "").strip(),
            user_context_summary=str(user_context_summary or "").strip(),
            response_text=str(response_text or "").strip(),
            accepted=True,
            created_at=datetime.now(timezone.utc).isoformat(),
            source="ollama_validated",
        )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_example(payload: dict) -> LearnedResponseExample | None:
    try:
        return LearnedResponseExample(
            example_id=str(payload.get("example_id") or "").strip(),
            surface=str(payload.get("surface") or "").strip(),
            context_signals=[
                str(s).strip() for s in list(payload.get("context_signals") or []) if str(s).strip()
            ],
            energy_signal=str(payload.get("energy_signal") or "unknown").strip() or "unknown",
            mood_signal=str(payload.get("mood_signal") or "unknown").strip() or "unknown",
            topic=str(payload.get("topic") or "").strip(),
            user_context_summary=str(payload.get("user_context_summary") or "").strip(),
            response_text=str(payload.get("response_text") or "").strip(),
            accepted=bool(payload.get("accepted", True)),
            created_at=str(payload.get("created_at") or "").strip(),
            source=str(payload.get("source") or "ollama_validated").strip() or "ollama_validated",
        )
    except Exception:
        return None


def load_learned_response_examples(surface: str, limit: int = 20) -> List[LearnedResponseExample]:
    path = LEARNED_RESPONSE_EXAMPLES_PATH
    if not path.exists():
        return []
    surface_key = str(surface or "").strip().lower()
    out: List[LearnedResponseExample] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = str(line or "").strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                ex = _parse_example(payload)
                if not ex:
                    continue
                if ex.surface.lower() != surface_key:
                    continue
                out.append(ex)
    except Exception:
        return []
    out.sort(key=lambda x: x.created_at, reverse=True)
    return out[: max(0, int(limit or 0))]


def _has_signal_overlap(a: List[str], b: List[str]) -> bool:
    set_a = {str(x).strip().lower() for x in list(a or []) if str(x).strip()}
    set_b = {str(x).strip().lower() for x in list(b or []) if str(x).strip()}
    return bool(set_a.intersection(set_b))


def record_learned_response_example(example: LearnedResponseExample) -> None:
    if not isinstance(example, LearnedResponseExample):
        return
    if not str(example.surface or "").strip():
        return
    if not str(example.response_text or "").strip():
        return
    existing = load_learned_response_examples(surface=example.surface, limit=2000)
    for ex in existing:
        if ex.response_text.strip().lower() != example.response_text.strip().lower():
            continue
        if _has_signal_overlap(ex.context_signals, example.context_signals):
            return

    path = LEARNED_RESPONSE_EXAMPLES_PATH
    _ensure_parent(path)
    payload = asdict(example)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def find_similar_response_examples(
    context_signals: List[str],
    surface: str,
    limit: int = 3,
) -> List[LearnedResponseExample]:
    desired = [str(x).strip().lower() for x in list(context_signals or []) if str(x).strip()]
    if not desired:
        return load_learned_response_examples(surface=surface, limit=limit)
    candidates = load_learned_response_examples(surface=surface, limit=2000)

    def _score(ex: LearnedResponseExample) -> tuple[int, str]:
        ex_signals = {str(x).strip().lower() for x in list(ex.context_signals or []) if str(x).strip()}
        overlap = len(ex_signals.intersection(set(desired)))
        return (overlap, ex.created_at)

    ranked = [ex for ex in candidates if _score(ex)[0] > 0]
    ranked.sort(key=_score, reverse=True)
    return ranked[: max(0, int(limit or 0))]
