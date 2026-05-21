"""Convert routing_failures.jsonl → data/evaluations/evaluations.jsonl.

Backfills historical failure records so ConfidenceFusion._intent_success_rate()
has real data to work with.  Also exposes write_turn_eval() for live per-turn
recording so the file stays current going forward.

Score mapping (ConfidenceFusion counts score >= 70 as success):
  - Known failure vetoes   → 10–45  (failure)
  - Turn success           → 80     (success)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

_FAILURES_PATH = Path("data/routing_failures.jsonl")
_EVAL_PATH = Path("data/evaluations/evaluations.jsonl")
_WATERMARK_PATH = Path("data/evaluations/.watermark")

# Map veto_reason → overall_score (all < 70 so they register as failures)
_VETO_SCORES: Dict[str, int] = {
    "user_reported_misroute":            10,
    "verification_failed":               15,
    "tool_route_vetoed":                 25,
    "local_execution_target_not_found":  30,
    "fallback_response_used":            35,
    "no_explicit_file_target":           40,
    "route_clarify":                     45,
    "local_execution_error":             30,
}
_DEFAULT_FAILURE_SCORE = 40
_SUCCESS_SCORE = 80


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_turn_eval(
    *,
    intent: str,
    trace_id: str,
    success: bool,
    veto_reason: str = "",
) -> None:
    """Append a single turn outcome to evaluations.jsonl.

    Called from contract_pipeline after every turn so success rates
    stay current. Silently swallows all errors.
    """
    try:
        _EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        score = _SUCCESS_SCORE if success else _VETO_SCORES.get(
            str(veto_reason or ""), _DEFAULT_FAILURE_SCORE
        )
        record = {
            "trace_id": str(trace_id or ""),
            "action_type": str(intent or "unknown"),
            "overall_score": score,
            "source": "live_turn",
            "timestamp": _now_iso(),
        }
        if veto_reason:
            record["veto_reason"] = veto_reason
        with open(_EVAL_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass


class FailureEvalConverter:
    """One-shot or incremental backfill from routing_failures.jsonl."""

    def run(self, max_records: int = 1000) -> Dict[str, int]:
        """Convert new (since last watermark) routing failures to eval records.

        Returns {"converted": N, "total_failures": M, "new": K}.
        Safe to call multiple times — the watermark prevents double-writes.
        """
        if not _FAILURES_PATH.exists():
            return {"converted": 0, "total_failures": 0, "new": 0}

        watermark = self._read_watermark()
        try:
            lines = _FAILURES_PATH.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
        except Exception:
            return {"converted": 0, "total_failures": 0, "new": 0}

        new_lines = lines[watermark:]
        if not new_lines:
            return {"converted": 0, "total_failures": len(lines), "new": 0}

        _EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        converted = 0
        batch = new_lines[:max_records]

        try:
            with open(_EVAL_PATH, "a", encoding="utf-8") as fh:
                for raw in batch:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    veto = str(rec.get("veto_reason") or "")
                    score = _VETO_SCORES.get(veto, _DEFAULT_FAILURE_SCORE)
                    intent = str(rec.get("final_intent") or "unknown")
                    eval_rec = {
                        "trace_id": str(rec.get("trace_id") or ""),
                        "action_type": intent,
                        "overall_score": score,
                        "veto_reason": veto,
                        "source": "routing_failure_backfill",
                        "timestamp": str(
                            rec.get("timestamp") or _now_iso()
                        ),
                    }
                    fh.write(json.dumps(eval_rec) + "\n")
                    converted += 1
        except Exception:
            pass

        self._write_watermark(watermark + len(batch))
        return {
            "converted": converted,
            "total_failures": len(lines),
            "new": len(new_lines),
        }

    @staticmethod
    def _read_watermark() -> int:
        if _WATERMARK_PATH.exists():
            try:
                return int(_WATERMARK_PATH.read_text(encoding="utf-8").strip())
            except Exception:
                pass
        return 0

    @staticmethod
    def _write_watermark(pos: int) -> None:
        try:
            _WATERMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
            _WATERMARK_PATH.write_text(str(pos), encoding="utf-8")
        except Exception:
            pass

    def weak_spot_report(self, top_n: int = 10) -> str:
        """Human-readable per-intent failure summary from evaluations.jsonl."""
        if not _EVAL_PATH.exists():
            return "No evaluation data yet. Run converter first."

        from collections import defaultdict
        totals: Dict[str, int] = defaultdict(int)
        failures: Dict[str, int] = defaultdict(int)

        try:
            lines = _EVAL_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
            for raw in lines[-2000:]:  # cap to recent 2k
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                intent_prefix = str(rec.get("action_type") or "unknown").split(":")[0]
                totals[intent_prefix] += 1
                if int(rec.get("overall_score", 0)) < 70:
                    failures[intent_prefix] += 1
        except Exception as e:
            return f"Error reading evaluations: {e}"

        if not totals:
            return "No evaluation records found."

        rows = []
        for intent, total in sorted(totals.items(), key=lambda x: -x[1]):
            fail = failures.get(intent, 0)
            rate = fail / total if total else 0.0
            rows.append((intent, total, fail, rate))

        rows.sort(key=lambda r: -r[3])  # sort by failure rate

        lines_out = [
            f"{'Intent':<30} {'Total':>6} {'Fail':>6} {'Fail%':>6}",
            "-" * 52,
        ]
        for intent, total, fail, rate in rows[:top_n]:
            lines_out.append(
                f"{intent:<30} {total:>6} {fail:>6} {rate * 100:>5.0f}%"
            )
        lines_out.append(f"\nTotal eval records: {sum(totals.values())}")
        return "\n".join(lines_out)


_converter: Optional[FailureEvalConverter] = None


def get_failure_eval_converter() -> FailureEvalConverter:
    global _converter
    if _converter is None:
        _converter = FailureEvalConverter()
    return _converter
