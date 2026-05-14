from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

APPROVAL_LEDGER_PATH = Path("data/approval_ledger.jsonl")


@dataclass
class ApprovalRecord:
    approval_id: str
    action_name: str
    risk_level: str
    user_id: str = "default"
    target: str = ""
    approved: bool = True
    consumed: bool = False
    created_at: str = ""
    expires_at: str = ""
    reason: str = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_dt(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_record(payload: dict[str, Any]) -> ApprovalRecord | None:
    try:
        return ApprovalRecord(
            approval_id=str(payload.get("approval_id") or "").strip(),
            action_name=str(payload.get("action_name") or "").strip(),
            target=str(payload.get("target") or "").strip(),
            risk_level=str(payload.get("risk_level") or "").strip(),
            user_id=str(payload.get("user_id") or "default").strip() or "default",
            approved=bool(payload.get("approved", True)),
            consumed=bool(payload.get("consumed", False)),
            created_at=str(payload.get("created_at") or "").strip(),
            expires_at=str(payload.get("expires_at") or "").strip(),
            reason=str(payload.get("reason") or "").strip(),
        )
    except Exception:
        return None


def _iter_records() -> list[ApprovalRecord]:
    if not APPROVAL_LEDGER_PATH.exists():
        return []
    out: list[ApprovalRecord] = []
    with APPROVAL_LEDGER_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            record = _parse_record(payload if isinstance(payload, dict) else {})
            if record:
                out.append(record)
    return out


def _write_records(records: list[ApprovalRecord]) -> None:
    APPROVAL_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with APPROVAL_LEDGER_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def create_approval(
    action_name: str,
    target: str,
    risk_level: str,
    user_id: str = "default",
    reason: str = "",
) -> ApprovalRecord:
    record = ApprovalRecord(
        approval_id=f"appr_{uuid4().hex[:16]}",
        user_id=str(user_id or "default").strip() or "default",
        action_name=str(action_name or "").strip(),
        target=str(target or "").strip(),
        risk_level=str(risk_level or "").strip(),
        approved=True,
        consumed=False,
        created_at=_now_iso(),
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        reason=str(reason or "").strip(),
    )
    APPROVAL_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with APPROVAL_LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return record


def load_approval(approval_id: str) -> ApprovalRecord | None:
    needle = str(approval_id or "").strip()
    if not needle:
        return None
    records = _iter_records()
    for record in reversed(records):
        if record.approval_id == needle:
            return record
    return None


def consume_approval(approval_id: str) -> bool:
    needle = str(approval_id or "").strip()
    if not needle:
        return False
    records = _iter_records()
    updated = False
    for i in range(len(records) - 1, -1, -1):
        if records[i].approval_id == needle:
            if records[i].consumed:
                return False
            records[i].consumed = True
            updated = True
            break
    if not updated:
        return False
    _write_records(records)
    return True


def approval_matches(record: ApprovalRecord | dict[str, Any] | None, request: Any) -> bool:
    if record is None:
        return False
    if isinstance(record, dict):
        parsed = _parse_record(record)
        if not parsed:
            return False
        record = parsed

    req_name = str(getattr(request, "name", "") or "").strip()
    req_target = str(getattr(request, "target", "") or "").strip()
    req_risk = str(getattr(request, "risk_level", "") or "").strip()

    if not record.approved or record.consumed:
        return False
    if record.action_name != req_name:
        return False
    if req_target and record.target != req_target:
        return False
    if record.risk_level != req_risk:
        return False
    if record.expires_at:
        expires = _to_dt(record.expires_at)
        if expires and datetime.now(timezone.utc) > expires:
            return False
    return True
