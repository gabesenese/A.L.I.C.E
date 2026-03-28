"""Approval ledger with confirmation provenance for high-risk operations."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    action: str
    scope: str
    summary: str
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    action: str
    scope: str
    summary: str
    approved: bool
    confirmation_text: str
    actor: str
    created_at: float
    recorded_at: float


class ApprovalLedger:
    def __init__(
        self,
        storage_path: str = "data/security/approval_ledger.jsonl",
        ttl_seconds: int = 300,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(30, int(ttl_seconds or 300))
        self._pending: Dict[str, ApprovalRequest] = {}
        self._scope_approvals: Dict[str, float] = {}

    def create_request(
        self, *, action: str, scope: str, summary: str
    ) -> ApprovalRequest:
        now = time.time()
        req = ApprovalRequest(
            approval_id=f"appr-{uuid.uuid4().hex[:12]}",
            action=str(action or "unknown"),
            scope=str(scope or "unknown"),
            summary=str(summary or ""),
            created_at=now,
            expires_at=now + float(self.ttl_seconds),
        )
        self._pending[req.approval_id] = req
        return req

    def get_pending(self, approval_id: str) -> Optional[ApprovalRequest]:
        req = self._pending.get(str(approval_id or ""))
        if not req:
            return None
        if req.expires_at < time.time():
            self._pending.pop(req.approval_id, None)
            return None
        return req

    def confirm(
        self, *, approval_id: str, confirmation_text: str, actor: str = "user"
    ) -> Optional[ApprovalRecord]:
        req = self.get_pending(approval_id)
        if not req:
            return None

        record = ApprovalRecord(
            approval_id=req.approval_id,
            action=req.action,
            scope=req.scope,
            summary=req.summary,
            approved=True,
            confirmation_text=str(confirmation_text or ""),
            actor=str(actor or "user"),
            created_at=req.created_at,
            recorded_at=time.time(),
        )
        self.note_scope_approval(req.scope, ttl_seconds=self.ttl_seconds)
        self._append_record(record)
        self._pending.pop(req.approval_id, None)
        return record

    def reject(
        self, *, approval_id: str, confirmation_text: str, actor: str = "user"
    ) -> Optional[ApprovalRecord]:
        req = self.get_pending(approval_id)
        if not req:
            return None

        record = ApprovalRecord(
            approval_id=req.approval_id,
            action=req.action,
            scope=req.scope,
            summary=req.summary,
            approved=False,
            confirmation_text=str(confirmation_text or ""),
            actor=str(actor or "user"),
            created_at=req.created_at,
            recorded_at=time.time(),
        )
        self._append_record(record)
        self._pending.pop(req.approval_id, None)
        return record

    def _append_record(self, record: ApprovalRecord) -> None:
        with self.storage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def note_scope_approval(self, scope: str, ttl_seconds: int | None = None) -> None:
        ttl = max(30, int(ttl_seconds or self.ttl_seconds))
        self._scope_approvals[str(scope or "unknown")] = time.time() + float(ttl)

    def is_scope_approved(self, scope: str) -> bool:
        key = str(scope or "unknown")
        exp = float(self._scope_approvals.get(key, 0.0))
        if exp <= 0.0:
            return False
        if time.time() > exp:
            self._scope_approvals.pop(key, None)
            return False
        return True


_approval_ledger: Optional[ApprovalLedger] = None


def get_approval_ledger(
    storage_path: str = "data/security/approval_ledger.jsonl",
) -> ApprovalLedger:
    global _approval_ledger
    if _approval_ledger is None:
        _approval_ledger = ApprovalLedger(storage_path=storage_path)
    return _approval_ledger
