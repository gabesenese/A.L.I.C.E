"""Role-based capability control for runtime tool execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional


class PermissionScope(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


class UserRole(str, Enum):
    OWNER = "owner"
    TRUSTED = "trusted"
    STANDARD = "standard"
    RESTRICTED = "restricted"


CONFIRM_MARKERS = (
    "confirm",
    "yes, do it",
    "yes do it",
    "proceed",
    "run it",
)


@dataclass(frozen=True)
class AccessRequest:
    intent: str
    user_input: str
    entities: Dict
    plugin_name: str = ""


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str
    required_scope: PermissionScope


ROLE_SCOPES = {
    UserRole.OWNER: {
        PermissionScope.READ,
        PermissionScope.WRITE,
        PermissionScope.EXECUTE,
        PermissionScope.DELETE,
        PermissionScope.ADMIN,
    },
    UserRole.TRUSTED: {
        PermissionScope.READ,
        PermissionScope.WRITE,
        PermissionScope.EXECUTE,
        PermissionScope.DELETE,
    },
    UserRole.STANDARD: {
        PermissionScope.READ,
        PermissionScope.WRITE,
    },
    UserRole.RESTRICTED: {PermissionScope.READ},
}


HIGH_RISK_SCOPES = {
    PermissionScope.DELETE,
    PermissionScope.EXECUTE,
    PermissionScope.ADMIN,
}


def _contains_any(text: str, words: Iterable[str]) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)


class RBACEngine:
    def __init__(
        self,
        default_role: UserRole = UserRole.STANDARD,
        confirmation_grace_seconds: int = 180,
    ) -> None:
        self.default_role = default_role
        self.confirmation_grace_seconds = max(0, int(confirmation_grace_seconds or 0))
        self._recent_scope_confirmation: Dict[str, float] = {}

    def _required_scope(self, request: AccessRequest) -> PermissionScope:
        intent = (request.intent or "").lower()
        user_input = (request.user_input or "").lower()

        if _contains_any(intent, ("system:", "shell", "execute")):
            return PermissionScope.EXECUTE
        if _contains_any(intent, ("file_operations:", "memory:")) and _contains_any(
            user_input, ("delete", "remove", "erase", "drop")
        ):
            return PermissionScope.DELETE
        if _contains_any(
            intent, ("calendar:", "email:", "notes:", "reminder:")
        ) and _contains_any(
            user_input, ("create", "add", "send", "write", "update", "edit")
        ):
            return PermissionScope.WRITE
        if _contains_any(
            user_input, ("delete", "remove", "shutdown", "restart", "format")
        ):
            return PermissionScope.DELETE
        return PermissionScope.READ

    def authorize(
        self, request: AccessRequest, role: Optional[UserRole] = None
    ) -> AccessDecision:
        active_role = role or self.default_role
        required = self._required_scope(request)
        granted = ROLE_SCOPES.get(active_role, ROLE_SCOPES[UserRole.STANDARD])

        if required not in granted:
            return AccessDecision(
                allowed=False,
                requires_confirmation=False,
                reason=f"Blocked by RBAC policy: role={active_role.value}, scope={required.value}",
                required_scope=required,
            )

        needs_confirmation = required in HIGH_RISK_SCOPES
        confirmed = _contains_any(request.user_input, CONFIRM_MARKERS)
        if confirmed:
            self._recent_scope_confirmation[required.value] = time.time()

        grace_until = self._recent_scope_confirmation.get(required.value, 0.0)
        grace_active = (time.time() - grace_until) <= float(
            self.confirmation_grace_seconds
        )
        if needs_confirmation and not confirmed:
            if grace_active:
                return AccessDecision(
                    allowed=True,
                    requires_confirmation=False,
                    reason=(
                        f"Allowed by recent confirmation grace for scope={required.value}."
                    ),
                    required_scope=required,
                )
            return AccessDecision(
                allowed=True,
                requires_confirmation=True,
                reason=(
                    f"Action requires confirmation for scope={required.value}. "
                    "Reply with 'confirm' to continue."
                ),
                required_scope=required,
            )

        return AccessDecision(
            allowed=True,
            requires_confirmation=False,
            reason=f"Allowed by RBAC policy: role={active_role.value}, scope={required.value}",
            required_scope=required,
        )


_rbac_engine: Optional[RBACEngine] = None


def get_rbac_engine(default_role: UserRole = UserRole.STANDARD) -> RBACEngine:
    global _rbac_engine
    if _rbac_engine is None:
        _rbac_engine = RBACEngine(default_role=default_role)
    return _rbac_engine
