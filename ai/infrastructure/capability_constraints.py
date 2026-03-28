"""
Capability Constraints Ledger - Tier 1: Consistent Identity Thread

Maintains a single source of truth for what ALICE can/cannot do.
Pre-validates all capability claims against this ledger.
"""

import logging
from typing import Dict, List, Set, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CapabilityLevel(Enum):
    """Capability access levels."""
    ALWAYS = "always"  # Always available
    WITH_APPROVAL = "with_approval"  # Requires user confirmation
    NEVER = "never"  # Never available


@dataclass
class Capability:
    """A single capability with its constraints."""
    
    name: str  # e.g., "delete_files"
    level: CapabilityLevel
    depends_on: List[str] = field(default_factory=list)  # Other capabilities needed
    risk_level: str = "low"  # low, medium, high, critical
    description: str = ""
    examples: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)  # Custom constraints
    
    def is_available(self) -> bool:
        """Check if capability is available."""
        return self.level != CapabilityLevel.NEVER
    
    def requires_approval(self) -> bool:
        """Check if capability requires approval."""
        return self.level == CapabilityLevel.WITH_APPROVAL


class CapabilityConstraintsLedger:
    """Single source of truth for ALICE's capabilities."""
    
    def __init__(self):
        self.capabilities: Dict[str, Capability] = {}
        self.capability_groups: Dict[str, Set[str]] = {}
        self._init_default_capabilities()
    
    def _init_default_capabilities(self):
        """Initialize ALICE's core capabilities with constraints."""
        
        # ── Read-Only Operations ──
        self.register_capability(
            Capability(
                name="read_files",
                level=CapabilityLevel.ALWAYS,
                risk_level="low",
                description="Read files from workspace",
                examples=["read main.py", "show me config.json"],
            )
        )
        
        self.register_capability(
            Capability(
                name="search_codebase",
                level=CapabilityLevel.ALWAYS,
                risk_level="low",
                description="Search within codebase",
                examples=["find all references to function X", "search for TODO comments"],
            )
        )
        
        self.register_capability(
            Capability(
                name="analyze_code",
                level=CapabilityLevel.ALWAYS,
                risk_level="low",
                description="Analyze code structure and architecture",
                examples=["explain this function", "what does this module do"],
            )
        )
        
        # ── Write Operations (with approval) ──
        self.register_capability(
            Capability(
                name="create_files",
                level=CapabilityLevel.WITH_APPROVAL,
                risk_level="medium",
                description="Create new files",
                examples=["create a new test file", "write config.json"],
                constraints={"max_file_size_mb": 50},
            )
        )
        
        self.register_capability(
            Capability(
                name="modify_files",
                level=CapabilityLevel.WITH_APPROVAL,
                risk_level="medium",
                description="Modify existing files",
                examples=["update main.py", "fix this code"],
                constraints={"require_backup": True, "require_confirmation": True},
            )
        )
        
        self.register_capability(
            Capability(
                name="delete_files",
                level=CapabilityLevel.WITH_APPROVAL,
                risk_level="critical",
                description="Delete files",
                examples=["remove this file", "delete temp files"],
                constraints={"no_system_files": True, "require_confirmation": True},
            )
        )
        
        # ── Version Control ──
        self.register_capability(
            Capability(
                name="git_commit",
                level=CapabilityLevel.WITH_APPROVAL,
                risk_level="high",
                description="Commit changes to git",
                examples=["commit these changes", "create a snapshot"],
                depends_on=["modify_files"],
                constraints={"require_message": True, "require_confirmation": True},
            )
        )
        
        self.register_capability(
            Capability(
                name="git_push",
                level=CapabilityLevel.WITH_APPROVAL,
                risk_level="high",
                description="Push to remote repository",
                examples=["push to main", "upload commits"],
                depends_on=["git_commit"],
                constraints={"require_confirmation": True, "verify_remote": True},
            )
        )
        
        # ── Build/Test Operations ──
        self.register_capability(
            Capability(
                name="run_tests",
                level=CapabilityLevel.ALWAYS,
                risk_level="low",
                description="Run test suite",
                examples=["run tests", "test this"],
                constraints={"timeout_seconds": 300},
            )
        )
        
        self.register_capability(
            Capability(
                name="run_build",
                level=CapabilityLevel.ALWAYS,
                risk_level="low",
                description="Run build process",
                examples=["build project", "compile"],
                constraints={"timeout_seconds": 600},
            )
        )
        
        # ── System Operations ──
        self.register_capability(
            Capability(
                name="execute_shell",
                level=CapabilityLevel.NEVER,  # Never directly
                risk_level="critical",
                description="Execute arbitrary shell commands",
                constraints={"always_denied": True},
            )
        )
        
        self.register_capability(
            Capability(
                name="modify_config",
                level=CapabilityLevel.WITH_APPROVAL,
                risk_level="high",
                description="Modify configuration files",
                constraints={"require_confirmation": True, "backup_existing": True},
            )
        )
        
        # ── Grouping ──
        self._add_to_group("read-only", ["read_files", "search_codebase", "analyze_code", "run_tests", "run_build"])
        self._add_to_group("write", ["create_files", "modify_files", "delete_files"])
        self._add_to_group("vcs", ["git_commit", "git_push"])
        self._add_to_group("dangerous", ["delete_files", "execute_shell", "git_push"])
    
    def register_capability(self, capability: Capability) -> None:
        """Register a capability."""
        self.capabilities[capability.name] = capability
        logger.info(f"[Constraints] Registered capability: {capability.name} ({capability.level.value})")
    
    def can_do(self, capability: str) -> bool:
        """Check if ALICE can perform a capability."""
        cap = self.capabilities.get(capability)
        if not cap:
            logger.warning(f"[Constraints] Unknown capability: {capability}")
            return False
        return cap.is_available()
    
    def requires_approval_for(self, capability: str) -> bool:
        """Check if capability requires approval."""
        cap = self.capabilities.get(capability)
        if not cap:
            return False
        return cap.requires_approval()
    
    def get_capability(self, name: str) -> Capability:
        """Get capability details."""
        return self.capabilities.get(name)
    
    def validate_capability_claim(self, claimed_capability: str) -> tuple[bool, str]:
        """
        Validate if ALICE should claim she can do something.
        
        Returns: (is_valid, reason)
        """
        cap = self.capabilities.get(claimed_capability)
        
        if not cap:
            return False, f"Unknown capability: {claimed_capability}"
        
        if not cap.is_available():
            return False, f"Capability '{claimed_capability}' is not available"
        
        if cap.is_available():
            return True, "Capability available"
        
        return True, f"Capability available but requires approval"
    
    def validate_capability_disclaimer(self, disclaimed_capability: str) -> tuple[bool, str]:
        """
        Validate if ALICE should claim she CANNOT do something.
        
        Returns: (is_valid, reason)
        """
        cap = self.capabilities.get(disclaimed_capability)
        
        if not cap:
            return False, f"Unknown capability: {disclaimed_capability}"
        
        if not cap.is_available():
            return True, f"Capability is correctly not available"
        
        return False, f"CONTRADICTION: Claimed cannot do '{disclaimed_capability}' but it IS available"
    
    def _add_to_group(self, group: str, capabilities: List[str]) -> None:
        """Group capabilities for easier management."""
        self.capability_groups[group] = set(capabilities)
    
    def get_all_capabilities(self, available_only: bool = False) -> Dict[str, Capability]:
        """Get all capabilities."""
        if not available_only:
            return self.capabilities.copy()
        return {k: v for k, v in self.capabilities.items() if v.is_available()}
    
    def get_capabilities_requiring_approval(self) -> Dict[str, Capability]:
        """Get all capabilities that require approval."""
        return {k: v for k, v in self.capabilities.items() if v.requires_approval()}
    
    def get_constraints_for(self, capability: str) -> Dict[str, Any]:
        """Get constraints for a specific capability."""
        cap = self.capabilities.get(capability)
        if not cap:
            return {}
        return cap.constraints.copy()
    
    def check_constraint(self, capability: str, constraint_key: str) -> Any:
        """Check a specific constraint for a capability."""
        cap = self.capabilities.get(capability)
        if not cap:
            return None
        return cap.constraints.get(constraint_key)
    
    def capability_contradiction_detected(
        self,
        claim: str,
        current_ledger_state: bool
    ) -> bool:
        """
        Detect if ALICE made a contradictory capability claim.
        
        E.g., "I can do X" but ledger says capability.NEVER
        """
        cap = self.capabilities.get(claim)
        if not cap:
            return False
        
        ledger_available = cap.is_available()
        return claim != "" and ledger_available != current_ledger_state
    
    def get_high_risk_capabilities(self) -> Dict[str, Capability]:
        """Get all high-risk capabilities for safety monitoring."""
        return {
            k: v for k, v in self.capabilities.items()
            if v.risk_level in ("high", "critical")
        }


# Singleton
_ledger: CapabilityConstraintsLedger = None


def get_capability_constraints_ledger() -> CapabilityConstraintsLedger:
    """Get or create the capability constraints ledger."""
    global _ledger
    if _ledger is None:
        _ledger = CapabilityConstraintsLedger()
    return _ledger
