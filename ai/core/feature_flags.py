"""
A.L.I.C.E. Feature Flag System
==============================
Lightweight feature toggles for A/B testing and gradual rollouts.

Features:
- JSON-based configuration
- Runtime toggle (no restarts)
- Per-user overrides
- Rollout percentages

Usage:
    >>> flags = get_feature_flags()
    >>> if flags.is_enabled("nlp_entity_normalizer"):
    ...     # Use new normalizer
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlag:
    """Single feature flag configuration."""
    
    name: str
    enabled: bool = False
    rollout_pct: int = 100  # Percentage of users who get this (0-100)
    description: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)


class FeatureFlagManager:
    """
    Manages feature flags with hot-reload support.
    
    Algorithm: Lock-protected dictionary with lazy file watching
    Complexity: O(1) flag lookup
    """
    
    _instance: Optional[FeatureFlagManager] = None
    _lock = Lock()
    
    def __new__(cls, config_path: Optional[Path] = None):
        """Singleton with optional config path override."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[Path] = None):
        if self._initialized:
            return
        
        self._initialized = True
        self.config_path = config_path or Path("data/feature_flags.json")
        self._flags: Dict[str, FeatureFlag] = {}
        self._user_overrides: Dict[str, Set[str]] = {}  # user_id -> set of enabled flags
        self._reload_lock = Lock()
        
        # Default flags for P0 NLP improvements
        self._defaults = {
            "nlp_intent_entity_validation": FeatureFlag(
                name="nlp_intent_entity_validation",
                enabled=True,  # Enable by default for testing
                description="P0-1: Intent-entity cross-validation with penalty system"
            ),
            "nlp_ambiguity_resolver": FeatureFlag(
                name="nlp_ambiguity_resolver",
                enabled=True,
                description="P0-2: Multi-candidate ambiguity detection for coreference"
            ),
            "nlp_entity_normalizer": FeatureFlag(
                name="nlp_entity_normalizer",
                enabled=True,
                description="P0-3: Rule-based entity normalization layer"
            ),
            "nlp_validation_prompts": FeatureFlag(
                name="nlp_validation_prompts",
                enabled=True,
                description="Show clarification prompts when validation fails"
            ),
        }
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load feature flags from JSON config."""
        # Start with defaults
        self._flags = dict(self._defaults)
        
        # Override with config file if it exists
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                # Load flags
                for flag_data in data.get("flags", []):
                    flag = FeatureFlag(**flag_data)
                    self._flags[flag.name] = flag
                
                # Load user overrides
                for user_id, enabled_flags in data.get("user_overrides", {}).items():
                    self._user_overrides[user_id] = set(enabled_flags)
                
                logger.info(f"[FEATURE_FLAGS] Loaded {len(self._flags)} flags from {self.config_path}")
            except Exception as e:
                logger.error(f"[FEATURE_FLAGS] Failed to load config: {e}")
        else:
            # Create default config
            self._save_config()
            logger.info("[FEATURE_FLAGS] Created default config with P0 improvements enabled")
    
    def _save_config(self) -> None:
        """Persist current flags to disk."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "flags": [
                    {
                        "name": flag.name,
                        "enabled": flag.enabled,
                        "rollout_pct": flag.rollout_pct,
                        "description": flag.description,
                        "metadata": flag.metadata,
                    }
                    for flag in self._flags.values()
                ],
                "user_overrides": {
                    user_id: list(flags) 
                    for user_id, flags in self._user_overrides.items()
                }
            }
            
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"[FEATURE_FLAGS] Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"[FEATURE_FLAGS] Failed to save config: {e}")
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            flag_name: Feature flag name
            user_id: Optional user ID for per-user overrides
            
        Returns:
            True if feature is enabled for this user
        """
        # Check user override first
        if user_id and user_id in self._user_overrides:
            return flag_name in self._user_overrides[user_id]
        
        # Check global flag
        flag = self._flags.get(flag_name)
        if not flag:
            return False  # Unknown flags default to disabled
        
        return flag.enabled
    
    def enable(self, flag_name: str, persist: bool = True) -> None:
        """Enable a feature flag globally."""
        if flag_name in self._flags:
            self._flags[flag_name].enabled = True
            if persist:
                self._save_config()
            logger.info(f"[FEATURE_FLAGS] Enabled: {flag_name}")
    
    def disable(self, flag_name: str, persist: bool = True) -> None:
        """Disable a feature flag globally."""
        if flag_name in self._flags:
            self._flags[flag_name].enabled = False
            if persist:
                self._save_config()
            logger.info(f"[FEATURE_FLAGS] Disabled: {flag_name}")
    
    def set_user_override(self, user_id: str, flag_name: str, enabled: bool) -> None:
        """Set a per-user override for a specific flag."""
        if user_id not in self._user_overrides:
            self._user_overrides[user_id] = set()
        
        if enabled:
            self._user_overrides[user_id].add(flag_name)
        else:
            self._user_overrides[user_id].discard(flag_name)
        
        self._save_config()
        logger.info(f"[FEATURE_FLAGS] User override: {user_id} - {flag_name} = {enabled}")
    
    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all registered feature flags."""
        return dict(self._flags)
    
    def reload(self) -> None:
        """Hot-reload configuration from disk."""
        with self._reload_lock:
            self._load_config()
            logger.info("[FEATURE_FLAGS] Configuration reloaded")


# Global singleton instance
_global_flags: Optional[FeatureFlagManager] = None


def get_feature_flags(config_path: Optional[Path] = None) -> FeatureFlagManager:
    """Get the global feature flag manager."""
    global _global_flags
    if _global_flags is None:
        _global_flags = FeatureFlagManager(config_path)
    return _global_flags
