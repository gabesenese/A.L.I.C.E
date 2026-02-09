"""
Pattern Miner: Detects clusters of similar interactions and proposes new patterns.
Groups logged interactions by intent/topic; when N similar examples exist,
proposes a pattern for human approval.
"""

import json
import os
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class PatternMiner:
    def __init__(self, logged_interactions_path: str = "data/training/logged_interactions.jsonl",
                 proposed_patterns_path: str = "data/training/proposed_patterns.json",
                 threshold: int = 3,
                 similarity_threshold: float = 0.7):
        """
        Initialize pattern miner.
        
        Args:
            logged_interactions_path: Path to logged interactions JSONL file
            proposed_patterns_path: Path to proposed patterns JSON file
            threshold: Minimum similar examples needed to propose pattern
            similarity_threshold: Similarity score (0-1) to consider patterns related
        """
        self.logged_interactions_path = logged_interactions_path
        self.proposed_patterns_path = proposed_patterns_path
        self.threshold = threshold
        self.similarity_threshold = similarity_threshold
        self.proposed_patterns = self._load_proposed_patterns()

    def _load_proposed_patterns(self) -> Dict[str, Any]:
        """Load existing proposed patterns."""
        if os.path.exists(self.proposed_patterns_path):
            with open(self.proposed_patterns_path, 'r') as f:
                return json.load(f)
        return {"proposals": [], "metadata": {"total_proposals": 0, "approved": 0, "rejected": 0}}

    def _save_proposed_patterns(self):
        """Save proposed patterns to disk."""
        os.makedirs(os.path.dirname(self.proposed_patterns_path) or '.', exist_ok=True)
        with open(self.proposed_patterns_path, 'w') as f:
            json.dump(self.proposed_patterns, f, indent=2)

    def _load_logged_interactions(self) -> List[Dict[str, Any]]:
        """Load logged interactions from JSONL file."""
        interactions = []
        if os.path.exists(self.logged_interactions_path):
            with open(self.logged_interactions_path, 'r') as f:
                for line in f:
                    if line.strip():
                        interactions.append(json.loads(line))
        return interactions

    def _similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings (0-1)."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _cluster_by_intent(self, interactions: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group interactions by intent."""
        clusters = defaultdict(list)
        for interaction in interactions:
            intent = interaction.get("intent", "unknown")
            clusters[intent].append(interaction)
        return clusters

    def _cluster_by_similarity(self, interactions: List[Dict[str, Any]]) -> List[List[Dict]]:
        """Cluster interactions by semantic similarity of user inputs."""
        if not interactions:
            return []

        clusters = []
        unclustered = interactions.copy()

        while unclustered:
            seed = unclustered.pop(0)
            cluster = [seed]

            # Find all similar interactions
            remaining = []
            for interaction in unclustered:
                sim = self._similarity(
                    seed.get("user_input", ""),
                    interaction.get("user_input", "")
                )
                if sim >= self.similarity_threshold:
                    cluster.append(interaction)
                else:
                    remaining.append(interaction)

            unclustered = remaining
            if len(cluster) >= self.threshold:
                clusters.append(cluster)

        return clusters

    def mine_patterns(self) -> List[Dict[str, Any]]:
        """
        Mine patterns from logged interactions.
        Returns list of proposed patterns.
        """
        interactions = self._load_logged_interactions()
        if not interactions:
            return []

        proposed = []

        # Cluster by intent
        intent_clusters = self._cluster_by_intent(interactions)

        for intent, intent_group in intent_clusters.items():
            # Further cluster by similarity within intent
            similarity_clusters = self._cluster_by_similarity(intent_group)

            for cluster in similarity_clusters:
                if len(cluster) >= self.threshold:
                    pattern = self._create_pattern_proposal(intent, cluster)
                    proposed.append(pattern)

        return proposed

    def _create_pattern_proposal(self, intent: str, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a pattern proposal from a cluster of interactions."""
        user_inputs = [i.get("user_input", "") for i in cluster]
        responses = [i.get("llm_response", "") for i in cluster]
        entities_list = [i.get("entities", []) for i in cluster]

        # Common entities
        common_entities = set()
        if entities_list and entities_list[0]:
            common_entities = set(entities_list[0])
            for ents in entities_list[1:]:
                common_entities &= set(ents)

        tool_used = cluster[0].get("tool", None)

        pattern = {
            "id": f"{intent}_{len(self.proposed_patterns['proposals'])}",
            "intent": intent,
            "example_inputs": user_inputs[:3],  # Top 3 examples
            "proposed_template": self._extract_template(responses),
            "template_variants": responses[:2],
            "common_entities": list(common_entities),
            "tool": tool_used,
            "cluster_size": len(cluster),
            "avg_quality": sum(c.get("quality_score", 0.8) for c in cluster) / len(cluster),
            "approval_status": "pending",
            "created": datetime.now().isoformat(),
            "approved_by": None,
            "confidence": min(0.99, 0.8 + (len(cluster) * 0.05))  # Higher with more examples
        }

        return pattern

    def _extract_template(self, responses: List[str]) -> str:
        """Extract a template from response examples."""
        if not responses:
            return ""

        # Use the most common response as template
        return responses[0][:100] + "..." if len(responses[0]) > 100 else responses[0]

    def propose_new_patterns(self, min_new: int = 1) -> List[Dict[str, Any]]:
        """
        Mine new patterns and add to proposals.
        Returns list of newly proposed patterns.
        """
        new_patterns = self.mine_patterns()

        # Filter out duplicates (by intent + template similarity)
        existing_ids = {p["id"] for p in self.proposed_patterns["proposals"]}
        truly_new = [p for p in new_patterns if p["id"] not in existing_ids]

        if truly_new:
            self.proposed_patterns["proposals"].extend(truly_new)
            self.proposed_patterns["metadata"]["total_proposals"] = len(self.proposed_patterns["proposals"])
            self._save_proposed_patterns()

        return truly_new[:min_new]

    def approve_pattern(self, pattern_id: str) -> bool:
        """Approve a proposed pattern."""
        for pattern in self.proposed_patterns["proposals"]:
            if pattern["id"] == pattern_id:
                pattern["approval_status"] = "approved"
                pattern["approved_by"] = "user"
                self.proposed_patterns["metadata"]["approved"] += 1
                self._save_proposed_patterns()
                return True
        return False

    def reject_pattern(self, pattern_id: str) -> bool:
        """Reject a proposed pattern."""
        for pattern in self.proposed_patterns["proposals"]:
            if pattern["id"] == pattern_id:
                pattern["approval_status"] = "rejected"
                self.proposed_patterns["metadata"]["rejected"] += 1
                self._save_proposed_patterns()
                return True
        return False

    def get_pending_patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns awaiting approval."""
        return [p for p in self.proposed_patterns["proposals"]
                if p.get("approval_status") == "pending"]

    def get_approved_patterns(self) -> List[Dict[str, Any]]:
        """Get all approved patterns."""
        return [p for p in self.proposed_patterns["proposals"]
                if p.get("approval_status") == "approved"]

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about proposed patterns."""
        pending = self.get_pending_patterns()
        approved = self.get_approved_patterns()

        return {
            "total_proposals": len(self.proposed_patterns["proposals"]),
            "pending_approval": len(pending),
            "approved": self.proposed_patterns["metadata"]["approved"],
            "rejected": self.proposed_patterns["metadata"]["rejected"],
            "approval_rate": self.proposed_patterns["metadata"]["approved"] / max(
                self.proposed_patterns["metadata"]["total_proposals"], 1
            ),
            "pending_patterns": pending
        }
