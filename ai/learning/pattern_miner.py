"""
Pattern Miner: Detects clusters of similar interactions and proposes new patterns.
Groups logged interactions by intent/topic; when N similar examples exist,
proposes a pattern for human approval.
"""

import json
import os
import threading
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple
from datetime import datetime


class PatternMiner:
    def __init__(
        self,
        logged_interactions_path: str = "data/training/logged_interactions.jsonl",
        proposed_patterns_path: str = "data/training/proposed_patterns.json",
        threshold: int = 3,
        similarity_threshold: float = 0.7,
    ):
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
            with open(self.proposed_patterns_path, "r") as f:
                return json.load(f)
        return {
            "proposals": [],
            "metadata": {"total_proposals": 0, "approved": 0, "rejected": 0},
        }

    def _save_proposed_patterns(self):
        """Save proposed patterns to disk."""
        os.makedirs(os.path.dirname(self.proposed_patterns_path) or ".", exist_ok=True)
        with open(self.proposed_patterns_path, "w") as f:
            json.dump(self.proposed_patterns, f, indent=2)

    def _load_logged_interactions(self) -> List[Dict[str, Any]]:
        """Load logged interactions from JSONL file."""
        interactions = []
        if os.path.exists(self.logged_interactions_path):
            with open(self.logged_interactions_path, "r") as f:
                for line in f:
                    if line.strip():
                        interactions.append(json.loads(line))
        return interactions

    def _similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings (0-1)."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _cluster_by_intent(
        self, interactions: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict]]:
        """Group interactions by intent."""
        clusters = defaultdict(list)
        for interaction in interactions:
            intent = interaction.get("intent", "unknown")
            clusters[intent].append(interaction)
        return clusters

    def _cluster_by_similarity(
        self, interactions: List[Dict[str, Any]]
    ) -> List[List[Dict]]:
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
                    seed.get("user_input", ""), interaction.get("user_input", "")
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

    def _create_pattern_proposal(
        self, intent: str, cluster: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
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
            "avg_quality": sum(c.get("quality_score", 0.8) for c in cluster)
            / len(cluster),
            "approval_status": "pending",
            "created": datetime.now().isoformat(),
            "approved_by": None,
            "confidence": min(
                0.99, 0.8 + (len(cluster) * 0.05)
            ),  # Higher with more examples
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
            self.proposed_patterns["metadata"]["total_proposals"] = len(
                self.proposed_patterns["proposals"]
            )
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
        return [
            p
            for p in self.proposed_patterns["proposals"]
            if p.get("approval_status") == "pending"
        ]

    def get_approved_patterns(self) -> List[Dict[str, Any]]:
        """Get all approved patterns."""
        return [
            p
            for p in self.proposed_patterns["proposals"]
            if p.get("approval_status") == "approved"
        ]

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about proposed patterns."""
        pending = self.get_pending_patterns()
        self.get_approved_patterns()

        return {
            "total_proposals": len(self.proposed_patterns["proposals"]),
            "pending_approval": len(pending),
            "approved": self.proposed_patterns["metadata"]["approved"],
            "rejected": self.proposed_patterns["metadata"]["rejected"],
            "approval_rate": self.proposed_patterns["metadata"]["approved"]
            / max(self.proposed_patterns["metadata"]["total_proposals"], 1),
            "pending_patterns": pending,
        }


# ---------------------------------------------------------------------------
# Habit Miner + HTN Planner  (recurring action sequences → macro automations)
# ---------------------------------------------------------------------------

ActionTuple = Tuple[str, str, str]  # (intent, plugin, action)


def _fmt(a: ActionTuple) -> str:
    return f"{a[0]}/{a[1]}/{a[2]}"


def _ngrams(seq: List[ActionTuple], n: int) -> Iterator[Tuple[ActionTuple, ...]]:
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i : i + n])


@dataclass
class HabitMacro:
    """
    A discovered recurring action sequence with confidence tracking.

    confidence = occurrence_count / threshold  (>= 1.0 means threshold met)
    trigger_length controls how many trailing actions must match to suggest.
    """

    name: str
    sequence: List[ActionTuple]
    trigger_length: int = 2
    confidence: float = 1.0
    first_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    fire_count: int = 0

    def trigger_pattern(self) -> List[ActionTuple]:
        return self.sequence[-self.trigger_length :]

    def matches_tail(self, recent: List[ActionTuple]) -> bool:
        pattern = self.trigger_pattern()
        if len(recent) < len(pattern):
            return False
        return recent[-len(pattern) :] == pattern

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sequence"] = [list(t) for t in self.sequence]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HabitMacro":
        seq = [tuple(t) for t in d.pop("sequence")]
        return HabitMacro(sequence=seq, **d)


class HabitMiner:
    """
    Discovers recurring action sequences from a sliding window and promotes
    high-frequency n-grams to HabitMacro objects.

    Improvement over PatternMiner: uses ActionTuples (intent+plugin+action)
    with PrefixSpan-style n-gram counting over a configurable window.
    """

    def __init__(
        self,
        window_size: int = 200,
        ngram_range: Tuple[int, int] = (2, 5),
        threshold: int = 3,
        persistence_path: Optional[str] = None,
    ) -> None:
        self._window: Deque[ActionTuple] = deque(maxlen=window_size)
        self._min_n, self._max_n = ngram_range
        self._threshold = threshold
        self._counts: Counter = Counter()
        self._macros: Dict[str, HabitMacro] = {}
        self._hm_lock = threading.Lock()
        self._path = Path(persistence_path) if persistence_path else None
        if self._path and self._path.exists():
            self._load()

    def update(self, intent: str, plugin: str, action: str) -> Optional[HabitMacro]:
        """Record a new action and run incremental mining. Returns a new macro if promoted."""
        with self._hm_lock:
            tup: ActionTuple = (intent, plugin, action)
            self._window.append(tup)
            window_list = list(self._window)
            newly_promoted: Optional[HabitMacro] = None
            for n in range(self._min_n, self._max_n + 1):
                for gram in _ngrams(window_list, n):
                    self._counts[gram] += 1
                    count = self._counts[gram]
                    gram_key = "|".join(_fmt(a) for a in gram)
                    if count >= self._threshold and gram_key not in self._macros:
                        macro = self._promote(gram, count)
                        self._macros[gram_key] = macro
                        newly_promoted = macro
                    elif gram_key in self._macros:
                        self._macros[gram_key].confidence = count / self._threshold
                        self._macros[gram_key].last_seen = datetime.utcnow().isoformat()
        if newly_promoted and self._path:
            self._save()
        return newly_promoted

    def suggest(
        self, recent: Optional[List[ActionTuple]] = None
    ) -> Optional[HabitMacro]:
        """Return the highest-confidence macro whose trigger matches the recent tail."""
        with self._hm_lock:
            tail = recent if recent is not None else list(self._window)
            candidates = [m for m in self._macros.values() if m.matches_tail(tail)]
            if not candidates:
                return None
            best = max(candidates, key=lambda m: (m.confidence, len(m.sequence)))
            best.fire_count += 1
            return best

    def all_macros(self) -> List[HabitMacro]:
        with self._hm_lock:
            return sorted(
                self._macros.values(), key=lambda m: m.confidence, reverse=True
            )

    def recent_actions(self, k: int = 10) -> List[ActionTuple]:
        with self._hm_lock:
            return list(self._window)[-k:]

    @staticmethod
    def _promote(gram: Tuple[ActionTuple, ...], count: int) -> HabitMacro:
        seq = list(gram)
        intent_labels = [a[0].replace(":", "_") for a in seq]
        name = "→".join(dict.fromkeys(intent_labels))
        return HabitMacro(
            name=name,
            sequence=seq,
            trigger_length=min(2, len(seq)),
            confidence=float(count),
        )

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lines = [json.dumps(m.to_dict()) for m in self._macros.values()]
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text("\n".join(lines), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            pass  # non-critical persistence

    def _load(self) -> None:
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                m = HabitMacro.from_dict(d)
                gram_key = "|".join(_fmt(a) for a in m.sequence)
                self._macros[gram_key] = m
        except Exception:
            pass


@dataclass
class HTNMethod:
    """A decomposition method: goal → list of sub-goals or primitive sequences."""

    goal: str
    conditions: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    priority: float = 1.0


class HTNPlanner:
    """
    Hierarchical Task Network planner. Decomposes abstract goals into sequences
    of primitive actions using mined HabitMacro objects as method options.
    """

    def __init__(self, habit_miner: Optional[HabitMiner] = None) -> None:
        self._methods: Dict[str, List[HTNMethod]] = {}
        self._miner = habit_miner
        self._htn_lock = threading.Lock()

    def add_method(self, method: HTNMethod) -> None:
        with self._htn_lock:
            self._methods.setdefault(method.goal, []).append(method)

    def decompose(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Return a flat list of primitive action strings for goal."""
        if self._miner:
            for macro in self._miner.all_macros():
                if goal.lower() in macro.name.lower():
                    return [_fmt(a) for a in macro.sequence]
        with self._htn_lock:
            methods = sorted(
                self._methods.get(goal, []), key=lambda m: m.priority, reverse=True
            )
        ctx = context or {}
        for method in methods:
            if all(ctx.get(cond) for cond in method.conditions):
                plan: List[str] = []
                for subtask in method.subtasks:
                    plan.extend(self.decompose(subtask, context))
                return plan
        return [goal]


_habit_miner_instance: Optional[HabitMiner] = None
_htn_planner_instance: Optional[HTNPlanner] = None
_habit_singletons_lock = threading.Lock()


def get_habit_miner(persistence_path: Optional[str] = None) -> HabitMiner:
    """Return the process-wide singleton HabitMiner."""
    global _habit_miner_instance
    if _habit_miner_instance is None:
        with _habit_singletons_lock:
            if _habit_miner_instance is None:
                _habit_miner_instance = HabitMiner(
                    persistence_path=persistence_path or "memory/habits.jsonl"
                )
    return _habit_miner_instance


def get_htn_planner() -> HTNPlanner:
    """Return the process-wide singleton HTNPlanner."""
    global _htn_planner_instance
    if _htn_planner_instance is None:
        with _habit_singletons_lock:
            if _htn_planner_instance is None:
                _htn_planner_instance = HTNPlanner(habit_miner=get_habit_miner())
    return _htn_planner_instance
