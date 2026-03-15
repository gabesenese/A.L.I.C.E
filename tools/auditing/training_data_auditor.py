"""
Training Data Auditor & Cleaner
================================
Analyzes Alice's training data and removes low-quality or outdated learning.
Alice should only learn from her best interactions.
"""

import argparse
import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    severity: str
    code: str
    area: str
    path: str
    message: str
    line: Optional[int] = None
    sample: Optional[str] = None
    record_locator: Optional[str] = None


class TrainingDataAuditor:
    """
    Audits and cleans Alice's training data.
    Removes:
    - Low quality responses
    - Hallucinated information
    - Outdated patterns
    - Inconsistent data
    - Poor examples
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training"
        self.knowledge_dir = self.data_dir / "knowledge"

        # Quality thresholds
        self.min_quality_score = 0.7
        self.min_response_length = 10
        self.max_response_length = 1000

        # Hallucination indicators
        self.hallucination_markers = [
            "stanford research",
            "sri international",
            "ladie system",
            "ladie project",
            "knowledge graph",  # If Alice doesn't actually have one
            "neural network",   # If not actually using one
        ]

        # Error indicators
        self.error_markers = [
            "i apologize",
            "i don't know",
            "i'm not sure",
            "error occurred",
            "something went wrong",
            "try again",
            "can't help with that"
        ]

    def audit_training_examples(self) -> Dict[str, Any]:
        """Audit all training data and return analysis"""

        training_file = self.training_dir / "training_data.jsonl"
        if not training_file.exists():
            logger.warning(f"Training file not found: {training_file}")
            return {"status": "no_data"}

        total = 0
        keep = []
        remove = []
        issues = defaultdict(int)

        with open(training_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                total += 1
                try:
                    example = json.loads(line)

                    # Analyze this example
                    should_keep, reason = self._should_keep_example(example)

                    if should_keep:
                        keep.append(example)
                    else:
                        remove.append(example)
                        issues[reason] += 1

                except json.JSONDecodeError:
                    issues['invalid_json'] += 1
                    continue

        return {
            'total_examples': total,
            'keep_count': len(keep),
            'remove_count': len(remove),
            'issues': dict(issues),
            'quality_rate': len(keep) / total if total > 0 else 0,
            'examples_to_keep': keep
        }

    def _should_keep_example(self, example: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if training example should be kept"""

        response = example.get('assistant_response', '')
        user_input = example.get('user_input', '')
        quality_score = example.get('quality_score', 0.5)

        # Check 1: Quality score too low
        if quality_score < self.min_quality_score:
            return False, 'low_quality_score'

        # Check 2: Response too short or too long
        if len(response) < self.min_response_length:
            return False, 'response_too_short'
        if len(response) > self.max_response_length:
            return False, 'response_too_long'

        # Check 3: Contains error markers
        response_lower = response.lower()
        for marker in self.error_markers:
            if marker in response_lower:
                return False, 'contains_error_marker'

        # Check 4: Contains hallucination markers
        for marker in self.hallucination_markers:
            if marker in response_lower:
                return False, 'hallucinated_content'

        # Check 5: Empty or useless
        if not response.strip() or not user_input.strip():
            return False, 'empty_content'

        # Check 6: Repetitive pattern (same response copied)
        if response.count(response[:20]) > 2:
            return False, 'repetitive_content'

        return True, 'passed'

    def clean_training_data(self, backup: bool = True):
        """Clean training data by removing low-quality examples"""

        logger.info("Starting training data audit...")

        # Audit first
        audit_result = self.audit_training_examples()

        if audit_result.get('status') == 'no_data':
            logger.warning("No training data to clean")
            return audit_result

        logger.info(f"Audit complete:")
        logger.info(f"  Total: {audit_result['total_examples']}")
        logger.info(f"  Keep: {audit_result['keep_count']}")
        logger.info(f"  Remove: {audit_result['remove_count']}")
        logger.info(f"  Quality rate: {audit_result['quality_rate']:.1%}")

        if audit_result['issues']:
            logger.info("  Issues found:")
            for issue, count in sorted(audit_result['issues'].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    - {issue}: {count}")

        # Backup original
        if backup:
            training_file = self.training_dir / "training_data.jsonl"
            backup_file = self.training_dir / f"training_data.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

            if training_file.exists():
                import shutil
                shutil.copy(training_file, backup_file)
                logger.info(f"Backup created: {backup_file}")

        # Write cleaned data
        training_file = self.training_dir / "training_data.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for example in audit_result['examples_to_keep']:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Cleaned training data written to {training_file}")

        return audit_result

    def audit_knowledge_entities(self) -> Dict[str, Any]:
        """Audit knowledge entities - remove noise words captured as entities"""

        entities_file = self.knowledge_dir / "entities.json"
        if not entities_file.exists():
            return {"status": "no_data"}

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities = json.load(f)

        # Noise words that shouldn't be entities
        noise_words = {
            'i', 'you', 'we', 'they', 'it', 'he', 'she',
            'my', 'your', 'our', 'their', 'his', 'her',
            'as', 'if', 'but', 'and', 'or', 'so', 'for',
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'how', 'what', 'when', 'where', 'who', 'why',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'can', 'may', 'might',
            'need', 'want', 'like', 'know', 'think', 'see', 'get',
            'make', 'take', 'give', 'go', 'come', 'say', 'tell',
            'let', 'nice', 'good', 'great', 'well', 'now', 'then',
            'since', 'given', 'while', 'however', 'therefore', 'thus',
            'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about'
        }

        keep = {}
        remove = {}

        for name, data in entities.items():
            name_lower = name.lower().strip()

            # Remove noise words
            if name_lower in noise_words:
                remove[name] = data
                continue

            # Remove if very short and low confidence
            if len(name) <= 2 and data['confidence'] < 0.8:
                remove[name] = data
                continue

            # Remove if never mentioned (mention_count = 0)
            if data['mention_count'] == 0:
                remove[name] = data
                continue

            keep[name] = data

        return {
            'total_entities': len(entities),
            'keep_count': len(keep),
            'remove_count': len(remove),
            'cleaned_entities': keep
        }

    def clean_knowledge_entities(self, backup: bool = True):
        """Clean knowledge entities"""

        logger.info("Starting knowledge entity audit...")

        audit_result = self.audit_knowledge_entities()

        if audit_result.get('status') == 'no_data':
            logger.warning("No knowledge data to clean")
            return audit_result

        logger.info(f"Entity audit complete:")
        logger.info(f"  Total: {audit_result['total_entities']}")
        logger.info(f"  Keep: {audit_result['keep_count']}")
        logger.info(f"  Remove: {audit_result['remove_count']}")

        # Backup original
        if backup:
            entities_file = self.knowledge_dir / "entities.json"
            backup_file = self.knowledge_dir / f"entities.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            if entities_file.exists():
                import shutil
                shutil.copy(entities_file, backup_file)
                logger.info(f"Backup created: {backup_file}")

        # Write cleaned data
        entities_file = self.knowledge_dir / "entities.json"
        with open(entities_file, 'w', encoding='utf-8') as f:
            json.dump(audit_result['cleaned_entities'], f, indent=2)

        logger.info(f"Cleaned entities written to {entities_file}")

        return audit_result

    def full_audit_and_clean(self):
        """Run full audit on all of Alice's learning data"""

        print("=" * 70)
        print("ALICE TRAINING DATA AUDITOR")
        print("=" * 70)
        print()

        results = {}

        # Clean training data
        print("1. Auditing training examples...")
        print("-" * 70)
        results['training'] = self.clean_training_data(backup=True)
        print()

        # Clean knowledge entities
        print("2. Auditing knowledge entities...")
        print("-" * 70)
        results['entities'] = self.clean_knowledge_entities(backup=True)
        print()

        # Summary
        print("=" * 70)
        print("AUDIT COMPLETE")
        print("=" * 70)

        if results.get('training', {}).get('status') != 'no_data':
            training = results['training']
            print(f"Training Data:")
            print(f"  Removed: {training['remove_count']} low-quality examples")
            print(f"  Kept: {training['keep_count']} high-quality examples")
            print(f"  Quality: {training['quality_rate']:.1%}")

        if results.get('entities', {}).get('status') != 'no_data':
            entities = results['entities']
            print(f"\nKnowledge Entities:")
            print(f"  Removed: {entities['remove_count']} noise entities")
            print(f"  Kept: {entities['keep_count']} valid entities")

        print()
        print("All backups saved to data/ directories")
        print("Alice will now learn only from high-quality data!")
        print()

        return results


class LearningDataQAAuditor:
    """Audit learned phrasing and related persisted learning artifacts."""

    AREAS = ("learned_phrasings", "entities", "relationships", "patterns")
    AREA_GROUPS = {
        "learned_phrasings": "learned_phrasings",
        "entities": "knowledge",
        "relationships": "knowledge",
        "patterns": "knowledge",
    }
    SEVERITY_ORDER = {"info": 1, "warning": 2, "critical": 3}

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "can", "could", "did", "do", "does", "for", "from", "get", "go",
        "had", "has", "have", "he", "her", "here", "hers", "him", "his", "how",
        "i", "if", "in", "is", "it", "its", "know", "like", "make", "me",
        "my", "need", "now", "of", "on", "or", "our", "say", "see", "she",
        "so", "something", "tell", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "those", "to", "us", "want", "was",
        "we", "were", "what", "when", "where", "who", "why", "will", "with",
        "would", "you", "your",
    }

    DOMAIN_KEYWORDS = {
        "music": {"music", "song", "play", "pause", "stop", "track", "audio"},
        "weather": {"weather", "forecast", "temperature", "rain", "snow", "umbrella", "coat", "jacket"},
        "notes": {"note", "notes", "list", "write down", "archive"},
        "email": {"email", "mail", "inbox", "reply", "send", "compose"},
        "calendar": {"calendar", "meeting", "schedule", "appointment", "event"},
        "file_operations": {"file", "folder", "directory", "open", "read", "write", "delete"},
        "memory": {"remember", "recall", "forget", "memory", "preference"},
    }

    LLM_META_MARKERS = (
        "this revised response",
        "here's a natural phrasing",
        "here is a natural phrasing",
        "with the specified tone",
        "still conveys the accurate information",
        "alice has formulated a response",
    )

    FAILURE_MARKERS = (
        "couldn't",
        "could not",
        "failed",
        "problem",
        "went wrong",
        "not able",
        "can't",
        "cannot",
    )

    SUCCESS_MARKERS = (
        "all done",
        "created",
        "deleted",
        "found",
        "went ahead",
        "successfully",
        "done!",
    )

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.findings: List[Finding] = []

    def audit(self) -> Dict[str, Any]:
        self.findings = []

        for file_path in self._iter_default_files():
            if not file_path.exists():
                continue
            if file_path.suffix == ".jsonl":
                self._audit_jsonl(file_path)
            elif file_path.suffix == ".json":
                self._audit_json(file_path)

        severity_counts = Counter(f.severity for f in self.findings)
        code_counts = Counter(f.code for f in self.findings)
        area_summaries = self._build_area_summaries()
        group_summaries = self._build_group_summaries(area_summaries)

        return {
            "generated_at": datetime.now().isoformat(),
            "root_dir": str(self.root_dir),
            "total_findings": len(self.findings),
            "severity_counts": dict(severity_counts),
            "issue_counts": dict(code_counts),
            "area_summaries": area_summaries,
            "group_summaries": group_summaries,
            "findings": [asdict(f) for f in self.findings],
        }

    def clean(
        self,
        min_severity: str = "critical",
        quarantine_dir: Optional[str] = None,
        backup: bool = True,
    ) -> Dict[str, Any]:
        if min_severity not in self.SEVERITY_ORDER:
            raise ValueError(f"Unsupported severity threshold: {min_severity}")

        report = self.audit()
        actionable = self._collect_actionable_records(min_severity)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_quarantine = self.root_dir / "data" / "qa" / "quarantine" / timestamp
        quarantine_root = Path(quarantine_dir) if quarantine_dir else default_quarantine
        quarantine_root.mkdir(parents=True, exist_ok=True)

        files_summary: Dict[str, Dict[str, Any]] = {}
        total_removed = 0

        for relative_path, locators in sorted(actionable.items()):
            path = self.root_dir / Path(relative_path)
            if not path.exists() or not locators:
                continue

            if path.name == "learned_phrasings.jsonl":
                summary = self._clean_learned_phrasings_file(path, locators, quarantine_root, backup, timestamp)
            elif path.name == "entities.json":
                summary = self._clean_entities_file(path, locators, quarantine_root, backup, timestamp)
            elif path.name == "relationships.json":
                summary = self._clean_relationships_file(path, locators, quarantine_root, backup, timestamp)
            else:
                summary = {
                    "path": relative_path,
                    "removed_records": 0,
                    "kept_records": None,
                    "quarantine_path": None,
                    "backup_path": None,
                    "skipped": True,
                    "reason": "No clean-up handler for this file type.",
                }

            total_removed += summary.get("removed_records", 0)
            files_summary[relative_path] = summary

        cleanup_summary = {
            "generated_at": datetime.now().isoformat(),
            "severity_threshold": min_severity,
            "quarantine_root": str(quarantine_root),
            "total_removed_records": total_removed,
            "files": files_summary,
        }

        return {
            "report": report,
            "cleanup": cleanup_summary,
        }

    def _iter_default_files(self) -> Iterable[Path]:
        return [
            self.root_dir / "data" / "learned_phrasings.jsonl",
            self.root_dir / "app" / "data" / "learned_phrasings.jsonl",
            self.root_dir / "data" / "entities.json",
            self.root_dir / "app" / "data" / "entities.json",
            self.root_dir / "data" / "relationships.json",
            self.root_dir / "app" / "data" / "relationships.json",
            self.root_dir / "data" / "patterns.json",
            self.root_dir / "app" / "data" / "patterns.json",
        ]

    def _add_finding(
        self,
        severity: str,
        code: str,
        path: Path,
        message: str,
        line: Optional[int] = None,
        sample: Optional[str] = None,
        area: Optional[str] = None,
        record_locator: Optional[str] = None,
    ) -> None:
        rel = path.relative_to(self.root_dir).as_posix()
        self.findings.append(
            Finding(
                severity=severity,
                code=code,
                area=area or self._area_for_path(path),
                path=rel,
                message=message,
                line=line,
                sample=sample[:240] if sample else None,
                record_locator=record_locator,
            )
        )

    def _area_for_path(self, path: Path) -> str:
        if path.name == "learned_phrasings.jsonl":
            return "learned_phrasings"
        if path.name == "entities.json":
            return "entities"
        if path.name == "relationships.json":
            return "relationships"
        if path.name == "patterns.json":
            return "patterns"
        return "other"

    def _build_area_summaries(self) -> Dict[str, Dict[str, Any]]:
        summaries: Dict[str, Dict[str, Any]] = {}
        for area in self.AREAS:
            area_findings = [finding for finding in self.findings if finding.area == area]
            summaries[area] = {
                "total_findings": len(area_findings),
                "severity_counts": dict(Counter(f.severity for f in area_findings)),
                "issue_counts": dict(Counter(f.code for f in area_findings)),
                "affected_paths": sorted({f.path for f in area_findings}),
            }
        return summaries

    def _build_group_summaries(self, area_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        group_summaries: Dict[str, Dict[str, Any]] = {}
        for group_name in sorted(set(self.AREA_GROUPS.values())):
            group_areas = [area for area, group in self.AREA_GROUPS.items() if group == group_name]
            severity_counts: Counter[str] = Counter()
            issue_counts: Counter[str] = Counter()
            total_findings = 0
            affected_paths = set()
            for area in group_areas:
                area_summary = area_summaries.get(area, {})
                total_findings += area_summary.get("total_findings", 0)
                severity_counts.update(area_summary.get("severity_counts", {}))
                issue_counts.update(area_summary.get("issue_counts", {}))
                affected_paths.update(area_summary.get("affected_paths", []))

            group_summaries[group_name] = {
                "areas": group_areas,
                "total_findings": total_findings,
                "severity_counts": dict(severity_counts),
                "issue_counts": dict(issue_counts),
                "affected_paths": sorted(affected_paths),
            }
        return group_summaries

    def _collect_actionable_records(self, min_severity: str) -> Dict[str, set[str]]:
        threshold = self.SEVERITY_ORDER[min_severity]
        actionable: Dict[str, set[str]] = defaultdict(set)
        for finding in self.findings:
            if finding.record_locator is None:
                continue
            if self.SEVERITY_ORDER.get(finding.severity, 0) < threshold:
                continue
            actionable[finding.path].add(finding.record_locator)
        return actionable

    def _backup_file(self, path: Path, timestamp: str) -> str:
        backup_path = path.with_name(f"{path.stem}.backup.{timestamp}{path.suffix}")
        shutil.copy2(path, backup_path)
        return str(backup_path)

    def _quarantine_output_path(self, path: Path, quarantine_root: Path, timestamp: str) -> Path:
        relative_parent = path.relative_to(self.root_dir).parent
        destination_dir = quarantine_root / relative_parent
        destination_dir.mkdir(parents=True, exist_ok=True)
        return destination_dir / f"{path.stem}.quarantine.{timestamp}{path.suffix}"

    def _clean_learned_phrasings_file(
        self,
        path: Path,
        locators: set[str],
        quarantine_root: Path,
        backup: bool,
        timestamp: str,
    ) -> Dict[str, Any]:
        kept_lines: List[str] = []
        removed_lines: List[str] = []

        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                locator = f"line:{line_no}"
                if locator in locators:
                    removed_lines.append(raw_line)
                else:
                    kept_lines.append(raw_line)

        summary = {
            "path": path.relative_to(self.root_dir).as_posix(),
            "removed_records": len(removed_lines),
            "kept_records": len(kept_lines),
            "quarantine_path": None,
            "backup_path": None,
            "skipped": False,
        }
        if not removed_lines:
            return summary

        if backup:
            summary["backup_path"] = self._backup_file(path, timestamp)

        path.write_text("".join(kept_lines), encoding="utf-8")
        quarantine_path = self._quarantine_output_path(path, quarantine_root, timestamp)
        quarantine_path.write_text("".join(removed_lines), encoding="utf-8")
        summary["quarantine_path"] = str(quarantine_path)
        return summary

    def _clean_entities_file(
        self,
        path: Path,
        locators: set[str],
        quarantine_root: Path,
        backup: bool,
        timestamp: str,
    ) -> Dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {
                "path": path.relative_to(self.root_dir).as_posix(),
                "removed_records": 0,
                "kept_records": None,
                "quarantine_path": None,
                "backup_path": None,
                "skipped": True,
                "reason": "File is not valid JSON and could not be rewritten safely.",
            }

        if not isinstance(payload, dict):
            return {
                "path": path.relative_to(self.root_dir).as_posix(),
                "removed_records": 0,
                "kept_records": None,
                "quarantine_path": None,
                "backup_path": None,
                "skipped": True,
                "reason": "entities.json is not a JSON object.",
            }

        flagged_keys = {locator.split(":", 1)[1] for locator in locators if locator.startswith("entity:")}
        removed = {key: value for key, value in payload.items() if key in flagged_keys}
        kept = {key: value for key, value in payload.items() if key not in flagged_keys}

        summary = {
            "path": path.relative_to(self.root_dir).as_posix(),
            "removed_records": len(removed),
            "kept_records": len(kept),
            "quarantine_path": None,
            "backup_path": None,
            "skipped": False,
        }
        if not removed:
            return summary

        if backup:
            summary["backup_path"] = self._backup_file(path, timestamp)

        path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
        quarantine_path = self._quarantine_output_path(path, quarantine_root, timestamp)
        quarantine_path.write_text(json.dumps(removed, indent=2), encoding="utf-8")
        summary["quarantine_path"] = str(quarantine_path)
        return summary

    def _clean_relationships_file(
        self,
        path: Path,
        locators: set[str],
        quarantine_root: Path,
        backup: bool,
        timestamp: str,
    ) -> Dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {
                "path": path.relative_to(self.root_dir).as_posix(),
                "removed_records": 0,
                "kept_records": None,
                "quarantine_path": None,
                "backup_path": None,
                "skipped": True,
                "reason": "File is not valid JSON and could not be rewritten safely.",
            }

        if not isinstance(payload, list):
            return {
                "path": path.relative_to(self.root_dir).as_posix(),
                "removed_records": 0,
                "kept_records": None,
                "quarantine_path": None,
                "backup_path": None,
                "skipped": True,
                "reason": "relationships.json is not a JSON array.",
            }

        flagged_indexes = {
            int(locator.split(":", 1)[1])
            for locator in locators
            if locator.startswith("relationship:") and locator.split(":", 1)[1].isdigit()
        }
        kept = [item for index, item in enumerate(payload) if index not in flagged_indexes]
        removed = [item for index, item in enumerate(payload) if index in flagged_indexes]

        summary = {
            "path": path.relative_to(self.root_dir).as_posix(),
            "removed_records": len(removed),
            "kept_records": len(kept),
            "quarantine_path": None,
            "backup_path": None,
            "skipped": False,
        }
        if not removed:
            return summary

        if backup:
            summary["backup_path"] = self._backup_file(path, timestamp)

        path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
        quarantine_path = self._quarantine_output_path(path, quarantine_root, timestamp)
        quarantine_path.write_text(json.dumps(removed, indent=2), encoding="utf-8")
        summary["quarantine_path"] = str(quarantine_path)
        return summary

    def _audit_jsonl(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    self._add_finding(
                        "critical",
                        "invalid_jsonl",
                        path,
                        f"Invalid JSONL entry: {exc}",
                        line=line_no,
                        sample=raw_line.strip(),
                        record_locator=f"line:{line_no}",
                    )
                    continue

                if path.name == "learned_phrasings.jsonl":
                    self._audit_learned_phrasing_entry(path, line_no, entry)

    def _audit_json(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._add_finding(
                "critical",
                "invalid_json",
                path,
                f"Invalid JSON file: {exc}",
            )
            return

        if path.name == "entities.json":
            self._audit_entities(path, payload)
        elif path.name == "relationships.json":
            self._audit_relationships(path, payload)
        elif path.name == "patterns.json":
            self._audit_patterns(path, payload)

    def _audit_learned_phrasing_entry(self, path: Path, line_no: int, entry: Dict[str, Any]) -> None:
        required = {
            "pattern": str,
            "alice_thought": dict,
            "ollama_phrasing": str,
            "context": dict,
            "timestamp": str,
            "tone": str,
        }
        for key, expected_type in required.items():
            value = entry.get(key)
            if not isinstance(value, expected_type) or (isinstance(value, str) and not value.strip()):
                self._add_finding(
                    "critical",
                    "missing_or_invalid_field",
                    path,
                    f"Field '{key}' is missing or invalid for learned phrasing entry.",
                    line=line_no,
                    sample=str(entry),
                    record_locator=f"line:{line_no}",
                )
                return

        thought = entry["alice_thought"]
        phrasing = entry["ollama_phrasing"].strip()
        context = entry["context"]
        thought_type = str(thought.get("type", "")).strip()
        user_input = self._extract_user_input(thought, context)

        expected_pattern = self._expected_pattern_for(thought)
        if expected_pattern and entry["pattern"] != expected_pattern:
            self._add_finding(
                "warning",
                "pattern_mismatch",
                path,
                f"Pattern '{entry['pattern']}' does not match expected '{expected_pattern}'.",
                line=line_no,
                sample=phrasing,
                record_locator=f"line:{line_no}",
            )

        if any(marker in phrasing.lower() for marker in self.LLM_META_MARKERS):
            self._add_finding(
                "critical",
                "llm_meta_artifact",
                path,
                "Learned phrasing contains LLM meta text instead of a user-facing answer.",
                line=line_no,
                sample=phrasing,
                record_locator=f"line:{line_no}",
            )

        if "none°c" in phrasing.lower() or "null°c" in phrasing.lower():
            self._add_finding(
                "critical",
                "invalid_temperature_render",
                path,
                "Learned phrasing contains an invalid rendered temperature.",
                line=line_no,
                sample=phrasing,
                record_locator=f"line:{line_no}",
            )

        if len(phrasing) >= 2 and phrasing[0] == phrasing[-1] == '"':
            self._add_finding(
                "info",
                "wrapped_response_quotes",
                path,
                "Learned phrasing is wrapped in quote characters.",
                line=line_no,
                sample=phrasing,
                record_locator=f"line:{line_no}",
            )

        if thought_type == "weather_advice" and self._contains_unwanted_personalization(phrasing):
            self._add_finding(
                "critical",
                "weather_personalization_leak",
                path,
                "Weather advice learned an unnecessary user-name or placeholder address.",
                line=line_no,
                sample=phrasing,
                record_locator=f"line:{line_no}",
            )

        if thought_type in {"operation_success", "operation_failure"} and self._contains_placeholder_name(phrasing):
            self._add_finding(
                "warning",
                "placeholder_user_name",
                path,
                "Operation phrasing contains a placeholder user name.",
                line=line_no,
                sample=phrasing,
                record_locator=f"line:{line_no}",
            )

        if thought_type == "operation_success":
            details = thought.get("details", {}) if isinstance(thought.get("details"), dict) else {}
            if details.get("found") is False and details.get("count") == 0 and re.search(r"\bfound\b", phrasing, re.IGNORECASE):
                self._add_finding(
                    "critical",
                    "contradictory_success_content",
                    path,
                    "Operation success phrasing says items were found even though the data says none were found.",
                    line=line_no,
                    sample=phrasing,
                    record_locator=f"line:{line_no}",
                )
            if any(marker in phrasing.lower() for marker in self.FAILURE_MARKERS):
                self._add_finding(
                    "warning",
                    "success_failure_tone_mismatch",
                    path,
                    "Operation success phrasing sounds like a failure or error.",
                    line=line_no,
                    sample=phrasing,
                    record_locator=f"line:{line_no}",
                )

        if thought_type == "operation_failure" and any(marker in phrasing.lower() for marker in self.SUCCESS_MARKERS):
            self._add_finding(
                "warning",
                "failure_success_tone_mismatch",
                path,
                "Operation failure phrasing sounds like a success.",
                line=line_no,
                sample=phrasing,
                record_locator=f"line:{line_no}",
            )

        if ":" in thought_type:
            domain = thought_type.split(":", 1)[0].strip().lower()
            if domain in self.DOMAIN_KEYWORDS and user_input:
                lowered_input = user_input.lower()
                if not any(keyword in lowered_input for keyword in self.DOMAIN_KEYWORDS[domain]):
                    self._add_finding(
                        "critical",
                        "wrong_domain_learning",
                        path,
                        f"Learned phrasing is stored under '{thought_type}' but the source user input does not look like that domain.",
                        line=line_no,
                        sample=user_input,
                        record_locator=f"line:{line_no}",
                    )

    def _audit_entities(self, path: Path, payload: Any) -> None:
        if not isinstance(payload, dict):
            self._add_finding("critical", "invalid_entities_shape", path, "entities.json must contain a JSON object.")
            return

        for entity_key, entity in payload.items():
            if not isinstance(entity, dict):
                self._add_finding("critical", "invalid_entity_record", path, f"Entity '{entity_key}' is not a JSON object.")
                continue

            name = str(entity.get("name", entity_key))
            confidence = entity.get("confidence")
            mention_count = entity.get("mention_count")

            if name != entity_key:
                self._add_finding("warning", "entity_key_name_mismatch", path, f"Entity key '{entity_key}' does not match stored name '{name}'.", sample=name, record_locator=f"entity:{entity_key}")

            if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
                self._add_finding("critical", "invalid_entity_confidence", path, f"Entity '{name}' has invalid confidence: {confidence}.", sample=str(entity), record_locator=f"entity:{entity_key}")

            if not isinstance(mention_count, int) or mention_count < 0:
                self._add_finding("critical", "invalid_entity_mention_count", path, f"Entity '{name}' has invalid mention_count: {mention_count}.", sample=str(entity), record_locator=f"entity:{entity_key}")

            normalized = name.strip().lower()
            if normalized in self.STOPWORDS:
                self._add_finding("warning", "noise_entity", path, f"Entity '{name}' looks like a stopword/noise capture.", sample=name, record_locator=f"entity:{entity_key}")

            if "\n" in name or re.search(r"\n\d+$", name):
                self._add_finding("critical", "entity_list_artifact", path, f"Entity '{name}' looks like a note-list extraction artifact.", sample=name, record_locator=f"entity:{entity_key}")

            if re.fullmatch(r"something(?:\s+something){2,}", normalized):
                self._add_finding("critical", "entity_repetition_artifact", path, f"Entity '{name}' looks like repeated placeholder text.", sample=name, record_locator=f"entity:{entity_key}")

    def _audit_relationships(self, path: Path, payload: Any) -> None:
        if not isinstance(payload, list):
            self._add_finding("critical", "invalid_relationships_shape", path, "relationships.json must contain a JSON array.")
            return

        for index, rel in enumerate(payload):
            if not isinstance(rel, dict):
                self._add_finding("critical", "invalid_relationship_record", path, "Relationship record is not a JSON object.", sample=str(rel), record_locator=f"relationship:{index}")
                continue

            source_entity = str(rel.get("source_entity", ""))
            target_entity = str(rel.get("target_entity", ""))
            confidence = rel.get("confidence")
            context = str(rel.get("context", ""))

            if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
                self._add_finding("critical", "invalid_relationship_confidence", path, "Relationship has invalid confidence.", sample=str(rel), record_locator=f"relationship:{index}")

            if any(name.strip().lower() in self.STOPWORDS for name in (source_entity, target_entity)):
                self._add_finding("warning", "relationship_noise_entity", path, "Relationship references a noise-word entity.", sample=f"{source_entity} -> {target_entity}", record_locator=f"relationship:{index}")

            if "\n" in source_entity or "\n" in target_entity:
                self._add_finding("critical", "relationship_list_artifact", path, "Relationship entity contains numbered-list or multiline extraction artifacts.", sample=f"{source_entity} -> {target_entity}", record_locator=f"relationship:{index}")

            if "**you have" in context.lower() or "…and" in context:
                self._add_finding("critical", "relationship_from_rendered_list", path, "Relationship appears to be extracted from rendered note-list output rather than real knowledge.", sample=context, record_locator=f"relationship:{index}")

    def _audit_patterns(self, path: Path, payload: Any) -> None:
        if not isinstance(payload, dict):
            self._add_finding("warning", "invalid_patterns_shape", path, "patterns.json is not a JSON object.")
            return
        if not payload:
            self._add_finding("warning", "empty_patterns_file", path, "patterns.json is empty.")

    def _extract_user_input(self, thought: Dict[str, Any], context: Dict[str, Any]) -> str:
        if isinstance(context.get("user_input"), str) and context["user_input"].strip():
            return context["user_input"].strip()
        data = thought.get("data") if isinstance(thought.get("data"), dict) else {}
        for key in ("user_input", "user_question"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        value = thought.get("user_question")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return ""

    def _expected_pattern_for(self, thought: Dict[str, Any]) -> str:
        thought_type = thought.get("type", "general")
        if thought_type == "capability_answer":
            return f"capability:{thought.get('can_do', False)}"
        if thought_type == "knowledge_answer":
            return "knowledge:general"
        if thought_type == "reasoning_result":
            return "reasoning:conclusion"
        if thought_type == "factual_answer":
            return "factual:answer"
        return f"general:{thought_type}"

    def _contains_placeholder_name(self, phrasing: str) -> bool:
        return bool(re.search(r"\b(?:testuser|user)\b", phrasing, re.IGNORECASE))

    def _contains_unwanted_personalization(self, phrasing: str) -> bool:
        return bool(
            re.search(r"^(?:for|hey|hi|hello)\s+[a-z][\w-]*\b", phrasing, re.IGNORECASE)
            or re.search(r",\s*[a-z][\w-]*,", phrasing, re.IGNORECASE)
            or self._contains_placeholder_name(phrasing)
        )


def _print_learning_qa_summary(report: Dict[str, Any]) -> None:
    print("LEARNING DATA QA REPORT")
    print("=" * 60)
    print(f"Generated: {report['generated_at']}")
    print(f"Total findings: {report['total_findings']}")

    severity_counts = report.get("severity_counts", {})
    if severity_counts:
        print("Severity counts:")
        for severity in ("critical", "warning", "info"):
            if severity in severity_counts:
                print(f"  - {severity}: {severity_counts[severity]}")

    issue_counts = report.get("issue_counts", {})
    if issue_counts:
        print("Top issues:")
        for code, count in sorted(issue_counts.items(), key=lambda item: item[1], reverse=True)[:12]:
            print(f"  - {code}: {count}")

    group_summaries = report.get("group_summaries", {})
    if group_summaries:
        print("Summary by cleanup area:")
        for group_name in ("learned_phrasings", "knowledge"):
            group_summary = group_summaries.get(group_name)
            if not group_summary:
                continue
            label = "Learned phrasings" if group_name == "learned_phrasings" else "Entities / relationships"
            print(f"  {label}: {group_summary['total_findings']} findings")
            for severity in ("critical", "warning", "info"):
                count = group_summary.get("severity_counts", {}).get(severity)
                if count:
                    print(f"    - {severity}: {count}")
            top_codes = sorted(group_summary.get("issue_counts", {}).items(), key=lambda item: item[1], reverse=True)[:6]
            for code, count in top_codes:
                print(f"    - {code}: {count}")

    area_summaries = report.get("area_summaries", {})
    if area_summaries:
        print("Area detail:")
        for area in LearningDataQAAuditor.AREAS:
            area_summary = area_summaries.get(area, {})
            if not area_summary.get("total_findings"):
                continue
            print(f"  - {area}: {area_summary['total_findings']}")

    print("Sample findings:")
    for finding in report.get("findings", [])[:15]:
        location = finding["path"]
        if finding.get("line"):
            location += f":{finding['line']}"
        print(f"  - [{finding['severity']}] {finding['code']} @ {location}")
        print(f"    {finding['message']}")
        if finding.get("sample"):
            print(f"    sample: {finding['sample']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Alice's training and learned data.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("clean", help="Run the original training/entity cleanup audit.")

    qa_parser = subparsers.add_parser("qa", help="Run learned phrasing and persisted learning QA checks.")
    qa_parser.add_argument("--root", default=".", help="Project root to audit.")
    qa_parser.add_argument("--report", help="Optional path to write the JSON report.")
    qa_parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any critical findings are present.")

    qa_clean_parser = subparsers.add_parser("qa-clean", help="Quarantine bad learned entries and rewrite cleaned learning data.")
    qa_clean_parser.add_argument("--root", default=".", help="Project root to audit.")
    qa_clean_parser.add_argument("--report", help="Optional path to write the JSON cleanup report.")
    qa_clean_parser.add_argument("--quarantine-dir", help="Optional directory to store quarantined records.")
    qa_clean_parser.add_argument(
        "--min-severity",
        choices=("critical", "warning", "info"),
        default="critical",
        help="Minimum finding severity to quarantine automatically.",
    )

    args = parser.parse_args(argv)

    if args.command in (None, "clean"):
        logging.basicConfig(level=logging.INFO)
        auditor = TrainingDataAuditor()
        auditor.full_audit_and_clean()
        return 0

    if args.command == "qa":
        auditor = LearningDataQAAuditor(root_dir=args.root)
        report = auditor.audit()
        _print_learning_qa_summary(report)
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nReport written to {report_path}")
        if args.strict and report.get("severity_counts", {}).get("critical", 0) > 0:
            return 1
        return 0

    if args.command == "qa-clean":
        auditor = LearningDataQAAuditor(root_dir=args.root)
        result = auditor.clean(min_severity=args.min_severity, quarantine_dir=args.quarantine_dir)
        _print_learning_qa_summary(result["report"])
        cleanup = result["cleanup"]
        print("\nQA CLEANUP")
        print("=" * 60)
        print(f"Severity threshold: {cleanup['severity_threshold']}")
        print(f"Quarantine root: {cleanup['quarantine_root']}")
        print(f"Total removed records: {cleanup['total_removed_records']}")
        for path_key, file_summary in cleanup.get("files", {}).items():
            print(f"  - {path_key}: removed={file_summary['removed_records']} kept={file_summary['kept_records']}")
            if file_summary.get("backup_path"):
                print(f"    backup: {file_summary['backup_path']}")
            if file_summary.get("quarantine_path"):
                print(f"    quarantine: {file_summary['quarantine_path']}")
            if file_summary.get("skipped") and file_summary.get("reason"):
                print(f"    skipped: {file_summary['reason']}")
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"\nCleanup report written to {report_path}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
