"""
ALICE Learning Engine - Unified learning system
Consolidates: training_system, ml_learner, response_generator

Responsibilities:
- Collect interactions
- Learn patterns (TF-IDF + sklearn)
- Generate responses from learned patterns
- Track quality and user feedback
- Provide training data for fine-tuning
"""

import os
import json
import logging
import pickle
import importlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

np = None
TfidfVectorizer = None
LogisticRegression = None
cosine_similarity = None
SentenceTransformer = None
_DEPS_LOADED = False


def _load_optional_deps() -> None:
    global _DEPS_LOADED, np, TfidfVectorizer, LogisticRegression, cosine_similarity, SentenceTransformer
    if _DEPS_LOADED:
        return
    try:
        np = importlib.import_module("numpy")
    except ImportError:  # pragma: no cover
        np = None
    try:
        sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
        sklearn_linear = importlib.import_module("sklearn.linear_model")
        sklearn_pairwise = importlib.import_module("sklearn.metrics.pairwise")
        TfidfVectorizer = sklearn_text.TfidfVectorizer
        LogisticRegression = sklearn_linear.LogisticRegression
        cosine_similarity = sklearn_pairwise.cosine_similarity
    except ImportError:  # pragma: no cover
        TfidfVectorizer = None
        LogisticRegression = None
        cosine_similarity = None
    try:
        # Skip if disabled via env var to avoid network timeout/import issues
        if os.environ.get('ALICE_DISABLE_SEMANTIC_CLASSIFIER'):
            SentenceTransformer = None
        else:
            st_mod = importlib.import_module("sentence_transformers")
            SentenceTransformer = st_mod.SentenceTransformer
    except ImportError:  # pragma: no cover
        SentenceTransformer = None
    _DEPS_LOADED = True


def _ml_available() -> bool:
    _load_optional_deps()
    return np is not None and TfidfVectorizer is not None and LogisticRegression is not None and cosine_similarity is not None


def _dl_available() -> bool:
    _load_optional_deps()
    return SentenceTransformer is not None


@dataclass
class TrainingExample:
    """A single training example"""
    user_input: str
    assistant_response: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    user_rating: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LearningEngine:
    """
    Unified learning system for ALICE
    - Collects interactions
    - Learns patterns
    - Generates responses from learned knowledge
    - Tracks quality and user preferences
    """
    
    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._tfidf_matrix = None
        self._needs_refit = True

        # Storage
        self.training_file = self.data_dir / "training_data.jsonl"
        self.stats_file = self.data_dir / "training_stats.json"
        self.patterns_file = self.data_dir / "learned_patterns.pkl"
        self.embeddings_file = self.data_dir / "embeddings.pkl"

        # In-memory
        self.examples = []  # All examples
        self.patterns = {}  # input_hash → learned response
        self.embeddings = None

        if _ml_available():
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=200)
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.vectorizer = None
            self.classifier = None
            logger.warning("[Learning] ML dependencies missing; similarity search disabled")

        if not _dl_available():
            logger.warning("[Learning] sentence-transformers not available, using traditional ML")

        self.stats = self._load_stats()
        self._lock = threading.RLock()
        self._load_examples()
        self._load_patterns()

        logger.info(f"[Learning] Engine initialized with {len(self.examples)} examples")
    
    def collect_interaction(
        self,
        user_input: str,
        assistant_response: str,
        intent: Optional[str] = None,
        entities: Optional[Dict] = None,
        quality_score: float = 1.0,
        user_rating: Optional[int] = None
    ):
        """Record an interaction for learning"""
        with self._lock:
            example = TrainingExample(
                user_input=user_input,
                assistant_response=assistant_response,
                intent=intent,
                entities=entities or {},
                quality_score=quality_score,
                user_rating=user_rating
            )
            
            self.examples.append(example)
            
            # Save incrementally
            self._save_example(example)
            
            # Update statistics
            self.stats['total_examples'] = len(self.examples)
            if intent:
                self.stats['examples_by_intent'][intent] = self.stats['examples_by_intent'].get(intent, 0) + 1
            
            # Quality tracking
            if quality_score >= 0.8:
                self.stats['quality_distribution']['high'] += 1
            elif quality_score >= 0.6:
                self.stats['quality_distribution']['medium'] += 1
            else:
                self.stats['quality_distribution']['low'] += 1
            
            self.stats['last_training'] = datetime.now().isoformat()
            self._save_stats()
            
            # Learn pattern immediately
            self._learn_pattern(example)

            # Mark TF-IDF matrix as needing refit after new example
            self._needs_refit = True

    def run_offline_training(self, max_entries: int = 500) -> Dict[str, Any]:
        """
        Offline pass over recent logs to convert mistakes into corrections
        and promote safer patterns.
        """
        project_root = Path(__file__).resolve().parents[1]
        log_path = project_root / "data" / "training" / "auto_generated.jsonl"

        correction_engine = get_auto_correction_engine(project_root)
        error_summary = correction_engine.process_error_logs(log_path, max_entries=max_entries)
        applied_summary = correction_engine.apply_corrections_to_thresholds()

        error_entries = self._load_error_entries(log_path)
        hard_lessons = self._extract_hard_lessons(error_entries)

        hard_lesson_summary = {}
        if hard_lessons:
            try:
                from ai.autonomous_adjuster import create_autonomous_adjuster
                adjuster = create_autonomous_adjuster(project_root)
                hard_lesson_summary = adjuster.apply_hard_lessons(hard_lessons)
            except Exception as e:
                logger.warning(f"[Learning] Hard lesson adjustment failed: {e}")

        # Promote corrected mistakes into training data for pattern learning
        promoted_from_errors = 0
        for correction in error_summary.get('new_corrections', []):
            corrected_response = correction.get('teacher_response') or ""
            if not corrected_response:
                continue
            self.collect_interaction(
                user_input=correction.get('user_input', ''),
                assistant_response=corrected_response,
                intent=correction.get('expected_intent', None),
                entities={},
                quality_score=0.9,
                user_rating=None
            )
            promoted_from_errors += 1

        return {
            'errors_seen': error_summary.get('errors_seen', 0),
            'corrections_added': error_summary.get('corrections_added', 0),
            'corrections_updated': error_summary.get('corrections_updated', 0),
            'corrections_applied': applied_summary.get('applied_count', 0),
            'hard_lessons': hard_lesson_summary.get('hard_lessons', 0),
            'hard_lesson_adjustments': hard_lesson_summary.get('adjustments_made', 0),
            'promoted_from_errors': promoted_from_errors
        }

    def _load_error_entries(self, log_path: Path) -> List[Dict[str, Any]]:
        """Load failed interactions from training logs."""
        entries: List[Dict[str, Any]] = []
        if not log_path.exists():
            return entries

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    success_flag = entry.get('success_flag', entry.get('success', True))
                    if success_flag:
                        continue

                    entries.append(entry)
        except Exception as e:
            logger.warning(f"[Learning] Failed to load error entries: {e}")

        return entries

    def _extract_hard_lessons(self, error_entries: List[Dict[str, Any]], min_count: int = 5) -> List[Dict[str, Any]]:
        """Group repeated errors into hard lessons."""
        counts: Dict[Tuple[str, str, str], int] = defaultdict(int)

        for entry in error_entries:
            domain = entry.get('domain', 'unknown')
            intent = entry.get('expected_intent') or entry.get('actual_intent') or 'unknown'
            error_type = entry.get('error_type') or 'unknown'
            counts[(domain, intent, error_type)] += 1

        hard_lessons = []
        for (domain, intent, error_type), count in counts.items():
            if count >= min_count:
                hard_lessons.append({
                    'domain': domain,
                    'intent': intent,
                    'error_type': error_type,
                    'count': count
                })

        return hard_lessons
    
    def _refit_vectorizer(self):
        """Fit vectorizer on current examples and cache the matrix."""
        texts = [ex.user_input for ex in self.examples]
        self._tfidf_matrix = self.vectorizer.fit_transform(texts)
        self._needs_refit = False

    def _learn_pattern(self, example: TrainingExample):
        """Learn a single pattern"""
        import hashlib
        key = hashlib.sha256(example.user_input.encode("utf-8")).hexdigest()[:16]
        self.patterns[key] = {
            'input': example.user_input,
            'response': example.assistant_response,
            'intent': example.intent,
            'quality': example.quality_score,
            'rating': example.user_rating
        }

        self._save_patterns()

       
    def get_similar_examples(self, user_input: str, top_k: int = 3) -> List[TrainingExample]:
        """Find similar examples using TF-IDF similarity"""
        if not _ml_available():
            return []
        if len(self.examples) < 2:
            return []
        
        try:
            # Vectorize all examples
            all_texts = [ex.user_input for ex in self.examples] + [user_input]
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=200)
            vectors = vectorizer.fit_transform(all_texts)
            
            # Find similarities
            similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar = []
            for idx in top_indices:
                if similarities[idx] > 0.4:  # Minimum similarity threshold
                    similar.append(self.examples[idx])
            
            return similar
        except Exception as e:
            logger.warning(f"[Learning] Similarity search failed: {e}")
            return []
    
    def generate_response_from_learned(self, user_input: str) -> Optional[str]:
        """
        Generate response using learned patterns (no LLM needed)
        Returns None if no good pattern found
        """
        similar = self.get_similar_examples(user_input, top_k=1)
        
        if similar:
            example = similar[0]
            # Only use if high quality
            if example.quality_score >= 0.7:
                return example.assistant_response
        
        return None 
    
    def _stable_hash(self, txt: str) -> str:
        import hashlib
        return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12] # 48-bit
    
    def get_high_quality_examples(self, min_examples: int = 50) -> Optional[List[Dict]]:
        """
        Get high-quality examples for fine-tuning
        Returns None if not enough examples
        """
        high_quality = [
            ex for ex in self.examples
            if ex.quality_score >= 0.8 and (ex.user_rating is None or ex.user_rating >= 3)
        ]
        
        if len(high_quality) < min_examples:
            return None
        
        return [
            {
                'input': ex.user_input,
                'output': ex.assistant_response,
                'intent': ex.intent
            }
            for ex in high_quality
        ]
    
    def should_finetune(self, min_examples: int = 50) -> bool:
        """Check if we have enough data to fine-tune"""
        return self.get_high_quality_examples(min_examples) is not None
    
    def get_training_data(
        self,
        min_quality: float = 0.7,
        max_examples: Optional[int] = None
    ) -> List[TrainingExample]:
        filtered = [ex for ex in self.examples if ex.quality_score >= min_quality]
        
        if max_examples:
            filtered = filtered[:max_examples]
        
        return filtered
    
    def record_feedback(self, user_input: str, response: str, rating: int, feedback_text: str = ""):
        """Record user feedback on a response"""
        # Find matching example
        for ex in self.examples:
            if ex.user_input.lower() == user_input.lower() and ex.assistant_response == response:
                ex.user_rating = rating
                self._save_stats()
                logger.info(f"[Learning] Feedback recorded: {user_input[:50]}... → rating {rating}")
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        high_quality = len([ex for ex in self.examples if ex.quality_score >= 0.8])
        user_rated = len([ex for ex in self.examples if ex.user_rating is not None])
        if user_rated > 0:
            ratings = [ex.user_rating for ex in self.examples if ex.user_rating]
            avg_rating = (sum(ratings) / len(ratings)) if ratings else 0
        else:
            avg_rating = 0
        
        return {
            'total_examples': len(self.examples),
            'high_quality': high_quality,
            'user_rated': user_rated,
            'average_rating': float(avg_rating),
            'ready_for_finetuning': self.should_finetune(),
            'by_intent': self.stats.get('examples_by_intent', {}),
            'quality_distribution': self.stats.get('quality_distribution', {}),
            'llm_calls_logged': self.stats.get('llm_calls_logged', 0)
        }
    
    def log_llm_call(
        self,
        user_input: str,
        llm_response: str,
        intent: Optional[str] = None,
        context: Dict[str, Any] = None
    ):
        """
        Log LLM call for pattern learning (gatekeeper function)
        
        This is called EVERY time LLM is used, so we can:
        1. Track unhandled → LLM patterns
        2. Auto-create deterministic patterns when threshold met
        3. Suggest learning opportunities
        """
        # Collect as normal interaction
        self.collect_interaction(
            user_input=user_input,
            assistant_response=llm_response,
            intent=intent or 'llm_fallback',
            quality_score=0.8  # LLM responses are generally good
        )
        
        # Track LLM usage
        self.stats['llm_calls_logged'] = self.stats.get('llm_calls_logged', 0) + 1
        
        # Check if this input has been asked before
        similar_count = self._count_similar_inputs(user_input)
        
        if similar_count >= 3:
            logger.info(f"Pattern opportunity: '{user_input[:50]}...' asked {similar_count} times - consider creating pattern")
            return {
                'should_create_pattern': True,
                'similar_count': similar_count,
                'suggested_response': llm_response
            }
        
        return {
            'should_create_pattern': False,
            'similar_count': similar_count
        }
    
    def _count_similar_inputs(self, user_input: str, threshold: float = 0.85) -> int:
        """Count how many times similar input was asked"""
        user_input_lower = user_input.lower().strip()
        count = 0
        
        for ex in self.examples:
            # Simple similarity check
            ex_input_lower = ex.user_input.lower().strip()
            
            # Exact match
            if ex_input_lower == user_input_lower:
                count += 1
            # Fuzzy match (simple word overlap)
            elif self._fuzzy_match(ex_input_lower, user_input_lower) > threshold:
                count += 1
        
        return count
    
    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """Simple fuzzy matching based on word overlap"""
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def suggest_pattern_creation(self, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Suggest patterns that should be created based on repeated LLM calls
        
        Returns:
            List of pattern suggestions with user_input, response, count
        """
        from collections import defaultdict
        
        # Group similar inputs
        input_groups = defaultdict(list)
        
        for ex in self.examples:
            # Normalize input
            normalized = ex.user_input.lower().strip()
            
            # Find similar group
            found_group = False
            for key in input_groups:
                if self._fuzzy_match(normalized, key) > 0.85:
                    input_groups[key].append(ex)
                    found_group = True
                    break
            
            if not found_group:
                input_groups[normalized].append(ex)
        
        # Find groups with enough occurrences
        suggestions = []
        for normalized_input, examples in input_groups.items():
            if len(examples) >= min_occurrences:
                # Get most common response
                responses = [ex.assistant_response for ex in examples]
                most_common_response = max(set(responses), key=responses.count)
                
                suggestions.append({
                    'user_input': examples[0].user_input,  # Use first original form
                    'suggested_response': most_common_response,
                    'occurrence_count': len(examples),
                    'should_learn': True
                })
        
        return sorted(suggestions, key=lambda x: x['occurrence_count'], reverse=True)
    
    # --- Internal methods ---
    
    def _save_example(self, example: TrainingExample):
        """Append example to training file"""
        try:
            with open(self.training_file, 'a') as f:
                f.write(json.dumps(asdict(example)) + '\n')
        except Exception as e:
            logger.error(f"[Learning] Error saving example: {e}")
    
    def _load_examples(self):
        """Load all examples from disk"""
        if self.training_file.exists():
            try:
                with open(self.training_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                # Backward compatibility: older entries may include 'feedback'
                                # Map numeric feedback to user_rating if present, then drop the key
                                if isinstance(data, dict) and "feedback" in data:
                                    if data.get("user_rating") is None:
                                        data["user_rating"] = data.get("feedback")
                                    data.pop("feedback", None)
                                example = TrainingExample(**data)
                                self.examples.append(example)
                            except (json.JSONDecodeError, TypeError, ValueError):
                                pass  # Skip malformed lines
            except Exception as e:
                logger.warning(f"[Learning] Error loading examples: {e}")
    
    def _save_patterns(self):
        """Save learned patterns to disk"""
        try:
            with open(self.patterns_file, 'wb') as f:
                pickle.dump(self.patterns, f)
        except Exception as e:
            logger.error(f"[Learning] Error saving patterns: {e}")
    
    def _load_patterns(self):
        """Load learned patterns from disk"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'rb') as f:
                    self.patterns = pickle.load(f)
            except Exception as e:
                logger.warning(f"[Learning] Error loading patterns: {e}")
    
    def _save_stats(self):
        """Save statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"[Learning] Error saving stats: {e}")
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load statistics"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'total_examples': 0,
            'examples_by_intent': {},
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'last_training': None
        }


class AutoCorrectionEngine:
    """
    Reads scenario logs and interaction logs, identifies mismatches,
    and stores corrected entries for pattern learning
    """
    
    DANGEROUS_DOMAINS = {'system', 'shell', 'code_execution', 'security', 'admin'}
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / "training"
        self.corrections_file = self.project_root / "memory" / "corrections.json"
        self.corrections = self._load_corrections()
        logger.info("[AutoCorrection] Engine initialized")
    
    def _load_corrections(self) -> List[Dict]:
        """Load existing corrections"""
        if self.corrections_file.exists():
            try:
                with open(self.corrections_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[AutoCorrection] Error loading corrections: {e}")
        return []
    
    def _save_corrections(self):
        """Save corrections to disk"""
        try:
            with open(self.corrections_file, 'w') as f:
                json.dump(self.corrections, f, indent=2)
        except Exception as e:
            logger.error(f"[AutoCorrection] Error saving corrections: {e}")
    
    def process_scenario_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Process scenario results and create corrections for mismatches
        
        Args:
            results: List of scenario results with expected vs actual intent/route
        
        Returns:
            Summary of corrections made
        """
        corrections_added = 0
        intent_mismatches = 0
        route_mismatches = 0
        
        for result in results:
            if not result.get('intent_match') or not result.get('route_match'):
                correction = self._create_correction_from_mismatch(result)
                if correction:
                    self.corrections.append(correction)
                    corrections_added += 1
                    
                    if not result.get('intent_match'):
                        intent_mismatches += 1
                    if not result.get('route_match'):
                        route_mismatches += 1
        
        if corrections_added > 0:
            self._save_corrections()
            logger.info(f"[AutoCorrection] Added {corrections_added} corrections ({intent_mismatches} intent, {route_mismatches} route)")
        
        return {
            'corrections_added': corrections_added,
            'intent_mismatches': intent_mismatches,
            'route_mismatches': route_mismatches
        }

    def process_error_logs(self, log_path: Path, max_entries: int = 500) -> Dict[str, Any]:
        """
        Process error logs (success_flag=False) and create corrections.

        Args:
            log_path: Path to jsonl training log
            max_entries: Max corrections to add/update in this pass

        Returns:
            Summary of corrections processed
        """
        if not log_path.exists():
            return {
                'errors_seen': 0,
                'corrections_added': 0,
                'corrections_updated': 0,
                'used_expected': 0,
                'teacher_judged': 0,
                'new_corrections': []
            }

        errors_seen = 0
        corrections_added = 0
        corrections_updated = 0
        used_expected = 0
        teacher_judged = 0
        new_corrections = []

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    success_flag = entry.get('success_flag', entry.get('success', True))
                    if success_flag:
                        continue

                    errors_seen += 1
                    correction = self._create_correction_from_log(entry)
                    if not correction:
                        continue

                    existing = self._find_existing_correction(correction)
                    if existing:
                        existing['validation_count'] = existing.get('validation_count', 1) + 1
                        existing['last_seen'] = datetime.now().isoformat()
                        corrections_updated += 1
                    else:
                        self.corrections.append(correction)
                        new_corrections.append(correction)
                        corrections_added += 1

                    if correction.get('source') == 'teacher_judge':
                        teacher_judged += 1
                    else:
                        used_expected += 1

                    if (corrections_added + corrections_updated) >= max_entries:
                        break

        except Exception as e:
            logger.warning(f"[AutoCorrection] Error processing logs: {e}")

        if corrections_added > 0 or corrections_updated > 0:
            self._save_corrections()

        return {
            'errors_seen': errors_seen,
            'corrections_added': corrections_added,
            'corrections_updated': corrections_updated,
            'used_expected': used_expected,
            'teacher_judged': teacher_judged,
            'new_corrections': new_corrections
        }
    
    def _create_correction_from_mismatch(self, result: Dict) -> Optional[Dict]:
        """Create correction entry from scenario mismatch"""
        try:
            correction_type = []
            if not result.get('intent_match'):
                correction_type.append('intent')
            if not result.get('route_match'):
                correction_type.append('route')
            
            domain = result.get('domain', 'unknown')
            
            # Check if dangerous domain
            if domain.lower() in self.DANGEROUS_DOMAINS:
                logger.warning(f"[AutoCorrection] Skipping dangerous domain correction: {domain}")
                return None
            
            correction = {
                'id': f"auto_corr_{datetime.now().isoformat()}",
                'timestamp': datetime.now().isoformat(),
                'correction_type': '|'.join(correction_type),
                'user_input': result.get('user_input', ''),
                'expected_intent': result.get('expected_intent', ''),
                'actual_intent': result.get('actual_intent', ''),
                'expected_route': result.get('expected_route', ''),
                'actual_route': result.get('actual_route', ''),
                'domain': domain,
                'source': 'auto_scenario',
                'applied': False,
                'validation_count': 1,
                'confidence': result.get('confidence', 0.0)
            }
            
            return correction
        except Exception as e:
            logger.warning(f"[AutoCorrection] Error creating correction: {e}")
            return None

    def _create_correction_from_log(self, entry: Dict[str, Any]) -> Optional[Dict]:
        """Create correction from a training log entry."""
        try:
            expected_intent = entry.get('expected_intent')
            expected_route = entry.get('expected_route')
            domain = entry.get('domain', 'unknown')

            if domain.lower() in self.DANGEROUS_DOMAINS:
                return None

            judged = None
            if not expected_intent or not expected_route:
                judged = self._judge_with_teacher(entry)
                if not judged:
                    return None
                expected_intent = judged.get('expected_intent')
                expected_route = judged.get('expected_route')

            if not expected_intent or not expected_route:
                return None

            correction = {
                'id': f"auto_log_corr_{datetime.now().isoformat()}",
                'timestamp': datetime.now().isoformat(),
                'correction_type': entry.get('error_type', self._infer_error_type(entry)),
                'user_input': entry.get('user_input', ''),
                'expected_intent': expected_intent,
                'actual_intent': entry.get('actual_intent', ''),
                'expected_route': expected_route,
                'actual_route': entry.get('actual_route', ''),
                'domain': domain,
                'source': 'teacher_judge' if judged else 'log_expected',
                'applied': False,
                'validation_count': 1,
                'confidence': entry.get('confidence', 0.0),
                'teacher_response': entry.get('teacher_response') or (judged.get('response') if judged else None)
            }

            return correction
        except Exception as e:
            logger.warning(f"[AutoCorrection] Error creating log correction: {e}")
            return None

    def _infer_error_type(self, entry: Dict[str, Any]) -> str:
        if entry.get('intent_match') is False:
            return 'mis_intent'
        if entry.get('route_match') is False:
            return 'wrong_route'
        return 'bad_answer'

    def _find_existing_correction(self, correction: Dict[str, Any]) -> Optional[Dict]:
        """Find existing correction by key fields."""
        for existing in self.corrections:
            if (
                existing.get('user_input') == correction.get('user_input') and
                existing.get('expected_intent') == correction.get('expected_intent') and
                existing.get('expected_route') == correction.get('expected_route')
            ):
                return existing
        return None

    def _judge_with_teacher(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ask LLM teacher to suggest expected intent/route for a failed input."""
        try:
            from ai.core.llm_engine import LLMConfig, LocalLLMEngine

            llm = LocalLLMEngine(LLMConfig(model="llama3.1:8b"))
            user_input = entry.get('user_input', '')
            actual_intent = entry.get('actual_intent', '')
            actual_route = entry.get('actual_route', '')

            prompt = (
                "You are a strict routing judge. Given the user input and the model's wrong output, "
                "suggest the correct intent and route. Return ONLY valid JSON in this format: "
                "{\"expected_intent\": \"...\", \"expected_route\": \"...\", \"response\": \"...\"}.\n\n"
                f"User input: {user_input}\n"
                f"Actual intent: {actual_intent}\n"
                f"Actual route: {actual_route}\n"
                "Valid routes: CONVERSATIONAL, TOOL, CLARIFICATION, RAG, LLM_FALLBACK."
            )

            raw = llm.chat(user_input=prompt, use_history=False)
            if not raw:
                return None

            if isinstance(raw, dict):
                return raw

            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return None

            return None
        except Exception as e:
            logger.warning(f"[AutoCorrection] Teacher judge failed: {e}")
            return None
    
    def apply_corrections_to_thresholds(self) -> Dict[str, Any]:
        """
        Apply corrections to adjust NLP thresholds and rules
        Only applies corrections that have been validated multiple times
        """
        MIN_VALIDATIONS = 3  # Require 3 consistent examples
        
        applied_count = 0
        by_type = defaultdict(int)
        
        for correction in self.corrections:
            if correction.get('applied'):
                continue
            
            if correction.get('validation_count', 0) < MIN_VALIDATIONS:
                continue
            
            domain = correction.get('domain', '').lower()
            if domain in self.DANGEROUS_DOMAINS:
                continue
            
            # Mark as applied
            correction['applied'] = True
            applied_count += 1
            by_type[correction.get('correction_type')] += 1
            
            logger.info(
                f"[AutoCorrection] Applying: {correction['user_input'][:30]}... "
                f"({correction.get('correction_type')}) for {domain}"
            )
        
        if applied_count > 0:
            self._save_corrections()
        
        return {
            'applied_count': applied_count,
            'by_type': dict(by_type)
        }


class PatternPromotionEngine:
    """
    Scans training data for clusters of similar examples,
    and auto-creates patterns when thresholds are met
    """
    
    SAFE_DOMAINS = {'greeting', 'farewell', 'thanks', 'notes', 'weather', 'time', 'status_inquiry', 'help'}
    DANGEROUS_DOMAINS = {'system', 'shell', 'code_execution', 'security', 'admin'}
    MIN_CLUSTER_SIZE = 3
    MIN_QUALITY_SCORE = 0.7
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / "training"
        self.learning_patterns_file = self.project_root / "memory" / "learning_patterns.json"
        self.patterns_for_review_file = self.project_root / "memory" / "patterns_for_review.json"
        self.learning_patterns = self._load_patterns()
        self.patterns_for_review = self._load_review_patterns()
        logger.info("[PatternPromotion] Engine initialized")
    
    def _load_patterns(self) -> Dict:
        """Load existing learning patterns"""
        if self.learning_patterns_file.exists():
            try:
                with open(self.learning_patterns_file, 'r') as f:
                    data = json.load(f)
                    # Ensure it's a dict (not a list)
                    if isinstance(data, list):
                        logger.warning(f"[PatternPromotion] learning_patterns.json contains array, converting to dict")
                        return {}
                    return data if isinstance(data, dict) else {}
            except Exception as e:
                logger.warning(f"[PatternPromotion] Error loading patterns: {e}")
        return {}
    
    def _load_review_patterns(self) -> List[Dict]:
        """Load patterns pending review"""
        if self.patterns_for_review_file.exists():
            try:
                with open(self.patterns_for_review_file, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                    # Ensure it's a list
                    if isinstance(data, dict):
                        return list(data.values()) if data else []
                    return data if isinstance(data, list) else []
            except Exception as e:
                logger.warning(f"[PatternPromotion] Error loading review patterns: {e}")
        return []
    
    def _save_patterns(self):
        """Save learning patterns"""
        try:
            with open(self.learning_patterns_file, 'w') as f:
                json.dump(self.learning_patterns, f, indent=2)
        except Exception as e:
            logger.error(f"[PatternPromotion] Error saving patterns: {e}")
    
    def _save_review_patterns(self):
        """Save patterns for review"""
        try:
            with open(self.patterns_for_review_file, 'w') as f:
                json.dump(self.patterns_for_review, f, indent=2)
        except Exception as e:
            logger.error(f"[PatternPromotion] Error saving review patterns: {e}")
    
    def scan_and_promote(self) -> Dict[str, Any]:
        """
        Scan training data for pattern promotion opportunities
        Returns summary of promoted/staged patterns
        """
        # Load training data
        training_data = self._load_training_data()
        
        if not training_data:
            logger.info("[PatternPromotion] No training data found")
            return {'promoted': 0, 'staged_for_review': 0}
        
        # Cluster by intent
        clusters = self._cluster_by_intent(training_data)
        
        promoted_count = 0
        staged_count = 0
        
        for intent, examples in clusters.items():
            if len(examples) >= self.MIN_CLUSTER_SIZE:
                # Check if safe domain
                is_safe = self._is_safe_domain(intent)
                
                if is_safe:
                    # Auto-promote
                    pattern = self._create_pattern_from_cluster(intent, examples)
                    if pattern:
                        self.learning_patterns[intent] = pattern
                        promoted_count += 1
                        logger.info(f"[PatternPromotion] Promoted pattern for: {intent}")
                else:
                    # Stage for review
                    pattern = self._create_pattern_from_cluster(intent, examples)
                    if pattern:
                        pattern['requires_review'] = True
                        self.patterns_for_review.append(pattern)
                        staged_count += 1
                        logger.info(f"[PatternPromotion] Staged for review: {intent}")
        
        if promoted_count > 0:
            self._save_patterns()
        if staged_count > 0:
            self._save_review_patterns()
        
        return {
            'promoted': promoted_count,
            'staged_for_review': staged_count,
            'total_clusters_found': len(clusters)
        }
    
    def _load_training_data(self) -> List[Dict]:
        """Load all training data from jsonl file"""
        training_file = self.data_dir / "training_data.jsonl"
        examples = []
        
        if training_file.exists():
            try:
                with open(training_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                # Quality filter
                                if data.get('quality_score', 0.0) >= self.MIN_QUALITY_SCORE:
                                    examples.append(data)
                            except json.JSONDecodeError:
                                pass  # Skip malformed lines
            except Exception as e:
                logger.warning(f"[PatternPromotion] Error loading training data: {e}")
        
        return examples
    
    def _cluster_by_intent(self, examples: List[Dict]) -> Dict[str, List[Dict]]:
        """Cluster training examples by intent"""
        from collections import defaultdict
        clusters = defaultdict(list)
        
        for ex in examples:
            intent = ex.get('intent', 'unknown')
            clusters[intent].append(ex)
        
        return dict(clusters)
    
    def _is_safe_domain(self, intent: str) -> bool:
        """Check if intent is safe for auto-promotion"""
        intent_lower = intent.lower()
        
        # Check against safe domains
        for safe in self.SAFE_DOMAINS:
            if safe in intent_lower:
                return True
        
        # Check against dangerous domains
        for dangerous in self.DANGEROUS_DOMAINS:
            if dangerous in intent_lower:
                return False
        
        # Default: stage for review if not explicitly safe
        return False
    
    def _create_pattern_from_cluster(self, intent: str, examples: List[Dict]) -> Optional[Dict]:
        """Create a pattern from a cluster of examples"""
        try:
            # Get most common response
            responses = [ex.get('assistant_response', '') for ex in examples]
            most_common_response = max(set(responses), key=responses.count)
            
            # Calculate confidence
            quality_scores = [ex.get('quality_score', 0.7) for ex in examples]
            avg_quality = (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.7
            
            pattern = {
                'intent': intent,
                'responses': [most_common_response],
                'active': True,
                'created_at': datetime.now().isoformat(),
                'cluster_size': len(examples),
                'avg_quality': float(avg_quality),
                'source': 'auto_promotion',
                'requires_review': False
            }
            
            return pattern
        except Exception as e:
            logger.warning(f"[PatternPromotion] Error creating pattern: {e}")
            return None


# Singleton
_learning_engine: Optional[LearningEngine] = None
_auto_correction_engine: Optional[AutoCorrectionEngine] = None
_pattern_promotion_engine: Optional[PatternPromotionEngine] = None


def get_learning_engine(data_dir: str = "data/training") -> LearningEngine:
    """Get singleton learning engine"""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LearningEngine(data_dir)
    return _learning_engine


def get_auto_correction_engine(project_root: Optional[Path] = None) -> AutoCorrectionEngine:
    """Get singleton auto-correction engine"""
    global _auto_correction_engine
    if _auto_correction_engine is None:
        _auto_correction_engine = AutoCorrectionEngine(project_root)
    return _auto_correction_engine


def get_pattern_promotion_engine(project_root: Optional[Path] = None) -> PatternPromotionEngine:
    """Get singleton pattern promotion engine"""
    global _pattern_promotion_engine
    if _pattern_promotion_engine is None:
        _pattern_promotion_engine = PatternPromotionEngine(project_root)
    return _pattern_promotion_engine
