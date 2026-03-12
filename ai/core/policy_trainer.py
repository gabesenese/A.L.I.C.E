"""
A.L.I.C.E. Policy Trainer
==========================
Trains two lightweight binary classifiers from TurnLogger data to replace
hand-tuned thresholds in InteractionPolicy:

1. **clarify_classifier** — predicts whether Alice should request clarification
   given turn features (mood, sentiment, urgency, intent_conf, frame_conf).
   Label: ``skip_clarification == False`` (i.e. 1 = should clarify, 0 = skip).

2. **confidence_classifier** — predicts whether Alice should be "cautious"
   (lower threshold, shorter response) vs "confident" (higher threshold,
   normal/detailed response).
   Label: derived from observed ``outcome_reward`` — reward < 0.4 → cautious.

Both classifiers are trained with scikit-learn's ``LogisticRegression`` (or a
``DecisionTreeClassifier`` if sklearn version is too old), using max 2 000 of
the most recent *labelled* entries (entries with ``outcome_reward != null``).

Model artefacts are persisted in ``data/models/policy_models.pkl``.  If the
file doesn't exist or training fails, ``InteractionPolicy`` falls back to
hand-tuned rules entirely.

Usage
-----
    from ai.core.policy_trainer import PolicyTrainer
    trainer = PolicyTrainer()
    trainer.fit()                # train on current log
    prob = trainer.predict_clarify(mood="frustrated", sentiment=-0.6,
                                   urgency="high", intent_conf=0.55,
                                   frame_conf=0.0)
    # prob is a float 0–1; >0.5 → should clarify

    proba = trainer.predict_confidence(mood="neutral", sentiment=0.1,
                                       urgency="none", intent_conf=0.82,
                                       frame_conf=0.75)
    # proba < 0.5 → cautious mode; ≥ 0.5 → confident mode
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: scikit-learn
# ---------------------------------------------------------------------------

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "[PolicyTrainer] scikit-learn not available. "
        "Policy trainer will not produce learned models."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOOD_ENCODING: Dict[str, float] = {
    "frustrated": -2.0,
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
    "urgent": 0.5,
}
_URGENCY_ENCODING: Dict[str, float] = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}

_DEFAULT_MODEL_PATH = Path("data") / "models" / "policy_models.pkl"
_MIN_TRAINING_SAMPLES = 30  # Don't train with fewer labelled examples than this


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _encode_features(
    mood: str,
    sentiment: float,
    urgency: str,
    intent_conf: float,
    frame_conf: float,
) -> List[float]:
    """Convert categorical + continuous turn features to a numeric vector."""
    m = _MOOD_ENCODING.get(mood, 0.0)
    u = _URGENCY_ENCODING.get(urgency, 0.0)
    # Clip sentiment to [-1, 1]
    s = max(-1.0, min(1.0, float(sentiment)))
    ic = max(0.0, min(1.0, float(intent_conf)))
    fc = max(0.0, min(1.0, float(frame_conf)))
    return [m, s, u, ic, fc]


def _extract_features_from_record(r: Dict[str, Any]) -> Optional[List[float]]:
    """Extract feature vector from a raw turn-log record dict."""
    try:
        return _encode_features(
            mood=r.get("mood", "neutral"),
            sentiment=float(r.get("sentiment", 0.0)),
            urgency=r.get("urgency", "none"),
            intent_conf=float(r.get("intent_conf", 0.5)),
            frame_conf=float(r.get("frame_conf", 0.0)),
        )
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# PolicyTrainer
# ---------------------------------------------------------------------------


class PolicyTrainer:
    """
    Fits and persists two logistic-regression policy classifiers.

    Interface is intentionally simple to keep the integration in
    InteractionPolicy lightweight.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = str(_DEFAULT_MODEL_PATH)
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self._clarify_model: Optional[Any] = None
        self._confidence_model: Optional[Any] = None
        self._lock = threading.Lock()

        # Try loading persisted models on init
        self._load()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, records: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Train both classifiers from labelled TurnLogger entries.

        Parameters
        ----------
        records:
            Optional list of raw turn-log dicts. If None, the module-level
            ``get_turn_logger()`` is used to load recent labelled entries.

        Returns True if models were successfully trained.
        """
        if not SKLEARN_AVAILABLE:
            return False

        if records is None:
            try:
                from ai.core.turn_logger import get_turn_logger

                records = get_turn_logger().load_labelled(2000)
            except Exception as exc:
                logger.warning("[PolicyTrainer] Could not load turn log: %s", exc)
                return False

        labelled = [r for r in records if r.get("outcome_reward") is not None]
        if len(labelled) < _MIN_TRAINING_SAMPLES:
            logger.info(
                "[PolicyTrainer] Only %d labelled samples — skipping training "
                "(need ≥ %d).",
                len(labelled),
                _MIN_TRAINING_SAMPLES,
            )
            return False

        X: List[List[float]] = []
        y_clarify: List[int] = []
        y_confidence: List[int] = []

        for r in labelled:
            feats = _extract_features_from_record(r)
            if feats is None:
                continue

            # Label 1: should clarify (i.e. skip_clarification was False)
            policy = r.get("policy") or {}
            should_clarify = 1 if not policy.get("skip_clarification", False) else 0

            # Label 2: reward < 0.4 → cautious (1), else confident (0)
            reward = float(r.get("outcome_reward", 0.5))
            cautious = 1 if reward < 0.4 else 0

            X.append(feats)
            y_clarify.append(should_clarify)
            y_confidence.append(cautious)

        if len(X) < _MIN_TRAINING_SAMPLES:
            logger.info(
                "[PolicyTrainer] After filtering, only %d valid samples.", len(X)
            )
            return False

        try:
            clarify_pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=300, C=1.0, class_weight="balanced"
                        ),
                    ),
                ]
            )
            clarify_pipe.fit(X, y_clarify)

            confidence_pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=300, C=1.0, class_weight="balanced"
                        ),
                    ),
                ]
            )
            confidence_pipe.fit(X, y_confidence)

            with self._lock:
                self._clarify_model = clarify_pipe
                self._confidence_model = confidence_pipe

            self._save()
            logger.info(
                "[PolicyTrainer] Trained on %d samples. Models saved to %s.",
                len(X),
                self.model_path,
            )
            return True

        except Exception as exc:
            logger.warning("[PolicyTrainer] Training failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_clarify(
        self,
        mood: str,
        sentiment: float,
        urgency: str,
        intent_conf: float,
        frame_conf: float,
    ) -> Optional[float]:
        """
        Return probability (0–1) that the model thinks Alice should ask for
        clarification.  Returns None when no trained model is available.
        """
        with self._lock:
            model = self._clarify_model
        if model is None:
            return None
        try:
            feats = [
                _encode_features(mood, sentiment, urgency, intent_conf, frame_conf)
            ]
            prob = model.predict_proba(feats)[0][1]
            return float(prob)
        except Exception as exc:
            logger.debug("[PolicyTrainer] predict_clarify error: %s", exc)
            return None

    def predict_confidence(
        self,
        mood: str,
        sentiment: float,
        urgency: str,
        intent_conf: float,
        frame_conf: float,
    ) -> Optional[float]:
        """
        Return probability (0–1) that the model predicts a *cautious* mode is
        appropriate (< 0.5 → confident; ≥ 0.5 → cautious).
        Returns None when no trained model is available.
        """
        with self._lock:
            model = self._confidence_model
        if model is None:
            return None
        try:
            feats = [
                _encode_features(mood, sentiment, urgency, intent_conf, frame_conf)
            ]
            prob = model.predict_proba(feats)[0][1]
            return float(prob)
        except Exception as exc:
            logger.debug("[PolicyTrainer] predict_confidence error: %s", exc)
            return None

    @property
    def is_ready(self) -> bool:
        """True when both models have been trained/loaded."""
        with self._lock:
            return (
                self._clarify_model is not None and self._confidence_model is not None
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            payload = {
                "clarify": self._clarify_model,
                "confidence": self._confidence_model,
            }
            with self.model_path.open("wb") as fh:
                pickle.dump(payload, fh)
        except Exception as exc:
            logger.warning("[PolicyTrainer] Could not save models: %s", exc)

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with self.model_path.open("rb") as fh:
                payload = pickle.load(fh)
            with self._lock:
                self._clarify_model = payload.get("clarify")
                self._confidence_model = payload.get("confidence")
            logger.info(
                "[PolicyTrainer] Loaded policy models from %s.", self.model_path
            )
        except Exception as exc:
            logger.warning("[PolicyTrainer] Could not load models: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[PolicyTrainer] = None
_instance_lock = threading.Lock()


def get_policy_trainer(model_path: Optional[str] = None) -> PolicyTrainer:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PolicyTrainer(model_path)
    return _instance
