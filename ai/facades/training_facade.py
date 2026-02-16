"""
Training Facade for A.L.I.C.E
Training data collection and model fine-tuning
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    from ai.training.scenario_generator import ScenarioGenerator
    _scenario_gen_available = True
except ImportError:
    _scenario_gen_available = False

try:
    from ai.training.ollama_evaluator import get_ollama_evaluator
    _evaluator_available = True
except ImportError:
    _evaluator_available = False


class TrainingFacade:
    """Facade for training and fine-tuning"""

    def __init__(self) -> None:
        # Scenario generator
        try:
            self.scenario_gen = ScenarioGenerator() if _scenario_gen_available else None
        except Exception as e:
            logger.warning(f"Scenario generator not available: {e}")
            self.scenario_gen = None

        # Evaluator
        try:
            self.evaluator = get_ollama_evaluator() if _evaluator_available else None
        except Exception as e:
            logger.warning(f"Evaluator not available: {e}")
            self.evaluator = None

        # Training data collector (not yet implemented)
        self.collector = None

        # Fine-tuning manager (not yet implemented)
        self.fine_tuner = None
        self.fine_tuning = None

        logger.info("[TrainingFacade] Initialized training systems")

    def collect_example(
        self,
        user_input: str,
        assistant_response: str,
        intent: str,
        quality_score: float = 1.0
    ) -> bool:
        """
        Collect training example

        Args:
            user_input: User's message
            assistant_response: Alice's response
            intent: Classified intent
            quality_score: Quality rating (0.0-1.0)

        Returns:
            True if collected successfully
        """
        if not self.collector:
            return False

        try:
            self.collector.add_example(
                user_input=user_input,
                assistant_response=assistant_response,
                intent=intent,
                quality_score=quality_score
            )
            return True
        except Exception as e:
            logger.error(f"Failed to collect example: {e}")
            return False

    def get_training_data(
        self,
        min_quality: float = 0.7,
        max_examples: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get collected training data

        Args:
            min_quality: Minimum quality threshold
            max_examples: Maximum examples to return

        Returns:
            List of training examples
        """
        if not self.collector:
            return []

        try:
            return self.collector.get_training_data(
                min_quality=min_quality,
                max_examples=max_examples
            )
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []

    def start_fine_tuning(
        self,
        model_name: str,
        training_file: str
    ) -> Optional[str]:
        """
        Start fine-tuning job

        Args:
            model_name: Base model to fine-tune
            training_file: Path to training data

        Returns:
            Job ID or None
        """
        if not self.fine_tuning:
            return None

        try:
            return self.fine_tuning.start_job(model_name, training_file)
        except Exception as e:
            logger.error(f"Failed to start fine-tuning: {e}")
            return None

    def get_fine_tuning_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get fine-tuning job status

        Args:
            job_id: Job identifier

        Returns:
            Status dictionary
        """
        if not self.fine_tuning:
            return {"error": "Fine-tuning manager not available"}

        try:
            return self.fine_tuning.get_status(job_id)
        except Exception as e:
            logger.error(f"Failed to get fine-tuning status: {e}")
            return {"error": str(e)}

    def export_training_data(
        self,
        output_file: str,
        format: str = "jsonl"
    ) -> bool:
        """
        Export training data to file

        Args:
            output_file: Output file path
            format: Export format (jsonl, csv)

        Returns:
            True if exported successfully
        """
        if not self.collector:
            return False

        try:
            self.collector.export_data(output_file, format=format)
            return True
        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return False


# Singleton instance
_training_facade: Optional[TrainingFacade] = None


def get_training_facade() -> TrainingFacade:
    """Get or create the TrainingFacade singleton"""
    global _training_facade
    if _training_facade is None:
        _training_facade = TrainingFacade()
    return _training_facade
