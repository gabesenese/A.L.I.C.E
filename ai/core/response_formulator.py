"""
Response Formulator for A.L.I.C.E
=================================
Transforms structured plugin data into natural language responses.

Instead of plugins returning hard-coded messages, they return structured data.
Alice learns to formulate natural responses from that data.

Philosophy:
- Plugins provide facts (data)
- Alice provides personality (phrasing)
- Learning happens through examples
- Progressive independence from Ollama
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResponseTemplate:
    """Template for a type of response"""
    action: str  # e.g., "create_note", "delete_notes", "search_results"
    example_data: Dict[str, Any]  # Example data structure
    example_phrasings: List[str]  # Multiple ways to phrase it
    formulation_rules: List[str]  # Rules for generating response


class ResponseFormulator:
    """
    Learns to formulate natural responses from structured data.

    Works with:
    - PhrasingLearner: Learns successful phrasings
    - LLM Gateway: Uses Ollama for initial formulations
    - Pattern system: Recognizes when Alice can formulate alone
    """

    def __init__(
        self,
        phrasing_learner=None,
        llm_gateway=None,
        storage_path: str = "data/response_templates"
    ):
        self.phrasing_learner = phrasing_learner
        self.llm_gateway = llm_gateway
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load response templates
        self.templates: Dict[str, ResponseTemplate] = {}
        self._load_templates()

        # Track what Alice can formulate independently
        self.independent_actions = set()
        self._load_independence_data()

    def _load_templates(self):
        """Load response templates from storage"""
        template_file = self.storage_path / "templates.json"
        if template_file.exists():
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for action, template_data in data.items():
                        self.templates[action] = ResponseTemplate(
                            action=action,
                            example_data=template_data.get('example_data', {}),
                            example_phrasings=template_data.get('example_phrasings', []),
                            formulation_rules=template_data.get('formulation_rules', [])
                        )
                logger.info(f"Loaded {len(self.templates)} response templates")
            except Exception as e:
                logger.error(f"Error loading response templates: {e}")

    def _load_independence_data(self):
        """Load actions Alice can formulate independently"""
        independence_file = self.storage_path / "independence.json"
        if independence_file.exists():
            try:
                with open(independence_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.independent_actions = set(data.get('independent_actions', []))
                logger.info(f"Alice can independently formulate {len(self.independent_actions)} action types")
            except Exception as e:
                logger.error(f"Error loading independence data: {e}")

    def formulate_response(
        self,
        action: str,
        data: Dict[str, Any],
        success: bool,
        user_input: str,
        tone: str = "helpful"
    ) -> str:
        """
        Formulate a natural language response from plugin data.

        Process:
        1. Check if Alice can formulate independently (learned patterns)
        2. If not, use Ollama with examples to formulate
        3. Learn from successful formulation
        4. Track progress toward independence
        """

        # Check if Alice can formulate this independently
        if action in self.independent_actions and self.phrasing_learner:
            alice_response = self._formulate_independently(action, data, success, tone)
            if alice_response:
                logger.info(f"[ResponseFormulator] Alice formulated '{action}' independently")
                return alice_response

        # Use Ollama to formulate with learning
        if self.llm_gateway:
            ollama_response = self._formulate_with_llm(action, data, success, user_input, tone)

            # Learn from this formulation
            if ollama_response and self.phrasing_learner:
                self._learn_formulation(action, data, ollama_response, tone)

            return ollama_response

        # Fallback: basic template-based formulation
        return self._formulate_basic(action, data, success)

    def _formulate_independently(
        self,
        action: str,
        data: Dict[str, Any],
        success: bool,
        tone: str
    ) -> Optional[str]:
        """Formulate response using Alice's learned patterns"""
        if not self.phrasing_learner:
            return None

        # Build a thought structure from the data
        thought = {
            "type": action,
            "success": success,
            "data": data
        }

        # Try to phrase it using learned patterns
        try:
            response = self.phrasing_learner.phrase_thought(thought, tone=tone)
            if response and response != "":
                return response
        except Exception as e:
            logger.debug(f"Could not formulate independently: {e}")

        return None

    def _formulate_with_llm(
        self,
        action: str,
        data: Dict[str, Any],
        success: bool,
        user_input: str,
        tone: str
    ) -> str:
        """Use LLM to formulate response with examples for learning"""

        # Get template and examples if available
        template = self.templates.get(action)
        examples_text = ""

        if template and template.example_phrasings:
            examples_text = "\n\nExamples of good responses for this action:\n"
            for i, example in enumerate(template.example_phrasings[:3], 1):
                examples_text += f"{i}. {example}\n"

        # Build prompt for LLM
        prompt = f"""The user said: "{user_input}"

Action performed: {action}
Success: {success}
Data: {json.dumps(data, indent=2)}

{examples_text}

Formulate a natural, {tone} response for Alice saying what was done.
- Be concise (1-2 sentences)
- Be specific using the data
- Match Alice's personality
- Don't use emojis
- Don't repeat the user's exact words

Response:"""

        try:
            from ai.core.llm_policy import LLMCallType

            result = self.llm_gateway.request(
                prompt=prompt,
                call_type=LLMCallType.RESPONSE_GENERATION,
                use_history=False,
                user_input=user_input
            )

            if result.success and result.response:
                return result.response.strip()
        except Exception as e:
            logger.error(f"Error formulating with LLM: {e}")

        # Fallback
        return self._formulate_basic(action, data, success)

    def _formulate_basic(
        self,
        action: str,
        data: Dict[str, Any],
        success: bool
    ) -> str:
        """Basic fallback formulation without LLM"""
        if not success:
            return "I wasn't able to complete that action."

        # Simple templates based on action type
        if "create" in action:
            return "Created successfully."
        elif "delete" in action or "remove" in action:
            count = data.get('count', 1)
            return f"Removed {count} item{'s' if count != 1 else ''}."
        elif "search" in action or "find" in action:
            count = data.get('count', 0)
            return f"Found {count} result{'s' if count != 1 else ''}."
        elif "update" in action or "edit" in action:
            return "Updated successfully."
        else:
            return "Done."

    def _learn_formulation(
        self,
        action: str,
        data: Dict[str, Any],
        response: str,
        tone: str
    ):
        """Learn from a successful formulation"""
        if not self.phrasing_learner:
            return

        # Record this as a learned phrasing
        thought = {
            "type": action,
            "data": data
        }

        try:
            self.phrasing_learner.learn_from_example(
                alice_thought=thought,
                ollama_phrasing=response,
                tone=tone
            )

            # Check if Alice has learned enough to be independent
            if self.phrasing_learner.can_phrase_myself(thought, tone):
                self.independent_actions.add(action)
                self._save_independence_data()
                logger.info(f"[ResponseFormulator] Alice achieved independence for '{action}'!")
        except Exception as e:
            logger.debug(f"Could not learn formulation: {e}")

    def _save_independence_data(self):
        """Save actions Alice can formulate independently"""
        independence_file = self.storage_path / "independence.json"
        try:
            with open(independence_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'independent_actions': list(self.independent_actions)
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving independence data: {e}")

    def add_template(
        self,
        action: str,
        example_data: Dict[str, Any],
        example_phrasings: List[str],
        formulation_rules: List[str] = None
    ):
        """Add a response template for an action type"""
        self.templates[action] = ResponseTemplate(
            action=action,
            example_data=example_data,
            example_phrasings=example_phrasings,
            formulation_rules=formulation_rules or []
        )
        self._save_templates()

    def _save_templates(self):
        """Save response templates to storage"""
        template_file = self.storage_path / "templates.json"
        try:
            data = {}
            for action, template in self.templates.items():
                data[action] = {
                    'example_data': template.example_data,
                    'example_phrasings': template.example_phrasings,
                    'formulation_rules': template.formulation_rules
                }

            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.templates)} response templates")
        except Exception as e:
            logger.error(f"Error saving templates: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get formulator statistics"""
        return {
            'total_templates': len(self.templates),
            'independent_actions': len(self.independent_actions),
            'learning_progress': f"{len(self.independent_actions)}/{len(self.templates)}" if self.templates else "0/0"
        }


# Singleton instance
_formulator = None

def get_response_formulator(phrasing_learner=None, llm_gateway=None) -> ResponseFormulator:
    """Get or create the response formulator singleton"""
    global _formulator
    if _formulator is None:
        _formulator = ResponseFormulator(
            phrasing_learner=phrasing_learner,
            llm_gateway=llm_gateway
        )
    return _formulator
