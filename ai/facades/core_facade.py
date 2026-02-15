"""
Core Facade for A.L.I.C.E
Central reasoning, language processing, and LLM orchestration
"""

from ai.core.nlp_processor import NLPProcessor
from ai.core.reasoning_engine import get_reasoning_engine
from ai.core.conversational_engine import get_conversational_engine
from ai.core.llm_engine import LocalLLMEngine, LLMConfig
from ai.core.llm_gateway import get_llm_gateway
from ai.core.response_formulator import get_response_formulator
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CoreFacade:
    """Central reasoning, language processing, and LLM orchestration"""

    def __init__(self, llm_model: str = "llama3.3:70b") -> None:
        # Initialize NLP processor
        self.nlp = NLPProcessor()

        # Initialize reasoning engine
        try:
            self.reasoning = get_reasoning_engine()
        except Exception as e:
            logger.warning(f"Reasoning engine not available: {e}")
            self.reasoning = None

        # Initialize LLM engine
        try:
            self.llm = LocalLLMEngine(LLMConfig(model=llm_model))
        except Exception as e:
            logger.error(f"Failed to initialize LLM engine: {e}")
            self.llm = None

        # Initialize LLM gateway
        try:
            self.gateway = get_llm_gateway(self.llm) if self.llm else None
        except Exception as e:
            logger.warning(f"LLM gateway not available: {e}")
            self.gateway = None

        # Initialize response formulator
        try:
            self.formulator = get_response_formulator()
        except Exception as e:
            logger.error(f"Failed to initialize response formulator: {e}")
            self.formulator = None

        # Initialize conversational engine
        try:
            self.conversational = get_conversational_engine()
        except Exception as e:
            logger.warning(f"Conversational engine not available: {e}")
            self.conversational = None

        logger.info("[CoreFacade] Initialized core reasoning and LLM systems")

    def process_input(
        self,
        text: str
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Process user input through NLP pipeline

        Args:
            text: User input text

        Returns:
            Tuple of (intent, confidence, entities)
        """
        try:
            intent, confidence = self.nlp.classify_intent(text)
            entities = self.nlp.extract_entities(text)
            return intent, confidence, entities
        except Exception as e:
            logger.error(f"Failed to process input: {e}")
            return "unknown", 0.0, {}

    def generate_response(
        self,
        action: str,
        data: Dict[str, Any],
        success: bool,
        user_input: str,
        tone: str = "warm and helpful"
    ) -> str:
        """
        Generate natural language response

        Args:
            action: Action that was performed
            data: Structured data from action
            success: Whether action succeeded
            user_input: Original user input
            tone: Response tone

        Returns:
            Natural language response
        """
        if not self.formulator:
            # Fallback if formulator not available
            if success:
                return "Done!"
            return "I encountered an issue with that request."

        try:
            return self.formulator.formulate_response(
                action=action,
                data=data,
                success=success,
                user_input=user_input,
                tone=tone
            )
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I completed the task but had trouble phrasing the response."

    def chat(self, user_input: str, use_history: bool = True) -> str:
        """
        Direct LLM chat interaction

        Args:
            user_input: User's message
            use_history: Include conversation history

        Returns:
            LLM response
        """
        if not self.llm:
            return "Language model not available. Please check configuration."

        try:
            return self.llm.chat(user_input, use_history=use_history)
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return f"I encountered an error: {str(e)}"

    def can_handle_conversationally(
        self,
        user_input: str,
        intent: str,
        context: Any
    ) -> bool:
        """
        Check if conversational engine can handle this without LLM

        Args:
            user_input: User input
            intent: Classified intent
            context: Conversation context

        Returns:
            True if conversational engine can handle
        """
        if not self.conversational:
            return False

        try:
            return self.conversational.can_handle(user_input, intent, context)
        except Exception as e:
            logger.error(f"Failed to check conversational handling: {e}")
            return False

    def generate_conversational_response(
        self,
        user_input: str,
        intent: str,
        context: Any
    ) -> Optional[str]:
        """
        Generate response using conversational engine (no LLM)

        Args:
            user_input: User input
            intent: Classified intent
            context: Conversation context

        Returns:
            Response or None if can't handle
        """
        if not self.conversational:
            return None

        try:
            return self.conversational.generate_response(
                user_input=user_input,
                intent=intent,
                context=context
            )
        except Exception as e:
            logger.error(f"Failed to generate conversational response: {e}")
            return None

    def query_knowledge(self, question: str) -> str:
        """
        Query LLM for factual knowledge

        Args:
            question: Question to ask

        Returns:
            Factual answer
        """
        if not self.llm:
            return "Knowledge query unavailable - LLM not initialized."

        try:
            return self.llm.query_knowledge(question)
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return f"Knowledge query error: {str(e)}"

    def phrase_with_tone(
        self,
        content: str,
        tone: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Phrase structured content with specific tone

        Args:
            content: Content to phrase
            tone: Desired tone
            context: Optional context

        Returns:
            Phrased response
        """
        if not self.llm:
            return str(content)

        try:
            return self.llm.phrase_with_tone(content, tone, context or {})
        except Exception as e:
            logger.error(f"Phrasing failed: {e}")
            return str(content)

    def get_llm_stats(self) -> Dict[str, Any]:
        """
        Get LLM usage statistics

        Returns:
            Statistics dictionary
        """
        if not self.llm:
            return {"error": "LLM not initialized"}

        try:
            return self.llm.get_stats()
        except Exception as e:
            logger.error(f"Failed to get LLM stats: {e}")
            return {"error": str(e)}


# Singleton instance
_core_facade: Optional[CoreFacade] = None


def get_core_facade(llm_model: str = "llama3.3:70b") -> CoreFacade:
    """Get or create the CoreFacade singleton"""
    global _core_facade
    if _core_facade is None:
        _core_facade = CoreFacade(llm_model=llm_model)
    return _core_facade
