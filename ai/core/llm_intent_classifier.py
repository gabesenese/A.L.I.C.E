"""
LLM-Based Intent Classifier with Advanced Prompting Techniques
Uses Chain-of-Thought, Few-Shot Learning, and Self-Consistency

Kicks in when semantic similarity classifier has low confidence (<0.6)
"""

import logging
from typing import Dict, Optional, List, Tuple
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMIntentResult:
    """Result from LLM intent classification"""
    intent: str
    confidence: float
    reasoning: str
    plugin: Optional[str] = None
    action: Optional[str] = None


class LLMIntentClassifier:
    """
    Advanced LLM-based intent classifier using prompting techniques:

    1. Chain-of-Thought (CoT): LLM explains its reasoning step-by-step
    2. Few-Shot Learning: Provide examples of intent classification
    3. Self-Consistency: Generate multiple responses, pick most common
    4. Role Prompting: Give LLM specific role as intent classifier
    """

    def __init__(self, llm_gateway=None):
        """
        Initialize LLM intent classifier

        Args:
            llm_gateway: Gateway to LLM for classification
        """
        self.llm_gateway = llm_gateway
        self.few_shot_examples = self._create_few_shot_examples()

    def _create_few_shot_examples(self) -> List[Dict[str, str]]:
        """
        Create few-shot examples for intent classification
        Shows LLM how to classify and reason about intents
        """
        return [
            {
                "query": "how many notes do i have",
                "reasoning": "User is asking for a count/quantity of notes. This is a query about existing data, not creating or modifying. Intent: notes, Action: count",
                "intent": "notes",
                "action": "count",
                "confidence": "0.95"
            },
            {
                "query": "what's the weather like",
                "reasoning": "User wants current weather information. 'what's' indicates present tense, 'weather like' is asking for conditions. Intent: weather, Action: current",
                "intent": "weather",
                "action": "current",
                "confidence": "0.98"
            },
            {
                "query": "remind me to call mom",
                "reasoning": "User wants to create a reminder for future action. 'remind me to' indicates setting up a future notification. Intent: notes (for tasks/reminders), Action: create_task",
                "intent": "notes",
                "action": "create_task",
                "confidence": "0.92"
            },
            {
                "query": "what did we talk about yesterday",
                "reasoning": "User is asking to recall past conversation content. 'what did we talk about' indicates memory retrieval. Intent: memory, Action: search",
                "intent": "memory",
                "action": "search",
                "confidence": "0.90"
            },
            {
                "query": "how are you",
                "reasoning": "User is asking about my status/wellbeing. This is a conversational greeting, not a command. Intent: status_inquiry, Action: respond",
                "intent": "status_inquiry",
                "action": "respond",
                "confidence": "0.85"
            }
        ]

    def _build_cot_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """
        Build Chain-of-Thought prompt for intent classification

        Args:
            query: User's query to classify
            context: Recent conversation context

        Returns:
            Formatted prompt with few-shot examples and CoT instructions
        """
        # Start with role definition
        prompt = """You are an expert intent classifier for an AI assistant named A.L.I.C.E.

Your task: Classify user queries into intents and actions by reasoning step-by-step.

Available intents:
- notes: Managing notes, tasks, reminders
- email: Email operations (list, read, send, search)
- weather: Weather information (current, forecast)
- time: Time and date queries
- music: Music playback control
- calendar: Calendar and schedule management
- file: File operations (create, read, delete, move, search)
- memory: Remember and recall information
- status_inquiry: Asking about assistant's status ("how are you")
- greeting: Greetings and farewells
- conversation: General conversation and questions

Examples of classification with reasoning:

"""

        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples, 1):
            prompt += f"""Example {i}:
Query: "{example['query']}"
Reasoning: {example['reasoning']}
Result: {{"intent": "{example['intent']}", "action": "{example['action']}", "confidence": {example['confidence']}}}

"""

        # Add context if available
        if context and len(context) > 0:
            prompt += f"\nRecent conversation context:\n"
            for i, ctx in enumerate(context[-3:], 1):  # Last 3 queries
                prompt += f"{i}. {ctx}\n"
            prompt += "\n"

        # Add current query with CoT instruction
        prompt += f"""Now classify this query:
Query: "{query}"

Think step-by-step:
1. What is the user asking for?
2. What action do they want performed?
3. Which intent category does this belong to?
4. How confident are you? (0.0-1.0)

Provide your analysis as JSON:
{{"reasoning": "your step-by-step reasoning here", "intent": "intent_name", "action": "action_name", "confidence": 0.XX}}

JSON response:"""

        return prompt

    def classify_with_cot(
        self,
        query: str,
        context: Optional[List[str]] = None,
        num_samples: int = 1
    ) -> Optional[LLMIntentResult]:
        """
        Classify intent using Chain-of-Thought reasoning

        Args:
            query: User query to classify
            context: Recent conversation context
            num_samples: Number of samples for self-consistency (1=no consistency check)

        Returns:
            LLMIntentResult with classification and reasoning
        """
        if not self.llm_gateway:
            logger.warning("No LLM gateway available for intent classification")
            return None

        from ai.core.llm_gateway import LLMCallType

        prompt = self._build_cot_prompt(query, context)

        # For self-consistency, generate multiple responses
        if num_samples > 1:
            return self._classify_with_self_consistency(prompt, query, num_samples)

        # Single response with CoT
        try:
            response = self.llm_gateway.request(
                prompt=prompt,
                call_type=LLMCallType.INTENT_CLASSIFICATION,
                use_history=False
            )

            if response.success and response.response:
                return self._parse_llm_response(response.response, query)
            else:
                logger.warning(f"LLM intent classification failed: {response.response if response else 'No response'}")
                return None

        except Exception as e:
            logger.error(f"Error in LLM intent classification: {e}")
            return None

    def _classify_with_self_consistency(
        self,
        prompt: str,
        query: str,
        num_samples: int
    ) -> Optional[LLMIntentResult]:
        """
        Use self-consistency: Generate multiple classifications and pick most common

        Args:
            prompt: CoT prompt
            query: Original query
            num_samples: Number of samples to generate (typically 3-5)

        Returns:
            Most consistent LLMIntentResult
        """
        from ai.core.llm_gateway import LLMCallType
        from collections import Counter

        results = []

        # Generate multiple responses
        for i in range(num_samples):
            try:
                response = self.llm_gateway.request(
                    prompt=prompt,
                    call_type=LLMCallType.INTENT_CLASSIFICATION,
                    use_history=False,
                    temperature=0.7  # Some variation for diversity
                )

                if response.success and response.response:
                    result = self._parse_llm_response(response.response, query)
                    if result:
                        results.append(result)
            except Exception as e:
                logger.warning(f"Error generating sample {i+1}: {e}")
                continue

        if not results:
            return None

        # Find most common intent
        intent_counts = Counter(r.intent for r in results)
        most_common_intent = intent_counts.most_common(1)[0][0]

        # Get the result with the most common intent and highest confidence
        matching_results = [r for r in results if r.intent == most_common_intent]
        best_result = max(matching_results, key=lambda r: r.confidence)

        # Boost confidence based on consistency
        consistency_ratio = intent_counts[most_common_intent] / len(results)
        best_result.confidence = min(0.99, best_result.confidence * (0.5 + 0.5 * consistency_ratio))

        logger.info(f"Self-consistency: {most_common_intent} ({consistency_ratio:.0%} agreement across {len(results)} samples)")

        return best_result

    def _parse_llm_response(self, response: str, query: str) -> Optional[LLMIntentResult]:
        """
        Parse LLM's JSON response into LLMIntentResult

        Args:
            response: LLM response string
            query: Original query

        Returns:
            Parsed LLMIntentResult or None if parsing fails
        """
        try:
            # Extract JSON from response (may have extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                logger.warning(f"No JSON found in LLM response: {response[:100]}")
                return None

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            return LLMIntentResult(
                intent=data.get('intent', 'conversation'),
                confidence=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', ''),
                action=data.get('action', 'respond')
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:200]}")
            return None
        except Exception as e:
            logger.error(f"Error parsing LLM intent response: {e}")
            return None

    def classify_hybrid(
        self,
        query: str,
        semantic_confidence: float,
        semantic_intent: Optional[str],
        context: Optional[List[str]] = None
    ) -> Tuple[str, float, str]:
        """
        Hybrid classification: Use LLM CoT for low semantic confidence

        Args:
            query: User query
            semantic_confidence: Confidence from semantic classifier
            semantic_intent: Intent from semantic classifier
            context: Recent conversation context

        Returns:
            (intent, confidence, source) tuple
            source: 'semantic', 'llm_cot', or 'llm_consistency'
        """
        # High semantic confidence - trust it
        if semantic_confidence >= 0.7:
            return (semantic_intent, semantic_confidence, 'semantic')

        # Medium confidence (0.5-0.7) - use single LLM CoT
        if 0.5 <= semantic_confidence < 0.7:
            logger.info(f"Medium confidence ({semantic_confidence:.2f}), using LLM CoT")
            result = self.classify_with_cot(query, context, num_samples=1)
            if result and result.confidence > semantic_confidence:
                return (result.intent, result.confidence, 'llm_cot')
            else:
                return (semantic_intent, semantic_confidence, 'semantic_fallback')

        # Low confidence (<0.5) - use self-consistency (3 samples)
        logger.info(f"Low confidence ({semantic_confidence:.2f}), using LLM self-consistency")
        result = self.classify_with_cot(query, context, num_samples=3)
        if result:
            return (result.intent, result.confidence, 'llm_consistency')
        else:
            # LLM failed, fallback to semantic
            return (semantic_intent if semantic_intent else 'conversation',
                    semantic_confidence if semantic_confidence else 0.3,
                    'fallback')


# Singleton
_llm_classifier = None

def get_llm_intent_classifier(llm_gateway=None) -> LLMIntentClassifier:
    """Get or create LLM intent classifier singleton"""
    global _llm_classifier
    if _llm_classifier is None:
        _llm_classifier = LLMIntentClassifier(llm_gateway)
    elif llm_gateway and not _llm_classifier.llm_gateway:
        _llm_classifier.llm_gateway = llm_gateway
    return _llm_classifier
