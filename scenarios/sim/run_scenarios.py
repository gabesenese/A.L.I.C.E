"""
Scenario Runner for A.L.I.C.E

Runs scripted conversation scenarios, captures routing decisions,
generates training data, and compares against teacher responses.
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

# Add project root to path
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SIM_DIR))
sys.path.insert(0, PROJECT_ROOT)

from .scenarios import (
    Scenario, ScenarioStep, ScenarioResult, ExpectedRoute,
    ALL_SCENARIOS, get_scenarios_by_domain, get_scenarios_by_tag
)
from .teacher import TeacherMode

# Import Alice components (without full initialization)
from ai.nlp_processor import NLPProcessor
from ai.conversational_engine import get_conversational_engine
from ai.llm_gateway import get_llm_gateway
from ai.llm_engine import LocalLLMEngine, LLMConfig
from ai.context_engine import get_context_engine
from ai.memory_system import MemorySystem
from ai.reasoning_engine import get_reasoning_engine
from ai.learning_engine import get_learning_engine
from ai.llm_policy import configure_minimal_policy, get_llm_policy, LLMCallType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Intent mapping from NLP output to scenario expectations
INTENT_MAPPING = {
    # Email intents
    "email:list": "list_emails",
    "email:read": "read_email",
    "email:search": "search_emails",
    "email:compose": "compose_email",
    "email:delete": "delete_email",
    # Notes intents
    "notes:create": "create_note",
    "notes:search": "search_notes",
    "notes:list": "list_notes",
    # Weather/Time intents
    "weather:current": "get_weather",
    "weather:forecast": "get_weather_forecast",
    "time:current": "get_time",
    # System intents
    "system:status": "system_status",
    # Conversational intents
    "greeting": "greeting",
    "farewell": "thanks",
    "thanks": "thanks",
    "status_inquiry": "status_inquiry",
    "conversation:question": "vague_question",
    "conversation:meta_question": "vague_question",
    "conversation:status": "status_inquiry",
    # Clarification triggers
    "clarification:needed": "vague_question",
    "schedule_action": "schedule_action",
    "vague_temporal_question": "vague_temporal_question",
}


class ScenarioRunner:
    """
    Runs conversation scenarios against Alice and generates training data
    """
    
    def __init__(
        self,
        llm_policy: str = "minimal",
        llm_model: str = "llama3.1:8b",
        use_teacher: bool = True
    ):
        """
        Initialize scenario runner
        
        Args:
            llm_policy: LLM policy to use (minimal/default/strict)
            llm_model: LLM model for both Alice and teacher
            use_teacher: Whether to use teacher mode for comparisons
        """
        self.llm_policy = llm_policy
        self.llm_model = llm_model
        self.use_teacher = use_teacher
        
        logger.info("=" * 70)
        logger.info("A.L.I.C.E Scenario Runner - Automated Testing & Training")
        logger.info("=" * 70)
        logger.info(f"LLM Policy: {llm_policy}")
        logger.info(f"Model: {llm_model}")
        logger.info(f"Teacher Mode: {'Enabled' if use_teacher else 'Disabled'}")
        logger.info("=" * 70)
        
        # Initialize core components (lightweight - no full ALICE)
        logger.info("Initializing core components...")
        
        # Configure policy first
        if llm_policy == "minimal":
            configure_minimal_policy()
            logger.info("[OK] Minimal policy configured")
        
        self.nlp = NLPProcessor()
        self.context = get_context_engine()
        self.memory = MemorySystem()
        self.reasoning_engine = get_reasoning_engine("SimUser")
        self.learning_engine = get_learning_engine()
        
        # LLM components
        llm_config = LLMConfig(model=llm_model)
        self.llm = LocalLLMEngine(llm_config)
        self.llm_gateway = get_llm_gateway(
            llm_engine=self.llm,
            learning_engine=self.learning_engine
        )
        
        # Conversational engine
        self.conversational_engine = get_conversational_engine(
            memory_system=self.memory,
            world_state=self.reasoning_engine
        )
        
        # Teacher mode
        self.teacher = TeacherMode(model=llm_model) if use_teacher else None
        
        logger.info("[OK] Core components ready\n")
        
        # Results storage
        self.results: List[ScenarioResult] = []
        self.training_data: List[Dict[str, Any]] = []
    
    def _determine_route(self, user_input: str, nlp_result) -> str:
        """
        Determine Alice's routing decision for input
        
        Args:
            user_input: User's input text
            nlp_result: NLP processing result (ProcessedQuery object)
        
        Returns:
            Route name (CONVERSATIONAL/TOOL/RAG/LLM_FALLBACK/CLARIFICATION)
        """
        intent = nlp_result.intent if hasattr(nlp_result, 'intent') else "unknown"
        confidence = nlp_result.intent_confidence if hasattr(nlp_result, 'intent_confidence') else 0.0
        text_lower = user_input.lower()

        # Clarification patterns (vague or meta questions)
        vague_phrases = [
            "tell me about",
            "can you do that",
            "what about",
            "i have a question about",
            "can i ask you about",
            "let me ask you about",
            "schedule it for"  # ambiguous schedule without what
        ]
        if any(phrase in text_lower for phrase in vague_phrases) or "vague" in intent or "unclear" in intent or intent == "conversation:meta_question" or intent in ["vague_temporal_question", "vague_question", "vague_request"]:
            return "CLARIFICATION"

        # Confidence-based clarification gate
        strong_domain_keywords = any(word in text_lower for word in [
            "email", "mail", "inbox", "note", "notes", "weather", "forecast", "temperature", "rain", "snow", "sunny", "cloudy",
            "time", "clock", "system", "status", "cpu", "memory", "disk", "battery",
            "calendar", "event", "schedule", "tomorrow", "today", "tonight"
        ])
        if confidence < 0.7 and not strong_domain_keywords:
            return "CLARIFICATION"

        # Tool requests
        if intent in [
            "email:list", "email:search", "email:read", "email:delete", "email:compose",
            "notes:create", "notes:search", "notes:list",
            "weather:current", "weather:forecast", "time:current"
        ]:
            return "TOOL"

        # System status
        if intent == "system:status":
            return "CONVERSATIONAL"

        # Check for RAG/memory queries
        if "remember" in text_lower or "recall" in text_lower:
            return "RAG"

        # Conversational engine
        if self.conversational_engine.can_handle(user_input, intent, None):
            return "CONVERSATIONAL"

        return "LLM_FALLBACK"
    
    def _get_alice_response(self, user_input: str, route: str) -> tuple:
        """
        Get Alice's response for a given input and route
        
        Args:
            user_input: User input
            route: Determined route
        
        Returns:
            Tuple of (response_text, llm_used)
        """
        llm_used = False
        
        if route == "CONVERSATIONAL":
            # Process with NLP to get intent
            nlp_result = self.nlp.process(user_input)
            intent = nlp_result.intent if hasattr(nlp_result, 'intent') else "unknown"
            entities = nlp_result.entities if hasattr(nlp_result, 'entities') else {}
            
            # Try to generate from conversational engine
            from ai.conversational_engine import ConversationalContext
            context = ConversationalContext(
                user_input=user_input,
                intent=intent,
                entities=entities,
                recent_topics=[],
                active_goal=None,
                world_state=self.reasoning_engine
            )
            response = self.conversational_engine.generate_response(context)
            return response or "I understand.", llm_used
        
        elif route == "CLARIFICATION":
            return "Could you clarify what you mean? I want to make sure I help you correctly.", llm_used
        
        elif route == "TOOL":
            return "I'll help you with that.", llm_used
        
        elif route == "RAG":
            return "Let me check my memory for that information.", llm_used
        
        else:  # LLM_FALLBACK
            # Use gateway which enforces policy
            llm_used = True
            response = self.llm_gateway.request(
                prompt=user_input,
                call_type=LLMCallType.GENERATION,
                user_input=user_input
            )
            if response.success:
                return response.response, llm_used
            else:
                return "I'm not sure how to help with that.", llm_used
    
    def run_scenario(self, scenario: Scenario) -> List[ScenarioResult]:
        """
        Run a single scenario through all its steps
        
        Args:
            scenario: Scenario to run
        
        Returns:
            List of results for each step
        """
        logger.info(f"\n[SCENARIO] Running scenario: {scenario.name}")
        logger.info(f"   {scenario.description}")
        
        scenario_results = []
        
        for i, step in enumerate(scenario.steps, 1):
            logger.info(f"\n  Step {i}/{len(scenario.steps)}: '{step.user_input}'")
            
            # Process with NLP
            nlp_result = self.nlp.process(step.user_input)
            actual_intent = nlp_result.intent if hasattr(nlp_result, 'intent') else "unknown"
            
            # Map intent from NLP output to scenario expectation format
            mapped_intent = INTENT_MAPPING.get(actual_intent, actual_intent)
            
            # Determine route
            actual_route = self._determine_route(step.user_input, nlp_result)
            
            # Get Alice's response and track LLM usage
            alice_response, llm_used = self._get_alice_response(step.user_input, actual_route)
            
            # Compare with teacher if enabled
            teacher_response = None
            teacher_quality = None
            needs_learning = False
            
            if self.teacher:
                comparison = self.teacher.compare_responses(
                    user_input=step.user_input,
                    alice_response=alice_response,
                    expected_route=step.expected_route.value
                )
                teacher_response = comparison.get("teacher_response")
                teacher_quality = comparison.get("quality_score", 0)
                
                # Check if should flag for learning
                needs_learning = self.teacher.should_flag_for_learning(
                    alice_route=actual_route,
                    expected_route=step.expected_route.value,
                    comparison=comparison
                )
            
            # Determine success: route + intent match = success (or has teacher approval)
            success = (actual_route == step.expected_route.value and 
                      mapped_intent == step.expected_intent)
            
            # Create result
            result = ScenarioResult(
                step=step,
                actual_route=actual_route,
                actual_intent=actual_intent,
                actual_response=alice_response,
                teacher_response=teacher_response,
                route_match=(actual_route == step.expected_route.value),
                intent_match=(mapped_intent == step.expected_intent),
                needs_learning=needs_learning,
                confidence=nlp_result.intent_confidence if hasattr(nlp_result, 'intent_confidence') else 0.0
            )
            
            scenario_results.append(result)
            
            # Log result
            route_status = "OK" if result.route_match else "ERR"
            intent_status = "OK" if result.intent_match else "ERR"
            logger.info(f"    Route: {route_status} {actual_route} (expected: {step.expected_route.value})")
            logger.info(f"    Intent: {intent_status} {mapped_intent} (expected: {step.expected_intent})")
            logger.info(f"    Alice: {alice_response[:80]}...")
            
            if teacher_response:
                logger.info(f"    Teacher: {teacher_response[:80]}...")
                if needs_learning:
                    logger.info(f"    [LEARNING] Flagged for learning")
        
        return scenario_results

    def _derive_error_type(self, result: ScenarioResult) -> str:
        """Derive a simple error type classification for training logs."""
        if result.route_match and result.intent_match:
            return "bad_answer" if result.needs_learning else "none"

        if not result.intent_match:
            return "mis_intent"

        if not result.route_match:
            return "wrong_route"

        return "unknown"
    
    def run_all(
        self,
        domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Run all scenarios (optionally filtered by domain/tags)
        
        Args:
            domains: List of domains to run (None = all)
            tags: List of tags to filter by (None = all)
        """
        # Filter scenarios
        scenarios = ALL_SCENARIOS
        
        if domains:
            scenarios = [s for s in scenarios if s.domain in domains]
        
        if tags:
            scenarios = [s for s in scenarios if any(tag in s.tags for tag in tags)]
        
        logger.info(f"\n[RUN] Running {len(scenarios)} scenarios...\n")
        
        # Run each scenario
        for scenario in scenarios:
            results = self.run_scenario(scenario)
            self.results.extend(results)
            
            # Convert to training data with enhanced fields
            for result in results:
                error_type = self._derive_error_type(result)
                training_item = {
                    "timestamp": datetime.now().isoformat(),
                    "scenario_name": scenario.name,
                    "user_input": result.step.user_input,
                    "expected_intent": result.step.expected_intent,
                    "actual_intent": result.actual_intent,
                    "expected_route": result.step.expected_route.value,
                    "actual_route": result.actual_route,
                    "route": result.actual_route,
                    "alice_response": result.actual_response,
                    "teacher_response": result.teacher_response,
                    "route_match": result.route_match,
                    "intent_match": result.intent_match,
                    "success": result.route_match and result.intent_match,
                    "success_flag": result.route_match and result.intent_match,
                    "error_type": error_type,
                    "needs_learning": result.needs_learning,
                    "confidence": result.confidence,
                    "domain": scenario.domain,
                    "llm_used": result.actual_route == "LLM_FALLBACK"
                }
                self.training_data.append(training_item)
    
    def save_results(self) -> None:
        """Save results to training data file and generate report"""
        # Ensure data/training directory exists
        training_dir = Path("data/training")
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training data as JSONL
        output_file = training_dir / "auto_generated.jsonl"
        with open(output_file, "a") as f:
            for item in self.training_data:
                f.write(json.dumps(item) + "\n")
        
        logger.info(f"\n[SAVED] Saved {len(self.training_data)} interactions to {output_file}")
        
        # Generate report
        self._generate_report()
    
    def _generate_report(self) -> None:
        """Generate and display summary report"""
        if not self.results:
            logger.info("No results to report")
            return
        
        total = len(self.results)
        route_matches = sum(1 for r in self.results if r.route_match)
        intent_matches = sum(1 for r in self.results if r.intent_match)
        needs_learning = sum(1 for r in self.results if r.needs_learning)
        
        logger.info("\n" + "=" * 70)
        logger.info("[REPORT] SCENARIO RUNNER REPORT")
        logger.info("=" * 70)
        logger.info(f"Total Steps: {total}")
        logger.info(f"Route Accuracy: {route_matches}/{total} ({route_matches/total*100:.1f}%)")
        logger.info(f"Intent Accuracy: {intent_matches}/{total} ({intent_matches/total*100:.1f}%)")
        logger.info(f"Learning Opportunities: {needs_learning} ({needs_learning/total*100:.1f}%)")
        
        # Route distribution
        route_dist = {}
        for r in self.results:
            route_dist[r.actual_route] = route_dist.get(r.actual_route, 0) + 1
        
        logger.info("\nRoute Distribution:")
        for route, count in sorted(route_dist.items(), key=lambda x: -x[1]):
            logger.info(f"  {route}: {count} ({count/total*100:.1f}%)")
        
        # Domain breakdown
        domain_accuracy = {}
        for r in self.results:
            domain = r.step.domain or "unknown"
            if domain not in domain_accuracy:
                domain_accuracy[domain] = {"total": 0, "correct": 0}
            domain_accuracy[domain]["total"] += 1
            if r.route_match and r.intent_match:
                domain_accuracy[domain]["correct"] += 1
        
        logger.info("\nDomain Accuracy:")
        for domain, stats in sorted(domain_accuracy.items()):
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            logger.info(f"  {domain}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
        
        logger.info("=" * 70)


def main():
    """Main entry point for scenario runner"""
    parser = argparse.ArgumentParser(
        description="Run conversation scenarios for A.L.I.C.E testing and training"
    )
    parser.add_argument(
        '--policy',
        type=str,
        choices=['default', 'minimal', 'strict'],
        default='minimal',
        help='LLM policy mode'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1:8b',
        help='LLM model to use'
    )
    parser.add_argument(
        '--no-teacher',
        action='store_true',
        help='Disable teacher mode (faster but less learning data)'
    )
    parser.add_argument(
        '--domains',
        nargs='+',
        help='Only run scenarios from these domains'
    )
    parser.add_argument(
        '--tags',
        nargs='+',
        help='Only run scenarios with these tags'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = ScenarioRunner(
        llm_policy=args.policy,
        llm_model=args.model,
        use_teacher=not args.no_teacher
    )
    
    # Run scenarios
    runner.run_all(domains=args.domains, tags=args.tags)
    
    # Save results
    runner.save_results()


if __name__ == "__main__":
    main()
