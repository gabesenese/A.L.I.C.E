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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.scenarios import (
    Scenario, ScenarioStep, ScenarioResult, ExpectedRoute,
    ALL_SCENARIOS, get_scenarios_by_domain, get_scenarios_by_tag
)
from sim.teacher import TeacherMode

# Import Alice components (without full initialization)
from ai.nlp_processor import NLPProcessor
from ai.conversational_engine import get_conversational_engine
from ai.llm_gateway import get_llm_gateway
from ai.llm_engine import LocalLLMEngine, LLMConfig
from ai.context_engine import get_context_engine
from ai.memory_system import MemorySystem
from ai.reasoning_engine import get_reasoning_engine
from ai.learning_engine import get_learning_engine
from ai.llm_policy import configure_minimal_policy, get_llm_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.info("✓ Minimal policy configured")
        
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
        
        logger.info("✓ Core components ready\n")
        
        # Results storage
        self.results: List[ScenarioResult] = []
        self.training_data: List[Dict[str, Any]] = []
    
    def _determine_route(self, user_input: str, nlp_result: Dict[str, Any]) -> str:
        """
        Determine Alice's routing decision for input
        
        Args:
            user_input: User's input text
            nlp_result: NLP processing result
        
        Returns:
            Route name (CONVERSATIONAL/TOOL/RAG/LLM_FALLBACK/CLARIFICATION)
        """
        # Check conversational engine first
        conv_response = self.conversational_engine.process(
            user_input=user_input,
            context={"nlp_result": nlp_result}
        )
        
        if conv_response and conv_response.get("handled"):
            return "CONVERSATIONAL"
        
        # Check for clarification need (vague patterns)
        intent = nlp_result.get("intent", "unknown")
        if "vague" in intent or "unclear" in intent:
            return "CLARIFICATION"
        
        # Check if it's a tool request
        if intent in ["list_emails", "search_emails", "read_email", "delete_email",
                      "create_note", "search_notes", "list_notes",
                      "get_weather", "get_time"]:
            return "TOOL"
        
        # Check for RAG/memory queries
        if "remember" in user_input.lower() or "recall" in user_input.lower():
            return "RAG"
        
        # Default to LLM fallback
        return "LLM_FALLBACK"
    
    def _get_alice_response(self, user_input: str, route: str) -> str:
        """
        Get Alice's response for a given input and route
        
        Args:
            user_input: User input
            route: Determined route
        
        Returns:
            Alice's response text
        """
        if route == "CONVERSATIONAL":
            nlp_result = self.nlp.process(user_input)
            conv_result = self.conversational_engine.process(
                user_input=user_input,
                context={"nlp_result": nlp_result}
            )
            return conv_result.get("response", "I understand.")
        
        elif route == "CLARIFICATION":
            return "Could you clarify what you mean? I want to make sure I help you correctly."
        
        elif route == "TOOL":
            return "I'll help you with that." # Placeholder - actual tool would be called
        
        elif route == "RAG":
            return "Let me check my memory for that information."
        
        else:  # LLM_FALLBACK
            # Use gateway which enforces policy
            response = self.llm_gateway.chat_completion(
                messages=[{"role": "user", "content": user_input}],
                call_type="fallback",
                context={"route": route}
            )
            return response.get("content", "I'm not sure how to help with that.")
    
    def run_scenario(self, scenario: Scenario) -> List[ScenarioResult]:
        """
        Run a single scenario through all its steps
        
        Args:
            scenario: Scenario to run
        
        Returns:
            List of results for each step
        """
        logger.info(f"\n📋 Running scenario: {scenario.name}")
        logger.info(f"   {scenario.description}")
        
        scenario_results = []
        
        for i, step in enumerate(scenario.steps, 1):
            logger.info(f"\n  Step {i}/{len(scenario.steps)}: '{step.user_input}'")
            
            # Process with NLP
            nlp_result = self.nlp.process(step.user_input)
            actual_intent = nlp_result.get("intent", "unknown")
            
            # Determine route
            actual_route = self._determine_route(step.user_input, nlp_result)
            
            # Get Alice's response
            alice_response = self._get_alice_response(step.user_input, actual_route)
            
            # Compare with teacher if enabled
            teacher_response = None
            needs_learning = False
            
            if self.teacher:
                comparison = self.teacher.compare_responses(
                    user_input=step.user_input,
                    alice_response=alice_response,
                    expected_route=step.expected_route.value
                )
                teacher_response = comparison.get("teacher_response")
                
                # Check if should flag for learning
                needs_learning = self.teacher.should_flag_for_learning(
                    alice_route=actual_route,
                    expected_route=step.expected_route.value,
                    comparison=comparison
                )
            
            # Create result
            result = ScenarioResult(
                step=step,
                actual_route=actual_route,
                actual_intent=actual_intent,
                actual_response=alice_response,
                teacher_response=teacher_response,
                route_match=(actual_route == step.expected_route.value),
                intent_match=(actual_intent == step.expected_intent),
                needs_learning=needs_learning,
                confidence=nlp_result.get("confidence", 0.0)
            )
            
            scenario_results.append(result)
            
            # Log result
            route_status = "✓" if result.route_match else "✗"
            intent_status = "✓" if result.intent_match else "✗"
            logger.info(f"    Route: {route_status} {actual_route} (expected: {step.expected_route.value})")
            logger.info(f"    Intent: {intent_status} {actual_intent} (expected: {step.expected_intent})")
            logger.info(f"    Alice: {alice_response[:80]}...")
            
            if teacher_response:
                logger.info(f"    Teacher: {teacher_response[:80]}...")
                if needs_learning:
                    logger.info(f"    🎓 Flagged for learning")
        
        return scenario_results
    
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
        
        logger.info(f"\n🚀 Running {len(scenarios)} scenarios...\n")
        
        # Run each scenario
        for scenario in scenarios:
            results = self.run_scenario(scenario)
            self.results.extend(results)
            
            # Convert to training data
            for result in results:
                self.training_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "scenario_name": scenario.name,
                    "user_input": result.step.user_input,
                    "expected_intent": result.step.expected_intent,
                    "actual_intent": result.actual_intent,
                    "expected_route": result.step.expected_route.value,
                    "actual_route": result.actual_route,
                    "alice_response": result.actual_response,
                    "teacher_response": result.teacher_response,
                    "route_match": result.route_match,
                    "intent_match": result.intent_match,
                    "needs_learning": result.needs_learning,
                    "domain": scenario.domain
                })
    
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
        
        logger.info(f"\n💾 Saved {len(self.training_data)} interactions to {output_file}")
        
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
        logger.info("📊 SCENARIO RUNNER REPORT")
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
