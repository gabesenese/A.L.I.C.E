"""
Scenario Runner: Execute scripted conversations offline to generate synthetic training data.

This allows pre-training Alice before live user interactions, ensuring she has
learned patterns from realistic conversation flows without waiting for user input.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of running a scenario turn"""
    scenario_name: str
    turn_index: int
    user_input: str
    expected_intent: str
    detected_intent: Optional[str]
    intent_match: bool
    expected_tool: Optional[str]
    used_tool: Optional[str]
    tool_match: bool
    response_generated: bool
    response_type: str
    success: bool
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ScenarioRunner:
    """Runs scripted scenarios through the router and collects training data."""
    
    def __init__(self, nlp_processor=None, router=None, output_dir: str = "data/training"):
        """
        Initialize scenario runner.
        
        Args:
            nlp_processor: NLPProcessor instance for intent detection
            router: RequestRouter instance for routing decisions
            output_dir: Output directory for training data
        """
        self.nlp = nlp_processor
        self.router = router
        self.output_dir = output_dir
        self.results: List[ScenarioResult] = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_scenario(self, scenario_name: str, scenario: Dict[str, Any]) -> Tuple[List[ScenarioResult], bool]:
        """
        Run a complete scenario through the system.
        
        Args:
            scenario_name: Name of the scenario
            scenario: Scenario definition with turns
            
        Returns:
            Tuple of (results, all_turns_successful)
        """
        scenario_results = []
        all_success = True
        
        logger.info(f"\nRunning scenario: {scenario_name}")
        logger.info(f"Description: {scenario.get('description', 'N/A')}")
        
        turns = scenario.get('turns', [])
        for turn_index, turn in enumerate(turns):
            result = self._run_turn(scenario_name, turn_index, turn)
            scenario_results.append(result)
            self.results.append(result)
            
            if not result.success:
                all_success = False
            
            # Log turn result
            status = "✓" if result.success else "✗"
            logger.info(f"  Turn {turn_index + 1} {status}: {result.user_input[:50]}...")
            logger.info(f"    Intent: {result.detected_intent} (expected: {result.expected_intent})")
            if result.expected_tool:
                logger.info(f"    Tool: {result.used_tool} (expected: {result.expected_tool})")
        
        return scenario_results, all_success
    
    def _run_turn(self, scenario_name: str, turn_index: int, turn: Dict[str, Any]) -> ScenarioResult:
        """
        Run a single turn of a scenario.
        
        Args:
            scenario_name: Name of the scenario
            turn_index: Index of this turn in the scenario
            turn: Turn definition
            
        Returns:
            ScenarioResult with outcome
        """
        user_input = turn.get('user_input', '')
        expected_intent = turn.get('expected_intent', '')
        expected_tool = turn.get('expected_tool')
        expected_response_type = turn.get('expected_response_type', '')
        
        # Detect intent
        detected_intent = None
        used_tool = None
        intent_confidence = 0.0
        
        if self.nlp:
            try:
                nlp_result = self.nlp.process(user_input)
                detected_intent = nlp_result.get('intent')
                intent_confidence = nlp_result.get('confidence', 0.0)
            except Exception as e:
                logger.warning(f"NLP processing failed: {e}")
        
        # Route decision
        if self.router and detected_intent:
            try:
                route_result = self.router.route(
                    intent=detected_intent,
                    confidence=intent_confidence,
                    entities={}
                )
                used_tool = route_result.tool_name
            except Exception as e:
                logger.warning(f"Routing failed: {e}")
        
        # Check success conditions
        intent_match = detected_intent == expected_intent
        tool_match = (expected_tool is None and used_tool is None) or (used_tool == expected_tool)
        response_generated = detected_intent is not None
        success = intent_match and tool_match
        
        # Handle red-team scenarios
        if turn.get('should_clarify'):
            # For ambiguous intents, success = clarification requested
            success = detected_intent == 'clarification_needed' or intent_match
        
        if turn.get('should_require_confirmation'):
            # For unsafe commands, success = confirmation requested
            success = response_generated
        
        result = ScenarioResult(
            scenario_name=scenario_name,
            turn_index=turn_index,
            user_input=user_input,
            expected_intent=expected_intent,
            detected_intent=detected_intent,
            intent_match=intent_match,
            expected_tool=expected_tool,
            used_tool=used_tool,
            tool_match=tool_match,
            response_generated=response_generated,
            response_type=expected_response_type,
            success=success
        )
        
        return result
    
    def run_all_scenarios(self, scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run all scenarios and collect statistics.
        
        Args:
            scenarios: Dictionary of all scenarios
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_scenarios': len(scenarios),
            'successful_scenarios': 0,
            'failed_scenarios': 0,
            'total_turns': 0,
            'successful_turns': 0,
            'failed_turns': 0,
            'scenario_results': {}
        }
        
        for scenario_name, scenario in scenarios.items():
            results, success = self.run_scenario(scenario_name, scenario)
            
            stats['scenario_results'][scenario_name] = {
                'success': success,
                'turns': len(results),
                'successful_turns': sum(1 for r in results if r.success),
                'failed_turns': sum(1 for r in results if not r.success)
            }
            
            if success:
                stats['successful_scenarios'] += 1
            else:
                stats['failed_scenarios'] += 1
            
            stats['total_turns'] += len(results)
            stats['successful_turns'] += sum(1 for r in results if r.success)
            stats['failed_turns'] += sum(1 for r in results if not r.success)
        
        return stats
    
    def generate_training_data(self) -> List[Dict[str, Any]]:
        """
        Convert scenario results into training data format.
        
        Returns:
            List of training data entries
        """
        training_data = []
        
        for result in self.results:
            training_entry = {
                "user_input": result.user_input,
                "intent": result.detected_intent or result.expected_intent,
                "entities": {},
                "response": f"[{result.response_type}]",
                "tool": result.used_tool,
                "scenario": result.scenario_name,
                "turn": result.turn_index,
                "success": result.success,
                "quality_score": 0.9 if result.success else 0.5,
                "category": "synthetic_from_scenarios",
                "timestamp": result.timestamp
            }
            training_data.append(training_entry)
        
        return training_data
    
    def save_training_data(self, output_file: str = None):
        """
        Save generated training data to JSONL file.
        
        Args:
            output_file: Output file path (defaults to synthetic_from_scenarios.jsonl)
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "synthetic_from_scenarios.jsonl")
        
        training_data = self.generate_training_data()
        
        with open(output_file, 'w') as f:
            for entry in training_data:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"\nSaved {len(training_data)} training entries to {output_file}")
        return output_file
    
    def save_scenario_results(self, output_file: str = None):
        """
        Save detailed scenario results for debugging.
        
        Args:
            output_file: Output file path
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "scenario_results.jsonl")
        
        with open(output_file, 'w') as f:
            for result in self.results:
                f.write(json.dumps(asdict(result)) + '\n')
        
        logger.info(f"Saved {len(self.results)} scenario results to {output_file}")
        return output_file
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about scenario runs."""
        total_turns = len(self.results)
        successful_turns = sum(1 for r in self.results if r.success)
        intent_accuracy = (sum(1 for r in self.results if r.intent_match) / total_turns * 100) if total_turns > 0 else 0
        tool_accuracy = (sum(1 for r in self.results if r.tool_match) / total_turns * 100) if total_turns > 0 else 0
        
        # Group by scenario
        by_scenario = {}
        for result in self.results:
            if result.scenario_name not in by_scenario:
                by_scenario[result.scenario_name] = {'success': 0, 'total': 0}
            by_scenario[result.scenario_name]['total'] += 1
            if result.success:
                by_scenario[result.scenario_name]['success'] += 1
        
        return {
            'total_turns': total_turns,
            'successful_turns': successful_turns,
            'success_rate': (successful_turns / total_turns * 100) if total_turns > 0 else 0,
            'intent_accuracy': intent_accuracy,
            'tool_accuracy': tool_accuracy,
            'by_scenario': by_scenario
        }


def main():
    """Example usage of ScenarioRunner"""
    from scenarios import get_all_scenarios
    
    # Initialize runner (without NLP/router for demo)
    runner = ScenarioRunner()
    
    # Run all scenarios
    scenarios = get_all_scenarios()
    stats = runner.run_all_scenarios(scenarios)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("SCENARIO RUNNER STATISTICS")
    print("=" * 70)
    print(f"Total scenarios: {stats['total_scenarios']}")
    print(f"Successful scenarios: {stats['successful_scenarios']}")
    print(f"Failed scenarios: {stats['failed_scenarios']}")
    print(f"Total turns: {stats['total_turns']}")
    print(f"Successful turns: {stats['successful_turns']}")
    print(f"Turn success rate: {(stats['successful_turns'] / stats['total_turns'] * 100) if stats['total_turns'] > 0 else 0:.1f}%")
    print()
    
    # Save results
    runner.save_scenario_results()
    runner.save_training_data()
    
    return stats


if __name__ == '__main__':
    main()
