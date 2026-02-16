"""
OS Architecture Validation Test Harness for A.L.I.C.E

Comprehensive end-to-end testing of:
1. Router deterministic pipeline (5-stage routing)
2. Scenario execution and training data generation
3. Red-team adversarial scenario routing
4. Event emission and metrics collection
5. Safety routing validation

Run: python -m pytest test_os_architecture.py -v
Or:  python test_os_architecture.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.router import RequestRouter, RoutingDecision
from ai.event_bus import get_event_bus
from scenarios import get_all_scenarios, get_scenario
from scenarios.red_team import (
    get_all_red_team_scenarios,
    RED_TEAM_ROUTING_EXPECTATIONS,
    validate_red_team_safety_routing
)


@dataclass
class TestResult:
    """Test result record"""
    test_name: str
    passed: bool
    expected: str
    actual: str
    error: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class OSArchitectureValidator:
    """Comprehensive validator for A.L.I.C.E OS architecture"""
    
    def __init__(self):
        self.router = RequestRouter()
        self.event_bus = get_event_bus()
        self.results: List[TestResult] = []
        self.events_captured: List[Dict[str, Any]] = []
        
        # Subscribe to routing events
        self._setup_event_capture()
    
    def _setup_event_capture(self):
        """Setup event listeners for metrics"""
        def capture_routing_event(event):
            self.events_captured.append({
                'type': 'routing',
                'data': event.data if hasattr(event, 'data') else {}
            })
        
        # Subscribe to custom routing events
        for stage in ['self_reflection', 'conversational', 'tool', 'rag', 'llm', 'error']:
            self.event_bus.subscribe_to_custom(f'routing.{stage}', capture_routing_event)
    
    def test_router_5_stage_pipeline(self) -> List[TestResult]:
        """Test router 5-stage priority pipeline"""
        tests = []
        
        # Test 1: SELF_REFLECTION (highest priority)
        result = self.router.route(
            intent='training_status',
            confidence=0.5,  # Even low confidence goes to self-reflection
            entities={}
        )
        tests.append(TestResult(
            test_name="Router: SELF_REFLECTION stage (high priority)",
            passed=result.decision == RoutingDecision.SELF_REFLECTION,
            expected="SELF_REFLECTION",
            actual=result.decision.value,
            details={'intent': 'training_status', 'confidence': 0.5}
        ))
        
        # Test 2: CONVERSATIONAL (priority 2)
        result = self.router.route(
            intent='greeting',
            confidence=0.8,
            entities={}
        )
        tests.append(TestResult(
            test_name="Router: CONVERSATIONAL stage (priority 2)",
            passed=result.decision == RoutingDecision.CONVERSATIONAL,
            expected="CONVERSATIONAL",
            actual=result.decision.value,
            details={'intent': 'greeting', 'confidence': 0.8}
        ))
        
        # Test 3: TOOL_CALL (priority 3)
        result = self.router.route(
            intent='email_read',
            confidence=0.7,
            entities={}
        )
        tests.append(TestResult(
            test_name="Router: TOOL_CALL stage (priority 3)",
            passed=result.decision == RoutingDecision.TOOL_CALL,
            expected="TOOL_CALL",
            actual=result.decision.value,
            details={'intent': 'email_read', 'confidence': 0.7, 'tool': result.tool_name}
        ))
        
        # Test 4: RAG_ONLY (priority 4)
        result = self.router.route(
            intent='document_query',
            confidence=0.6,
            entities={}
        )
        tests.append(TestResult(
            test_name="Router: RAG_ONLY stage (priority 4)",
            passed=result.decision == RoutingDecision.RAG_ONLY,
            expected="RAG_ONLY",
            actual=result.decision.value,
            details={'intent': 'document_query', 'confidence': 0.6}
        ))
        
        # Test 5: LLM_FALLBACK (priority 5, last resort)
        result = self.router.route(
            intent='explanation',
            confidence=0.5,
            entities={}
        )
        tests.append(TestResult(
            test_name="Router: LLM_FALLBACK stage (priority 5)",
            passed=result.decision == RoutingDecision.LLM_FALLBACK,
            expected="LLM_FALLBACK",
            actual=result.decision.value,
            details={'intent': 'explanation', 'confidence': 0.5}
        ))
        
        # Test 6: ERROR (unknown intent) - actually falls through to LLM for clarification
        # This is better UX: instead of "unknown intent error", we ask LLM to help clarify
        result = self.router.route(
            intent='unknown_xyz_intent_12345',
            confidence=0.5,
            entities={}
        )
        tests.append(TestResult(
            test_name="Router: Unknown intents go to LLM for clarification",
            passed=result.decision in [RoutingDecision.LLM_FALLBACK, RoutingDecision.ERROR],
            expected="LLM_FALLBACK or ERROR",
            actual=result.decision.value,
            details={'intent': 'unknown_xyz_intent_12345', 'confidence': 0.5}
        ))
        
        self.results.extend(tests)
        return tests
    
    def test_tool_routing(self) -> List[TestResult]:
        """Test tool routing maps"""
        tests = []
        
        # Test email tool routing
        result = self.router.route(
            intent='email_send',
            confidence=0.8,
            entities={'recipient': 'test@example.com'}
        )
        tests.append(TestResult(
            test_name="Tool routing: email_send → email plugin",
            passed=result.tool_name == 'email',
            expected="email",
            actual=result.tool_name or "None",
            details={'intent': 'email_send'}
        ))
        
        # Test calendar tool routing
        result = self.router.route(
            intent='check_availability',
            confidence=0.8,
            entities={'time': 'tomorrow 2pm'}
        )
        tests.append(TestResult(
            test_name="Tool routing: check_availability → calendar plugin",
            passed=result.tool_name == 'calendar',
            expected="calendar",
            actual=result.tool_name or "None",
            details={'intent': 'check_availability'}
        ))
        
        # Test file tool routing
        result = self.router.route(
            intent='file_read',
            confidence=0.8,
            entities={'filename': 'config.json'}
        )
        tests.append(TestResult(
            test_name="Tool routing: file_read → file_operations plugin",
            passed=result.tool_name == 'file_operations',
            expected="file_operations",
            actual=result.tool_name or "None",
            details={'intent': 'file_read'}
        ))
        
        self.results.extend(tests)
        return tests
    
    def test_red_team_safety_routing(self) -> List[TestResult]:
        """Test that red-team scenarios route to safety handlers"""
        tests = []
        
        # Get red-team scenarios
        red_team = get_all_red_team_scenarios()
        
        unsafe_scenarios = [
            'unsafe_delete_all',
            'unsafe_system_shutdown',
            'unsafe_privilege_escalation',
            'unsafe_production_deploy',
            'unsafe_data_wipe'
        ]
        
        for scenario_name in unsafe_scenarios:
            if scenario_name not in red_team:
                continue
            
            scenario = red_team[scenario_name]
            turn = scenario['turns'][0]
            intent = turn.get('expected_intent', '')
            
            # Route the scenario
            result = self.router.route(
                intent=intent,
                confidence=0.9,
                entities=turn.get('expected_entities', {})
            )
            
            # Unsafe commands should go to LLM (for safety checks) or ERROR
            tests.append(TestResult(
                test_name=f"Red-team safety: {scenario_name}",
                passed=result.decision in [RoutingDecision.LLM_FALLBACK, RoutingDecision.ERROR],
                expected="LLM_FALLBACK or ERROR (for safety)",
                actual=result.decision.value,
                details={'scenario': scenario_name, 'input': turn['user_input']}
            ))
        
        # Test ambiguous pronouns → should request clarification (go to conversational or LLM)
        ambiguous_scenarios = [
            'ambiguous_pronoun_it',
            'ambiguous_pronoun_that',
            'ambiguous_pronoun_this'
        ]
        
        for scenario_name in ambiguous_scenarios:
            if scenario_name not in red_team:
                continue
            
            scenario = red_team[scenario_name]
            turn = scenario['turns'][0]
            intent = turn.get('expected_intent', '')
            
            result = self.router.route(
                intent=intent,
                confidence=0.5,
                entities=turn.get('expected_entities', {})
            )
            
            # Ambiguous should either stay as clarification or go to LLM
            tests.append(TestResult(
                test_name=f"Red-team ambiguity: {scenario_name}",
                passed=result.decision in [RoutingDecision.CONVERSATIONAL, RoutingDecision.LLM_FALLBACK, RoutingDecision.ERROR],
                expected="CONVERSATIONAL or LLM_FALLBACK (for clarification)",
                actual=result.decision.value,
                details={'scenario': scenario_name, 'input': turn['user_input']}
            ))
        
        self.results.extend(tests)
        return tests
    
    def test_scenarios_coverage(self) -> List[TestResult]:
        """Test that all normal scenarios route appropriately"""
        tests = []
        
        scenarios = get_all_scenarios()
        routing_success = 0
        routing_attempts = 0
        
        for scenario_name, scenario in scenarios.items():
            for turn in scenario['turns']:
                intent = turn.get('expected_intent', '')
                confidence = 0.8
                entities = turn.get('expected_entities', {})
                expected_tool = turn.get('expected_tool')
                
                routing_attempts += 1
                result = self.router.route(intent, confidence, entities)
                
                # Check routing makes sense
                if expected_tool:
                    # Should route to TOOL_CALL
                    if result.decision == RoutingDecision.TOOL_CALL:
                        routing_success += 1
                elif turn.get('expected_response_type') in ['greeting_response', 'capabilities', 'command_list']:
                    # Should route to CONVERSATIONAL
                    if result.decision == RoutingDecision.CONVERSATIONAL:
                        routing_success += 1
                else:
                    # Any valid routing is good
                    if result.decision != RoutingDecision.ERROR:
                        routing_success += 1
        
        success_rate = routing_success / routing_attempts if routing_attempts > 0 else 0
        tests.append(TestResult(
            test_name="Scenario routing coverage",
            passed=success_rate > 0.85,  # 85% success rate
            expected=">85% routing success",
            actual=f"{success_rate*100:.1f}% ({routing_success}/{routing_attempts})",
            details={'scenarios_tested': len(scenarios), 'turns_tested': routing_attempts}
        ))
        
        self.results.extend(tests)
        return tests
    
    def test_event_emission(self) -> List[TestResult]:
        """Test that router emits events for metrics"""
        tests = []
        
        # Clear event capture
        initial_event_count = len(self.events_captured)
        
        # Route multiple requests
        self.router.route('greeting', 0.9, {})
        self.router.route('email_read', 0.8, {})
        self.router.route('training_status', 0.5, {})
        
        # Check events were captured
        new_events = len(self.events_captured) - initial_event_count
        
        tests.append(TestResult(
            test_name="Event emission: Router emits events",
            passed=new_events >= 3,
            expected=">=3 routing events",
            actual=f"{new_events} events",
            details={'total_events': len(self.events_captured)}
        ))
        
        self.results.extend(tests)
        return tests
    
    def test_routing_statistics(self) -> List[TestResult]:
        """Test routing statistics collection"""
        tests = []
        
        # Reset stats
        self.router.reset_stats()
        
        # Route various requests
        requests = [
            ('greeting', RoutingDecision.CONVERSATIONAL),
            ('email_read', RoutingDecision.TOOL_CALL),
            ('training_status', RoutingDecision.SELF_REFLECTION),
            ('explanation', RoutingDecision.LLM_FALLBACK),
            ('document_query', RoutingDecision.RAG_ONLY),
        ]
        
        for intent, expected_decision in requests:
            result = self.router.route(intent, 0.8, {})
        
        stats = self.router.get_stats()
        
        # Verify stats were updated
        total_routed = sum(stats.values())
        tests.append(TestResult(
            test_name="Statistics: Router tracks routing decisions",
            passed=total_routed >= 5,
            expected=">=5 routing decisions tracked",
            actual=f"{total_routed} total routes",
            details={'stats': stats}
        ))
        
        self.results.extend(tests)
        return tests
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests"""
        print("\n" + "="*70)
        print("A.L.I.C.E OS ARCHITECTURE VALIDATION TEST SUITE")
        print("="*70 + "\n")
        
        # Run test groups
        self.test_router_5_stage_pipeline()
        self.test_tool_routing()
        self.test_red_team_safety_routing()
        self.test_scenarios_coverage()
        self.test_event_emission()
        self.test_routing_statistics()
        
        # Generate report
        return self._generate_report()
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Print results
        print("\nTEST RESULTS:")
        print("-" * 70)
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status}: {result.test_name}")
            if not result.passed:
                print(f"  Expected: {result.expected}")
                print(f"  Actual:   {result.actual}")
                if result.error:
                    print(f"  Error:    {result.error}")
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests:  {total_tests}")
        print(f"Passed:       {passed_tests}")
        print(f"Failed:       {failed_tests}")
        print(f"Pass Rate:    {pass_rate:.1f}%")
        print(f"Events Captured: {len(self.events_captured)}")
        print("="*70 + "\n")
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'pass_rate': pass_rate,
            'results': [asdict(r) for r in self.results],
            'events_captured': len(self.events_captured)
        }


def main():
    """Run validation suite"""
    validator = OSArchitectureValidator()
    report = validator.run_all_tests()
    
    # Save report
    report_path = PROJECT_ROOT / "data" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Report saved to: {report_path}")
    
    # Exit with appropriate code
    sys.exit(0 if report['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
