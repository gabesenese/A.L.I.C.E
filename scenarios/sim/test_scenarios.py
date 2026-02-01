"""
Quick test of the simulation system
"""

import sys
import os

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SIM_DIR))
sys.path.insert(0, PROJECT_ROOT)

from .scenarios import (
    ALL_SCENARIOS, 
    get_scenarios_by_domain,
    get_scenarios_by_tag,
    ExpectedRoute
)

def test_scenarios():
    """Test scenario definitions"""
    print("=" * 70)
    print("Testing Scenario Definitions")
    print("=" * 70)
    
    # Count scenarios
    print(f"\nTotal scenarios: {len(ALL_SCENARIOS)}")
    
    # Count by domain
    domains = {}
    for s in ALL_SCENARIOS:
        domains[s.domain] = domains.get(s.domain, 0) + 1
    
    print("\nScenarios by domain:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")
    
    # Count total steps
    total_steps = sum(len(s.steps) for s in ALL_SCENARIOS)
    print(f"\nTotal test steps: {total_steps}")
    
    # Test filtering
    email_scenarios = get_scenarios_by_domain("email")
    print(f"\nEmail scenarios: {len(email_scenarios)}")
    
    clarification_scenarios = get_scenarios_by_tag("clarification")
    print(f"Clarification scenarios: {len(clarification_scenarios)}")
    
    # Show a sample scenario
    print("\n" + "=" * 70)
    print("Sample Scenario: Vague Sun Question")
    print("=" * 70)
    
    for scenario in ALL_SCENARIOS:
        if "sun" in scenario.name.lower():
            print(f"\nName: {scenario.name}")
            print(f"Description: {scenario.description}")
            print(f"Domain: {scenario.domain}")
            print(f"Steps: {len(scenario.steps)}")
            
            for i, step in enumerate(scenario.steps, 1):
                print(f"\n  Step {i}:")
                print(f"    Input: '{step.user_input}'")
                print(f"    Expected Intent: {step.expected_intent}")
                print(f"    Expected Route: {step.expected_route.value}")
                print(f"    Notes: {step.notes}")
            break
    
    print("\n✅ Scenario definitions test passed!\n")


if __name__ == "__main__":
    test_scenarios()
