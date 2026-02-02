#!/usr/bin/env python3
"""
Generate learning scenarios from test errors
Creates training data so Alice can improve from mistakes
"""

import json
from pathlib import Path

print("📚 Generating learning scenarios from test errors...\n")

error_file = Path("data/training/comprehensive_test_errors.jsonl")
scenario_file = Path("data/training/auto_generated_corrections.jsonl")

if not error_file.exists():
    print(f"No errors file found at {error_file}")
    exit(1)

scenarios = []

with open(error_file, 'r') as f:
    errors = [json.loads(line) for line in f]

print(f"Found {len(errors)} errors to learn from\n")

# Generate correction scenarios
for error in errors:
    category = error.get("category", "unknown")
    query = error.get("query", "")
    response = error.get("response", "")
    error_type = "uncertain_response" if "unfortunately" in response.lower() or "not sure" in response.lower() else "wrong_context"
    
    # Create a corrected scenario
    if "Notes: List" in category:
        corrected_response = "Here are your notes:\n• meeting tomorrow\n• Important task\n• Follow up with team"
        correction_intent = "List user's notes with proper formatting"
    elif "Music: Play" in category:
        corrected_response = "I've started playing jazz music for you. Enjoy!"
        correction_intent = "Execute music playback command"
    else:
        corrected_response = response
        correction_intent = "Standard response"
    
    scenario = {
        "error_category": category,
        "user_input": query,
        "incorrect_response": response,
        "corrected_response": corrected_response,
        "error_type": error_type,
        "correction_intent": correction_intent,
        "context": "Learned from comprehensive test",
        "success": False,
        "should_learn": True
    }
    
    scenarios.append(scenario)
    
    # Print details
    print(f"📌 {category}")
    print(f"   Query: {query}")
    print(f"   Error Type: {error_type}")
    print(f"   Should respond: {corrected_response[:60]}...")
    print()

# Save scenarios
if scenarios:
    print(f"\n💾 Saving {len(scenarios)} correction scenarios...\n")
    
    scenario_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(scenario_file, 'w') as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + '\n')
    
    print(f"✓ Saved to {scenario_file}\n")
    
    # Also add to error log for scenario generator
    error_log = Path("data/training/auto_generated.jsonl")
    error_log.parent.mkdir(parents=True, exist_ok=True)
    
    with open(error_log, 'a') as f:
        for error in errors:
            f.write(json.dumps(error) + '\n')
    
    print(f"✓ Added to error log: {error_log}\n")

print("✓ Learning scenarios ready for next training cycle!")
