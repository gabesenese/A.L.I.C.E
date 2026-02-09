#!/usr/bin/env python3
"""Debug pattern promotion with updated logic"""

import json
from collections import defaultdict, Counter
from pathlib import Path

# Simulate the UPDATED analyze_logs logic
log_file = Path("data/training/auto_generated.jsonl")

groups = defaultdict(lambda: {
    "inputs": [],
    "teacher_responses": [],
    "alice_responses": [],
    "route_matches": [],
    "intent_matches": []
})

with open(log_file) as f:
    for line in f:
        try:
            entry = json.loads(line)
            
            # Skip if no teacher response
            if not entry.get("teacher_response"):
                continue
            
            # Group by (intent, domain)
            key = (entry.get("actual_intent"), entry.get("domain"))
            
            groups[key]["inputs"].append(entry.get("user_input", ""))
            groups[key]["teacher_responses"].append(entry.get("teacher_response", ""))
            groups[key]["alice_responses"].append(entry.get("alice_response", ""))
            groups[key]["route_matches"].append(entry.get("route_match", False))
            groups[key]["intent_matches"].append(entry.get("intent_match", False))
            
        except json.JSONDecodeError:
            continue

print("Groups analysis with NEW criteria:")
print("=" * 60)

min_frequency = 3
min_teacher_consistency = 0.8  # 80% teacher response rate
min_alice_agreement = 0.7  # 70% route+intent agreement

candidates = 0

for (intent, domain), data in groups.items():
    frequency = len(data["inputs"])
    
    # Check min_frequency
    if frequency < min_frequency:
        continue
    
    # Calculate teacher response rate
    teacher_response_rate = sum(
        1 for tr in data["teacher_responses"] if tr
    ) / frequency if frequency > 0 else 0.0
    
    # Check teacher response rate
    if teacher_response_rate < min_teacher_consistency:
        print(f"  ({intent}, {domain}): frequency={frequency}")
        print(f"    teacher_response_rate={teacher_response_rate:.1%} < 80% -> SKIP")
        continue
    
    # Calculate agreement
    correct_routes = sum(data["route_matches"])
    correct_intents = sum(data["intent_matches"])
    route_agreement = correct_routes / frequency if frequency > 0 else 0.0
    intent_agreement = correct_intents / frequency if frequency > 0 else 0.0
    avg_agreement = (route_agreement + intent_agreement) / 2
    
    # Check agreement
    if avg_agreement < min_alice_agreement:
        print(f"  ({intent}, {domain}): frequency={frequency}")
        print(f"    teacher_response_rate={teacher_response_rate:.1%} (OK)")
        print(f"    avg_agreement={avg_agreement:.1%} < 70% -> SKIP")
        continue
    
    # CANDIDATE!
    print(f"  ({intent}, {domain}): frequency={frequency}")
    print(f"    teacher_response_rate={teacher_response_rate:.1%}")
    print(f"    route_agreement={route_agreement:.1%}, intent_agreement={intent_agreement:.1%}")
    print(f"    avg_agreement={avg_agreement:.1%}")
    candidates += 1

print("=" * 60)
print(f"TOTAL CANDIDATES: {candidates}")

