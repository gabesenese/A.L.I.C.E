#!/usr/bin/env python3
"""Check training data structure"""

import json

with open('data/training/auto_generated.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            entry = json.loads(line)
            print('Entry', i+1)
            print('  user_input:', entry.get('user_input'))
            print('  teacher_response:', entry.get('teacher_response'))
            print('  alice_response:', entry.get('alice_response'))
            print('  route_match:', entry.get('route_match'))
            print('  intent_match:', entry.get('intent_match'))
            print()

# Count entries with teacher_response
count_with_teacher = 0
count_without_teacher = 0
with open('data/training/auto_generated.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get('teacher_response'):
                count_with_teacher += 1
            else:
                count_without_teacher += 1

print('Summary:')
print('  Entries with teacher_response:', count_with_teacher)
print('  Entries without teacher_response:', count_without_teacher)
