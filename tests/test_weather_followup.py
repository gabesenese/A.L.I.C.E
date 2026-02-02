#!/usr/bin/env python3
"""
Test weather forecast follow-up handling
Tests the fix for: "is that wednesday?" not returning Wednesday's forecast
"""

from app.main import ALICE
import logging
import time

# Suppress verbose logging
logging.getLogger('ai').setLevel(logging.WARNING)
logging.getLogger('app.main').setLevel(logging.WARNING)

# Initialize Alice
print('\n🚀 Initializing A.L.I.C.E...')
alice = ALICE(debug=False)
print('✓ A.L.I.C.E ready!\n')

# Test 1: Ask for forecast
print('═' * 70)
print('TEST 1: Asking for weather forecast for tomorrow')
print('═' * 70)
response1 = alice.process_input('What is the weather forecast for tomorrow?', use_voice=False)
print(f'Response:\n{response1}\n')

# Small delay to ensure entity is stored
time.sleep(0.5)

# Test 2: Ask about specific day (THE PROBLEMATIC QUERY)
print('═' * 70)
print('TEST 2: Follow-up - "is that wednesday?"')
print('Expected: Should return Wednesday forecast from stored data')
print('Previous behavior: Returned current weather instead')
print('═' * 70)
response2 = alice.process_input('is that wednesday?', use_voice=False)
print(f'Response:\n{response2}\n')

# Verify it's different from current weather
if 'wednesday' in response2.lower() or '2026-02-' in response2:
    print('✓ SUCCESS: Response mentions Wednesday or has a date (forecast returned!)')
else:
    print('⚠ Response may not be from forecast data')

# Test 3: Another weekday follow-up
print('\n' + '═' * 70)
print('TEST 3: Follow-up - "what about thursday?"')
print('═' * 70)
response3 = alice.process_input('what about thursday?', use_voice=False)
print(f'Response:\n{response3}\n')

# Test 4: Verify intent detection
print('═' * 70)
print('TEST 4: Verify NLP intent detection for weekday queries')
print('═' * 70)
from ai.nlp_processor import NLPProcessor
nlp = NLPProcessor()

test_inputs = [
    'is that wednesday?',
    'what about thursday?',
    'what is friday like?',
    'how about monday?'
]

for test_input in test_inputs:
    result = nlp.process(test_input)
    print(f'  Input: "{test_input}"')
    print(f'    → Intent: {result.intent} (confidence: {result.confidence:.2f})')
    print()

print('═' * 70)
print('✓ Test completed')
print('═' * 70)
