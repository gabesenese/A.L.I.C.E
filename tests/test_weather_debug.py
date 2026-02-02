#!/usr/bin/env python3
"""Test weather follow-up with detailed tracing"""

from app.main import ALICE
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    filename='test_weather_debug.log',
    filemode='w'
)

# Also print to console for ERROR and above
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger('').addHandler(console)

alice = ALICE(debug=False)

print('\n' + '=' * 70)
print('TEST 1: Request weather forecast')
print('=' * 70)
r1 = alice.process_input('What is the weather forecast?', use_voice=False)
print(f'Response:\n{r1}\n')

import time
time.sleep(0.5)

print('=' * 70)
print('TEST 2: Follow-up with "is that wednesday?"')
print('=' * 70)
r2 = alice.process_input('is that wednesday?', use_voice=False)
print(f'Response:\n{r2}\n')

print('\nCheck test_weather_debug.log for detailed trace')
