#!/usr/bin/env python3
"""Count scenarios"""

import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scenarios.sim.scenarios import ALL_SCENARIOS

print(f'Total scenarios: {len(ALL_SCENARIOS)}')
print('By domain:')
domains = collections.Counter([s.domain for s in ALL_SCENARIOS])
for d, count in sorted(domains.items()):
    print(f'  {d}: {count}')
    
print('\nBy tags:')
tags = collections.Counter()
for s in ALL_SCENARIOS:
    for tag in s.tags:
        tags[tag] += 1
for tag, count in tags.most_common(10):
    print(f'  {tag}: {count}')
