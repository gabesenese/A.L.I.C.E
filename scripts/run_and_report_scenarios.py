#!/usr/bin/env python3
"""Run scenarios and extract accuracy report"""

import subprocess
import sys
import re
from pathlib import Path

# Run scenarios
project_root = Path(__file__).resolve().parent.parent
result = subprocess.run(
    [sys.executable, "-m", "scenarios.sim.run_scenarios", "--policy", "minimal"],
    capture_output=True,
    text=True,
    cwd=project_root
)

output = result.stdout + result.stderr

# Extract report section
lines = output.split('\n')
in_report = False
report_lines = []

for line in lines:
    if '[REPORT]' in line:
        in_report = True
    if in_report:
        report_lines.append(line)
    if in_report and '=======' in line and report_lines.index(line) > 5:
        break

# Print report
print('\n'.join(report_lines))
