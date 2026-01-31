"""
Run the offline training loop (train from logs, evaluate, update thresholds, deploy).
Call this daily or weekly, e.g. via cron or Task Scheduler.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.training_system import run_offline_loop

if __name__ == "__main__":
    result = run_offline_loop(min_examples=20)
    print("Offline loop result:", result)