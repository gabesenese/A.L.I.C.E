#!/usr/bin/env python3
"""Debug the import process"""

import os
import json
from collections import defaultdict
from datetime import datetime

# Read existing corrections
corrections_file = os.path.join("memory", "corrections.json")
if os.path.exists(corrections_file):
    with open(corrections_file, 'r', encoding='utf-8') as f:
        existing = json.load(f)
        print(f"Existing corrections in memory/corrections.json: {len(existing)}")
        existing_ids = {c.get('id') for c in existing}
        print(f"Existing IDs sample: {list(existing_ids)[:3]}")

# Now check import logic
training_file = os.path.join("data", "training", "auto_generated.jsonl")
if os.path.exists(training_file):
    existing_ids_in_import = existing_ids.copy()
    imported_count = 0
    
    with open(training_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entry_id = f"training_{entry.get('user_input', '')}_{entry.get('timestamp', '')}"[:80]
                
                # Skip if already imported
                if entry_id in existing_ids_in_import:
                    continue
                
                # Would be imported
                imported_count += 1
                existing_ids_in_import.add(entry_id)
                
                if imported_count <= 3:
                    print(f"Would import: {entry_id}")
                    
            except Exception as e:
                pass
    
    print(f"Would import: {imported_count} new corrections")
