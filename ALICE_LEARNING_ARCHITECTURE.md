
Alice Learning Architecture - Putting Knowledge to Work
========================================================

Problem: Hard-coded plugin messages prevent Alice from learning

Solution: Data-driven response formulation with progressive learning


## ARCHITECTURE OVERVIEW

Old Way (Hard-coded):
```
User: "delete my notes"
  ↓
Plugin: execute_command()
  ↓
return {"success": True, "message": "✅ Archived 5 notes..."}  ← HARD-CODED!
  ↓
Alice speaks it verbatim (no learning)
```

New Way (Learning):
```
User: "delete my notes"
  ↓
Plugin: execute_command()
  ↓
return {"success": True, "action": "delete_notes", "data": {"count": 5}}  ← DATA ONLY
  ↓
ResponseFormulator: formulate_response()
  ├─ Check if Alice learned this (independent?)
  │   └─ YES: Use learned pattern
  │   └─ NO:  Ask Ollama for help
  ↓
Learn from Ollama's phrasing (store in PhrasingLearner)
  ↓
After 3 examples: Alice can phrase independently!
```


## HOW TO REFACTOR A PLUGIN

### Step 1: Identify Hard-Coded Messages

BEFORE:
```python
def _delete_note(self, command: str) -> Dict[str, Any]:
    # ... deletion logic ...
    return {
        "success": True,
        "message": f"✅ Archived {count} notes. They're not permanently deleted."  # BAD!
    }
```

### Step 2: Return Structured Data Instead

AFTER:
```python
def _delete_note(self, command: str) -> Dict[str, Any]:
    # ... deletion logic ...
    return {
        "success": True,
        "action": "delete_notes",  # What action was performed
        "data": {                   # Just the facts
            "count": count,
            "archived": True,
            "permanent": False
        },
        "formulate": True  # Signal that Alice should formulate response
    }
```

### Step 3: Handle in Main Processing Loop

In main.py plugin execution:
```python
# After plugin execution
if plugin_result.get('formulate', False):
    # Let Alice formulate the response
    response = self.response_formulator.formulate_response(
        action=plugin_result['action'],
        data=plugin_result['data'],
        success=plugin_result['success'],
        user_input=user_input,
        tone=tone
    )
else:
    # Use old message field (for backward compatibility)
    response = plugin_result.get('message', 'Done.')
```


## TEACHING ALICE NEW RESPONSES

### Method 1: Seed Templates (Quickest)

Run the seed script to give Alice examples:
```bash
python scripts/automation/seed_response_templates.py
```

This creates templates with 3 example phrasings for each action type.

### Method 2: Interactive Teaching

```python
# In Python/script
formulator = get_response_formulator()

formulator.add_template(
    action="play_music",
    example_data={"song": "Bohemian Rhapsody", "artist": "Queen"},
    example_phrasings=[
        "Playing Bohemian Rhapsody by Queen.",
        "Now playing: Bohemian Rhapsody - Queen.",
        "I've started Bohemian Rhapsody for you.",
    ],
    formulation_rules=[
        "Mention song title and artist",
        "Use present progressive (playing/starting)",
        "Be concise"
    ]
```

### Method 3: Learning from Ollama (Automatic)

Just use the system! Alice will:
1. Ask Ollama how to phrase it (with examples)
2. Learn from Ollama's response
3. After 3 similar examples, formulate independently

Progress tracking:
```python
stats = formulator.get_stats()
# {
#   'total_templates': 20,
#   'independent_actions': 15,
#   'learning_progress': '15/20'
# }
```


## PROGRESSIVE INDEPENDENCE

Alice learns in stages:

**Stage 1: Beginner (0-1 examples)**
- Needs Ollama for everything
- Learning each new action type

**Stage 2: Student (2-3 examples)**
- Seeing multiple examples
- Starting to recognize patterns

**Stage 3: Independent (3+ examples)**
- Can formulate without Ollama
- Uses learned patterns
- 🎉 Marked as independent_action

**Stage 4: Expert (10+ examples)**
- Handles variations confidently
- Adapts to context naturally


## MIGRATION PLAN

Refactor plugins gradually (no breaking changes):

### Phase 1: Add Response Formulator
- ✓ Create response_formulator.py
- ✓ Seed initial templates
- ✓ Integrate into main.py

### Phase 2: Refactor High-Traffic Plugins First
Priority order:
1. Notes Plugin (148 messages!)
2. Weather Plugin
3. Time Plugin
4. System Control Plugin
5. File Operations Plugin

### Phase 3: Backward Compatibility
Keep both paths working:
```python
if 'formulate' in result and result['formulate']:
    # New way: formulate from data
    response = formulator.formulate_response(...)
else:
    # Old way: use hard-coded message
    response = result.get('message', 'Done.')
```

### Phase 4: Training & Validation
- Run nightly training scenarios
- Monitor independence progress
- Validate response quality
- Track which actions are independent


## MONITORING ALICE'S LEARNING

### Check Independence Status
```bash
# Add to /status command
Alice can independently formulate 15/20 action types (75%)
```

### View Learning Progress
```python
# GET /api/learning/formulation_stats
{
  "independent_actions": ["create_note", "delete_notes", "weather_current", ...],
  "learning_actions": ["play_music", "send_email"],
  "total_templates": 20,
  "independence_rate": 0.75
}
```

### Reset Learning (for testing)
```bash
rm data/response_templates/independence.json
rm data/learned_phrasings.jsonl
```


## EXAMPLE: COMPLETE REFACTOR

Let's refactor the notes plugin's _delete_all_notes method:

### Before:
```python
def _delete_all_notes(self) -> Dict[str, Any]:
    active_notes = [note for note in self.manager.notes.values() if not note.archived]
    count = len(active_notes)

    for note in active_notes:
        self.manager.archive_note(note.id)

    return {
        "success": True,
        "message": f"✅ Archived {count} notes.
(Notes are archived, not permanently deleted...)",
        "count": count
    }
```

### After:
```python
def _delete_all_notes(self) -> Dict[str, Any]:
    active_notes = [note for note in self.manager.notes.values() if not note.archived]
    count = len(active_notes)

    for note in active_notes:
        self.manager.archive_note(note.id)

    return {
        "success": True,
        "action": "delete_notes",
        "data": {
            "count": count,
            "archived": True,
            "permanent": False,
            "restorable": True
        },
        "formulate": True  # Let Alice formulate response
    }
```

Alice learns to say things like:
- "I archived 5 notes for you. They're not permanently deleted."
- "Done. I moved 5 notes to your archive."
- "Archived 5 notes. You can restore them anytime."


## BENEFITS

1. **Alice Learns**: No more hard-coded responses
2. **Personality Develops**: Alice's voice becomes unique
3. **Context-Aware**: Responses adapt to situation/tone
4. **Progressive Independence**: Less Ollama usage over time
5. **Maintainable**: Change templates, not code
6. **Scalable**: Add new actions easily


## NEXT STEPS

1. Run seed script to create initial templates
2. Integrate ResponseFormulator into main.py
3. Refactor one plugin as proof-of-concept
4. Monitor learning progress
5. Gradually migrate remaining plugins


## FILES CREATED

- ai/core/response_formulator.py - Main formulation logic
- scripts/automation/seed_response_templates.py - Initial template seeding
- data/response_templates/templates.json - Response templates
- data/response_templates/independence.json - Independence tracking


Alice will truly learn and develop her own voice! 🎓
