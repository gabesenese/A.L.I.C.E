# A.L.I.C.E Simulation & Training System

Automated testing and overnight learning system for A.L.I.C.E.

## Overview

This system enables A.L.I.C.E to improve automatically through:
1. **Scripted scenario testing** - Run conversations against Alice with expected outcomes
2. **Teacher-guided learning** - Ollama provides "ideal" responses for comparison
3. **Automated pattern promotion** - High-confidence patterns auto-learn overnight
4. **Nightly training pipeline** - Complete improvement cycle runs automatically

## Quick Start

### Run Scenario Simulations

```bash
# Run all scenarios with minimal policy and teacher mode
python -m sim.run_scenarios --policy minimal

# Run specific domains only
python -m sim.run_scenarios --domains email notes

# Run specific tags
python -m sim.run_scenarios --tags clarification vague

# Disable teacher mode (faster)
python -m sim.run_scenarios --no-teacher
```

### Promote Patterns from Logs

```bash
# Analyze simulation logs and auto-promote high-confidence patterns
python -m ai.promote_patterns

# Require higher frequency threshold
python -m ai.promote_patterns --min-frequency 5

# Disable auto-apply (manual review only)
python -m ai.promote_patterns --no-auto-apply
```

### Run Nightly Training Pipeline

```bash
# Complete training pipeline (scenarios + pattern promotion + fallback analysis)
python scripts/nightly_training.py
```

## Architecture

### Simulation Components

**sim/scenarios.py** - Scenario definitions
- Email flows (list, read, search, delete, compose)
- Notes operations (create, search, list)
- Weather/time/system queries
- Conversational patterns (greetings, thanks, status)
- Clarification tests (vague/ambiguous inputs)

**sim/run_scenarios.py** - Scenario runner
- Instantiates Alice core components
- Feeds scenario inputs
- Captures routing decisions (CONVERSATIONAL/TOOL/RAG/LLM_FALLBACK/CLARIFICATION)
- Compares expected vs actual routes and intents
- Writes to `data/training/auto_generated.jsonl`
- Generates accuracy report

**sim/teacher.py** - Ollama teacher mode
- Provides "ideal" assistant responses
- Compares Alice's responses to teacher's
- Flags deviations for learning
- Detects mismatches in tone, length, content

### Learning Components

**ai/promote_patterns.py** - Pattern promotion
- Groups simulation logs by (intent, domain)
- Calculates teacher consistency
- Detects variable slots in patterns
- Auto-promotes when:
  - Frequency ≥ 5
  - Teacher consistency ≥ 80%
  - No negative feedback exists
- Saves to `memory/learning_patterns.json`
- Manual review for edge cases → `memory/patterns_for_review.json`

**ai/teacher_loop.py** - Real interaction analysis
- Analyzes `data/training/logged_interactions.jsonl`
- Groups LLM fallbacks by similarity
- Auto-learns high-frequency patterns
- Complements simulation-based learning

### Automation

**scripts/nightly_training.py** - Overnight improvement pipeline
1. Run all scenarios with minimal policy
2. Promote patterns from simulation logs
3. Analyze real interaction fallbacks
4. Auto-learn high-confidence patterns
5. Generate summary report

Logs to: `data/training/nightly_training.log`

## Scenario Format

```python
Scenario(
    name="Scenario Name",
    description="What this tests",
    domain="email",  # Domain category
    steps=[
        ScenarioStep(
            user_input="show me my recent emails",
            expected_intent="list_emails",
            expected_route=ExpectedRoute.TOOL,
            domain="email",
            expected_entities={"limit": 10}
        )
    ],
    tags=["email", "list"]  # For filtering
)
```

## Output Files

| File | Purpose |
|------|---------|
| `data/training/auto_generated.jsonl` | Simulation interaction logs with teacher responses |
| `memory/learning_patterns.json` | Auto-promoted patterns (versioned) |
| `memory/patterns_for_review.json` | Pattern candidates needing manual review |
| `data/training/teacher_suggestions.json` | Real interaction learning opportunities |
| `data/training/nightly_training.log` | Nightly pipeline execution log |

## Guardrails

Pattern promotion only happens when:
- ✅ Frequency ≥ minimum threshold (default: 3)
- ✅ Teacher consistency ≥ 80% (teacher gives similar answers)
- ✅ No negative user feedback exists for that pattern
- ✅ Pattern doesn't already exist

Auto-apply requires:
- ✅ All above guardrails
- ✅ Frequency ≥ 5 (higher bar)

## Scheduling Nightly Training

### Windows (Task Scheduler)

```powershell
# Run daily at 2 AM
schtasks /create /tn "ALICE_NightlyTraining" /tr "python C:\path\to\alice\scripts\nightly_training.py" /sc daily /st 02:00
```

### Linux (cron)

```bash
# Add to crontab (run daily at 2 AM)
0 2 * * * cd /path/to/alice && python scripts/nightly_training.py
```

## Monitoring

Check training results:
```bash
# View nightly training log
cat data/training/nightly_training.log

# Check promoted patterns
cat memory/learning_patterns.json

# Review manual candidates
cat memory/patterns_for_review.json
```

## Advanced Usage

### Custom Scenarios

Add to `sim/scenarios.py`:

```python
CUSTOM_SCENARIOS = [
    Scenario(
        name="Your Custom Test",
        description="Test description",
        domain="custom",
        steps=[
            ScenarioStep(
                user_input="your test input",
                expected_intent="custom_intent",
                expected_route=ExpectedRoute.TOOL,
                domain="custom"
            )
        ],
        tags=["custom"]
    )
]

# Add to ALL_SCENARIOS
ALL_SCENARIOS = (
    EMAIL_SCENARIOS +
    # ... other scenarios ...
    CUSTOM_SCENARIOS
)
```

### Teacher System Prompt

Modify in `sim/teacher.py`:

```python
TEACHER_SYSTEM_PROMPT = """Your custom teacher instructions..."""
```

### Pattern Template Extraction

Custom logic in `ai/promote_patterns.py`:

```python
def _create_response_template(self, teacher_responses, variable_slots):
    # Your template extraction logic
    pass
```

## Benefits

1. **Continuous Improvement** - Alice learns overnight without manual intervention
2. **Quality Assurance** - Scenarios catch regressions automatically
3. **Scalable Learning** - Thousands of interactions can be simulated
4. **Teacher-Guided** - LLM provides gold standard responses
5. **Safe Auto-Learning** - Multiple guardrails prevent bad pattern promotion

## The "Tony Stark" Vision

> "I want Alice to wake up smarter than she was yesterday."

- **Day**: Alice handles real user interactions
- **Night**: System runs scenarios, compares to teacher, learns patterns
- **Morning**: Alice has new patterns, reduced LLM dependency, faster responses
- **Repeat**: Continuous improvement cycle

This is intelligent automation - Alice teaches herself from simulated and real experiences, with quality guardrails ensuring she only learns good patterns.
