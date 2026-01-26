# A.L.I.C.E System Architecture Update

## Addressing the Gaps: Reactive → Anticipatory

### Gap 1: ALICE now anticipates (Pattern Learning)

**Before:** Waited for input
**Now:** Notices patterns and suggests proactively

#### Pattern Learner ([ai/pattern_learner.py](ai/pattern_learner.py))
- **Temporal patterns**: "Every Sunday at 3pm, user reviews finance notes"
- **Sequential patterns**: "After checking email, user opens calendar"
- **Contextual patterns**: "When system is idle >10min, user takes a break"

**How it works:**
```python
# ALICE observes actions
learner.observe_action("review_notes:finance", context={'day': 'Sunday', 'hour': 15})

# Later, when Sunday 3pm arrives
suggestions = learner.get_suggestions()
# → "You usually review finance notes around this time. Want me to prepare a summary?"
```

**Smart suggestions:**
- Confidence threshold (>60%)
- Acceptance rate tracking (learns what you actually want)
- Cooldown periods (won't spam)
- Only suggests reliable patterns (≥3 occurrences)

---

### Gap 2: Planning vs Execution - Strictly Separated

**Before:** Blended logic flow
**Now:** Clean pipeline: Understand → Plan → Execute

#### Task Planner ([ai/task_planner.py](ai/task_planner.py))
**Job:** Create plans, NOT execute them

```python
# Input: Intent + Entities
plan = planner.create_plan(
    intent="summarize_notes",
    entities={'topic': 'finance', 'timeframe': 'last_month'},
    context={}
)

# Output: ExecutionPlan with steps
# Step 1: plugin.notes.list (topic=finance, timeframe=last_month)
# Step 2: plugin.notes.read_multiple (depends on Step 1)
# Step 3: llm.summarize (depends on Step 2)
```

#### Plan Executor ([ai/plan_executor.py](ai/plan_executor.py))
**Job:** Execute plans, NOT create them

```python
# Takes a plan, executes step-by-step
result = executor.execute(plan)

# Handles:
# - Dependency resolution (Step 2 waits for Step 1)
# - Parameter interpolation ({{step_1_result}})
# - Error recovery
# - Progress tracking
# - Event emission
```

**Separation benefits:**
- Planner can explain without executing
- Executor can retry without re-planning
- Plans can be validated before execution
- Can ask "confirm this plan?" before running

---

### Gap 3: ALICE now "owns the system"

**Before:** Knew context when asked
**Now:** Continuously observes system state

#### System Monitor ([ai/system_monitor.py](ai/system_monitor.py))
**Tracks:**
- Running apps (Chrome opened, VS Code closed)
- Process state (CPU, memory per app)
- File changes (watched files modified)
- Notable events (Spotify started, Outlook closed)

**Integration with Event Bus:**
```python
monitor.start_monitoring()

# Now ALICE knows:
- "Chrome just opened → Maybe user wants to browse?"
- "VS Code running for 3 hours → Suggest a break?"
- "Outlook closed → User finished email session"
```

**Smart filtering:**
- Ignores system processes (dwm.exe, svchost.exe)
- Only tracks "notable" apps (browsers, IDEs, office, dev tools)
- Emits events for app launch/close

---

## The Complete Pipeline Now

### Input → Understanding → Planning → Execution → Response

```
User: "Summarize my finance notes from last month"
  ↓
[1. UNDERSTAND] NLP Processor
  → Intent: summarize_notes
  → Entities: {topic: finance, timeframe: last_month}
  ↓
[2. PLAN] Task Planner
  → Step 1: Load finance notes
  → Step 2: Read note contents
  → Step 3: LLM summarize
  ↓
[3. EXECUTE] Plan Executor
  → Runs steps with dependency management
  → Emits progress events
  → Handles errors
  ↓
[4. RESPOND] Rich Terminal UI
  → Display summary
  → Log to memory
  ↓
[5. LEARN] Pattern Learner
  → Observe: "User asked for finance summary on Sunday"
  → Next Sunday → Suggest proactively
```

---

## Background Intelligence

### Observers Watch Everything

**TaskObserver:**
- Long-running tasks (>2min) → Notify user
- Task failures → Always interrupt
- Important completions → Notify

**SystemHealthObserver:**
- Memory >85% → Warning (with cooldown)
- Storage <10% → Alert
- System errors → Critical notification

**ReminderObserver:**
- Due reminders → High priority interrupt
- Approaching events → Normal priority

**BackgroundActivityObserver:**
- Important emails → Notify
- Calendar changes → Low priority
- File changes → Configurable

### Pattern-Based Proactivity

**SystemMonitor** sees: User opened VS Code at 9am
**PatternLearner** knows: User codes 9am-12pm on weekdays
**Suggestion**: "Want me to set 'focus mode' and mute notifications?"

---

## Event-Driven Architecture

Everything communicates via events:
- **State changes** → Events
- **Observers** → Watch events
- **Decide** → When to interrupt
- **Notify** → Only when justified

**Example flow:**
```
Task takes >2min
  → TaskObserver sees TASK_PROGRESS event
  → Checks: Already notified? No
  → Checks: Important enough? Yes
  → observer.notify("Task taking longer than expected")
  → ALICE speaks/displays message
```

---

## What This Enables

### Proactive ALICE:
- "You usually check email now. 5 new messages waiting."
- "Outlook has been open for 2 hours. Want me to close it?"
- "Chrome using 4GB RAM. Should I suggest closing tabs?"
- "You typically review notes on Sundays. Want a summary?"

### Smart Interruptions:
- Won't spam (cooldowns)
- Context-aware (user active vs idle)
- Priority-filtered (only important stuff)
- Learning (tracks acceptance rates)

### Clean Architecture:
- Planner doesn't execute
- Executor doesn't plan
- Observers don't act directly
- Components communicate via events

---

## Status: Gaps Closed ✓

**Gap 1 - Anticipation:** ✓ Pattern learning + proactive suggestions  
**Gap 2 - Separation:** ✓ TaskPlanner + PlanExecutor split  
**Gap 3 - System Ownership:** ✓ SystemMonitor + continuous observation

---

## Next Integration Step

Connect to main ALICE:
1. Initialize all systems on startup
2. Wire pattern learner to action logging
3. Integrate planner/executor into process_input()
4. Start background monitors
5. Set up observer notification callbacks

This architecture is now **Jarvis-grade** - anticipatory, separated, and system-aware.
