# A.L.I.C.E Architecture

This document describes the intended architecture: runtime flow (router → tool vs LLM → resolver → policy → verifier → logger) and the offline training loop.

---

## Runtime flow

```
User input
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. ROUTER                                                        │
│    Predicts tool/action with confidence                          │
│    (intent + intent_confidence)                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ├── If confident ──────────────────────────────────────────────► TOOL PATH
    │       • Code/self-reflection, training status, weather follow-up
    │       • Plugins: execute_for_intent(intent, user_input, entities)
    │       • Planner/executor for structured tasks
    │
    └── If not confident ─────────────────────────────────────────► GOAL PATH
            │
            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 2. GOAL (when unclear)                                       │
    │    LLM produces Goal JSON  ← [GAP: currently goal from        │
    │    (target intent, target item, ask vs execute)     resolver] │
    └─────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 3. RESOLVER                                                  │
    │    Picks target item (references: "it", "that email", etc.)  │
    │    • ReferenceResolver → bindings, resolved_input            │
    │    • GoalResolver → current goal, cancel/revise/reference     │
    └─────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 4. POLICY                                                    │
    │    Decides: execute vs ask (clarify / confirm)               │
    │    [GAP: today this is implicit in main loop:                │
    │     try plugin → if match execute; low confidence → LLM]     │
    └─────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 5. EXECUTE / ASK                                             │
    │    • Execute: plugin run, then response from generator/Ollama│
    │    • Ask: clarification (reasoning_engine) or LLM response   │
    └─────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 6. VERIFIER                                                  │
    │    Checks result: action_succeeded, goal_fulfilled           │
    │    → suggest follow-up or mark goal completed                │
    └─────────────────────────────────────────────────────────────┘
            │
            ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 7. LOGGER                                                    │
    │    Writes training row (training_collector.collect_interaction)│
    │    • user_input, assistant_response, intent, context, quality│
    └─────────────────────────────────────────────────────────────┘
```

### Current code mapping

| Step | Component | Location |
|------|-----------|----------|
| **Router** | NLP (intent + confidence), reasoning_engine (uncertainty → clarify) | `nlp_processor`, `intent_classifier`, `reasoning_engine`; confidence used in `main.py` (~984, 1672, 1696) |
| **Tool path** | Code handler, training handler, plugins, planner/executor | `_handle_code_request`, `_handle_training_request`, `plugin_system.execute_for_intent`, `_use_planner_executor` |
| **Goal** | Goal resolution from user input (no LLM Goal JSON yet) | `goal_resolver.resolve(user_input, intent, entities)` |
| **Resolver** | Reference + goal | `reference_resolver.resolve`, `goal_resolver.resolve` |
| **Policy** | Implicit: plugin match → execute; low confidence → LLM | `main.py` plugin branch vs LLM branch |
| **Verifier** | After plugin result | `verifier.verify(plugin_result, goal_intent, goal_description)` |
| **Logger** | Training row | `training_collector.collect_interaction(...)` |

---

## Offline training loop (daily or weekly)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Train router + policy from logs                                │
│    • Router: intent (and optionally tool) prediction from        │
│      user_input + context in training_data.jsonl                 │
│    • Policy: execute vs ask from outcomes (success, clarification)│
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Evaluate accuracy                                             │
│    • Router: intent accuracy, tool prediction accuracy            │
│    • Policy: correct execute/ask decisions                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Update thresholds                                             │
│    • Confidence thresholds for “confident → tool” vs “→ Goal path”│
│    • Optional: per-intent or per-tool thresholds                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Deploy model file locally                                     │
│    • Save router/policy model (e.g. ml_learner.pkl, or new files) │
│    • Load on next A.L.I.C.E startup                              │
└─────────────────────────────────────────────────────────────────┘
```

### Current code mapping

| Step | Component | Location |
|------|-----------|----------|
| **Train from logs** | `TrainingDataCollector` (JSONL), `SimpleMLLearner` (TF-IDF + LR, optional sentence_transformers) | `training_system.py`: `collect_interaction`, `SimpleMLLearner.train_from_examples` |
| **Evaluate accuracy** | Not yet a dedicated pipeline | Could use same JSONL + holdout or time-based split |
| **Update thresholds** | Confidence thresholds are hardcoded (e.g. 0.6, 0.7) | `main.py` (e.g. intent_confidence &lt; 0.6 / 0.7) |
| **Deploy locally** | ML learner saved as `data/training/ml_learner.pkl`, loaded on init | `SimpleMLLearner._load`, `model_path` |

---

## Gaps vs target architecture

1. **Goal JSON from LLM**  
   Goal is currently derived only from `GoalResolver` (user input + intent). A dedicated “LLM produces Goal JSON” step (target intent, target item, ask vs execute) is not implemented.

2. **Explicit Policy module**  
   “Policy decides execute/ask” is implicit in the main loop (plugin try → execute; low confidence → LLM/clarify). A separate Policy that consumes resolver output and returns execute vs ask would align with the diagram.

3. **Offline loop automation**  
   Training and model persistence exist; there is no scheduled “daily/weekly” job that (1) trains router + policy from logs, (2) evaluates accuracy, (3) updates thresholds, (4) deploys model file. Thresholds are fixed in code.

4. **Router as explicit “tool/action” predictor**  
   The router currently outputs intent (and entities); the decision to take the “tool path” is based on intent + confidence + plugin match. A single router that predicts (tool/action, confidence) and drives the “if confident → tool path” branch would match the diagram more directly.

---

## Summary

- **Runtime:** Router (intent + confidence) → confident → tool path (plugins, code, etc.); not confident → resolver (reference + goal) → implicit policy (execute vs LLM/ask) → verifier → logger. Existing pieces are mostly in place; the main gaps are LLM-produced Goal JSON and an explicit Policy module.
- **Offline:** Training and local model deploy exist; missing pieces are automated evaluation, threshold updates from evaluation, and a scheduled training pipeline.

This file can be updated as you add Goal JSON from LLM, a Policy module, and the full offline loop (evaluate → thresholds → deploy).
