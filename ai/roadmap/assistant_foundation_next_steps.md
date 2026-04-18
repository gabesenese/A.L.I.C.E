# A.L.I.C.E → Advanced Assistant Foundation Next Steps

This plan is based on the current architecture and code paths in:

- `ai/core/nlp_processor.py`
- `ai/core/foundation_layers.py`
- `ai/core/unified_action_engine.py`
- `ai/core/cognitive_orchestrator.py`
- `ai/core/world_state_memory.py`
- `ai/core/live_state_service.py`

The goal is not "voice gimmicks" yet; it is *agent reliability + autonomy quality*.

---

## P0 (Highest Impact, Do Next)

### 1) Intent Calibration + Guardrails: learn from mistakes automatically
**Why now**
- NLP now has staged inference and clarification policy, but thresholds are mostly static.

**Implement**
- Add an online threshold tuner that adjusts:
  - `clarification_policy` confidence and margin cutoffs
  - `should_run_deep_stage` gate in `FoundationLayers`
- Feed it data from:
  - false clarification triggers
  - wrong-tool executions
  - user corrections

**Where**
- `ai/core/foundation_layers.py`
- `ai/core/nlp_processor.py`
- `ai/core/adaptive_intent_calibrator.py` (already exists; extend usage)

---

### 2) World-state freshness contracts for tool answers
**Why now**
- `LiveStateService` has snapshot selection logic, but no strict freshness policy per domain.

**Implement**
- Add per-domain TTL + staleness policy:
  - weather, calendar, system metrics
- If stale:
  - force refresh
  - or respond with explicit stale warning

**Where**
- `ai/core/live_state_service.py`
- `ai/core/world_state_memory.py`
- action flow integration in `ai/core/unified_action_engine.py`

---

### 3) Recovery planner for failed actions
**Why now**
- `UnifiedActionEngine` already has rollback and retry structure.
- Missing: deterministic multi-step "what to try next".

**Implement**
- Add a recovery policy graph:
  - retry with adjusted params
  - alternative plugin/action
  - ask targeted clarification
- Persist attempted recovery paths in journal and world state.

**Where**
- `ai/core/unified_action_engine.py`
- `ai/core/rollback_executor.py`
- `ai/core/execution_journal.py`

---

## P1 (Second Wave)

### 4) Goal decomposition and continuity across turns
**Why now**
- `CognitiveOrchestrator` tracks long-horizon goals, but execution-level decomposition can be tighter.

**Implement**
- Convert high-level goals into explicit step graphs with completion criteria.
- Store progress and blockers in world-state memory.

**Where**
- `ai/core/cognitive_orchestrator.py`
- `ai/core/goal_tracker.py`
- `ai/core/world_state_memory.py`

---

### 5) Clarification UX upgrades (short, high-information prompts)
**Why now**
- Clarification exists, but you can make it feel more "operator-grade":
  - concise
  - optioned
  - context aware

**Implement**
- Clarification templates that include:
  - top 2 likely interpretations
  - exactly one follow-up question
  - recommended default

**Where**
- `ai/core/foundation_layers.py`
- `ai/core/clarification_resolver.py`
- `ai/core/route_coordinator.py`

---

### 6) Policy simulation before risky execution
**Why now**
- Action engine has simulation gates; deepen them for high-risk domains.

**Implement**
- Run "dry-run simulation" for:
  - delete/update actions
  - broad system commands
- Require either:
  - explicit confirmation
  - or low-risk verification score

**Where**
- `ai/core/unified_action_engine.py`
- `ai/core/goal_action_verifier.py`
- `ai/core/execution_verifier.py`

---

## P2 (Stability + Productization)

### 7) Eval harness expansion (agent benchmarks, not just unit tests)
**Why now**
- You have many integration tests already. Convert them into KPI-style dashboards.

**Implement**
- Nightly benchmark suite:
  - route accuracy
  - tool success rate
  - recovery success rate
  - clarification precision/recall
  - latency buckets (P50/P95)

**Where**
- `tools/auditing/core_benchmark_gate.py`
- `tests/integration/*`
- `scripts/automation/nightly_audit_scheduler.py`

---

### 8) Personalization and operator style model
**Why now**
- High-quality assistant feel = predictable personalization.

**Implement**
- Add stable user preference profile:
  - response brevity
  - preferred confirmation style
  - risk tolerance
- Use profile in response planner and routing.

**Where**
- `ai/core/adaptive_response_style.py`
- `ai/runtime/user_state_model.py`
- `ai/core/response_planner.py`

---

## Suggested execution order (4 sprints)

1. **Sprint 1**: P0-1 (intent calibration), P0-2 (freshness contracts)  
2. **Sprint 2**: P0-3 (recovery planner), P1-5 (clarification UX)  
3. **Sprint 3**: P1-4 (goal decomposition), P1-6 (risk simulation hardening)  
4. **Sprint 4**: P2-7 (benchmark KPI pipeline), P2-8 (personalization model)

---

## Definition of "closer to advanced assistant quality" (measurable)

- Clarification precision > 85%
- Wrong-tool execution rate < 3%
- Recovery success after first failure > 70%
- Stale-state answer rate < 2%
- P95 routing latency < 200ms (without LLM generation)
- Multi-turn task completion rate +20% over current baseline
