# A.L.I.C.E Development Roadmap

This file is the single source of truth for planning and delivery status.

## Canonical Runtime Direction

Objective: keep app/main.py as a composition root while executing turns through explicit runtime phases:
route -> execute -> verify -> respond.

## Delivery Contract Status

### A. Response Authority and Fallback Policy
- [x] Enforce LLM response authority contract:
  - if LLM response is accepted, publish it.
  - if rejected, try refinement first.
  - if refinement fails, use deterministic fallback.
- [x] Add shared runtime fallback policy with per-turn deterministic caching.
- [x] Remove duplicate deterministic fallback generation paths from hot routes.

### B. Orchestration Extraction
- [x] Extract route/execute/verify/respond turn phases into runtime orchestrator module.
- [x] Route contract pipeline through explicit phases instead of inline orchestration.
- [ ] Promote contract pipeline out of quarantine so canonical phase flow is active by default.
- [ ] Continue shrinking app/main.py by moving non-lifecycle logic into runtime modules.

### C. Governance and CI Gates
- [x] Wire benchmark gate into CI as merge-blocking workflow.
- [ ] Add per-domain regression artifact publication in CI.
- [ ] Require scorecard refresh policy per benchmark-affecting PR.

### D. Testing and Quality
- [x] Add transcript-style end-to-end pipeline tests.
- [x] Expand pytest discovery to include e2e transcript tests.
- [ ] Add transcript fixtures for multi-turn tool chaining and failure recovery.

### E. Startup and Operational Hygiene
- [x] Lazy-load non-essential startup modules used only for optional flows.
- [x] Quarantine stale module deprecation script behind explicit acknowledgment.
- [ ] Remove remaining non-essential eager imports from app/main.py startup path.

### F. Documentation and Command Consistency
- [x] Set roadmap.md as canonical planning artifact.
- [x] Standardize canonical launch command as: python app/main.py
- [ ] Align all secondary docs to reference this roadmap without duplicating planning state.

## 7-Day Execution Plan

### Day 1: Authority + Fallback Lock
- Finalize regression coverage for accepted-vs-rejected LLM publication paths.
- Add explicit test cases for refine-then-deterministic sequence.

### Day 2: Transcript Evaluation Pack
- Build a 50-turn transcript benchmark pack from real interaction traces.
- Add assertions for phase ordering and final-output authority.

### Day 3: main.py Reduction Pass 1
- Move fallback decision branches into ai/runtime/fallback_policy.py.
- Move response authority handling helpers into ai/runtime/response_authority.py.

### Day 4: main.py Reduction Pass 2
- Move remaining route/execute/verify/respond glue into ai/runtime/turn_orchestrator.py.
- Keep app/main.py focused on lifecycle, feature wiring, and IO boundaries.

### Day 5: CI Evidence and Reporting
- Publish benchmark gate result summary as CI artifact.
- Add per-domain delta table in workflow summary output.

### Day 6: Reliability Hardening
- Add transcript tests for verifier rejection and fallback behavior.
- Add one chaos-style test for tool failure with deterministic recovery.

### Day 7: Freeze and Acceptance
- Run integration + e2e suite.
- Run benchmark gate and confirm no regression.
- Update this roadmap with final metrics and completion notes.

## Definition of Done

- Canonical route -> execute -> verify -> respond flow is active by default.
- Accepted LLM responses are never overridden by unrelated deterministic paths.
- Rejected LLM responses follow refine first, deterministic second.
- Benchmark gate blocks regressions in CI.
- Transcript tests validate end-to-end turn behavior.
- app/main.py continues trending toward composition-root-only responsibilities.
