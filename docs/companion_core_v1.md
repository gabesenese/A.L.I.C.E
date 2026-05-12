# Companion Core v1

Companion Core v1 adds five local-first primitives that sit beside existing runtime foundations.

## New primitives
- `CompanionState` (`ai/runtime/companion_state.py`): session/human summary state (topic, goal, energy, mood, safe next action, privacy flags).
- `PerceptionFrame` (`ai/runtime/perception_frame.py`): rule-based split of social context vs actual request before routing.
- `ActionBus` (`ai/runtime/action_bus.py`): structured action request/result execution with evidence and verification flags.
- `ClaimVerifier` (`ai/runtime/claim_verifier.py`): blocks unsupported claims (inspection, memory, deletion, background work, fictional provenance).
- `MemoryRightsService` (`ai/memory/memory_rights_service.py`): preview, confirm-delete, suppress-topic, and show-topic flows for local memory rights.

## How this fits current architecture
- `OperatorState` remains operator execution state.
- `ProjectMemory` remains durable project memory.
- `AgentLoop` remains bounded planning/execution flow.
- `CompanionState` summarizes user/session state and perception metadata.

## Integration points
- Routing boundary builds a `PerceptionFrame` early and routes on `actual_request` when present.
- Operator response surface can use perception + companion state for one-line grounded acknowledgement.
- Verification boundary runs `ClaimVerifier` before accepting final responses.
- Memory plugin uses `MemoryRightsService` for topic preview/delete semantics and honest confirmation flow.
