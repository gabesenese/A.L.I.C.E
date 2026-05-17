# Alice Companion Core v1 Architecture Map

This sprint borrows architecture concepts from `DavidLiuXh/jarvis-personal-ai` (Apache-2.0) and translates them into Alice-native, local-first Python modules.

## Attribution and License Note

- Reference architecture: `https://github.com/DavidLiuXh/jarvis-personal-ai`
- License: Apache-2.0
- Approach in Alice: concept translation, not direct runtime dependency.
- If code is ever copied or closely adapted from the reference repository, preserve upstream attribution and license notices in the relevant files.

## What Alice Borrows Conceptually

- Local memory service with SQLite-backed records and retrieval scoring.
- Context refresh before model generation.
- Local turn routing for mode/subject/evidence/tool decisions.
- Separation of verified memories vs hint memories in prompt context.
- Recency-aware memory retrieval.

## What Alice Intentionally Does Not Integrate

- WeChat integrations
- Feishu integrations
- Gemini CLI dependency
- Cloud-first model routing
- Web UI surface
- Phone/text-app channels
- TypeScript runtime
- Channel-specific commands

## Mapping: Reference Concept -> Alice-Native Module

| Reference Concept | Alice-Native Module |
|---|---|
| MemoryService | `ai/memory/alice_memory_service.py` |
| Memory schema entities | `ai/memory/alice_memory_schema.py` |
| refreshContext | `ai/runtime/context_refresh_service.py` |
| LocalModelRouter | `ai/runtime/alice_turn_router.py` |
| BackgroundTaskRunner | Future: `ai/runtime/background_engine.py` |
| Tool loop guards | Future: `ai/runtime/agent_loop_guards.py` |

## Practical Scope of This Sprint

- Add a working SQLite memory service for companion/core memory.
- Add a context refresh service that builds compact, mode-aware model context.
- Add a turn router that classifies mode/subject and evidence/tool requirements.
- Wire context refresh into local model generation in a minimal, compatibility-safe way.
- Add tests proving concept thread carry-forward behavior.
