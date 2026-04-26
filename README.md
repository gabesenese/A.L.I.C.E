# A.L.I.C.E

Advanced Linguistic Intelligence Companion Entity.

This repository is organized around a central turn loop and contract pipeline.
The current goal is companion-quality behavior: coherent state, disciplined actions,
and continuity across turns.

## Current Runtime Model

The canonical runtime path is in the contract pipeline:

1. Route
2. Execute (if needed)
3. Verify
4. Respond
5. State update

The companion loop (central brain) keeps per-turn state and policy decisions centralized:

```python
def process_turn(user_input):
    perception = perceive(user_input)
    state = update_companion_state(perception)
    decision = policy_engine.decide(state)
    ...
```

## Quick Start

Use a virtual environment, then install the lean default dependencies:

```bash
pip install -r requirements.txt
```

Run the main CLI runtime:

```bash
python app/main.py
```

Run the user-facing UI wrapper:

```bash
python app/alice.py
```

Dev mode with auto-reload:

```bash
python app/dev.py
```

Windows helper:

```bash
dev.bat
```

## Optional Dependency Bundles

Install only what you need:

```bash
pip install -r requirements-dev.txt
pip install -r requirements-voice.txt
pip install -r requirements-integrations.txt
pip install -r requirements-ops.txt
```

## Tests

Canonical integration tests:

```bash
python -m pytest -q tests/integration/test_contract_pipeline.py
```

Broader integration/e2e suites:

```bash
python -m pytest -q tests/integration tests/e2e
```

Small startup smoke test:

```bash
python test_init.py
```

## Docker

Default compose path is intentionally minimal (alice service).
Optional profiles:

1. `llm` for local Ollama service
2. `ops` for Redis cache service

Examples:

```bash
docker compose up --build
docker compose --profile llm up --build
docker compose --profile ops up --build
```

## Repository Notes

1. Experimental and low-frequency scripts are archived under `archive/2026-04/`.
2. Core runtime is under `app/` and `ai/runtime/`.
3. Keep default-path changes focused on policy/state/verification quality.

## License

Private project. All rights reserved.
