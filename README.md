# A.L.I.C.E

Advanced Linguistic Intelligence Companion Entity.

A local, Jarvis-shaped assistant: it runs on your machine, talks to a local
model, acts through real tools, and remembers across sessions.

**Read [`docs/north_star.md`](docs/north_star.md) first.** It says what Alice is
for, which parts of the Jarvis comparison are achievable with today's technology
and which are set dressing, and gives the decision rules to apply when a change
is ambiguous. Most disagreements about this codebase are settled there.

This repository is organized around a central turn loop and contract pipeline.
The current goal is companion-quality behavior: coherent state, disciplined actions,
and continuity across turns.

## Engineering Standard: Verified Growth

Verified Growth is the permanent engineering standard for Alice. Every accepted change
must define:

1. Capability added
2. Behavior replaced
3. Invariant enforced
4. Evidence produced
5. Visible user flow improved
6. Old code deleted or simplified where possible

Reference: `docs/verified_growth.md`

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
python scripts/setup_nltk.py   # optional: NLTK corpora, for better tokenizing
```

Add `-r requirements-api.txt` if you want the HTTP API (`app/api`, Docker).
The terminal companion does not need it.

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

### Choosing the model

The model is set in one place, for every entry point (CLI, UI, API, quality
harness):

```bash
export ALICE_MODEL=qwen3:14b                 # default: llama3.1:8b
export ALICE_OLLAMA_HOST=http://localhost:11434
export ALICE_NUM_CTX=8192                    # context window sent on every request
export OLLAMA_API_KEY=...                    # only when ALICE_OLLAMA_HOST is ollama.com
```

`--model` on the command line overrides `ALICE_MODEL`. Thinking models
(qwen3, gpt-oss) work: their reasoning is never shown. Cloud models
(`gpt-oss:120b-cloud`) work through a local Ollama that is signed in. A model
you name is never swapped for another; if it is not pulled, Alice says so.

## Optional Dependency Bundles

Install only what you need:

```bash
pip install -r requirements-dev.txt
pip install -r requirements-voice.txt
pip install -r requirements-integrations.txt
pip install -r requirements-ops.txt
```

## Measuring answer quality

The test suite runs without Ollama, so it proves the plumbing and nothing more.
To measure how well she actually answers, run the harness against your own local
model:

```bash
python scripts/quality_harness.py
python scripts/quality_harness.py --model llama3.1:8b --json before.json
# make a change, then:
python scripts/quality_harness.py --compare before.json
```

It checks each turn mechanically — did she call the tool that had the answer,
did she state a fact she never read, did she reach for a tool on a turn that was
just conversation — and reports per-scenario FIXED / REGRESSED between runs.
Scenarios live in `scenarios/quality/suite.json`.

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

`docker compose up --build` starts Alice plus a local Ollama and waits for the
model server to report healthy.

```bash
docker compose up --build
docker compose --profile gpu up --build   # attach an NVIDIA GPU to Ollama
```

The GPU is opt-in because a `reservations.devices` block is a hard requirement
rather than a preference: with it always on, `docker compose up` fails outright
on any machine without an NVIDIA card instead of running on CPU.

## Repository Notes

1. Experimental and low-frequency scripts are archived under `archive/2026-04/`.
2. Core runtime is under `app/` and `ai/runtime/`.
3. Keep default-path changes focused on policy/state/verification quality.

## License

Private project. All rights reserved.
