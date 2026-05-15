# Model-First Runtime Principle

Alice's `alice-ollama` model is the response brain for user-facing language.

Runtime systems guide, validate, and protect model output. They do not replace Alice's voice with scripted personality.

## Deterministic Runtime Responsibilities

- Route and classify intent.
- Execute tools and local actions.
- Verify claims and evidence.
- Enforce approval and risk controls.
- Validate model outputs against policy and safety constraints.
- Ask the model to regenerate when validation fails.

## Deterministic Runtime Non-Responsibilities

- Returning canned companion greetings.
- Returning canned emotional acknowledgements.
- Faking warmth/personality with scripted lines.
- Replacing failed model output with scripted personality text.

## Safety Contract

The runtime must continue to block unsupported claims, including fake memory, fake action, fake deletion, fake background work, and unsupported continuity.

Safety validators remain active. Model-first means "model writes, runtime verifies," not "runtime disables guardrails."
