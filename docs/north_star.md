# What Alice is for

Alice is an attempt at J.A.R.V.I.S. with 2026 technology, running locally, on
one person's machine.

That comparison is useful because it is specific. Jarvis is not "an AI
assistant" — he has a recognisable character that most assistants lack, and most
of what makes him recognisable is achievable now. This document separates the
part that is achievable from the part that is set dressing, and gives rules for
deciding when it is not obvious.

It exists because this codebase has repeatedly chosen the shape of the thing over
the thing. Read it when a change feels ambiguous.

---

## The one-line test

> **Would Jarvis have gone and looked, or would he have talked about looking?**

Almost every decision in this repository reduces to that. Alice had a
`list_workspace_files` tool and a `get_current_weather` tool for months, and
could not call either of them unless a keyword router happened to recognise the
phrasing. She would *discuss* the contents of a directory she had never read.
That is the failure this project keeps making, in new costumes.

---

## What is real, and what is not

### Real, and therefore in scope

| Jarvis does | We can do | Notes |
|---|---|---|
| Answers from the actual state of the world | Tool calls against the real filesystem, real services, real system metrics | The point is not that a tool exists; it's that the model can *decide* to use it |
| Remembers across sessions | SQLite-backed memory with semantic recall | Continuity is most of what separates a companion from a chatbot |
| Takes multi-step action | Reason/act/observe loop with a step budget | List, then read what you found, then answer |
| Knows what he is allowed to do | Graduated trust: auto / confirm / refuse, by blast radius | Reversible things run; irreversible ones ask |
| Has a consistent voice | One personality, no mode switching mid-conversation | |
| Volunteers relevant things | Heartbeat, ambient monitoring, proactive surfacing | Rare and earned, not a notification firehose |
| Speech in and out | Local STT/TTS | Real today, quality varies |
| Says "I don't know" | Grounding checks, refusal to state unverified facts | Jarvis is never confidently wrong |

### Not real, and therefore out of scope

- **Holograms, suit telemetry, flying anything.** Obviously.
- **General reasoning at human level.** A local 8B model is not Jarvis's mind.
  Design around what it is good at: deciding *which tool*, and phrasing what
  came back.
- **Perfect recall of everything ever said.** Storage is cheap; retrieval is
  hard. Recall is lossy and must present itself that way.
- **Reading intent from tone of voice, sarcasm, or a raised eyebrow.** We get
  text, and sometimes audio. Do not build features that assume more.
- **Acting on the physical world without a human in the loop.** Even where the
  API exists, the trust model does not.

### The interesting middle

These are the ones worth arguing about, and where this document earns its keep.

- **Anticipation.** Jarvis says "Sir, you have a call from Ms. Potts" before
  being asked. Real version: surface something only when it is both *timely* and
  *unambiguously relevant*. The bar is high because the failure mode — an
  assistant that interrupts with noise — is worse than one that stays quiet.
- **Personality.** Jarvis is dry, economical, occasionally arch. This is
  achievable through a system prompt and, critically, through *not* padding.
  It is not achievable through a table of canned witty lines. If a personality
  needs a lookup table, it is not a personality.
- **Initiative.** Jarvis runs analyses unasked. Real version: bounded background
  work on things the user has already expressed interest in, with results held
  until asked or until they matter. Not: speculative work that burns the
  machine's CPU on a guess.

---

## Decision rules

When a change is ambiguous, these decide it.

### 1. Go and look. Never describe looking.

If a tool could answer the question, the model must be given the chance to call
it. If no tool exists and one plausibly could, that is a feature request, not a
reason to generate a plausible-sounding answer.

**Corollary:** any number, filename, or fact in a reply that did not come from a
tool or from memory is a fabrication. Weather figures are the canonical case.

### 2. A canned string is a bug, not a feature.

If the answer to "how does Alice handle X?" is "there's a function that returns
the right sentence for X", that is a decision tree wearing an assistant's voice.
It does not generalise, it cannot be improved, and it produces exactly the
uncanny flatness this project is trying to escape.

Hand-written heuristics are acceptable in exactly two places:
- **Safety**, where deterministic behaviour is the requirement (the trust tiers).
- **Grounding checks**, where we verify the model did not invent something.

Everywhere else, the model decides and a check verifies.

**Substituting is not overriding.** Two things here look alike and are not. A
template that *substitutes* for a missing answer — the model was unreachable, or
returned nothing — is a fallback, and it stays; something has to be said. A
template that *overrides* an answer that already exists, because it failed a
shape test (under 70 characters, no comma, no question mark), is the bug. That
is how a direct reply became a menu and a one-line confirmation became an essay.

The override paths are behind `ALICE_ENABLE_SCRIPTED_OVERRIDES`, off by default.
They are kept rather than deleted so the two behaviours can be compared with
`scripts/quality_harness.py` on a machine with a real model, instead of argued
about. If the comparison says the templates win on some turn, that is a finding
about the prompt or the model, not a reason to restore the default.

### 3. Say what happened, in the user's words.

The user did not ask about a route, a lane, a decision band, or a pipeline.
"Falling back to language model response" is accurate and useless. So is
"Working on it" when nothing is working on it.

Failure messages are the ones people actually read. They should state what
happened and what to do, and nothing else.

### 4. Never be confidently wrong.

Ranked worst to best:
1. A confident, plausible, false answer. *(Unrecoverable — the user cannot tell.)*
2. Silence or a crash.
3. "I don't have that."
4. The right answer.

Design so that (1) is structurally impossible, even at the cost of more (3).
Answering "what did I say on March 3rd" with a memory from May is (1), which is
why it was worth fixing even though the reply looked helpful.

### 5. Startup is part of the product.

An assistant you wait a minute for is one you stop opening. Nothing blocking in
the constructor: no model loads, no network calls, no sleeps, no retries against
something that will not answer. Warm things in the background; degrade when they
are not ready.

Budget: **usable prompt in under 5 seconds**, cold, offline.

### 6. Local means local.

No cloud dependency on the default path. No telemetry. No API keys required to
have a conversation. Rate limits and quotas make no sense against a model running
on your own machine — if something is throttled, it should be to catch a runaway
loop, not to ration access.

Offline is the *normal* case, not the error case. Everything must degrade to
something honest when the network is gone.

### 7. Reversible things run. Irreversible things ask.

Asking permission for everything is a confirmation dialog with a personality.
Asking for none is not something you leave running. The line is blast radius:
reads and workspace edits are recoverable from a checkpoint, so they run.
Anything that leaves the machine, reaches another person, or cannot be undone
stops and asks.

### 8. Delete more than you add.

This codebase is ~155k lines, roughly half of it unreachable. Every subsystem
that is built but never called is a thing a future reader must understand before
they can change anything. A feature that does not affect a reply is not a
feature.

Before adding a layer, check whether the existing one is actually wired up. It
is frequently not.

### 9. If you cannot measure it, you cannot claim it.

`scripts/quality_harness.py` runs real turns against a real model and checks them
mechanically. Any claim about Alice being "better" should be a diff between two
harness runs, not an impression.

Startup time, recall latency, and tool-use rate are numbers. Use them.

---

## Anti-patterns, with examples from this repository

Each of these was real.

- **Ceremony without effect.** A 2,300-line executive controller whose decision
  was re-derived from the route it was handed, and whose output only ever reached
  metadata.
- **Logs that lie.** `"All 10 tier improvements initialized successfully,
  active_systems=10"` printed unconditionally while all ten were quarantined and
  none were constructed.
- **Guards that never fire.** An intent classifier referencing an enum member
  that did not exist, raising every call, caught by a blanket `except`, returning
  `None` 100% of the time while appearing to work.
- **Statistics from a biased sample.** "Success rate" computed over a log that
  only failures are written to, so every intent scored near zero forever, and
  every turn was demoted a confidence band.
- **Precision about nothing.** Blending uninformative priors into a confidence
  score so that knowing *nothing* about a turn cost it 15 points.
- **Fake intelligence.** `_project_ideation_narrowing_question`,
  `_is_weather_clothing_time_range_request`, `_native_conceptual_answer` — ~120
  methods of regex dispatch returning templated strings.

The common thread: each *looks* like sophistication in the source and produces
nothing a user could notice. When in doubt, ask what a user would observe if the
code were deleted. If the answer is "nothing", delete it.

---

## What "done" would look like

Not a checklist to complete — a description of the target, so progress is
legible.

- She starts in under five seconds and is useful immediately.
- Asked something a tool can answer, she calls the tool, every time, whatever
  words were used.
- Asked something conversational, she answers, without reaching for a tool.
- She remembers across restarts, and is accurate about what she does and does
  not remember.
- She acts on the machine within a trust boundary the user understands, and
  every irreversible action is either confirmed or reversible.
- When the model is down, the network is gone, or a tool fails, she says so in a
  sentence a person would say.
- She never states a fact she did not verify.
- The codebase is small enough that one person can hold the turn path in their
  head.

---

## Using this document

When you are unsure:

1. Apply the one-line test.
2. Check whether the decision rules already settle it.
3. If it is in "the interesting middle", the burden is on the change to show a
   user-visible improvement.
4. If it is sci-fi, say so and move on. There is enough real work.

If a rule here turns out to be wrong, change it here rather than working around
it in code.
