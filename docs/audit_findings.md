# Audit findings

Fifteen dimensions of Alice's turn path, audited in parallel, each finding then
handed to a separate agent whose only instruction was to refute it. This is the
surviving set, and the work queue for what comes next.

It is recorded here because the fleet ran in an ephemeral container and these
would otherwise have evaporated with it. Findings are the auditors' words, not
mine — read them as reports to verify, not as facts. Where a skeptic has ruled,
the count of what it refuted is given; where it had not yet reported when this
was written, the findings are marked unverified and deserve more suspicion.

`docs/north_star.md` decides anything ambiguous. Rule 9 in particular: a claim
that Alice is better is a diff between two `scripts/quality_harness.py` runs,
not an impression.

---

## context assembly

*5 survived an adversarial review; 4 refuted.*

On a normal conversational turn Alice sends the model ~7,200 characters, of which 79% is a static persona preamble, 4% is a garbled memory dump, 17% is raw history and 0.4% is the actual question — and the 370-line context builder that was supposed to supply episodic recall, semantic matches, conversation summaries and cross-restart continuity (`app/main.py:_build_llm_context`) is never called by anything but a test. The only real continuity she has is an in-process Python list that is wiped on restart, skipped entirely by the tool path, and polluted with engineered prompt scaffolding and drafts she never actually said.

**The biggest one:** Alice writes every turn into an episodic store, a semantic index, a conversation summarizer and a persisted conversation state file — and reads none of it back into the prompt, because the only function that assembles those into context, `app/main.py:_build_llm_context` (line 2328, ~370 lines), has zero production call sites. The live turn path never invokes it. So all the continuity machinery is write-only, and the model's entire memory of the relationship is an in-process `list` on the LLM engine plus a three-bullet `user=.../assistant=...` transcript fragment. That is exactly why she…

### Tool turns run with no companion context and are never written back to the transcript

`ai/runtime/boundaries/boundary_factory.py:151` — high, effort medium

`ReactLoop.run` accepts `context: Optional[str] = None` (ai/core/react_loop.py:150-153) and injects it as a second system message (react_loop.py:166-168), but this call passes nothing. The tool turn therefore runs on bare `SYSTEM_PROMPT` — no name, no memory block, no intent framing, and `chat_with_tools` (llm_engine.py:875) never touches `conversation_history`, so the loop also sees zero prior turns. This path is on by default: `_may_reach_for_tools` (line 105-121) returns True for any `route == "llm"` turn…

**Fix.** In `_try_tool_grounded_answer` pass the same block the conversational path builds: `loop.run(str(req.user_input or ""), context=<companion context>)`. `_build_companion_context` is currently a closure at boundary_factory.py:712 while `_try_tool_grounded_answer` is module-level at line 124 — lift it to module scope (or thread it in as a parameter from the call site at line…

### Recalled memories are pasted into the system prompt as a raw `user=…/assistant=…` log, cut at the first period

`ai/runtime/boundaries/boundary_factory.py:909` — high, effort medium

Verified end to end. `ai/runtime/contract_pipeline.py:1274` stores every turn as `"content": f"user={user_input}\nassistant={response_text}"`. That payload carries no `domain`/`kind`/`scope`, so `_store` (boundary_factory.py:2266-2279) falls to `alice.memory.store_memory(content=text, memory_type="episodic")`. On an ordinary turn `_recall` uses `alice.memory.search(...)` (line 2197), which returns those rows, and they land here as `memory_items`. The block she is handed under "Relevant context from memory:" is…

**Fix.** Two places. In `contract_pipeline.py` around line 1274, store the episodic content as prose (or as separate `user_text`/`assistant_text` fields), not a `key=value` blob. In `_build_companion_context` (boundary_factory.py:712, memory block at 900-916), replace `content.split(".")[0]` with a sentence-boundary truncation that ignores periods inside tokens, strip any leading…

### "Do you remember…" is answered by a canned heading plus a bullet dump of database rows

`ai/runtime/boundaries/boundary_factory.py:1076` — high, effort medium

`_is_personal_memory_query` (line 969-976) matches `\bdo you remember\b`, `\bwhat do you remember\b`, `\bwhat did i (talk|say|mention|share)\b`, `\babout me\b` — all ordinary phrasings. The caller at line 2840 short-circuits generation entirely: `_render_personal_memory_summary` returns this string and it is handed straight to `ResponseOutput`. `_surface_text` (line 1402-1439) only runs style clamps; no model ever sees the question. So the most human question in the product returns a literal heading and a bullet…

**Fix.** Keep both grounding gates — `_has_sufficient_personal_evidence` (line 1012) and the `requested_period`/`items_within` filter (lines 1055-1062), which are what stop her inventing a memory. Change only the rendering: pass the surviving rows into `_build_companion_context` as a labelled evidence block, call `alice.llm.chat(req.user_input, use_history=True, context=...)`, and run…

<details><summary>2 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/runtime/boundaries/boundary_factory.py:2965` | The hedge retry — the answer the user actually reads — is generated with the conversation switched off |
| medium | `ai/core/llm_engine.py:356` | conversation_history is never persisted, so every restart begins at turn zero |

</details>

---

## conversation memory feel

*11 findings, **not yet adversarially verified**.*

Alice does maintain a real role-based transcript (`llm_engine.conversation_history`, last 30 messages), so the raw model usually does know what "it" refers to — but almost every layer built on top of that either corrupts the user's sentence before classification, or strips the evidence of remembering out of the answer before the user sees it. I traced "I'm thinking about rewriting the memory layer" → "what would you do first?" through every module in scope and ran the pieces that don't need Ollama: the topic tracker never learns the topic, the reference resolver is handed hardcoded empty state, the coreference engine splices note titles into the middle of words, and a post-hoc guard deletes any sentence in which she refers back to the conversation. The net effect is a reply that is correct-ish but flat, occasionally a "what do you mean by 'it'?", and never once "you mentioned…" — which…

**The biggest one:** Alice is structurally forbidden from sounding like she remembers. `assess_continuity_claims` runs on the text of every model reply (boundary_factory.py:2998-3003) and deletes any sentence containing "you mentioned", "we were discussing", "when we last spoke", "as usual" unless it finds a retrieved memory whose `context["source"]` is one of four literal strings — and the code that writes turn memories (contract_pipeline.py:1273) never sets that key at all, so the check can never pass on in-session recall. Verified live: "You mentioned the memory layer. I'd pull the read path out first." comes…

### Continuity guard deletes every in-session callback because turn memories carry no `source` key

`/home/user/A.L.I.C.E/ai/runtime/continuity_claim_guard.py:210` — critical, effort medium

Alice never says "you mentioned" or "we were discussing" — the sentence is silently cut out of her reply. She answers the follow-up correctly but with no acknowledgement that a previous turn happened, so every answer reads as a fresh, context-free response to a standalone command. Worse, it is arbitrary: "you said storage was the messy part" survives (not in the pattern list) while "you mentioned the memory layer" is deleted, so her callbacks appear and vanish for no reason a user could infer.

**Fix.** The current conversation is not a claim that needs proving — it is the thing Alice is looking at. In `assess_continuity_claims`, add the live transcript as a first-class evidence source: accept an `in_session_turns` argument (pass `alice.llm.conversation_history` from boundary_factory.py:2998) and have `_recent_session_items` treat any claim whose topic tokens overlap the last…

### When the callback was the whole reply, the answer is replaced with a canned status banner

`/home/user/A.L.I.C.E/ai/runtime/continuity_claim_guard.py:311` — critical, effort small

The user asks "what would you do first?" and gets back "I am here. No active task is loaded yet, and we can continue an existing project or start fresh." — a boot banner, in the middle of a conversation, in place of an answer she had already generated. I reproduced this exactly: the reply "We were discussing the memory rewrite, so I'd start with the read path." becomes that sentence verbatim. It is the most terminal-like output in the entire codebase and it is reachable from an ordinary two-turn exchange.

**Fix.** Delete the canned string from `assess_continuity_claims`. When stripping empties the text, return the ORIGINAL `text` unchanged and set `unsupported_continuity_claim=True` so the caller can decide; a slightly over-confident callback is strictly better than a status banner where an answer should be. If a regeneration is wanted, have the caller in boundary_factory re-ask the…

### Coreference substitution uses `str.replace`, splicing entity titles into the middle of unrelated words

`/home/user/A.L.I.C.E/ai/core/coreference.py:477` — critical, effort small

The user's sentence is corrupted before intent classification. Verified outputs after a single earlier note titled "Groceries": "what should I do with it?" → 'what should I do w"Groceries"h it?'; "I'm thinking about rewriting the memory layer. Is it worth it?" → 'I'm thinking about rewr"Groceries"ing the memory layer. Is it worth it?'. The user sees the downstream effect: a wildly wrong intent, irrelevant memory recall, or "I didn't follow that" — on a perfectly ordinary sentence, with no way to guess why.

**Fix.** In `AdvancedCoreferenceResolver._apply_resolutions`, replace every `text.replace(old, replacement, 1)` with position-based splicing using the match object already in hand: `text = text[:m.start()] + replacement + text[m.end():]`, exactly as the ORDINAL branch at line 369 already does. Add a regression test asserting that "what should I do with it?" resolves only the standalone…

### Reference resolver is handed hardcoded empty topic/subject, so ordinary follow-ups route to "what do you mean?"

`/home/user/A.L.I.C.E/ai/runtime/boundaries/boundary_factory.py:1932` — high, effort medium

One turn after the user states a subject, a short follow-up containing a pronoun gets sent to the clarify route with an empty options list — Alice asks what "it" means about the thing the user just said. Verified: with this exact state dict, "how would you approach it?" → needs_clarification=True, unresolved=['it'], options=[]; "can you help me with that" → the same. With `last_subject` populated (the only difference), the same input resolves cleanly to "how would you approach rewriting the memory layer?" at…

**Fix.** Populate `_pre_state` from the trackers that already exist: `current_topic` and `active_goal` from `alice.conversation_state_tracker.get_state_summary()` (`conversation_topic`, `user_goal`), `last_subject` from the last non-pronoun noun phrase of the previous user turn, and `referenced_entities` from `alice.entity_registry` recent labels. That change alone turns the clarify…

### The hedge-retry regenerates the user's answer with the conversation history switched off

`/home/user/A.L.I.C.E/ai/runtime/boundaries/boundary_factory.py:2965` — high, effort medium

On exactly the turns where context matters most, the answer the user is shown was written by a model that had never seen the previous turn. "I'm thinking about rewriting the memory layer" / "what would you do first?" — the first pass says "It depends…", which is in the hedge list, so a second pass runs with zero history and the user gets a generic answer about first steps in the abstract. The reply is coherent and completely disconnected from what they just said, which is the exact texture of talking to a terminal.

**Fix.** Keep the transcript and drop only the bad turn. Give `LocalLLMEngine.chat` a `history_override` (or add `drop_last_exchange: bool`) so the retry can pass `use_history=True` with the hedged exchange excluded, and change this call site to use it. The retry instruction in `_retry_ctx` already tells the model not to hedge; it does not need amnesia to comply.

### "Do you remember…" returns a bulleted dump of raw database rows instead of an answer

`/home/user/A.L.I.C.E/ai/runtime/boundaries/boundary_factory.py:1076` — high, effort medium

Ask "do you remember what I said about the memory layer?" and Alice prints a heading and a bullet list of stored rows in their storage format. Because turn memories are written as `f"user={user_input}\nassistant={response_text}"` (ai/runtime/contract_pipeline.py:1274 — confirmed in data/memory/alice.db), the user literally sees `- user=what should I work on next?` / `assistant=I didn't follow that.` This is a SELECT statement rendered to the screen. It is also the only way Alice ever refers to the past, since…

**Fix.** Change `_render_personal_memory_summary` to return evidence, not prose: hand the deduped snippets to `alice.llm.chat` as context with an instruction to answer from them in one or two sentences and to say so if they do not cover the question, then run the existing `MemoryAnswerVerifier` on the model's text. Keep the out-of-period guard at line 1051-1060 (that one is a grounding…

### Every "it"/"that" in conversation anchors to the last note title, however old or unrelated

`/home/user/A.L.I.C.E/ai/core/coreference.py:502` — high, effort small

Days-old subjects leak into new conversations. Verified: after one `notes:create` for "Groceries" followed by eight unrelated turns, "what do you think about that?" is rewritten to 'what do you think about "Groceries"?' and "delete the file" to 'delete "Groceries"'. Alice then answers confidently about the wrong thing, with no hedge — north-star rule 4's worst case, where the user cannot tell she is wrong.

**Fix.** In `_last_note_ref`, bound the anchor by recency: return the NOTE_REF only if `self._turn - mention.turn_index <= 2`, otherwise return None so the pronoun stays unresolved and the model resolves it from the transcript. Additionally, gate PRONOUN_GENERIC and DOMAIN_PRONOUN on the current turn's intent being in the same plugin family as `mention.plugin` — a notes title has no…

<details><summary>4 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `/home/user/A.L.I.C.E/ai/memory/conversation_state.py:243` | The topic tracker only learns a topic from six lookup phrases, so it is empty for real conversation |
| medium | `/home/user/A.L.I.C.E/app/main.py:2328` | The entire conversation-state and recall context block is computed every turn and never reaches the model |
| medium | `/home/user/A.L.I.C.E/ai/core/nlp_processor.py:5272` | ConversationMemory records every turn with an empty response, and its context hint never reaches the model |
| medium | `/home/user/A.L.I.C.E/ai/core/llm_engine.py:355` | The one memory that actually works is in-process only and is lost on every restart |

</details>

---

## dead code round 2

*11 findings, **not yet adversarially verified**.*

Round 1 deleted code nothing could *reach*; what remains is worse for the user's complaint — roughly 6,700 lines that are reached, constructed, ticked every turn, and then discarded. Ten "tier improvement" modules, a 33-object "roadmap completion stack", ten "cognitive engines", a per-turn entity-relationship graph, and a memory consolidator all run and write into attributes or dict keys that zero code reads, while a dead 819-line keyword router and a dead canned-greeting table sit in the tree looking like the live path.

**The biggest one:** Every subsystem that would make Alice feel like someone thinking — tone trajectory, weak-spot detection, cross-session pattern detection, goal arbitration, memory consolidation, a relationship graph over the user's life — exists as working code and is wired to nothing. `app/main.py:777` does `setattr(self, attribute, ...)` for ten "tier improvements" and no line anywhere reads those attributes; `self.roadmap_stack` is built from 33 objects at line 884 and read zero times; ten more engines are constructed at lines 1230-1241 and read zero times. What actually survives to the user is the bare…

### Ten "tier improvement" subsystems are constructed into attributes nothing ever reads

`app/main.py:777` — critical, effort medium

Alice never adapts her tone over a session, never notices she keeps failing the same kind of question, never arbitrates between competing goals, never summarises a long session. The user turning on ALICE_ENABLE_TONE_TRAJECTORY_ENGINE=1 gets a log line saying it is active and literally no behavioural change, because the object is stored and never consulted.

**Fix.** Delete the tier_specs table and the `enable_advanced_tiers` branch in app/main.py (lines ~697-802), the 10 entries from QUARANTINED_SUBSYSTEMS in ai/infrastructure/runtime_flags.py, tests/integration/test_all_tier_improvements.py, and the 10 modules: ai/memory/session_summarizer.py, ai/infrastructure/capability_constraints.py, ai/core/result_quality_scorer.py,…

### self.roadmap_stack builds 33 objects and a 4-thread pool at startup and is never read

`app/main.py:884` — high, effort small

Startup pays for a ThreadPoolExecutor, two `data/security/` directory creations, and a full line-by-line hash-chain replay of two JSONL audit logs that grow forever — on every launch, for an object no code touches. The audit logs grow, are re-parsed at every boot, and nothing ever writes to or reads from them at runtime.

**Fix.** Delete line 884 and the `from ai.roadmap import get_roadmap_completion_stack` import at app/main.py:19, then delete ai/roadmap/completion_stack.py, ai/roadmap/__init__.py's re-exports, and tests/integration/test_roadmap_completion_stack.py, test_security_and_ops_stack.py, test_route_contracts_and_recovery.py. These tests assert that constructors construct; they pin nothing a…

### Ten "cognitive engine" objects constructed unconditionally on every startup, read by nothing

`app/main.py:1230` — high, effort medium

"What did I ask you an hour ago?" gets no temporal reasoning; a multi-part request gets no multi-step decomposition; a stated constraint ("keep it short", "only Python files") is never extracted into anything that shapes the answer. All of that machinery is loaded and instantiated on every boot and then sits untouched, which is a direct contributor to the flat, single-shot feel.

**Fix.** In app/main.py, delete the ten unread assignments in the block at 1230-1245, their imports at lines 99-110, their entries in ai/core/__init__.py, and the ten modules…

### ai/infrastructure/router.py: an 819-line keyword router that nothing imports, kept alive by one test

`ai/infrastructure/router.py:45` — high, effort small

No direct user symptom — the harm is that this is the most prominent, most readable routing implementation in the tree, and it is a regex/keyword decision tree of exactly the kind the north star names as the project's recurring failure. Anyone (human or model) trying to fix "why doesn't she call the tool" finds this file, edits it, and changes nothing, because the live path is ai/core/routing/route_arbiter.py plus ai/runtime/boundaries/boundary_factory.py.

**Fix.** Delete ai/infrastructure/router.py and tests/integration/test_conversational_guardrails.py's RequestRouter cases (or rewrite those cases against the live RouteArbiter so the guardrail they claim to protect is actually protected). If the guardrails matter, they belong as golden tests against turn_orchestrator, not against an orphan class.

### Per-turn entity relationship extraction runs twice and the result goes only to logger.debug

`app/main.py:6564` — high, effort medium

Alice silently builds a graph of the people, places and things in the user's life on every single turn — twice per turn — and can never use it. Ask "who is Sarah?" or "what do you know about my team?" and nothing consults the graph; the only way to see any of it is the `/entities` slash command, which prints a count. That is the difference between a companion and a terminal, and the data to be a companion is already there and thrown away.

**Fix.** Either wire it in or cut it. To wire it in: in the context-assembly function that builds `context_parts` (app/main.py around line 2409-2460, where episodic and semantic hits are already injected), add a block that calls `self.relationship_tracker.get_entity_relationships(entity)` for entities the NLP layer found in the current turn and appends them as grounded context. To cut…

<details><summary>6 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `app/main.py:6516` | memory_consolidator runs every 15 turns and writes to a dict key no code reads |
| medium | `features/welcome.py:17` | features/welcome.py: a 289-line canned greeting table that is unreachable, plus two other greeting systems |
| medium | `ai/core/decision_engine.py:38` | ai/core/decision_engine.py and ai/core/semantic_similarity.py have zero references in the entire repository |
| medium | `ai/planning/goal_tracker.py:152` | ai/planning/goal_tracker.py is dead and shadows the live ai/core/goal_tracker.py by name |
| medium | `app/main.py:421` | Four more startup constructions whose attributes are never read |
| low | `ai/runtime/pipeline/state_updater.py:1` | Thirteen placeholder and shim modules advertise a boundary layer that is one 3,225-line file |

</details>

---

## degraded paths

*12 findings, **not yet adversarially verified**.*

I booted Alice with Ollama genuinely unreachable and ran real turns: she never once says the model is down, she blames the user instead, and on two turns she throws away evidence she had already gathered and claims she has none. A tight grep finds ~81 distinct hand-written user-facing failure sentences across ai/ and app/ (489 including internal ones); the reachable ones leak plugin class names, raw error codes ("unknown_location"), and planner vocabulary ("active objective", "agent loop", "next safe step"), and three of them announce a retry that no code performs.

**The biggest one:** With Ollama down, Alice never names the outage — she blames the user, in the same sentence, forever. Verbatim from a real offline run: asking "what do you think about the router design?" four times in a row returns "I didn't follow that. Say it once more and I'll answer directly." four times, character-identical. The instruction is a lie (saying it once more changes nothing), and the one honest message in the codebase — "I can't reach the AI service. Make sure Ollama is running." at ai/infrastructure/errors.py:155 — lives in a module that is imported by zero files. A user in this state cannot…

### Model-unreachable is swallowed and re-reported as the user's fault

`ai/runtime/boundaries/boundary_factory.py:2994` — critical, effort medium

Real offline run, four turns with Ollama down: turn 1 "hey" -> "Hey Gabriel. What's the move?" (cheerful, nothing wrong); turn 4 "what do you think about the router design?" -> "I didn't follow that. Say it once more and I'll answer directly." At no point in the session is the outage mentioned. The user concludes Alice is stupid, not that her model is off.

**Fix.** In `_generate` (boundary_factory.py), change the handler at 2994 to bind the exception and set a local `llm_unreachable = True` when the message or cause is a requests ConnectionError/Timeout (or matches the engine's own 'Ollama not running' / 'Request timeout' raises). When that flag is set, return a ResponseOutput whose text names the outage in one sentence — e.g. "My model…

### "Say it once more and I'll answer directly" is a verbatim infinite loop

`app/main.py:2214` — critical, effort small

Verified on a real offline run — the same question asked four times returns the identical string four times: T1..T4 ALICE: "I didn't follow that. Say it once more and I'll answer directly." The message explicitly instructs the user to repeat themselves and promises a direct answer. Repeating produces the same message. A user can sit in this loop indefinitely.

**Fix.** Make `_fallback_from_intent` take the previous turn into account. Compare `user_input` against `self.last_user_input`; if this recovery already fired for a near-identical input, do not repeat it — state the blocker instead ("Same problem as last time: I can't reach the model, so I can't answer that one"). And delete the promise: never emit "I'll answer directly" from a path…

### A successful 400-file workspace listing is deleted, then Alice says she has no result

`ai/runtime/operator_response_surface.py:667` — high, effort small

"what files are in the workspace?" -> "I don't have a result for that yet. Tell me what to look at and I'll go and check." She already looked: local_execution reported {'action': 'code:list_files', 'workspace_file_count': 400, 'success': True}. Instrumenting render_operator_response shows it is handed the full listing as base_text (five highest-value files plus ten more) and returns the "no result" line. This is the north star's worst category: a confident false statement about her own state, and talking about…

**Fix.** In `render_operator_response`, strip the label SPAN, not the line. Replace the 666-668 block with a per-line `re.sub(r"(?i)\s*\bnext best move\b\s*:.*$", "", ln)` plus the same for `finding:`, keep any line that still has text, and only discard a line that is empty afterwards. Add a guard: if `base_stripped` came out empty but `base_text` was non-empty and…

### Every per-tool entry in FallbackGraph is unreachable — all tool failures get one generic line

`ai/runtime/turn_orchestrator.py:51` — high, effort small

"what's the weather in Boston?" with geocoding failing -> "I couldn't get that from the tool." The graph contains the right sentence for this exact error — ("weather", "unknown_location") -> "I couldn't find that location. Could you try a nearby city?" — and it is never used. Whatever fails, whatever the reason, the user gets the same six words about "the tool", a thing they never mentioned.

**Fix.** Two-line fix at the call site: in `_verification_fallback`, derive the lookup key from the routed intent, not the plugin class. Pass `decision.intent` down from `respond_phase` (it is available as `verify_phase`'s RouterDecision) and use `intent.split(':')[0]` — "weather", "notes", "local". As a belt-and-braces measure, normalise in `FallbackGraph.get_steps`: lower-case, strip…

### Second failure of the same tool prints the plugin class name and the raw error token

`ai/runtime/turn_orchestrator.py:59` — high, effort small

Verified second consecutive weather failure: "The WeatherPlugin action ran into an issue (unknown_location). Try rephrasing or check that the target exists. (Weather data unavailable — try again in a moment.)" The user sees an internal Python class name, an internal error constant, advice that makes no sense for weather ("check that the target exists" — which target?), and a second contradictory tail. Four registers in one sentence, none of them a person's.

**Fix.** Never interpolate `diagnostics['tool']` or `diagnostics['error']` into user text. In `_verification_fallback`, map the tool to a plain noun first (a small dict: weather -> "the weather service", notes -> "your notes", local/code -> "the workspace"), and map error_type through the same normaliser used by `humanize_local_execution_error`. If either has no mapping, drop it rather…

### Three failure messages announce a retry that no code ever performs

`ai/runtime/fallback_policy.py:120` — high, effort medium

On a weather timeout the user is told "Weather service timed out — retrying." and the turn ends there. Nothing retries. They sit waiting for a result that will never arrive, then type again and get the next message in the ladder. This is the plainest form of promising work that never happens.

**Fix.** Either make the action real or stop claiming it. Cheapest honest fix: change the three retry-step messages so they describe only what is true — "The weather service didn't answer in time.", "Couldn't reach the weather service.", "That request timed out." Better fix: have `_verification_fallback` actually honour `step.action == 'retry'` by re-invoking `boundaries.tools.execute`…

### A completed file analysis is thrown away in favour of a five-word canned label

`ai/runtime/operator_response_surface.py:647` — high, effort medium

"can you look at app/main.py and tell me what it does?" -> "Looked at the runtime pipeline." She did read the file: the analysis dict in hand has line_count, import_count 126, class_count, function_count, risk_flags and suggested_next_files. None of it is said. Worse, she does not even name app/main.py — the user cannot tell she opened the right file. Ask about a different file and the same shape comes back ("Looked at the routing layer.").

**Fix.** In `render_operator_response`, demote the responsibility branch below the evidence. When `summary` is empty but `analysis` is populated, compose one sentence from the facts that were actually measured — file name, line count, class/function counts, any risk_flags — e.g. "app/main.py — 277 lines here, 126 imports, one class. It's the runtime pipeline." Only fall to the bare…

### The correct weather clarification is computed, stored, then overwritten one phase later

`ai/runtime/turn_orchestrator.py:326` — high, effort small

The user asks for weather with no location set and gets "I couldn't get that from the tool." instead of "What city should I check the weather for?" — a question they could have answered. The useful message was already built for this turn and is discarded unread.

**Fix.** In `respond_phase`, check the proposed response before overriding it: if `proposed.metadata.get('type')` is a fallback the execute phase deliberately authored (e.g. "weather_location_clarification") or the tool_result data carries `use_fallback_message`, publish `proposed.text` as-is and skip `_verification_fallback`. The verifier's job is to stop unverified claims, not to…

<details><summary>4 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/runtime/fallback_policy.py:273` | The repeat-failure counter never resets, so Alice reports a streak that is not true |
| medium | `ai/runtime/contract_pipeline.py:1207` | A parenthetical system note is bolted onto messages that already said it |
| medium | `ai/runtime/response_momentum_policy.py:279` | "Let me try again" is returned by a function that then stops |
| medium | `ai/runtime/turn_orchestrator.py:72` | Alice teaches the user command syntax instead of calling the tool she already has |

</details>

---

## greeting and openers

*10 survived an adversarial review; 2 refuted.*

On the path the user actually runs (`python -m app.alice`), Alice never speaks first at all — the first thing on screen is an ASCII banner ending in "System ready. Type /help for available commands", then a bare `❯` prompt. The 658-line greeting policy only runs *after* the user types, and when it runs it hard-bypasses the conversation path, switches memory off, forbids every continuity phrase by regex, and — as I confirmed by executing it — rejects 7 of 12 realistic model greetings and falls back to a loop of six hardcoded sentences.

**The biggest one:** Turn one is a boot message, not a person. `app/alice.py` prints a banner that says "System ready" and waits. There is no opener, no "hey", no mention of anything Alice knows. Three separate greeting systems exist (a 658-line policy, a 96-combination template bank in the UI, and a curated pool in features/welcome.py) and the production entry point calls none of them. The user's complaint — "it feels like I'm talking to a terminal" — is a literal description of the first screen.

### Alice never speaks first; the opening screen is a boot banner

`app/alice.py:94` — high, effort medium

`python -m app.alice` is the documented user entry point (the module docstring says so). After construction it prints the ASCII panel, a goal line, and an info panel whose last line is `System ready. Type /help for available commands` (ui/rich_terminal.py:249), then drops to a bare prompt. Alice emits no sentence of her own until addressed. The entire first screen is machine status.

**Fix.** Add `ALICE.open_session()` that runs one ordinary turn through `process_input`'s normal path — seeded with the top goal from `get_goal_engine().top_goal()` and the current time — and call it in `start_alice_rich` between `ui.show_welcome()` and the input loop, printing via `ui.print_assistant_response`. Remove the `System ready.` line from `RichTerminalUI.show_welcome`…

### Greeting turns short-circuit the conversation path before history or memory

`ai/runtime/boundaries/boundary_factory.py:2468` — high, effort medium

`greeting_turn` is set at boundary_factory.py:2423 purely from `decision.intent`, and ai/core/nlp_processor.py:4824-4825 returns `"greeting", 0.9` for any input of <=4 words containing hi/hey/hello/yo/sup/hiya — on every turn, not just the first. So "hey" on turn 20 of a live thread returns before the normal LLM branch is ever reached, and `_build_grounded_greeting` calls `alice.llm.chat(prompt, intent="greeting", use_history=False)` with a prompt containing only greeting rules, the user name and recent greetings.…

**Fix.** Delete the `if greeting_turn: return _build_grounded_greeting()` short-circuit at 2468-2469 so greeting turns fall through to the normal LLM branch with history and memory, as any other conversational turn does. Keep `render_grounded_greeting` only at the `if not llm_text` guard around line 3018 — as a substitute for a missing answer, never as an override of one that exists.

### Memory recall is switched off on exactly the turn where continuity matters

`ai/runtime/boundaries/boundary_factory.py:2196` — high, effort medium

In `_recall`, `greeting_turn` is computed at line 2149 from the request intent, and the `elif not greeting_turn` branch at 2196 means `alice.memory.search` is never called on a greeting turn; `items` stays `[]` and metadata reports `mode: greeting_active_state_only`, `broad_memory_suppressed: True`. You say "hi" after a week away and the SQLite store full of what you were doing is not consulted. She cannot mention anything that happened because she was not allowed to look — the north star's "go and look, never…

**Fix.** Remove the `greeting_turn` exclusion in `_recall` so `alice.memory.search` runs on greeting turns, and thread the recalled items into the greeting prompt built by `_try_constrained_llm_greeting`. Then pass those real items into `assess_continuity_claims` (currently called with `memory_items=[]` at greeting_surface_policy.py:226 and :387) so the grounding check does the work…

### With the model unreachable, the greeting is a six-line carousel

`ai/runtime/greeting_surface_policy.py:277` — high, effort small

Confirmed by running `render_grounded_greeting(user_name="Gabriel", llm_generate=None, user_input="hey")` eight times, threading `session_state` through: turns 1-6 emit the six pool entries in order, turn 7 wraps back to "Hey Gabriel. What's the move?", turn 8 to "Alright — what are we building or breaking?". Every one reports `generated_by='fallback'`, `validation_reasons=['unsafe_llm_greeting_rejected']`. Six rotating lines pretending to be variety. North star: "If a personality needs a lookup table, it is not a…

**Fix.** Cut `_FALLBACK_POOL` to a single entry `"Hey {name}."` and simplify `_fallback_greeting` (line 286) to return it. When the model is unreachable one flat honest line is correct; rotating six manufactures fake variety. Surface `generated_by == "fallback"` to the UI so the thin opener is visibly a degraded mode rather than her voice.

### The greeting prompt's "good examples" are verbatim the canned fallback pool

`ai/runtime/greeting_surface_policy.py:330` — high, effort small

These four sentences are character-for-character `_FALLBACK_POOL[0:4]` (lines 278-281) with `{name}` filled in. The model is four-shot primed with the exact strings the no-model path also emits, and `"Hey Gabriel. What's the move?"` passes `validate_greeting_candidate` cleanly (verified). So when Ollama is up an 8B model copies the examples, and the user cannot tell whether a model ran at all — the output is the same either way. The canned string was not removed, it was laundered through the model.

**Fix.** Delete the `Good examples:` line from `_build_prompt`. Describe the register instead (dry, short, specific, addresses him by name, may reference something she actually knows) and give the model the recalled memory items and top goal as the material to be specific about. Never hand the model sentences a fallback path can also emit.

### Most natural greetings are rejected by the banned-phrase lists, leaving only flat ones

`ai/runtime/greeting_surface_policy.py:493` — high, effort medium

Ran `validate_greeting_candidate(user_input="hey", time_period="", allow_focus_reference=False, user_name="Gabriel")` over realistic candidates. Rejected: "Hey Gabriel. What's up?" -> ['emotional_assumption']; "Hey. Something on your mind?" -> ['emotional_assumption']; "Hello Gabriel, how's the Django project going?" -> ['hallucinated_entity_in_greeting']; "Hey Gabriel" -> ['missing_sentence']; "Morning." -> ['low_signal_greeting']. Accepted: "Hey Gabriel. What's the move?" — a fallback-pool line. Whatever the…

**Fix.** In `validate_greeting_candidate`, drop the `emotional_assumption` map entry and the `_has_hallucinated_entity` call (line 445), and replace both with `assess_continuity_claims` given the turn's real `memory_items` and `operator_state`, rejecting only claims those do not support. A proper noun that came out of recalled memory is what you want; one that came from nowhere is what…

<details><summary>4 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/runtime/greeting_surface_policy.py:586` | time_period is always empty in production, so any mention of time of day is rejected |
| medium | `ai/runtime/greeting_surface_policy.py:63` | The active objective can never reach the greeting: the gate requires two fields, only one is ever set |
| medium | `ui/rich_terminal.py:484` | Saying goodbye never reaches Alice; the session closes with a shutdown notice |
| medium | `ui/rich_terminal.py:271` | A fake two-second progress bar runs before the real six-second load |

</details>

---

## nlp gating

*5 survived an adversarial review; 3 refuted.*

The NLP layer has two designed escape hatches from keyword matching — a sentence-transformers semantic classifier and an LLM intent arbiter — and both are non-functional in a default install: the arbiter's gateway is never wired (`attach_llm_gateway` has zero callers) and the classifier loads with `model=None` while logging "[OK] ... loaded". Everything therefore routes through ~700 lines of hardcoded phrase lists, which I confirmed by live probe misclassify ordinary conversation into tool intents at 0.9 confidence ("how much memory does a python dict use per entry" → `system:status`; "remind me why we chose SQLite" → `reminder:set`, a plugin that does not exist), while a "typo corrector" silently rewrites "talk"→"task" and "lost"→"list" before classification even begins.

**The biggest one:** Intent routing is 100% regex in practice. Both non-regex paths are dead — `NLPProcessor.llm_gateway` is set to None at construction and `attach_llm_gateway()` is never called anywhere in the codebase, so `_maybe_apply_llm_intent_fallback` returns unchanged on every turn; and `SemanticIntentClassifier._load_model` swallows its download failure, leaving a truthy classifier object with `model=None` that returns `[]` forever, which `_ensure_semantic_classifier` reports as "[OK] Semantic intent classifier loaded". The result is that every utterance is matched against keyword tuples before the…

### Any sentence containing a resource word plus a filler verb becomes system:status at 0.90 and gets answered with a CPU/RAM readout

`ai/core/nlp_processor.py:4482` — critical, effort small

Both sets are matched as SUBSTRINGS (nlp_processor.py:335-339: resources include 'ram', 'memory', 'disk', 'battery', 'network'; verbs include 'is', 'check', 'open', 'running'). 'ram' is inside 'program', 'framework', 'parameter'; 'is' is inside 'this', 'list', 'raise'. I ran the real pipeline: 'how much memory does a python dict use per entry' -> system:status 0.90; 'is the disk scheduler in linux still using CFQ' -> system:status 0.90; 'can you check if the battery on my bike light is dead' -> system:status 0.90;…

**Fix.** In _detect_intent_semantic (nlp_processor.py:4378), delete the bare conjunction at 4482-4485 and require the resource noun to be bound to THIS machine: match on word boundaries, not substrings (re.search(r"\b(my|this|the)\s+(cpu|ram|memory|disk|battery|gpu)\b") or r"\bsystem (status|health|load|usage)\b"). Drop 'is', 'open', 'check', 'running' from _P1_SYSTEM_RESOURCE_VERBS…

### "remind me why/what/how ..." is forced to reminder:set at 0.95, and no reminder plugin exists anywhere in the codebase

`ai/core/nlp_processor.py:3413` — high, effort small

The override fires on the bare substring 'remind me', before any interrogative check. Live probe: 'remind me why we chose SQLite over Postgres' -> reminder:set 0.95; 'remind me what you said about the scheduler' -> reminder:set 0.95. 'reminder:' is in every tool-domain tuple (app/main.py:2790, 2866, 2961, 3783), so main.py:2866 returns ('operator','tool_intent_domain') and the turn goes to plugin execution. ls ai/plugins/ has no reminder plugin; PluginManager.execute_for_intent('reminder:set', ...) returns None,…

**Fix.** In the reminder early-override block (nlp_processor.py:3397-3413), bail out before setting _reminder_intent when the words after 'remind me' are interrogative or retrospective: if re.search(r"remind me (why|what|how|who|when we|about (the|our|that)|of (the|our|that))", _tl) -> leave _reminder_intent None and fall through to normal routing. Additionally require creation…

### The typo corrector rewrites the ordinary English word "talk" into the command word "task" before classification, destroying the phrase cues that…

`ai/core/nlp_processor.py:2025` — high, effort medium

_closest_lexicon_term (nlp_processor.py:1977) accepts any alphabetic token >= 4 chars and snaps it to the nearest command-lexicon term at edit distance 1, with no check that the token is already a real English word. Live probe of the real pipeline: 'what did we talk about yesterday' normalizes to 'what did we task about yesterday' -> conversation:clarification_needed 0.62, while 'what did we discuss yesterday' -> memory:search 0.95. The corrector eats the exact word the memory-recall rule keys on…

**Fix.** In _closest_lexicon_term (nlp_processor.py:1977), return None immediately for tokens that are known English words: load a bundled ~20k-word frequency list into a frozenset at init (or NLTK's words corpus when present) and check membership before the distance search. Also raise the minimum length from 4 to 6 (nlp_processor.py:1979) — the 4-5 letter band is where real words…

### LLM intent arbitration is dead — attach_llm_gateway() has no callers, so the ambiguity fallback returns immediately on every turn

`ai/core/nlp_processor.py:1747` — high, effort small

self.llm_gateway is set to None at nlp_processor.py:1578 and the only writer is attach_llm_gateway (nlp_processor.py:1732), which grep finds zero callers for across ai/, app/ and tests/ — app/main.py builds self.llm_gateway at line 811 and hands it to the response formulator and summarizer, never to the NLP processor. So _ensure_llm_intent_classifier always returns None, and _maybe_apply_llm_intent_fallback (nlp_processor.py:1788, called on the ordinary-turn path at nlp_processor.py:3520) returns the keyword route…

**Fix.** In AliceCore.__init__ (app/main.py, right after self.llm_gateway = get_llm_gateway(...) at line 811), call self.nlp.attach_llm_gateway(self.llm_gateway). Then add a boot assertion/doctor check that getattr(alice.nlp, 'llm_gateway', None) is not None so this cannot silently regress. Note the arbitration only triggers below confidence 0.68, so it will NOT rescue the 0.90…

<details><summary>1 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/core/nlp_processor.py:4109` | Emotion, urgency and sentiment are computed on every turn and never reach the response; the only consumer field is never assigned |

</details>

---

## personality identity

*10 findings, **not yet adversarially verified**.*

Alice's character exists almost entirely as numbers in JSON files and as prohibitions in a prompt she often never sees. I traced every path from "personality trait" to "a word that differs in the output" and found that the 6-axis PersonalityEvolutionEngine currently contributes zero characters to the live prompt, the 4-axis world-model drift renders the same four hedged sentences it did on day one (verified against 463 sessions of real state in data/world_model.json), and the tool-grounded answer path — now the default for ordinary turns — sends a personality-free six-rule system prompt that literally reads like a CLI spec.

**The biggest one:** Most turns now reach the ReAct loop, and ReactLoop.run builds its message list with only react_loop.SYSTEM_PROMPT — a six-numbered-rule instruction sheet with no persona, no voice, no name for the user, and the explicit directive "Answer in at most four sentences... No preamble... no offers to help further. Never mention tools... State the finding directly, as if you simply looked." chat_with_tools never calls _build_system_prompt, and boundary_factory passes the loop's raw answer straight through as the user-facing reply. So the exact turns where Alice does something real are the turns…

### ReAct loop answers with a personality-free rules sheet; the 4.5KB persona is never sent

`ai/core/react_loop.py:169` — critical, effort small

Any turn that touches a tool — and since the routing fix that is most ordinary turns, because `conversational_tool_use` is not quarantined so `is_enabled` returns True by default (ai/infrastructure/runtime_flags.py:41, boundary_factory.py:119) — comes back in a register no human uses. The only system message is react_loop.py:29-44, whose rule 4 is "Answer in at most four sentences unless asked for detail. No preamble, no restating the question, no offers to help further" and whose rule 5 is "State the finding…

**Fix.** Give ReactLoop a persona. Add a `persona_prompt: str = ""` parameter to ReactLoop.__init__ and, in run(), build `messages` as [persona system message, tool-discipline system message, user message] instead of one. In boundary_factory._try_tool_grounded_answer, construct the loop with `persona_prompt=llm._build_system_prompt(intent=str(req.decision.intent or ''),…

### PersonalityEvolutionEngine contributes zero characters to the live prompt

`ai/core/llm_engine.py:288` — high, effort medium

Nothing. Literally nothing — and that is the finding. I ran it against the live store: `get_traits_for_user("default")` returns the untouched defaults `verbosity=0.5, formality=0.3, humor=0.4, directness=0.6`, and at those values every threshold above is false, so `trait_hints == []` and the "Personality calibration:" line is never appended. I confirmed this by assembling a real system prompt — 5,662 chars — and the string "Personality calibration" does not appear in it. A whole 417-line subsystem the startup log…

**Fix.** Delete the dead axes and make the live ones audible. In PersonalityTraits, drop `enthusiasm` and `empathy` (nothing reads them). In _build_companion_context's Layer 3, stop bucketing on absolute value and render deviation instead: compare each trait to PersonalityTraits() defaults and emit a hint only when |trait - default| > 0.1, phrased as a change ('Gabriel writes in…

### Personality drift renders the same four sentences after 463 sessions — the dials cannot escape the dead band

`brain/personality.py:106` — high, effort medium

I loaded the live state from data/world_model.json — 463 sessions of real conversation — and ran personality_to_system_instructions on it. The drifted values are curiosity 0.45, directness 0.59, humor_threshold 0.59, concern 0.6. All four are inside the 0.35-0.65 mid band. The rendered block is byte-identical to a fresh install except one line: 'Follow-up behavior' moved from 'show active curiosity and ask relevant follow-ups' to 'ask one useful follow-up only when it moves the task forward' — i.e. the only thing…

**Fix.** In personality_to_system_instructions, render deviation from DEFAULT_PERSONALITY rather than absolute buckets: for each dial, if |value - default| <= 0.08 emit nothing for it, and when the whole block is empty return "" so apply_personality_to_system_prompt leaves the persona as the last thing the model reads. Rename the block header from 'Current ALICE personality drift:'…

### The humor dial is wired backwards — the user laughing makes Alice drier

`brain/personality.py:144` — high, effort small

Every time Gabriel writes 'thanks', 'haha', 'lol', 'nice', or 'love that', Alice is nudged one step toward being told 'keep responses focused; humor only when clearly invited'. The live value has already climbed from the 0.5 default to 0.59 — 463 sessions of the user being warm have moved her 9 points toward suppressing humor, and she is 0.06 away from crossing into the band where the prompt explicitly tells her to stop being funny. The user's positive feedback is being read as a signal to shut up.

**Fix.** In PersonalityLayer.update_after_turn, flip the sign: `personality["humor_threshold"] = _clamp(float(personality["humor_threshold"]) - 0.015)` on _WARM_SIGNAL. Then rename the field throughout (world_model.DEFAULT_PERSONALITY, _as_personality's clamp list, personality_to_system_instructions) to `humor_openness` with high=more humor, and swap the low/high strings at…

### The persona is 25 bullets, 15 of them prohibitions, and not one example of how Alice actually talks

`ai/core/llm_engine.py:362` — high, effort medium

This is the mechanical answer to 'does Alice have a voice'. She has a list of things not to say. I counted: 25 bullets, 15 containing Never/No/Don't/DO NOT. The voice itself is specified only as abstract adjectives — 'warm but not soft; opinionated but not dogmatic; dry humor when earned' (ai/identity/alice_identity.py:44) — plus 'Dry humor is fine' at line 381. A local 8B model handed 15 prohibitions, 10 abstractions, and zero demonstrations resolves the constraint set by finding the safe intersection: short,…

**Fix.** Add a `VOICE_EXEMPLARS: List[Dict[str,str]]` constant to ai/core/llm_engine.py — four to six short user/assistant pairs written in the target register (a blunt correction, a dry one-liner, an 'I don't know', a refusal to pad) — and have _build_system_prompt's callers prepend them as real `role: user` / `role: assistant` messages ahead of conversation_history in both chat() and…

<details><summary>5 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/identity/opinion_extractor.py:26` | Zero opinions recorded in 463 sessions — the 'opinionated' persona has no substrate |
| medium | `app/main.py:687` | ResponseVarianceEngine is constructed and never called, while startup logs it as active |
| medium | `ai/core/adaptive_response_style.py:124` | A post-processor shreds Alice's prose into bullets, contradicting the persona that produced it |
| medium | `app/main.py:2068` | The only warmth control in the response path is two hard-coded strings, and it is unreachable |
| medium | `ai/identity/alice_identity.py:222` | The identity block feeds Alice session telemetry the persona forbids her to use |

</details>

---

## response postprocessing

*8 survived an adversarial review; 3 refuted.*

Nine live functions and roughly forty regexes rewrite a reply after the model produces it, and I measured what survives: a realistic 138-word structured answer came out of the pipeline at 56 words (41%) with 0 of its 14 line breaks intact, ending mid-list on the literal string "Three things make it worse in practice: 1." The layers that were supposed to add character — `response_self_critic.py`, `response_variance_engine.py` (via `FoundationIntegration.process_interaction`, never called), `response_adapter.py` — are all dead code, while every layer that strips, caps, flattens, or substitutes is live and unconditional; three of the substitution rules are plain substring matches that discard a correct answer and print a stock sentence, and they are not behind `ALICE_ENABLE_SCRIPTED_OVERRIDES`.

**The biggest one:** Every reply is hard-capped at 4 or 5 "sentences" by `apply_response_discipline` at ai/runtime/boundaries/boundary_factory.py:2982, and the splitter counts "1." and "- " list items as sentences — so any answer with structure is guillotined at the first list item, and two layers later `strip_meta_response_artifacts` deletes every remaining newline. Alice physically cannot produce a reply longer than four sentences or one with a paragraph break, on any turn, which is the mechanical reason she reads as a terminal rather than as someone thinking.

### _clamp_final_response replaces any answer containing "as an ai" or "language model" with a canned line

`app/main.py:1334` — critical, effort small

CONFIRMED. `lower` is the response text. Reachable on the ordinary conversational path: boundary_factory.py:3031 calls `alice._clamp_final_response(llm_text, ..., route="contract_pipeline_llm")` on every LLM answer, and this branch is NOT gated by scripted_overrides_enabled(). Any reply containing the substring "language model" is discarded whole; the substring also matches inside words ("such as an airline" contains "as an ai"). The substitute is either a stock prompt or one of the hardcoded paragraphs in…

**Fix.** In `_clamp_final_response`, delete this branch. Replace it with an anchored strip of a leading self-identification clause only: `re.sub(r"^(?:as an ai(?: (?:language )?model)?|i am an ai(?: language model)?)[,:]?\s*", "", text, flags=re.I)` applied once at the start of the text. Never return a canned string in place of a non-empty model answer here; if nothing remains, fall…

### Every model answer is hard-capped at 4 or 5 sentences, truncating lists at the first item

`ai/runtime/boundaries/boundary_factory.py:2993` — critical, effort medium

CONFIRMED (auditor cited 2982; actual line is 2993). Unconditional on the main LLM generation path in `_generate` — no flag, no route guard. `limit_sentences` (ai/runtime/response_discipline.py:121) splits on `(?<=[.!?])\s+`, so "1." in a numbered list counts as a sentence boundary. I ran the real function on a 5-sentence answer containing a 3-item list: it returned "...Three things make it worse in practice: 1." — the user is promised three things and shown the numeral 1 followed by nothing. On non-discussion…

**Fix.** Two changes. (1) In `limit_sentences`, make the splitter list-aware: never split inside a line matching `^\s*(?:[-*+]|\d+[.)])\s`, and treat a contiguous run of list lines as one unit. (2) At boundary_factory.py:2993, drop the cap on ordinary turns — call `apply_response_discipline(llm_text)` for filler stripping only, and reserve `max_sentences` for the genuinely short system…

### strip_meta_response_artifacts flattens every newline in every reply, on every route

`ai/runtime/operator_response_surface.py:141` — high, effort medium

CONFIRMED and universally reachable. `apply_response_momentum` calls this unconditionally on its first line (response_momentum_policy.py:176), before any turn_mode branch or early return, and contract_pipeline.py:1118 calls apply_response_momentum on every turn of the live path (process_input -> run_default_turn -> pipeline.run_turn). I ran the real function on the memory-recall string built at boundary_factory.py:1078: "Here is what I have saved in memory:\n- You decided...\n- You want..." came back as one line,…

**Fix.** Rewrite `strip_meta_response_artifacts` to work block-by-block: split on `\n\n+` into paragraphs, run the existing banned-marker sentence filter within each paragraph, preserve any line matching `^\s*(?:[-*+]|\d+[.)])\s` as its own line, rejoin list lines with `\n` and paragraphs with `\n\n`. Delete the global `re.sub(r"\s+", " ")` and collapse only `[ \t]+`.

### strip_passive_followup_sentences cuts a sentence mid-phrase, leaving "Let me know" dangling

`ai/runtime/response_momentum_policy.py:147` — high, effort small

CONFIRMED live. Reachable on three ordinary intents — called at response_momentum_policy.py:230 (conversation:clarification_needed), :236 (conversation:educational_explain), :248 (conversation:concept_refinement). Running the real function in mode="educational_explain": "Local models are closing the gap. Let me know if you want the benchmark numbers." -> "Local models are closing the gap. Let me know" — a two-word fragment with no punctuation. "That's the tradeoff. Do you want me to sketch the schema?" -> "That's…

**Fix.** In `strip_passive_followup_sentences`, (a) delete the `preserved` prefix logic and drop the whole matched sentence instead, the same shape as `sanitize_operator_chatter`; and (b) cut `banned_tokens` down to contentless closers ("let me know if you need anything else", "i'm here to help", "hope this helps") — remove "if you want", "let me know", "do you want me to" and "would…

### Passive-marker stripping excises phrases mid-sentence, producing broken grammar

`ai/runtime/response_momentum_policy.py:333` — high, effort small

CONFIRMED live through the real `apply_response_momentum`. Input "I'd watch the 30B class over the next year — that's where it flips. Just let me know if you want a deeper dive on the benchmarks." with intent="conversation:educational_explain", route="llm", user_input="what's a good file name for this" returns "...that's where it flips. Just a deeper dive on the benchmarks." Reachable on non-operator turns because `operator_turn` is True when intent is conversation:educational_explain and…

**Fix.** In `apply_response_momentum`, replace this loop with sentence-level removal: split on `(?<=[.!?])\s+|\n+`, drop any whole sentence whose lowercase form contains a passive marker, and rejoin — the same shape as `sanitize_operator_chatter` in operator_response_surface.py. Never call `str.replace` on a marker phrase inside a sentence.

<details><summary>3 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/runtime/response_momentum_policy.py:41` | _is_operator_application_request matches the bare words "file", "runtime", "inspect" anywhere in the message |
| medium | `app/main.py:1332` | _clamp_final_response swaps a personal reply for a productivity-coach line on two substrings |
| medium | `ai/core/adaptive_response_style.py:14` | _strip_fillers removes "sure", "absolutely", "in summary" from mid-sentence, changing meaning |

</details>

---

## routing coverage

*10 findings, **not yet adversarially verified**.*

I enumerated every branch in `_generate` that returns text without a model call, then measured coverage two ways: by driving the real router over 20 ordinary utterances with a stub Alice (14 route=llm, 5 route=tool, 1 route=local), and against 586 real logged turns in data/evaluations/evaluations.jsonl (52.6% conversation, 16.9% weather, 23.4% code/operator, 7.2% hard branch). Routing coverage is not the disease — roughly 13 of 20 turns do reach the model — the disease is that four post-model gates and one tool-response path throw the model's words away after it has spoken, so what the user reads is a hand-written string even on turns the model answered.

**The biggest one:** `_clamp_final_response` (app/main.py:1334) deletes the model's entire answer whenever it contains the substring "as an ai" OR "language model", and substitutes "I can help with that. Tell me the exact result you want." This runs on every LLM reply (boundary_factory.py:3027-3037 passes response_type="general_response"), and it is NOT gated by ALICE_ENABLE_SCRIPTED_OVERRIDES. So "explain transformers to me" — a turn that routes correctly to the model, gets a good answer, and passes verification — is shown to the user as a helpdesk stub, because any competent explanation of transformers says…

### Any model answer containing "language model" or "as an ai" is deleted and replaced with a helpdesk stub

`app/main.py:1334` — critical, effort small

User: "explain transformers to me". The router sends it to the model (conversation:question, conf 0.96), the model writes a real explanation, and the user reads: "I can help with that. Tell me the exact result you want." Ask "how do language models work?" and instead he gets a fixed encyclopedia sentence from a lookup table (app/main.py:6002) that does not answer what he asked. Every AI/ML question — the exact subject he built Alice to discuss — comes back as a stub.

**Fix.** In _clamp_final_response, delete the 1334-1339 block and the two content-sniffing replacements above it (1327-1331 "analysis: project_ideation", 1332-1333 "when i unwind"). If refusal-boilerplate stripping is still wanted, replace it with a sentence-level filter: split the text, drop only sentences matching r'^\W*(as an ai|i am an ai|i'm an ai|as a language model)\b', and keep…

### "What's the news?" returns a byte-identical canned policy sentence and can never do the lookup it offers

`ai/runtime/boundaries/boundary_factory.py:2635` — critical, effort medium

I ran the formulator directly: "what's going on in the world" and "what's the news today" both return the SAME string, character for character: "This needs live sources because current events changes quickly. I will not summarize it from model memory; narrow it by topic, region, or market and I can fetch current context." The user narrows it — and gets the same sentence again, because the tool catalog (ai/core/tool_catalog.py:294-447) contains list_workspace_files, read_workspace_file, search_workspace,…

**Fix.** Two parts. (a) In _may_reach_for_tools, add `if str(getattr(req.decision, "intent", "")) == "freshness:current_events": return True` so the turn reaches the ReactLoop. (b) Add a `web_search` ToolSpec to ai/core/tool_catalog.py at RISK_READ backed by a real HTTP search, and let the loop call it. Until (b) lands, change _formulate_freshness_guard_response to state the true limit…

### Every tool-route turn prints the plugin's raw CLI string; weather is the only one the model is allowed to speak

`ai/runtime/boundaries/boundary_factory.py:2744` — critical, effort medium

"remind me to call mom", "what time is it", "help me write an email to my landlord", "do you remember what I said about my sister" and every notes turn all route to tools (verified: reminder:set 0.95, time:current 0.95, email:compose 0.97, memory:recall 0.95), and the user reads the plugin's literal string. Real examples shipping today: "Note created: 'X'." (notes_plugin.py:3871), "Try 'list my notes' to see what's available." (notes_plugin.py:2480), "I found multiple matching notes. Which one?\n{titles}\nReply…

**Fix.** Generalise the weather wrap into one `_narrate_tool_result(tool_response, user_input, intent)` helper called for every successful tool turn, using the same contract the weather prompt already uses: the data string must be reproduced exactly and the model may only add a natural sentence around it. Keep the raw string as the fallback when alice.llm is None or the call raises, so…

### "What do you struggle with?" answers with a fenced eval table

`ai/runtime/contract_pipeline.py:545` — high, effort small

I verified the phrase list at contract_pipeline.py:522-533 contains the bare substring "what do you struggle", so "what do you struggle with?", "what do you struggle with the most" and "honestly what do you struggle with" all match. The user asks a companion question about her limits and receives, in a markdown code fence: "Intent Total Fail Fail%" / "notes 134 134 100%" / "weather 278 144 52%" / "Total eval records: 1854". That is the actual output — I ran weak_spot_report(). This branch is the very first thing…

**Fix.** Narrow the phrase list at contract_pipeline.py:522-533 to unambiguous operator commands ("routing report", "failure report", "weak spot report") and drop "what do you struggle", "where are you failing", "show me your failures" and "alice report". Better: make it a slash command (/report) so it cannot collide with speech at all, and feed the report dict into the model's context…

### Greetings are answered from a six-line lookup table behind a validator the model usually fails

`ai/runtime/greeting_surface_policy.py:277` — high, effort medium

"hi", "hey", "hello", "hi alice" all classify as intent=greeting (verified, conf 0.95), and boundary_factory.py:2468-2469 diverts them out of the normal generation path into render_grounded_greeting. The model gets two attempts; if both are rejected the user gets one of six fixed lines, cycled. The first thing he says every session is answered from a table — and the table is small enough to exhaust, after which _fallback_greeting returns "Hey {name}. What's the move?" forever.

**Fix.** Cut the ban lists down to the two things that are actually grounding failures — fabricated continuity (already covered by assess_continuity_claims at :386) and hallucinated entities (:446-448) — and delete the prose-shape rules: the bare "assist"/"support" tokens, the emotional_assumption list, the <=3-word rule, the sentence cap. Raise the retry count in…

### Any pronoun pushes the turn toward the notes plugin, so one notes turn poisons the rest of the conversation

`ai/core/nlp_processor.py:3091` — high, effort small

Reproduced against the real router: on a fresh NLPProcessor, "i'm frustrated with this code" classifies as conversation:clarification_needed @0.41 and routes to the model. Run "add milk to my notes" first, and the same sentence becomes notes:list @0.97, route="tool" — the user vents about a bug and Alice lists his notes back at him. The plugin_scores trace shows notes=1.55 vs conversation=0.20. Every sentence containing "this", "that" or "it" carries this bias, which is most of casual speech.

**Fix.** In _compute_plugin_scores, gate the pronoun bonus on there actually being a referent: replace `if parsed.references:` with `if parsed.references and (self.context.mentioned_notes or str(self.context.last_intent or "").startswith("notes:")):`, and cut the bonus to 0.3. Add decay to the last_plugin bonus at :3103 — apply it only when the previous turn is the immediately…

### The code:request fallback narrates going to look, then returns without looking

`ai/runtime/boundaries/boundary_factory.py:2622` — high, effort small

"can you look at my code" or "analyse your codebase" — 73 code:request turns in the live log, the second most common intent after conversation:general — and when _handle_code_request comes back empty the user is told, in the future tense, exactly what Alice is about to do. Then the turn ends. Nothing was listed, nothing was inspected, and the next turn starts from scratch. This is the sentence in the north star's opening complaint, still shipping.

**Fix.** Delete the fallback string at 2622-2633. When _handle_code_request returns empty, re-enter _try_tool_grounded_answer with tool_names=["list_workspace_files","search_workspace","read_workspace_file"] and allow the loop its full step budget; if that also produces nothing, fall through to the ordinary LLM path at 2865 with the tool failure in the companion context so the model…

### Memory questions are answered by a database dump or a fixed "not enough saved memory" line

`ai/runtime/boundaries/boundary_factory.py:2831` — high, effort medium

"do you remember when I said I was thinking about leaving my job?" hits _is_personal_memory_query (boundary_factory.py:969-976 matches the bare phrase "do you remember"). Below the similarity threshold the user reads "I do not have enough saved memory yet to answer that accurately." (:1027) — a status line about the memory subsystem, not a reply. Above it he reads "Here is what I have saved in memory:" followed by a bulleted list of raw stored strings (:1076-1077). Neither is a person recalling something; both are…

**Fix.** Keep the evidence gate, drop the rendering. When _has_sufficient_personal_evidence passes, pass the retrieved items into _build_companion_context and fall through to the LLM path so the model narrates the recall in its own words, then keep memory_answer_verifier.verify_answer as the post-check on what it produced (that is the sanctioned use of a heuristic — verification, not…

<details><summary>2 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/runtime/boundaries/boundary_factory.py:2989` | Explanations are hard-truncated at four sentences with no marker |
| medium | `ai/runtime/boundaries/boundary_factory.py:3096` | The terminal fallback ladder tells the user to rephrase instead of asking a person's question |

</details>

---

## sampling parameters

*2 survived an adversarial review; 6 refuted.*

Across every call path I traced, decoding is a single flat temperature with no per-intent variation: the live conversational path is always 0.7, the multi-LLM router path is 0.15–0.25, `query_knowledge` is 0.3, and the final polish pass is 0.35 — and `top_p`, `top_k`, `repeat_penalty` and `seed` are never set anywhere in the repo. Worse than the numbers themselves is which prompt each temperature is paired with: the paths that carry Alice's 5,662-character persona run hot and the paths that run cold ("You are a concise assistant", "No personality, no decisions - just knowledge", "Answer in at most four sentences. No preamble") are exactly the paths that handle tool results, greetings, structured answers and factual questions — so the more Alice actually does something, the more she sounds like a command line.

**The biggest one:** The moment Alice does the Jarvis thing — goes and looks with a tool — she loses her personality entirely. `ReactLoop.run` builds its message list from a bare rules-list SYSTEM_PROMPT and `chat_with_tools` is the one engine entry point that never calls `_build_system_prompt`, so the tool-grounded answer is composed by a model that was never told who Alice is, has no conversation history, and was explicitly instructed "at most four sentences. No preamble." That is a literal specification for terminal output, and it fires on precisely the turns the user would find most impressive.

### Tool-grounded turns are invisible to the conversation: no history in, nothing recorded out

`ai/runtime/boundaries/boundary_factory.py:151` — high, effort small

ReactLoop.run is called with only the raw user line (context defaults to None). The loop builds messages from SYSTEM_PROMPT + user input (react_loop.py:166-169) and answers via chat_with_tools, which — unlike chat() — never calls record_exchange; record_exchange has exactly one non-test caller, llm_engine.py:763, inside chat(). boundary_factory.py:2519-2521 returns the grounded answer early, so alice.llm.chat never runs. A tool turn therefore reads no history and writes none: a hole in the transcript in both…

**Fix.** In _try_tool_grounded_answer (boundary_factory.py:124) build the same companion context the chat path builds at boundary_factory.py:2868 and pass it as loop.run(user_input, context=...); in ReactLoop.run, insert self.llm.conversation_history[-self.llm.config.max_history:] between the system messages and the user message. On a successful grounded answer, call…

<details><summary>1 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `models/base.py:81` | Greeting and farewell decode at temperature 0.2-0.25; the gateway's 0.85 is computed after the branch that would use it |

</details>

---

## streaming and latency

*10 findings, **not yet adversarially verified**.*

Alice has a complete, working token-streaming implementation (`stream_chat`) that no user-facing code path calls — the CLI wraps the entire turn in a blocking spinner and prints the finished string, so time-to-first-visible-token is identical to time-to-last-token on every turn. On top of that, an ordinary conversational turn spends two full model generations (sometimes three) before a single character is shown, because the agent loop runs first and its answer is discarded whenever no tool was called.

**The biggest one:** Time-to-first-visible-token equals time-to-last-token, always — and that time is roughly doubled by a generation whose output is thrown away. `app/alice.py:131-133` holds a "thinking…" spinner across the whole of `alice.process_input()`, which returns a `str`, then prints it in one shot. Meanwhile `ai/runtime/boundaries/boundary_factory.py:2518` runs the ReactLoop first; when the model answers conversationally without calling a tool (the normal case for "how's it going"), `boundary_factory.py:158` discards that answer and `boundary_factory.py:2878` generates the whole reply a second time.…

### Streaming is implemented and never reachable — the CLI blocks on the whole turn

`app/alice.py:131` — critical, effort large

The user types, gets an animated "thinking…" dot for 4-12 seconds with zero output, then the entire reply appears at once as a block of text. It reads as a command that ran and printed its result, not as someone answering. `ai/core/llm_engine.py:802 stream_chat()` exists, works, and handles errors — but a repo-wide grep for its callers finds only `llm_engine.py`'s own `__main__` demo block (line 1457), `achat`/`astream_chat` (which nothing calls), and two test files. Zero application code streams.

**Fix.** Add an optional `on_token: Callable[[str], None]` parameter threaded from `start_alice_rich` -> `ALICE.process_input` -> `run_default_turn` -> `ContractPipeline._run_turn_async` -> the response boundary's `_generate`. In `boundary_factory._generate`, when `on_token` is present, replace the `alice.llm.chat(...)` call at line 2878 with an accumulating loop over…

### An ordinary conversational turn pays for two full generations and throws the first one away

`ai/runtime/boundaries/boundary_factory.py:158` — critical, effort medium

Every non-greeting turn routed to `llm` takes roughly twice as long as it needs to. "What do you think about X" makes Alice generate a complete answer inside the ReactLoop (react_loop.py:184, `chat_with_tools` with the full tool catalog in the prompt), and because the model correctly decided no tool was needed, `result.used_tools` is False and that finished answer is dropped on the floor. Then boundary_factory.py:2878 generates the reply again from scratch. The user waits through both and is shown only the second.

**Fix.** In `_try_tool_grounded_answer`, stop discarding the no-tool answer's cost. Change `ReactLoop.run` to also return the plain answer it got, and in `_try_tool_grounded_answer` return a `ResponseOutput` with `metadata['type']='loop_plain_answer'` when `result.stopped_reason == 'answered'` and `result.answer` is non-empty, so `_generate` returns at line 2521 instead of falling…

### No token-callback seam exists anywhere in the turn path; run_turn is sync-only

`ai/runtime/contract_pipeline.py:1461` — high, effort medium

Even a developer who wants to fix the spinner cannot: there is no place to hook a partial result. `run_turn` calls `asyncio.run(self._run_turn_async(...))` (line 1467), creating and tearing down a fresh event loop per turn, and returns a `PipelineResult` whose `response_text` is a complete string. `run_default_turn` (ai/runtime/turn_orchestrator.py:362) returns `str`. `process_input` (app/main.py:6278) returns `str`. Every layer between the socket and the screen is shaped so that nothing can leave until everything…

**Fix.** Add an `on_token: Optional[Callable[[str], None]] = None` keyword to `run_turn`, `run_turn_sync`, `_run_turn_async`, `TurnOrchestrator.verify_phase`, and `ResponseRequest`. Only the generation call in `boundary_factory._generate` needs it. Keep the verification contract by streaming into a buffer and, if verification rejects the text, clearing the streamed line…

### The spinner says "thinking…" and nothing else, forever, then erases itself

`ui/rich_terminal.py:278` — high, effort medium

At second 8 of a slow turn the user cannot distinguish "she is composing a long answer", "she is on her second model round trip", "she is reading a file", and "Ollama is dead and this will time out at 90 seconds" (`ai/core/llm_engine.py:125` sets a 90s timeout, retried up to `transport.max_attempts` times). The spinner text never changes and carries no elapsed counter. `transient=True` then wipes it, so afterwards there is no trace that anything took time at all — which is exactly why a slow assistant feels like a…

**Fix.** Change `thinking_spinner` to `thinking_spinner(self)` yielding a small handle object with `.set_phase(text: str)`, and render `f"{phase} · {elapsed:.0f}s"` by driving the `Live` from a daemon thread that ticks the elapsed seconds. Have the pipeline call `on_phase("reading main.py")` / `("checking weather")` / `("writing")` at the same points it appends to `stages`. Drop…

<details><summary>6 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ui/rich_terminal.py:271` | A fake 2-second progress bar runs to 100%, then the terminal goes dead for the real ~6s startup |
| medium | `ai/runtime/contract_pipeline.py:1359` | The turn blocks on memory persistence and metrics after the answer is already known |
| medium | `ai/runtime/boundaries/boundary_factory.py:2962` | The hedge retry adds a silent third generation to the turns that were already slowest |
| medium | `ai/core/llm_engine.py:962` | achat(stream=True) materializes the entire generator before returning — the one streaming API in the engine does not stream |
| medium | `ai/core/llm_engine.py:817` | stream_chat silently drops companion context, intent and half the context window — a trap for whoever wires it up |
| low | `app/main.py:6813` | run_interactive shows no progress indicator at all — the plain console is fully silent during a turn |

</details>

---

## system prompts

*2 survived an adversarial review; 7 refuted.*

Alice does not have a system prompt — she has five, on five user-reachable paths, and three of them contain no character at all ("You are a concise assistant", the ReAct rule list, "You are a natural language generator for Alice"). Where a persona does exist it is 764 words containing 39 negative constraints and 14 of 25 bullets that are prohibitions, so the model's cheapest way to comply is a flat declarative sentence — which is exactly what the user is describing as "a terminal".

**The biggest one:** There is no single Alice. The persona string in llm_engine.py never reaches three of the paths a user actually hits: the session greeting and farewell go through the multi-LLM router, whose entire system prompt is "You are a concise assistant. Prioritize short, clear answers." at temperature 0.2; every tool-using turn goes through react_loop.SYSTEM_PROMPT, six numbered rules with no name, no memory and a four-sentence cap; and structured replies go through PHRASER_PROMPT, which explicitly tells the model it is not Alice. The character the repo wrote is only present on one branch, and even…

### Tool-grounded turns run with the persona but zero context: no date, no memory, no goals, no opinions

`ai/runtime/boundaries/boundary_factory.py:151` — high, effort small

ReactLoop.run accepts a `context` argument and injects it as a second system message (react_loop.py:167-168), but the one and only call site never passes it, and `chat_with_tools` (llm_engine.py:868) forwards the message list verbatim without adding anything. The loop is reachable on ordinary turns — `is_enabled("conversational_tool_use")` at boundary_factory.py:119 resolves through ai/infrastructure/runtime_flags.py:33, and the name is not in QUARANTINED_SUBSYSTEMS, so it is on by default for any route=="llm"…

**Fix.** In `_try_tool_grounded_answer` (boundary_factory.py:124), build the same block the conversational path uses and pass it through: `loop.run(str(req.user_input or ""), context=_build_companion_context(memory_items=list(req.memory.items or []), operator_state=operator_state, alice=alice, intent=str(req.decision.intent or ""), user_input=str(req.user_input or "")))`.…

<details><summary>1 lower-severity findings</summary>

| | where | what |
|---|---|---|
| low | `ai/core/llm_engine.py:246` | Raw user utterances are injected into the prompt as Alice's list of Gabriel's goals |

</details>

---

## terminal ux

*10 findings, **not yet adversarially verified**.*

The terminal layer takes whatever the model wrote and re-typesets it as a machine record: any reply over ~120 characters, or containing a hyphen-space, is first chopped into one-sentence paragraphs and then framed in a rounded ASCII box titled "A.L.I.C.E" with a timestamp in the bottom border — I rendered this live and confirmed it. Around that, the session opens with a fake 2-second progress bar followed by ~6 seconds of frozen blank screen, a backronym banner and the words "System ready.", never a greeting; slash commands bypass the Rich console entirely and dump `[OK]`, `[ERROR]` and `=====` banners; and errors surface as `ERROR: 'NoneType' object has no attribute 'generate'`.

**The biggest one:** `print_assistant_response` (ui/rich_terminal.py:382-398) puts Alice's reply inside a bordered `Panel` with `title="A.L.I.C.E"` and a timestamp subtitle, and `_format_assistant_terminal_text` feeds that branch by splitting ordinary two- and three-sentence replies into separate paragraphs — so the moment she says anything longer than one short line, her answer stops being speech and becomes a labelled, timestamped record drawn in box characters. That single branch is why she reads as program output.

### Every non-trivial reply is framed in a bordered box labelled "A.L.I.C.E" with a timestamp

`ui/rich_terminal.py:388` — critical, effort small

I rendered real replies through this code. "Done - saved." and "1.5 million rows, give or take." both come out as a full-terminal-width rounded box with A.L.I.C.E stamped into the top border and 18:00 stamped into the bottom one. A person does not get a frame and a timestamp; a log record does. This is the single strongest reason the user says it feels like a terminal rather than someone talking.

**Fix.** Delete the Panel branch (lines 382-398) from `print_assistant_response`. Print every reply through one path: a blank line, then `self.console.print(Text(display_text, overflow="fold"))` at the left margin, with at most a dim, unpunctuated `A.L.I.C.E` on the preceding line — the user's own `❯` prompt already separates turns, so no label is needed at all. Keep Rich rendering…

### _format_assistant_terminal_text explodes a normal reply into one sentence per paragraph

`ui/rich_terminal.py:356` — critical, effort small

What the function does to a reply: it strips it, collapses all whitespace to single spaces, and if the result is a single line of 120+ characters with two or more sentences, it returns those sentences joined by blank lines. A three-sentence answer the model wrote as one flowing paragraph is served back as three orphaned stanzas — and because it now contains newlines, it also trips the Panel branch above and gets boxed. Confirmed live: "I can map a practical AI build path from here. A solid next set of tracks is...…

**Fix.** Reduce `_format_assistant_terminal_text` to `return str(text or "").strip()` — pass the model's own line breaks through untouched — and delete lines 300-356 entirely. Then update tests/test_ui_terminal_formatting.py, whose `test_long_plain_response_gets_sentence_spacing_for_terminal_readability` and `test_long_structured_single_line_response_gets_section_and_list_breaks`…

### Section-label reflow cuts sentences in half, leaving dangling words at the end of a paragraph

`ui/rich_terminal.py:333` — high, effort small

Verified live. Input: "I looked at your notes from last night. The Approach: you wanted to batch the writes, which still seems right. The Timeline: two weeks. I would not change either of those yet." Output, inside a box: paragraph one is "I looked at your notes from last night. The" — the sentence is guillotined mid-clause and the word "The" is left hanging. The user sees an answer that looks corrupted, which is worse than terminal-flavoured; it looks broken.

**Fix.** Delete lines 309-341 (the `section_labels` tuple through the `return structured`). If genuinely structured output needs headings, the place to get them is the system prompt asking the model for Markdown headings, which the renderer already handles — not a keyword table rewriting prose after the fact.

### Markdown rendering is applied to prose and silently corrupts filenames — __init__.py displays as init.py

`ui/rich_terminal.py:374` — high, effort small

Confirmed live. Reply text "I found ai/memory/__init__.py and ai/runtime/turn_orchestrator.py - both were modified today." renders on screen as "I found ai/memory/init.py and ...". The double underscores were consumed as Markdown emphasis. Alice looked at the filesystem, read a real filename back, and the terminal showed the user a filename that does not exist. Per north star rule 4 that is category (1) — confident, plausible, false, and the user cannot tell.

**Fix.** In `print_assistant_response`, replace the substring sniff with a line-anchored test: treat the reply as Markdown only if it contains a ``` fence, or if some line after `lstrip()` starts with `#`, `- `, `* `, `> ` or `N. ` — i.e. markers at the start of their own line. Mid-sentence `-`, `*` and `_` must never qualify. Everything else goes through `Text(...)` verbatim.

### A fake 2-second progress bar runs, completes, and is then erased — while the real 6s startup happens behind a frozen blank screen

`ui/rich_terminal.py:273` — high, effort small

Launch sequence as the user experiences it: a bar labelled "Initializing A.L.I.C.E systems" fills smoothly to 100% over exactly 2 seconds while nothing is being initialized; then the screen goes completely still for ~6 seconds with no cursor, no spinner and no output, because app/alice.py:78 has redirected stdout and stderr to a StringIO while `ALICE(...)` actually constructs; then `ui.clear()` wipes the bar away. The one moment the user has real evidence that something is loading is the moment they are shown a…

**Fix.** Delete `show_loading` (ui/rich_terminal.py:256-273) and its call at app/alice.py:69. Wrap the real `ALICE(...)` construction at app/alice.py:82 in the existing `ui.thinking_spinner()` context manager (or a Live spinner reading "starting up") so the animation covers the actual 6 seconds of work and stops when the prompt is genuinely ready.

### The Rich session never greets — it opens with a backronym banner and the words "System ready."

`ui/rich_terminal.py:249` — high, effort medium

First contact via the documented entry point (`python -m app.alice`) is: ASCII block-letter art in a box, a goal in square brackets, "A.L.I.C.E >> Advanced Linguistic Intelligence Computer Entity", a date/time line, and "System ready. Type /help for available commands". Alice never says anything. The first sentence in the session belongs to a boot loader. Everything the user sees before they type sets the expectation that they are operating a program, and they then read her first reply through that frame.

**Fix.** In `start_alice_rich`, after `ui.show_welcome()`, call `ui.print_assistant_response(alice._get_greeting())` so the session opens with her voice. Drop the "System ready." sentence and the backronym expansion from `info_text`, keeping at most the date line. Delete `RichTerminalUI.greeting_bank`, `agentic_prompts` and `_get_greeting` (lines 44-139 and 161-188) — a canned bank is…

<details><summary>4 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `app/alice.py:122` | Slash commands bypass the Rich console and raw-print [OK], [ERROR] and ===== banners |
| medium | `ui/rich_terminal.py:286` | The user's own typed line is echoed back at them with a name and a timestamp |
| medium | `ui/rich_terminal.py:412` | Errors reach the user as an ALL-CAPS marker plus a raw Python exception string |
| medium | `app/alice.py:131` | Replies appear all at once after a silent spinner, even though token streaming exists and works |

</details>

---

## tool result rendering

*11 findings, **not yet adversarially verified**.*

On a tool turn, Alice has no "say this in a sentence" step at all: the response boundary publishes the string the plugin's own f-strings built, verbatim. The single exception is weather, and that prompt explicitly orders the model "Copy the weather data above exactly as written — do not rephrase it", so even there the template is the final answer and the model is reduced to a wrapper.

**The biggest one:** ai/runtime/boundaries/boundary_factory.py `_generate` reads `tool_payload["response"]` — a string the plugin already rendered — and returns it as the reply. The general LLM path further down (line 2865) is only reached when there is NO tool result. So the moment Alice actually goes and looks at something, she stops thinking and starts printing: every notes list, calendar view, file listing and code inspection the user sees is literally a Python f-string, and the model that would have said "you've got three things today, the dentist is the awkward one" never receives the data.

### Tool answers are published verbatim from the plugin's own f-strings; the model is never given the data

`ai/runtime/boundaries/boundary_factory.py:2700` — critical, effort medium

Every question a tool can answer comes back as machine output. "What's on my calendar?" returns " Your calendar today:\n • Standup - Mon, Sep 15 at 09:00-09:15". "List my notes" returns "You have 12 note(s).\n1. Groceries — milk, eggs...". It reads like `ls` because it is `ls`: the user is looking at a string a Python f-string wrote, then handed straight to the screen.

**Fix.** In `_generate`, after `req.tool_result.success`, stop using `tool_payload["response"]` as the answer. Build a phrasing turn from the structured `req.tool_result.data["data"]` dict (the plugin's real payload: events, notes, files, temps) plus `req.user_input`, call `alice.llm.chat(prompt, intent="tool_wrap", use_history=False)`, and publish that. Keep the plugin's `response`…

### The one place tool output reaches the model, the prompt forbids the model from phrasing it

`ai/runtime/boundaries/boundary_factory.py:2749` — high, effort medium

Weather is the only tool whose result goes to the model, and the reply is still "Today: light rain, up to 14°C. Tomorrow: cloudy, up to 16°C." — the template string, unchanged, with at best a question bolted on the end. Jarvis would say "it's going to rain this afternoon, take the umbrella"; Alice reads out the row she was handed.

**Fix.** Change `_weather_prompt` to interpolate the raw fields from `tool_payload["data"]` (temperature, condition, high/low per day, location) as a labelled fact block, and instruct the model to answer the user's actual question in one or two sentences using only those numbers. Keep grounding by running the existing `_verify_weather_claims`/`weather_claim_validation` check (already…

### _alice_direct_phrase runs before the model, so the LLM phrasing prompts written right below it are unreachable

`app/main.py:1876` — high, effort medium

Confirmations sound like exit codes: "Done: create note for 'Groceries'.", "I couldn't complete delete note: not_found.", "Status: in progress." The user asked Alice to do something and got a log line back.

**Fix.** Invert the order in `_generate_natural_response`: attempt the model prompt path (Step 3, lines 1924+) first for every `response_type`, and call `_alice_direct_phrase` only when `self.llm` is unavailable, `strict_no_llm` is set, or the model returns empty. That makes the ladder a fallback (permitted) instead of an override (the bug). Gate the inversion behind the existing…

### Calendar renders a bulleted timetable inside the plugin; the structured events never reach the model

`ai/plugins/calendar_plugin.py:316` — high, effort small

"What does my day look like?" produces:\n\n Your calendar today:\n • Standup - Mon, Sep 15 at 09:00-09:15\n • Dentist - Mon, Sep 15 at 14:00-15:00\n\nNo sense of shape — nothing about the gap, the clash, or which one matters. It is a printed table with a bullet character.

**Fix.** Delete the `response_lines` assembly in `_view_events` and return `"response": ""` with a normalized `data` dict — `{"events": [{title, start_iso, end_iso, all_day, location}], "count", "time_range", "plugin_type": "calendar"}` — exactly as WeatherPlugin already does at plugin_system.py:578. Then let the tool-wrap prompt from finding 1 turn that into a sentence. Same change…

### Notes list is emitted as a numbered dump with "note(s)" pluralization

`ai/plugins/notes_plugin.py:4193` — high, effort small

"You have 12 note(s).\n1. Groceries — milk, eggs, bread\n2. Standup notes — discussed..." — the literal "(s)" is the tell. No human writes "note(s)"; forms and CLI tools do. The user asked what they'd written down and got a paginated index.

**Fix.** In `_list_notes`, set `"response": ""` and drop the `note_lines`/`summary` block; the `data` dict at 4209-4218 already carries count, shown, notes with previews, overdue_count and upcoming_reminders. Let the tool-wrap prompt from finding 1 phrase it, and cap what is handed to the model at ~15 notes with a count of the rest so the model can say "twelve, mostly to-dos" instead…

### Asked to read a file, Alice replies with a statistics sheet instead of what the file does

`ai/runtime/local_actions/code_response_builder.py:20` — high, effort medium

"Read ai/runtime/agent_loop.py" answers: "Inspected `ai/runtime/agent_loop.py`. Primary responsibility: runtime pipeline. Structural stats: 419 lines, 13122 chars, 5 imports, 4 classes, 12 functions, 0 TODO/FIXME, 0 fallback phrases. Recommended next files:\n- ..." She went and looked — the north star's whole test — and then read out `wc -l` instead of telling you what the file does.

**Fix.** In `local_action_executor.execute`, for `code:analyze_file`/`code:read_file`, return `"response": ""` and put the file text (head + tail, budgeted to ~6k chars) into `data` alongside the stats. Reduce `build_analysis_response` to a fallback used only when the model is unreachable. The tool-wrap prompt should ask the model what the file is responsible for, given the source.

### File listing tells you the count and throws away the filenames; file read tells you it read the file and throws away the content

`ai/plugins/file_operations_plugin.py:376` — high, effort small

"What's in my documents folder?" → "Found 42 files in ." That is the entire reply. "Read notes.txt" → "Read file: notes.txt", with the contents nowhere. She did the work and reported only that work occurred.

**Fix.** In `FileOperationsPlugin.execute`, stop the `response = message` collapse: set `"response": ""` and move `message` into `data` as `summary`, keeping `files`/`content`/`count` in `data` with `"plugin_type": "file_operations"`. Then the tool-wrap prompt from finding 1 receives the filenames and the file body and can answer the question that was actually asked.

<details><summary>4 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/runtime/local_actions/code_analyzer.py:11` | "Primary responsibility" is a seven-branch substring lookup that is confidently wrong on most files |
| medium | `ai/plugins/notes_plugin.py:4219` | Notes plugin marks 104 results "formulate": True and nothing in the runtime reads the flag |
| medium | `app/main.py:1805` | codebase_listing renders a markdown directory dump as the answer to "show me your code" |
| low | `ai/core/llm_gateway.py:572` | The only function designed to hand tool results to the model has no callers, and the 1,100-line formatter library it guards is… |

</details>

---

## verification over restriction

*11 findings, **not yet adversarially verified**.*

The guard layers are not editors, they are switches: when any check fires, the model's entire answer is discarded and replaced with a fixed sentence from a table, and the checks fire on ordinary English — "the working directory", "taking longer than usual", "language model", "conftest.py", the word "which". None of these replacement paths sit behind ALICE_ENABLE_SCRIPTED_OVERRIDES, so on a default install they are the live behaviour; the result is an assistant whose failure mode is a short machine sentence telling you what command to type. Separately, `ai/core/semantic_similarity.py` has zero callers anywhere outside itself, so no similarity threshold is over-rejecting there — the 0.3 `find_best_match` threshold is inert.

**The biggest one:** Verification is all-or-nothing substitution rather than repair. `_verification_fallback` (ai/runtime/turn_orchestrator.py:30-93) maps a rejection reason to one hard-coded sentence and throws the whole model answer away — including the 90% of it that was fine — and the sentences it substitutes are terminal help text ("Use 'inspect <filename>' to get accurate info"). Because the triggering checks are token-level regexes over ordinary prose, a good answer that happens to contain the phrase "working directory" or a filename like `conftest.py` is replaced wholesale. That single design choice is…

### Verification failure discards the whole answer and prints a canned line

`ai/runtime/turn_orchestrator.py:68` — critical, effort medium

Any turn that trips a guard produces one of six fixed sentences instead of an answer. The user asks a real question, gets a paragraph's worth of reasoning generated behind the scenes, and sees "I don't have the file details memorized. Use 'inspect <filename>'..." — a help string telling them a command to type. Repeat it a few times and the assistant is indistinguishable from a CLI error handler.

**Fix.** Change `_verification_fallback` from a substitution to a repair. It already receives `diagnostics` naming the exact offending spans (`missing_paths`, `missing_directories`, `unsupported_claims`). For those reasons, strip or qualify only the offending sentences of `proposed_text` and return the remainder; return a canned string only when nothing survives. For `tool_failed` /…

### Any answer containing "<word> directory" is rejected as an unverified codebase claim

`ai/runtime/boundaries/boundary_factory.py:1211` — critical, effort small

Alice says "Run it from the working directory." or "pytest walks each directory looking for conftest.py" and the entire reply is thrown away and replaced with "I don't have the file details memorized. Use 'inspect <filename>'...". The user asked a general question, got a help string about a file they never mentioned.

**Fix.** In `_extract_directory_claims`, require the captured token to look like a path claim rather than an English adjective: accept only candidates containing `/` or `.`, or that match a known top-level directory name case-insensitively. Additionally require a possessive/deictic frame (`in the X directory`, `under X/`) rather than any `X directory`. Drop "directory"/"directories"…

### The continuity guard silently deletes sentences containing "as usual" or "than usual"

`ai/runtime/continuity_claim_guard.py:19` — critical, effort medium

Sentences vanish mid-paragraph. I ran it: "That build is taking longer than usual. I would check the test discovery step." comes back as "I would check the test discovery step." — the observation that motivated the advice is gone, so the reply reads like a non-sequitur. Worse, a single-sentence reply that trips a pattern is replaced outright: "You mentioned the parser was slow." becomes "I am here. No active task is loaded yet, and we can continue an existing project or start fresh." (line 312), which answers…

**Fix.** Split `_CLAIM_PATTERNS` into two tiers. Tier 1 (`last time we talked about`, `we left off on`, `our previous conversation was about`) are assertions about prior sessions and stay verifiable. Tier 2 (`than usual`, `as usual`, `still on your mind`) are stylistic and should be removed from the list entirely — they assert nothing checkable. Then change `assess_continuity_claims`…

### Any answer containing the phrase "language model" is replaced with a canned line

`app/main.py:1334` — critical, effort small

Ask "how does a language model work?" — a question the owner of this project will ask constantly — and the model's answer is discarded. I traced the branch: `_is_answerability_direct_question("how does a language model work?")` returns True (question term "how", domain term "model"), so `_answerability_gate_fallback_response` fires and, because "model" is in the text, returns "Short answer: model training learns parameters from data by minimizing a loss function..." — a canned paragraph about a different topic.…

**Fix.** In `_clamp_final_response`, replace the substring test with an anchored self-identification pattern — `^\s*(as an ai|as a large language model)\b` or `\bI(?:'m| am)\s+(?:just\s+)?an?\s+(?:ai|language model)\b` — so the phrase is only caught when Alice is disclaiming about herself, and strip just that clause rather than returning a substitute. Delete…

### Mentioning any Python filename that is not in this repo rejects the whole answer

`ai/runtime/boundaries/boundary_factory.py:1204` — high, effort small

Ask a general Python question — "how does pytest find my tests?", "where does Django put settings?" — and a correct answer mentioning `conftest.py`, `setup.py` or `manage.py` is deleted, replaced by "I don't have the file details memorized. Use 'inspect <filename>'...". Alice appears unable to discuss Python at all, only her own files.

**Fix.** Only enforce path verification when the claim is about this workspace. In `_verify_codebase_claims`, require `_looks_like_code_request(user_text)` unconditionally (move the check above the `explicit_paths` branch), and skip any claimed path that has no directory separator and is a well-known ecosystem filename. When paths do fail, return them so `_verification_fallback` can…

### The continuity guard cannot see the current turn, so referencing what the user just said is "unsupported"

`ai/runtime/continuity_claim_guard.py:222` — high, effort small

The user writes "the parser is slow and I think it's the tokenizer", Alice replies "You mentioned the tokenizer — let's profile it," and the first clause is deleted because nothing in recalled memory overlaps. Alice appears to forget something said one line earlier, which is precisely the opposite of the continuity the guard exists to protect.

**Fix.** Add a `user_input: str = ""` parameter to `assess_continuity_claims` and treat token overlap with the current user message as a support reason (`current_turn_overlap`), checked first in the loop at line 267. Pass `req.user_input` from both boundary_factory call sites. Also consider the last N turns of live conversation history, not just the memory store, as an evidence source.

### The word "which" or "delete" anywhere in the input forces a clarifying question

`ai/runtime/anti_overclarification_policy.py:25` — high, effort small

I ran these: "which file should I start with?", "how do I delete a git branch?", "why was that row deleted?", and "I am not sure which approach is better, thoughts?" all return False, i.e. do not answer, ask instead. The user asks a direct question and gets "What exact result should I produce next?" back. Asking "which" is the most natural way to ask a question, and it is the one phrasing guaranteed to get a question in return.

**Fix.** In `should_answer_instead_of_clarify`, move the `conversation:` escape above the blockers. Narrow `risky` to imperative forms with a target — match `\b(delete|wipe)\b` only when the input is an instruction to Alice (no leading `how|why|what|when|can you explain`), since this policy governs answer-vs-ask, not execute-vs-refuse (the trust tiers already handle that). Delete the…

### The entire clarification path is two hard-coded questions chosen by one regex

`ai/runtime/boundaries/boundary_factory.py:2670` — high, effort medium

Every clarification Alice has ever asked is one of two sentences. "What exact result should I produce next?" is the voice of a job scheduler, not a person, and it is the single most common thing a user sees when confidence dips below 0.60. It never references what the user actually said, so it reads as a parse failure.

**Fix.** Replace `clarify_base` with a single-shot model call: pass the user's message plus the specific ambiguity the router recorded (`metadata["options"]` / `metadata["pronouns"]` are already populated at lines 2087-2095) and ask for one short question naming the ambiguous thing. Keep the two fixed strings only as the fallback when `alice.llm` is unreachable, which is the legitimate…

### The "keep the LLM's response on conversational turns" escape hatch preserves the canned string instead

`ai/runtime/contract_pipeline.py:903` — high, effort small

Someone added this branch specifically so that a casual turn would not be flattened by a grounding guard, and it does nothing in the case it was written for. A conversational turn whose answer was rejected still shows the fixed hedge.

**Fix.** Capture the model's text before it is overwritten — keep `verify_phase.proposed_response.text` in a local (e.g. `proposed_text`) alongside the line 846 call — and in this branch assign `response_text = proposed_text` when it is non-empty. Also drop the unconditional `respond_requires_follow_up = True` at line 933 for this branch, so a casual turn that was allowed to stand does…

<details><summary>2 lower-severity findings</summary>

| | where | what |
|---|---|---|
| medium | `ai/core/response_quality_tracker.py:86` | Honest hedging is scored as a quality failure and ratchets the clarify bias upward |
| medium | `ai/core/executive_controller.py:1967` | The executive response gate penalises hedging by 0.35 against a 0.48/0.52 accept threshold |

</details>

---

*15 dimensions, 128 findings carried forward, 25 refuted.*
