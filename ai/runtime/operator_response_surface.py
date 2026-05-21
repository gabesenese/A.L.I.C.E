from __future__ import annotations

import re
from typing import Any, Dict, List
import json
from pathlib import Path
from types import SimpleNamespace

_EXAMPLES_PATH = Path(__file__).parent.parent.parent / "data" / "learned_response_examples.jsonl"
_examples_cache: list[dict] | None = None


def _load_examples() -> list[dict]:
    global _examples_cache
    if _examples_cache is None:
        try:
            _examples_cache = [
                json.loads(line)
                for line in _EXAMPLES_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (FileNotFoundError, OSError):
            _examples_cache = []
    return _examples_cache


def find_similar_response_examples(
    *,
    context_signals: list[str],
    surface: str,
    limit: int = 3,
) -> List[SimpleNamespace]:
    signal_set = set(str(s).lower().strip() for s in (context_signals or []))
    results: list[tuple[int, dict]] = []
    for ex in _load_examples():
        if str(ex.get("surface") or "") != surface:
            continue
        if not bool(ex.get("accepted", True)):
            continue
        ex_signals = set(str(s).lower().strip() for s in (ex.get("context_signals") or []))
        overlap = len(signal_set & ex_signals)
        results.append((overlap, ex))
    results.sort(key=lambda t: t[0], reverse=True)
    return [
        SimpleNamespace(
            response_text=str(ex.get("response_text") or ""),
            context_signals=list(ex.get("context_signals") or []),
        )
        for _, ex in results[:limit]
    ]

_META_ARTIFACT_PATTERNS = (
    r"\(note:\s*i[‘’]?ve rewritten the response[^)]*\)",
    r"\(note:\s*i[‘’]?ve kept the main points intact[^)]*\)",
    r"note:\s*i[‘’]?ve rewritten[^.\n]*\.?",
    r"note:\s*i[‘’]?ve kept[^.\n]*\.?",
    r"while making minor adjustments for tone and flow[^.\n]*\.?",
    r"i[‘’]?ve rewritten the response to sound[^.\n]*\.?",
    r"i[‘’]?ve kept the main points intact[^.\n]*\.?",
    r"while keeping the same facts[^.\n]*\.?",
    r"here is a rewritten version[:\s]*",
    r"rewritten:\s*",
    r"sure,\s*here[‘’]?s a more natural version[:\s]*",
)

# Phrases where the model narrates its own reasoning — strip from the start of a response
# or from any standalone sentence that is pure meta-commentary.
_THINKING_ALOUD_PREFIXES = (
    r"^let me think about (this|that)[.!,]?\s*",
    r"^let me analyze (this|that)[.!,]?\s*",
    r"^let me break (this|that) down[.!,]?\s*",
    r"^let me work through (this|that)[.!,]?\s*",
    r"^let me consider (this|that)[.!,]?\s*",
    r"^let me dive into (this|that)[.!,]?\s*",
    r"^let\x27?s? dive into (this|that|what)[^.!?\n]{0,60}[.!?]?\s*",
    r"^from my understanding[,.]?\s*",
    r"^from my perspective[,.]?\s*",
    r"^so,?\s+to (answer|address|tackle) (your|that|this)[^.!?\n]{0,40}[,.]?\s*",
    r"^to (answer|address) (your|that|this)[^.!?\n]{0,40}[,.]?\s*",
    r"^ill? (try to |go ahead and )?(help|assist|address|answer|explain) (you |that |this )?[^.!?\n]{0,30}[.!]?\s*",
)
_PASSIVE_OPERATOR_LINES = (
    "what would you like to start working on",
    "what would you like to work on",
    "what should we work on",
    "where should we start",
    "ready when you are",
    "how can i help",
    "let me know",
)
# Trailing wrap-up sentences that signal the LLM is summarizing rather than engaging.
# These are stripped from the END of responses on discussion/brainstorm turns.
_TRAILING_LECTURE_PATTERNS = re.compile(
    r"\s*(now (it\x27?s |it is )?time to (think|consider|focus|explore|discuss)[^.!?\n]{0,80}[.!?]?"
    r"|the key takeaway (is|here)[^.!?\n]{0,80}[.!?]?"
    r"|in (summary|conclusion)[,\s][^.!?\n]{0,80}[.!?]?)\s*$",
    re.IGNORECASE,
)
_FORBIDDEN_OPERATOR_CHATTER_MARKERS = (
    "i'm here to assist",
    "i am here to assist",
    "assist you with",
    "ready for our conversation",
    "ready when you are",
    "how can i help",
    "what would you like",
    "let me know",
    "what should we work on",
)


def strip_meta_response_artifacts(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("\r\n", "\n")
    for pattern in _META_ARTIFACT_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Strip thinking-aloud prefixes from the start of the whole response
    for pattern in _THINKING_ALOUD_PREFIXES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    banned_markers = (
        "note: i've kept",
        "note: i've rewritten",
        "while making minor adjustments for tone and flow",
        "while keeping the same facts",
        "here is a rewritten version",
        "rewritten:",
    )
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    kept: list[str] = []
    for part in parts:
        sentence = str(part or "").strip()
        if not sentence:
            continue
        low = sentence.lower()
        if any(marker in low for marker in banned_markers):
            continue
        # Drop standalone thinking-aloud sentences mid-response
        is_pure_filler = any(
            re.match(p, low, re.IGNORECASE) for p in _THINKING_ALOUD_PREFIXES
        )
        if is_pure_filler and len(kept) == 0:
            continue
        kept.append(sentence)
    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = _TRAILING_LECTURE_PATTERNS.sub("", cleaned).strip()
    return cleaned

def _suppress_passive_operator_chatter(text: str) -> str:
    out_lines: list[str] = []
    for line in re.split(r"\n+", str(text or "")):
        line_clean = str(line or "").strip()
        if not line_clean:
            continue
        low = line_clean.lower()
        if any(marker in low for marker in _PASSIVE_OPERATOR_LINES):
            continue
        out_lines.append(line_clean)
    return "\n".join(out_lines).strip()


def sanitize_operator_chatter(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""
    fragments = re.split(r"(?<=[.!?])\s+|\n+", source)
    kept: list[str] = []
    for fragment in fragments:
        sentence = str(fragment or "").strip()
        if not sentence:
            continue
        low = sentence.lower()
        if any(marker in low for marker in _FORBIDDEN_OPERATOR_CHATTER_MARKERS):
            continue
        kept.append(sentence)
    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    return cleaned


def detect_context_signal(user_input: str, perception_frame: Dict[str, Any] | None = None) -> Dict[str, Any]:
    text = str(user_input or "").strip().lower()
    pf = dict(perception_frame or {})
    signals: list[str] = []
    energy_signal = str(pf.get("user_energy_signal") or "unknown").strip().lower() or "unknown"
    mood_signal = str(pf.get("user_mood_signal") or "unknown").strip().lower() or "unknown"

    if any(token in text for token in ("woke up from a nap", "just woke up", "nap")):
        signals.append("nap")
    if "long day" in text:
        signals.append("long_day")
    if "monday" in text:
        signals.append("monday")
    if "positive" in text:
        signals.append("positive")
    if any(token in text for token in ("tired", "exhausted", "drained")):
        signals.append("tired")
    if any(token in text for token in ("going to bed", "go to bed", "bed right now")):
        signals.append("bedtime")
    if any(token in text for token in ("work on alice", "work on a.l.i.c.e", "work on alice for", "work session")):
        signals.append("work_session")

    if energy_signal == "unknown" and any(s in signals for s in ("nap", "tired", "bedtime", "long_day")):
        energy_signal = "low"
    if mood_signal == "unknown":
        if "positive" in signals:
            mood_signal = "positive"
        elif "tired" in signals:
            mood_signal = "tired"
        else:
            mood_signal = "neutral"

    topic = "Alice" if ("alice" in text or "a.l.i.c.e" in text) else ""
    summary_parts: list[str] = []
    if "nap" in signals:
        summary_parts.append("user just woke up from a nap")
    elif "long_day" in signals:
        summary_parts.append("user had a long day")
    elif "bedtime" in signals:
        summary_parts.append("user is near bedtime")
    elif "tired" in signals:
        summary_parts.append("user is tired")
    if "monday" in signals:
        summary_parts.append("it is Monday")
    if "positive" in signals:
        summary_parts.append("user is staying positive")
    if "work_session" in signals and topic:
        summary_parts.append("wants a short Alice work session")
    elif "work_session" in signals:
        summary_parts.append("wants a short work session")

    return {
        "has_context": bool(signals),
        "signals": signals,
        "energy_signal": energy_signal,
        "mood_signal": mood_signal,
        "topic": topic,
        "user_context_summary": ", ".join(summary_parts).strip(),
    }


def _examples_style_block(context_signals: list[str]) -> str:
    examples = find_similar_response_examples(
        context_signals=list(context_signals or []),
        surface="operator_context_ack",
        limit=3,
    )
    if not examples:
        return "[]"
    lines = [f'- "{ex.response_text}" | signals={list(ex.context_signals or [])}' for ex in examples]
    return "\n".join(lines)


def generate_context_acknowledgement(
    context_signal: Dict[str, Any],
    *,
    user_input: str,
    llm_generate=None,
) -> str:
    if not llm_generate:
        return ""
    if not bool((context_signal or {}).get("has_context")):
        return ""
    style_examples = _examples_style_block(list(context_signal.get("signals") or []))
    prompt = (
        "Write one short acknowledgement sentence for Alice before an operator/code-work response.\n\n"
        "Use only the current user context.\n"
        "Sound natural, calm, and focused.\n"
        "Do not ask a question.\n"
        "Do not sound like therapy.\n"
        "Do not sound motivational.\n"
        "Do not sound corporate.\n"
        "Do not be cute, poetic, or dramatic.\n"
        "Do not mention memory or previous conversations.\n"
        "Do not mention files, tools, code, or internal systems.\n"
        "Do not say how can I help.\n"
        "Do not simply announce the work.\n"
        "Do not say 'let's dive in', 'let's get started', 'let's work on', or similar.\n"
        "Do not exceed 12 words.\n\n"
        f"Context signals:\n{dict(context_signal or {})}\n\n"
        f"User message:\n{str(user_input or '').strip()}\n\n"
        "Similar accepted examples for style only:\n"
        f"{style_examples}\n\n"
        "Return only the acknowledgement sentence.\n"
        "Return an empty string if no natural acknowledgement is needed."
    )
    try:
        try:
            return str(llm_generate(prompt=prompt) or "").strip()
        except TypeError:
            return str(llm_generate(prompt) or "").strip()
    except Exception:
        return ""


def validate_context_ack(text: str, context_signal: Dict[str, Any]) -> tuple[bool, list[str]]:
    candidate = str(text or "").strip()
    if not candidate:
        return (True, [])
    reasons: list[str] = []
    words = [w for w in re.split(r"\s+", candidate) if w]
    if len(words) > 12:
        reasons.append("too_long")
    low = candidate.lower()
    therapy_phrases = (
        "that must have been hard",
        "your feelings are valid",
        "i'm sorry you feel",
    )
    motivational_phrases = (
        "you've got this",
        "proud of you",
    )
    banned_phrases = (
        "how can i help",
        "what should we work on",
        "what do you need",
        "i feel",
        "i remember",
        "last time",
        "cozy",
        "agent_loop.py",
        "runtime",
        "tool",
        "file",
        "inspect",
    )
    generic_task_kickoff = (
        "let's dive",
        "let's get started",
        "let's work on",
        "let's begin",
        "let's tackle",
        "time to work on",
        "ready to work on",
    )
    if any(phrase in low for phrase in therapy_phrases):
        reasons.append("therapy_tone")
    if any(phrase in low for phrase in motivational_phrases):
        reasons.append("motivational_tone")
    if any(phrase in low for phrase in generic_task_kickoff):
        reasons.append("generic_task_kickoff")
    for phrase in banned_phrases:
        if phrase in low:
            if phrase in {"agent_loop.py", "runtime", "tool", "file", "inspect"}:
                reasons.append("internal_reference")
    if "?" in candidate:
        reasons.append("contains_question")
    energy = str((context_signal or {}).get("energy_signal") or "").lower()
    signals = {str(s).lower().strip() for s in list((context_signal or {}).get("signals") or [])}
    if "you are tired" in low and "tired" not in signals and energy != "low":
        reasons.append("ungrounded_context_ack")
    grounding_map = {
        "nap": ("nap", "woke", "awake", "fresh", "reset", "slow start", "back up"),
        "long_day": ("long day", "long one", "low-energy", "light", "steady"),
        "bedtime": ("bedtime", "late", "night", "one pass", "wind down"),
        "tired": ("tired", "low-energy", "light", "tight", "steady"),
    }
    for signal, anchors in grounding_map.items():
        if signal in signals and not any(anchor in low for anchor in anchors):
            reasons.append("ungrounded_context_ack")
            break
    reasons = list(dict.fromkeys(reasons))
    return (len(reasons) == 0, reasons)


def _generate_context_ack_with_retry(
    *,
    context_signal: Dict[str, Any],
    user_input: str,
    llm_generate=None,
) -> tuple[str, list[str], Dict[str, Any]]:
    attempt_count = 0
    first = generate_context_acknowledgement(
        context_signal,
        user_input=user_input,
        llm_generate=llm_generate,
    )
    attempt_count += 1 if llm_generate else 0
    first_clean = str(first or "").strip()
    valid, reasons = validate_context_ack(first_clean, context_signal)
    if first_clean and valid:
        return (
            first_clean,
            [],
            {
                "attempt_count": attempt_count,
                "retry_used": False,
                "model_used": True,
                "validation_applied": True,
                "accepted": True,
            },
        )
    if not llm_generate:
        return (
            "",
            reasons,
            {
                "attempt_count": 0,
                "retry_used": False,
                "model_used": False,
                "validation_applied": True,
                "accepted": False,
            },
        )

    style_examples = _examples_style_block(list(context_signal.get("signals") or []))
    stricter_prompt = (
        "Write one short acknowledgement sentence for Alice before an operator/code-work response.\n\n"
        "Use only the current user context.\n"
        "Sound natural, calm, and focused.\n"
        "Do not ask a question.\n"
        "Do not sound like therapy.\n"
        "Do not sound motivational.\n"
        "Do not sound corporate.\n"
        "Do not be cute, poetic, or dramatic.\n"
        "Do not mention memory or previous conversations.\n"
        "Do not mention files, tools, code, or internal systems.\n"
        "Do not say how can I help.\n"
        "Do not simply announce the work.\n"
        "Do not say 'let's dive in', 'let's get started', 'let's work on', or similar.\n"
        "Do not exceed 12 words.\n\n"
        f"Context signals:\n{dict(context_signal or {})}\n\n"
        f"User message:\n{str(user_input or '').strip()}\n\n"
        "Similar accepted examples for style only:\n"
        f"{style_examples}\n\n"
        "Your previous acknowledgement was rejected because:\n"
        f"{', '.join(reasons) if reasons else 'invalid_output'}\n\n"
        "It was too generic or not grounded in the user context.\n"
        "Write one short sentence that acknowledges the human context, not the task.\n"
        "Return an empty string if no safe natural acknowledgement is needed."
    )
    try:
        try:
            second = str(llm_generate(prompt=stricter_prompt) or "").strip()
        except TypeError:
            second = str(llm_generate(stricter_prompt) or "").strip()
    except Exception:
        return (
            "",
            reasons,
            {
                "attempt_count": attempt_count + 1,
                "retry_used": True,
                "model_used": True,
                "validation_applied": True,
                "accepted": False,
            },
        )
    attempt_count += 1
    valid_second, reasons_second = validate_context_ack(second, context_signal)
    if second and valid_second:
        return (
            second,
            [],
            {
                "attempt_count": attempt_count,
                "retry_used": True,
                "model_used": True,
                "validation_applied": True,
                "accepted": True,
            },
        )
    return (
        "",
        reasons_second or reasons,
        {
            "attempt_count": attempt_count,
            "retry_used": True,
            "model_used": True,
            "validation_applied": True,
            "accepted": False,
        },
    )


def _extract_next_move(next_step: str, operator_state: Dict[str, Any]) -> str:
    def _normalize_reason(raw_reason: str) -> str:
        cleaned = re.sub(r"\.+$", "", str(raw_reason or "").strip())
        if not cleaned:
            return ""
        return cleaned[0].lower() + cleaned[1:]

    def _normalize_sentence(raw: str) -> str:
        cleaned = re.sub(r"\.+$", "", str(raw or "").strip())
        if not cleaned:
            return ""
        return cleaned + "."

    structured = dict(operator_state.get("last_recommended_action") or {})
    target = str(structured.get("target") or "").strip()
    reason = str(structured.get("reason") or "").strip()
    action = str(structured.get("action") or "inspect_file").strip()
    if target:
        verb = "inspect" if action in {"inspect_file", "analyze_file", "read_file"} else action.replace("_", " ")
        if reason:
            cleaned_reason = _normalize_reason(reason)
            if cleaned_reason:
                return _normalize_sentence(f"Next best move: {verb} {target} because {cleaned_reason}")
        return _normalize_sentence(f"Next best move: {verb} {target}")
    raw = str(next_step or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("next best move:"):
        return _normalize_sentence(raw)
    return _normalize_sentence(f"Next best move: {raw}")


def humanize_local_execution_error(local_execution: Dict[str, Any]) -> str:
    payload = dict(local_execution or {})
    error = str(payload.get("error") or "").strip()
    if not error:
        return "The local execution step failed before I could verify the result."

    if error == "target_not_found":
        target = str(payload.get("target") or "").strip()
        requested_target = str(payload.get("requested_target") or "").strip()
        if target:
            return f"I could not find {target}."
        if requested_target:
            return f"I could not find {requested_target}."
        return "I could not find the requested target."
    if error == "approval_required":
        return "That action needs approval before I can run it."
    if error == "unknown_action":
        return "I do not have a registered action for that request."
    if error == "tool_failed":
        return "The local tool failed before returning verified evidence."
    if error == "contract_local_execution_error":
        return "The local execution contract failed before I could verify the result."

    token = error.lower().strip()
    if re.fullmatch(r"[a-z0-9_]+", token):
        readable = token.replace("_", " ").strip()
        if not readable:
            return "Local execution failed."
        return readable[0].upper() + readable[1:] + "."

    compact = re.sub(r"\s+", " ", str(error or "")).strip()
    if not compact:
        return "Local execution failed."
    if len(compact) > 120:
        return "Local execution failed."
    return compact if compact.endswith(".") else (compact + ".")


def normalize_response_paragraphs(text: str) -> str:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return ""
    # Restore known paragraph headings when prior layers flatten whitespace.
    raw = re.sub(r"\s+(Blocker:\s*)", r"\n\n\1", raw)
    raw = re.sub(r"\s+(Next best move:\s*)", r"\n\n\1", raw)
    lines = raw.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        no_tabs = str(line).replace("\t", " ")
        compact_spaces = re.sub(r" {2,}", " ", no_tabs)
        cleaned_lines.append(compact_spaces.rstrip())
    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def render_local_execution_error_response(
    *,
    user_input: str,
    base_text: str,
    operator_state: Dict[str, Any],
    local_execution: Dict[str, Any],
    next_step: str,
) -> str:
    _ = user_input
    _ = base_text
    parts: list[str] = ["I couldn't verify the local step."]
    humanized_error = humanize_local_execution_error(dict(local_execution or {}))
    parts.append(f"Blocker: {humanized_error}")
    next_move = _extract_next_move(next_step, dict(operator_state or {}))
    if next_move:
        parts.append(next_move)
    rendered = "\n\n".join(parts)
    return normalize_response_paragraphs(rendered)


def render_operator_response(
    *,
    user_input: str,
    base_text: str,
    operator_state: Dict[str, Any],
    local_execution: Dict[str, Any],
    next_step: str,
    llm_generate=None,
    perception_frame: Dict[str, Any] | None = None,
    response_metadata: Dict[str, Any] | None = None,
    base_text_is_verified: bool = False,
) -> str:
    _ = user_input
    _ = llm_generate
    _ = perception_frame
    local = dict(local_execution or {})
    state = dict(operator_state or {})

    if response_metadata is not None:
        response_metadata["operator_ack"] = {
            "context_detected": False,
            "model_used": False,
            "validation_applied": True,
            "retry_used": False,
            "accepted": False,
            "attempt_count": 0,
        }

    local_success = local.get("success")
    if local_success is False or bool(str(local.get("error") or "").strip()):
        return render_local_execution_error_response(
            user_input=user_input,
            base_text=base_text,
            operator_state=state,
            local_execution=local,
            next_step=next_step,
        )

    # Build a user-facing summary without internal labels or raw file paths.
    # "Finding:", "Next best move:", and "I inspected {path}" are Alice's internal
    # planning tokens and must never appear in user-facing output.
    analysis = dict(local.get("analysis") or {})
    summary = str(
        analysis.get("summary")
        or local.get("summary")
        or ""
    ).strip()
    responsibility = str(analysis.get("responsibility") or "").strip()

    # Priority 1: explicit summary from the local execution result.
    if summary:
        return summary.strip()

    # Priority 2: responsibility label → natural sentence.
    if responsibility:
        if responsibility == "agent loop":
            return "Looked at the operator loop — that's where plan/act/verify runs."
        elif responsibility == "runtime pipeline":
            return "Looked at the runtime pipeline."
        else:
            return f"Looked at the {responsibility} layer."

    # Priority 2.5: when a file was actually inspected successfully, skip base_text.
    # base_text in this case is LLM-generated context that should not surface in operator output.
    # The momentum layer will append "I inspected {file}." with grounded evidence.
    inspected_file_evidence = str(local.get("inspected_file") or "").strip()
    if local_success is True and inspected_file_evidence:
        return "Working on it."

    # Strip internal planning labels that executors may append to base_text.
    # These must not reach the contract check or the user surface.
    # Filter line-by-line so leading labels (e.g. a response that IS "Next best move:...")
    # are fully removed rather than partially matched.
    _raw_lines = re.split(r"\n", str(base_text or ""))
    _kept_lines = [
        ln for ln in _raw_lines
        if not re.search(r"(?i)\bnext best move\b|\bfinding:", ln)
    ]
    base_stripped = "\n".join(_kept_lines).strip()

    # Priority 3: sanitized base_text — always preferred over a generic ack.
    cleaned = sanitize_operator_chatter(
        _suppress_passive_operator_chatter(strip_meta_response_artifacts(base_stripped))
    )
    if cleaned and len(cleaned) > 8:
        return cleaned.strip()

    # Priority 4: action-aware fallback when base_text was entirely internal labels.
    action = str(local.get("action") or "")
    if action == "code:request":
        return "I can inspect local source code in this workspace and inspect the local workspace."
    return "Working on it."


