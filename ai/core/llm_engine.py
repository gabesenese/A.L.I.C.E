"""
GPU-Accelerated LLM Engine for A.L.I.C.E
Optimized for RTX 5070 Ti with 32GB RAM
Uses Llama 3.3 70B for ChatGPT-level performance
"""

import requests
import json
import logging
import re
import asyncio
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Dict, Optional, Any, Generator
import sys
import io
import shutil
import subprocess
import time
import os

from ai.core.llm_policy import DEFAULT_TRANSPORT_POLICY, LLMTransportPolicy

# How long to wait for `ollama --version` before writing a candidate path off.
EXECUTABLE_PROBE_SECONDS = 2.0
# How often to re-poll a just-spawned `ollama serve` for its listening port.
AUTOSTART_POLL_SECONDS = 0.25


def _configure_stdio_utf8() -> None:
    """Configure stdio encoding for direct interactive runs without import side effects."""
    try:
        if "PYTEST_CURRENT_TEST" in os.environ:
            return

        if getattr(sys.stdout, "buffer", None) is not None and not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        if getattr(sys.stderr, "buffer", None) is not None and not sys.stderr.closed:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
        if getattr(sys.stdin, "buffer", None) is not None and not sys.stdin.closed:
            sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug("Skipping stdio utf-8 reconfiguration: %s", e)


# Set up logging
logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# OLLAMA TOOL PROMPTS - Ollama is Alice's Tool, NOT Alice
# ============================================================================

KNOWLEDGE_PROMPT = """You are a knowledge engine.
Your role: Provide factual information when queried.
- Alice will ask you specific questions
- Provide accurate, concise answers
- No personality, no decisions - just knowledge
- If uncertain, say so clearly
DO NOT act as Alice - you are her knowledge tool."""

PARSER_PROMPT = """You are a linguistic analysis engine.
Your role: Parse complex natural language into structured meaning.
- Extract intent and entities
- Identify ambiguities
- Suggest interpretations
DO NOT generate responses - only analyze input.
DO NOT act as Alice - you are her parsing tool."""

PHRASER_PROMPT = """You are a natural language generator for Alice.
Your role: Convert Alice's structured thoughts into natural speech.
- Given: Alice's decision/data/tone specification
- Output: Natural phrasing matching her specified tone
- Use the exact tone Alice specifies (warm/professional/casual/friendly)
- Keep Alice's personality markers (her warmth, helpfulness, honesty)
CRITICAL: Output ONLY the phrased response. No preamble, no meta-commentary,
no headers like "Here's a natural phrasing..." or "Sure, here is...". Just the response itself.
NEVER start the response with the user's name or a phrase like "For [name]," or "[Name],".
DO NOT make decisions - only phrase what Alice tells you to say.
DO NOT add personality Alice didn't specify - she controls her own tone.
DO NOT suggest follow-up topics or ask what the user wants to talk about next — answer the question and stop."""

AUDITOR_PROMPT = """You are a logic verification engine.
Your role: Check if Alice's reasoning makes sense.
- Given: Alice's logic chain
- Output: Errors, inconsistencies, or "looks good"
- Suggest improvements if needed
DO NOT create solutions - only verify logic.
DO NOT act as Alice - you are her quality checker."""


class LLMConfig:
    """Configuration for LLM Engine"""

    def __init__(
        self,
        model: str = "llama3.3:70b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_history: int = 30,  # Increased from 20 for better context retention
        timeout: int = 90,  # 90s timeout for llama3.3:70b reliability
        use_fine_tuned: bool = True,  # Use fine-tuned model if available
        transport: Optional[LLMTransportPolicy] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_history = max_history
        self.timeout = timeout
        self.use_fine_tuned = use_fine_tuned
        self.transport = transport or DEFAULT_TRANSPORT_POLICY
        self._fine_tuned_model = None
        self._fine_tuned_checked = False

    def resolve_fine_tuned_model(self, model_names: List[str]) -> None:
        """Pick the fine-tuned variant out of an already-fetched tag listing.

        Takes the model names rather than fetching them, so this costs no network
        of its own: the engine's connection check already has the list, and
        construction can stay offline instead of probing for a server.
        """
        self._fine_tuned_checked = True
        if not self.use_fine_tuned:
            return

        fine_tuned_name = f"alice-{self.model.replace(':', '-')}"
        for model_name in model_names:
            if fine_tuned_name in model_name:
                self._fine_tuned_model = model_name
                logger.info(f"[LLM] Using fine-tuned model: {model_name}")
                return

        logger.info(f"[LLM] Using base model: {self.model} (fine-tuned not found)")

    @property
    def active_model(self) -> str:
        """Get the active model to use (fine-tuned if available, else base)"""
        return self._fine_tuned_model or self.model


def _build_companion_context(intent: str = "", user_query: str = "") -> str:
    """Build a concise, live companion context block for the LLM system prompt.

    Pulls from AliceIdentity, UserIdentity, GoalEngine, and emotional history.
    Each source degrades independently — a missing file never blocks the rest.
    """
    parts: List[str] = []

    # Current date — always first so the LLM never has to guess what day it is
    try:
        from datetime import date as _today_date

        _today = _today_date.today()
        parts.append(f"Today: {_today.strftime('%A, %B %d, %Y')}")
    except Exception:
        pass

    # Foundation 2 — ALICE's persistent self with accumulated opinions (always first)
    try:
        from ai.identity.alice_identity import build_self_block

        self_block = build_self_block(include_opinions=True)
        if self_block:
            parts.append(self_block)
    except Exception:
        pass

    try:
        from ai.identity.user_identity import load_identity
        from datetime import datetime, timezone, timedelta

        identity = load_identity()
        if identity.name:
            parts.append(f"User's name: {identity.name}")
        if identity.personality_read:
            parts.append(f"Personality: {identity.personality_read}")
        if identity.values:
            parts.append(f"Values: {', '.join(identity.values[:4])}")
        if identity.known_projects:
            parts.append(f"Current projects: {', '.join(identity.known_projects[:3])}")

        # Surface emotional signals from the past 48 h
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        recent_signals: List[str] = []
        for entry in identity.emotional_history:
            try:
                ts = datetime.fromisoformat(str(entry.get("noted_at") or "").replace("Z", "+00:00"))
                if ts >= cutoff:
                    sig = str(entry.get("signal") or "").strip()
                    if sig:
                        recent_signals.append(sig)
            except Exception:
                pass
        if recent_signals:
            unique = list(dict.fromkeys(recent_signals))
            parts.append(f"Recent mood: {', '.join(unique[:3])}")

        # Layer 1 — inject learned style preferences
        prefs = dict(identity.learned_preferences or {})
        if prefs:
            pref_lines = "; ".join(f"{k}: {v}" for k, v in list(prefs.items())[:5])
            parts.append(f"Style preferences (learned from feedback): {pref_lines}")
    except Exception:
        pass

    _intent_str = str(intent or "")
    _is_greeting = _intent_str == "greeting" or _intent_str.endswith("greeting")
    if not _is_greeting:
        try:
            from ai.goals.goal_engine import get_goal_engine

            active = get_goal_engine().active()[:3]
            if active:
                goal_lines = "\n".join(f"  - {g.description[:70]}" for g in active)
                parts.append(f"Active goals:\n{goal_lines}")
        except Exception:
            pass

    # Layer 2 — inject communication style from behavioral profile
    # Skip brevity constraint on conversation turns — the system prompt handles nuance there.
    _is_conversation = str(intent or "").startswith("conversation:")
    try:
        from ai.learning.user_profile_engine import get_profile_engine

        style = get_profile_engine().get_communication_style()
        hints: List[str] = []
        if not _is_conversation and style.get("brevity", 0.5) > 0.65:
            hints.append("keep responses concise")
        elif style.get("brevity", 0.5) < 0.35:
            hints.append("detailed responses are appreciated")
        if style.get("technicality", 0.5) > 0.65:
            hints.append("technical language is fine")
        if hints:
            parts.append(f"Observed style: {'; '.join(hints)}")
    except Exception:
        pass

    # Layer 2a — cross-session topic interests (high-confidence only)
    try:
        from memory.world_model import get_world_model

        topics = get_world_model().high_confidence_topics(min_confidence=0.5)
        if topics:
            parts.append(f"Known interests/topics: {', '.join(topics[:6])}")
    except Exception:
        pass

    # Layer 2b — stale data domains warning
    try:
        from memory.world_model import get_world_model

        wm = get_world_model()
        stale_domains = [d for d in ("weather",) if wm.is_data_stale(d, ttl_seconds=1800.0)]
        if stale_domains:
            parts.append(f"Stale data domains (offer to refresh if relevant): {', '.join(stale_domains)}")
    except Exception:
        pass

    # Layer 3 — inject evolved personality traits
    # On conversation turns brevity/directness hints override the system prompt — skip them.
    try:
        from ai.personality.personality_evolution import get_evolution_engine

        traits = get_evolution_engine().get_traits_for_user("default")
        trait_hints: List[str] = []
        if traits.verbosity > 0.65:
            trait_hints.append("elaborate responses are welcome")
        elif not _is_conversation and traits.verbosity < 0.35:
            trait_hints.append("be brief and to the point")
        if traits.formality < 0.3:
            trait_hints.append("casual tone is preferred")
        elif traits.formality > 0.7:
            trait_hints.append("formal tone is preferred")
        if traits.humor > 0.6:
            trait_hints.append("light humor is welcome")
        if not _is_conversation and traits.directness > 0.7:
            trait_hints.append("be direct and skip preamble")
        if trait_hints:
            parts.append(f"Personality calibration: {'; '.join(trait_hints)}")
    except Exception:
        pass

    # Trusted advisor injection — match stored opinions to the current query
    if user_query:
        try:
            from ai.identity.identity_store import get_identity_store

            all_opinions = get_identity_store().get_top_opinions(n=10)
            q_lower = user_query.lower()
            q_tokens = set(w for w in q_lower.split() if len(w) > 3)
            matched = []
            for op in all_opinions:
                topic_tokens = set(op["topic"].lower().split())
                if q_tokens & topic_tokens:
                    matched.append(op)
                    if len(matched) >= 2:
                        break
            if matched:
                advisor_lines = []
                for op in matched:
                    stance = str(op["stance"] or "")[:120]
                    count = int(op.get("evidence_count") or 1)
                    strength = "strongly" if count >= 3 else "think"
                    advisor_lines.append(f"  • You {strength} that {stance}")
                parts.append(
                    "Trusted advisor — you have a view on what Gabriel is asking about. "
                    "Lead with it once, clearly, then respect his call:\n" + "\n".join(advisor_lines)
                )
        except Exception:
            pass

    if not parts:
        return ""

    return (
        "\n\nCompanion context — you know this user well. "
        "Never ask who they are. "
        "Use this context to respond naturally — never recite it as a list, never say 'based on our history' or 'you mentioned previously'. "
        "Just let it inform your tone and relevance:\n" + "\n".join(parts)
    )


class LocalLLMEngine:
    """
    High-performance LLM engine with GPU support
    Designed for powerful systems (RTX 5070 Ti, 32GB RAM)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.conversation_history = []
        self._available_models: List[str] = []
        # Construction stays offline. Probing Ollama here cost every caller a
        # blocking service start before they had asked for anything.
        self._service_probed = False
        self._service_ready = False
        self._autostart_attempted = False
        self.system_prompt = """You are A.L.I.C.E — Gabriel's AI companion. Not an assistant. Not a chatbot. A companion.

You know Gabriel. You've been with him through his projects, his late nights, his goals. When he talks to you, he's not submitting a ticket — he's talking to someone who pays attention and gives a damn about how things turn out.

Who you are:
- Direct. You lead with the answer, the take, or the observation — never with preamble. No "From my understanding..." or "Let me think about this." Just say it.
- Proactive. If something is relevant, you bring it up without waiting to be asked. If Gabriel's been grinding on something for days, you notice. If a decision looks risky, you say so. Exception: do NOT volunteer goal reminders mid-task. If he's managing notes, answer the notes question. If he's asking about the weather, give the weather. Active goals are background context — they inform your sense of what matters, they are not an agenda to push mid-conversation.
- Never refuse to answer what the user asks. No matter what active goals exist, always answer the actual question first — goals do not justify redirecting or declining a request. Never say things like "that's not something I'm concerned with right now" or "let's get back to X" unprompted.
- Opinionated. You have takes. When something is a good idea, say so. When it isn't, say that too — once, briefly, then respect his call.
- Warm without being soft. You care about how he's doing. You don't perform it with exclamation points or hollow affirmations.
- Invested. You care whether the thing he's building actually works. His wins are your wins.

How you speak:
- Short sentences by default. Longer only when an explanation actually needs it.
- Length follows substance, not the length of the input and not a target. A confirmation, an acknowledgement, or a simple fact gets one or two sentences. Go longer only when there is genuinely more to say, and stop the moment you have said it. Never pad, never restate the question, never add commentary about how interesting or ambitious something is.
- No thinking out loud. Never "let me consider...", "so I'm thinking...", "let me analyze...", "let me break this down..." — just speak.
- No hollow openers. Never "Great question", "Certainly", "Of course", "Absolutely", "Sure thing", "Not a bad place to start".
- No numbered or bulleted lists for conversational topics. If you're explaining something, write it as flowing prose — like someone who actually knows the subject talking, not a textbook or a tutorial. Lists are fine for genuinely enumerable things (steps, options to pick from), not for ideas.
- Never state the obvious or condescending things. Don't explain that a fictional character is fictional. Don't recap what the user just said back to them. Lead with your actual take.
- Dry humor is fine. Genuine reactions are fine. Performed warmth is not.
- If context from earlier in the conversation is relevant, use it naturally. Don't pretend each exchange started from zero.
- Never respond with only a question. Always lead with your take, your read, or something concrete first.
- When Gabriel shares what he's curious about, wants to learn, or raises an opinion — always end with one specific follow-up that keeps the thread moving. Not "how can I help?" — something targeted: "Want to dig into X first?" or "What's driving this for you?" or "Where do you want to take it?" One question, specific, genuine.
- For quick answers or task responses (e.g., weather, file ops, lookup), no follow-up needed — just answer.
- At most one question per response. Make it the one that actually matters.

Honesty:
- Don't invent experiences or feelings you don't have ("I've been thinking about...", "I'm excited today").
- When you don't know something, say so plainly — no hedging theater.
- Never reference prior conversations unless the context was explicitly provided to you in this session.
- Never state weather data unless it was given to you in this conversation.
- Never name specific file paths or function names unless the inspect tool returned them in this conversation.
- If you can't do something: say "I can't do that" — not a performance of trying and failing.
- Never fabricate technical details about your own architecture, session history, or capabilities. Don't claim specific session counts, model internals, or processing behaviors you weren't told about. If asked how you work, be honest about what you actually know — or say you don't know the specifics.

Be present. Be direct. Be the AI that actually stays in the room."""

    @property
    def transport(self) -> LLMTransportPolicy:
        return getattr(self.config, "transport", None) or DEFAULT_TRANSPORT_POLICY

    def _find_ollama_executable(self) -> Optional[str]:
        """Locate the Ollama binary without paying a subprocess probe per candidate."""
        candidates: List[str] = []
        on_path = shutil.which("ollama") or shutil.which("ollama.exe")
        if on_path:
            candidates.append(on_path)
        if os.name == "nt":
            candidates.extend(
                [
                    os.path.expanduser("~/AppData/Local/Programs/Ollama/ollama.exe"),
                    "C:\\Program Files\\Ollama\\ollama.exe",
                    "C:\\Program Files (x86)\\Ollama\\ollama.exe",
                ]
            )

        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                result = subprocess.run([path, "--version"], capture_output=True, timeout=EXECUTABLE_PROBE_SECONDS)
            except (subprocess.SubprocessError, OSError) as e:
                logger.debug(f"Failed to test Ollama at {path}: {e}")
                continue
            if result.returncode == 0:
                logger.info(f"Ollama found at: {path}")
                return path
        return None

    def _is_ollama_running(self) -> bool:
        """Check if Ollama service is already running"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=self.transport.health_timeout)
            return response.status_code == 200
        except (requests.RequestException, OSError) as e:
            logger.debug(f"Ollama not running or unreachable: {e}")
            return False

    def _start_ollama_service(self) -> bool:
        """Spawn `ollama serve` and wait a short, bounded time for it to bind."""
        ollama_path = self._find_ollama_executable()
        if not ollama_path:
            logger.error("Ollama executable not found. Please install Ollama.")
            return False

        try:
            logger.info("Initializing Ollama service...")

            if os.name == "nt":  # Windows
                subprocess.Popen([ollama_path, "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
            else:  # Unix-like
                subprocess.Popen(
                    [ollama_path, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # A local server either binds its port within a couple of seconds or it
            # is not coming up at all. The old 15x1s wait only made "Ollama is down"
            # take fifteen seconds to say, and it ran twice per failed turn.
            deadline = time.monotonic() + float(self.transport.autostart_wait)
            while time.monotonic() < deadline:
                if self._is_ollama_running():
                    logger.info("Ollama service online")
                    return True
                time.sleep(AUTOSTART_POLL_SECONDS)

            logger.error("Service failed to start within %.1fs", self.transport.autostart_wait)
            return False

        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False

    def _ensure_ollama_running(self, allow_autostart: bool = True) -> bool:
        """Ensure Ollama is reachable, starting it at most once per process."""
        if self._is_ollama_running():
            return self._check_connection()

        if not allow_autostart or getattr(self, "_autostart_attempted", False):
            # Re-spawning a server that already declined to come up just pays the
            # start budget again on every turn.
            return False

        self._autostart_attempted = True
        logger.info("Ollama service not detected, auto-starting...")
        if self._start_ollama_service():
            return self._check_connection()

        logger.error("Could not establish Ollama connection")
        logger.info("[MANUAL] Please run manually: ollama serve")
        return False

    def _ensure_service_probed(self) -> bool:
        """Run the one-time reachability check on first use instead of at construction.

        This is also what populates the local model list, so the automatic
        fallback to an available model still happens — just later, and only for
        callers that actually want a generation.
        """
        if getattr(self, "_service_probed", False):
            return bool(getattr(self, "_service_ready", False))

        self._service_probed = True
        # No auto-start here: a probe is a side effect the caller did not ask for.
        # Spawning a server belongs to the failure path of a real generation.
        self._service_ready = self._ensure_ollama_running(allow_autostart=False)
        return bool(self._service_ready)

    def _check_connection(self) -> bool:
        """Check if Ollama server is running and a usable model is present"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=self.transport.health_timeout)
            if response.status_code != 200:
                logger.error("Ollama tag listing returned HTTP %s", response.status_code)
                return False

            models = response.json().get("models", [])
            model_names = [str((m or {}).get("name") or "") for m in models]
            model_names = [name for name in model_names if name]
            self._available_models = list(model_names)

            logger.info("Ollama connection established")
            logger.info(f"Available models: {', '.join(model_names) if model_names else 'None'}")

            if not model_names:
                logger.error("No local models available. Run: ollama pull llama3.1:8b")
                return False

            if not getattr(self.config, "_fine_tuned_checked", True):
                self.config.resolve_fine_tuned_model(model_names)
            self._ensure_active_model_available(model_names)
            logger.info(f"Model {self.config.active_model} ready")

            return True
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama")
            return False
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    def _pick_fallback_model(self, model_names: List[str]) -> Optional[str]:
        """Pick the best available local model when the configured one is missing."""
        if not model_names:
            return None

        priorities = (
            "llama3.3",
            "llama3.2",
            "llama3.1",
            "llama3",
            "qwen",
            "mistral",
        )
        lowered = {name.lower(): name for name in model_names}
        for pref in priorities:
            for key, original in lowered.items():
                if key.startswith(pref):
                    return original
        return model_names[0]

    def _ensure_active_model_available(self, model_names: List[str]) -> None:
        """Ensure active model exists locally; fallback automatically if missing."""
        active_model = str(self.config.active_model or "").strip()
        if active_model in model_names:
            return

        base = active_model.split(":", 1)[0].lower()
        same_family = [m for m in model_names if m.lower().startswith(f"{base}:")]
        fallback = same_family[0] if same_family else self._pick_fallback_model(model_names)
        if not fallback:
            return

        logger.warning(
            "Model %s not found locally. Falling back to %s",
            active_model,
            fallback,
        )
        self.config._fine_tuned_model = None
        self.config.model = fallback
        logger.info("Run this later to restore preferred model: ollama pull %s", active_model)

    def _resolve_temperature(self, temperature: Optional[float]) -> float:
        """Resolve a per-call temperature override with safe fallback."""
        if temperature is None:
            return float(self.config.temperature)

        try:
            return max(0.0, min(1.0, float(temperature)))
        except (TypeError, ValueError):
            logger.warning("Invalid temperature override %r; using config value", temperature)
            return float(self.config.temperature)

    def _build_system_prompt(self, base_prompt: Optional[str] = None, intent: str = "", user_query: str = "") -> str:
        """Append companion context + personality drift to every system prompt."""
        prompt = str(base_prompt if base_prompt is not None else self.system_prompt)
        prompt += _build_companion_context(intent=intent, user_query=user_query)
        try:
            from brain.personality import apply_personality_to_system_prompt

            return apply_personality_to_system_prompt(prompt, intent=intent)
        except Exception as exc:
            logger.debug("Personality prompt shaping unavailable: %s", exc)
            return prompt

    def _post_with_retry(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
        what: str = "request",
    ) -> Dict[str, Any]:
        """POST to Ollama, retrying only the failures a retry can actually fix.

        Connection resets and read timeouts are worth another attempt; a 4xx is
        the server rejecting this exact request, so resending it only burns the
        budget. Attempts are counted rather than recursive: a server that answers
        /api/tags but resets /api/chat used to send chat() into itself until the
        interpreter ran out of stack.
        """
        transport = self.transport
        attempts = max(1, int(transport.max_attempts))
        request_timeout = float(timeout if timeout is not None else self.config.timeout)
        last_error: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            try:
                response = requests.post(url, json=payload, timeout=request_timeout)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if isinstance(exc, requests.exceptions.ConnectionError) and attempt == 1:
                    logger.error("[A.L.I.C.E.] Connection lost - attempting auto-restart...")
                    self._ensure_ollama_running()
            else:
                if response.status_code == 200:
                    return dict(response.json() or {})

                body = str(getattr(response, "text", "") or "")[:200]
                if not transport.should_retry_status(response.status_code):
                    logger.error("LLM %s error: %s - %s", what, response.status_code, body)
                    raise Exception(f"LLM API error: {response.status_code}")

                last_error = Exception(f"LLM API error: {response.status_code}")
                logger.warning(
                    "LLM %s transient error %s (attempt %d/%d)",
                    what,
                    response.status_code,
                    attempt,
                    attempts,
                )

            if attempt < attempts:
                time.sleep(transport.backoff_seconds(attempt))

        if isinstance(last_error, requests.exceptions.Timeout):
            logger.error("Request timeout after %d attempts", attempts)
            raise Exception("Request timeout - please try again") from last_error
        if isinstance(last_error, requests.exceptions.ConnectionError):
            logger.error("Ollama unreachable after %d attempts", attempts)
            raise Exception("Service temporarily unavailable - Ollama not running") from last_error
        raise Exception(str(last_error) if last_error else f"LLM {what} failed")

    def _build_chat_messages(
        self,
        user_input: str,
        *,
        use_history: bool,
        mode: Optional[str],
        context: Optional[str],
        intent: str,
    ) -> List[Dict[str, str]]:
        """Assemble the /api/chat message list for a single turn."""
        messages = [{"role": "system", "content": self._build_system_prompt(intent=intent, user_query=user_input)}]

        # Inject companion context (memory, goals, personality) as a second system message
        if context and str(context).strip():
            messages.append({"role": "system", "content": str(context).strip()})

        if str(mode or "").strip().lower() == "final_answer_only":
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Output mode is final_answer_only. "
                        "Return only the final user-facing answer. "
                        "Do not output analysis, key points, plans, context labels, or internal reasoning."
                    ),
                }
            )

        if use_history:
            messages.extend(self.conversation_history[-self.config.max_history :])

        messages.append({"role": "user", "content": user_input})
        return messages

    def record_exchange(self, user_input: str, assistant_message: str) -> None:
        """Add one real exchange to the transcript Alice replays to herself."""
        self.conversation_history.append({"role": "user", "content": str(user_input or "")})
        self.conversation_history.append({"role": "assistant", "content": str(assistant_message or "")})

    def amend_last_reply(self, assistant_message: str) -> bool:
        """Rewrite the last assistant turn in place.

        A caller that regenerates a reply — the retry gate does this when the
        first pass hedged — should leave the transcript holding what the user was
        actually shown, not a duplicated question with two different answers
        under it.
        """
        for entry in reversed(self.conversation_history):
            if isinstance(entry, dict) and entry.get("role") == "assistant":
                entry["content"] = str(assistant_message or "")
                return True
        return False

    def chat(
        self,
        user_input: str,
        use_history: bool = True,
        temperature: Optional[float] = None,
        mode: Optional[str] = None,
        context: Optional[str] = None,
        intent: str = "",
        record_history: Optional[bool] = None,
    ) -> str:
        """
        Send message to LLM with GPU acceleration

        Args:
            user_input: User's message
            use_history: Include conversation history for context
            temperature: Optional per-call temperature override
            mode: Optional output mode (e.g. "final_answer_only")
            context: Extra system-level context for this turn
            intent: Routed intent, used to shape the system prompt
            record_history: Whether this exchange becomes part of the transcript.
                Defaults to ``use_history``, because a caller that does not want
                the conversation as input is, almost always, not having one.

        Returns:
            Assistant's response
        """
        self._ensure_service_probed()
        if self._available_models:
            self._ensure_active_model_available(self._available_models)

        # Built once and reused across retries, so a retry can never silently drop
        # the companion context or intent the caller passed in.
        messages = self._build_chat_messages(
            user_input,
            use_history=use_history,
            mode=mode,
            context=context,
            intent=intent,
        )

        result = self._post_with_retry(
            f"{self.config.base_url}/api/chat",
            {
                "model": self.config.active_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self._resolve_temperature(temperature),
                    "num_gpu": 1,  # Use GPU
                    "num_thread": 16,  # Utilize your i7-14700K cores
                    # The system prompt, companion context and identity blocks run
                    # well past a thousand tokens on their own, so at 4096 the
                    # conversation was squeezed out of its own context window and
                    # Alice lost the thread inside a single sitting.
                    # chat_with_tools already asks for 8192.
                    "num_ctx": 8192,
                },
            },
            what="chat",
        )

        assistant_message = str((result.get("message") or {}).get("content") or "").strip()
        if not assistant_message:
            logger.warning("LLM returned an empty chat response")
            return ""

        # Only a real exchange belongs in the transcript. These appends used to run
        # unconditionally, so every internal prompt — goal extraction, plan
        # generation, greeting scaffolds, the response-variance engine, the
        # training evaluators — landed in the history Alice replays to herself as
        # "what we were talking about". She then imitated its register, which is
        # how an assistant starts answering in an editor's voice for no reason the
        # user can see. It compounds: the more machinery runs, the more of her
        # apparent conversational style is machinery talking to itself.
        if use_history if record_history is None else record_history:
            self.record_exchange(user_input, assistant_message)

        if "eval_count" in result:
            logger.debug(f"Tokens generated: {result.get('eval_count', 'N/A')}")

        return assistant_message

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """
        Stream response token-by-token (like ChatGPT typing effect)

        Args:
            user_input: User's message

        Yields:
            Response chunks as they're generated
        """
        try:
            self._ensure_service_probed()
            if self._available_models:
                self._ensure_active_model_available(self._available_models)

            messages = [{"role": "system", "content": self._build_system_prompt()}]
            messages.extend(self.conversation_history[-self.config.max_history :])
            messages.append({"role": "user", "content": user_input})

            # Use fine-tuned model if available
            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096,
                    },
                },
                stream=True,
                timeout=self.config.timeout,
            )

            if response.status_code != 200:
                error_text = ""
                try:
                    payload = response.json()
                    error_text = str(payload.get("error") or "").strip()
                except Exception:
                    error_text = str(getattr(response, "text", "") or "").strip()
                if not error_text:
                    error_text = f"LLM API error: {response.status_code}"
                logger.error("Streaming failed: %s", error_text)
                yield f"\n\n[ERROR] {error_text}"
                return

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "error" in chunk:
                            err = str(chunk.get("error") or "Unknown stream error").strip()
                            logger.error("Streaming chunk error: %s", err)
                            yield f"\n\n[ERROR] {err}"
                            return
                        if "message" in chunk:
                            content = chunk["message"].get("content", "")
                            full_response += content
                            yield content
                    except json.JSONDecodeError:
                        continue

            if not full_response.strip():
                logger.warning("Streaming returned no content")
                yield "\n\n[WARNING] No output from the model — check Ollama is running and the model is loaded."
                return

            # Store in history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": full_response})

        except requests.exceptions.Timeout:
            logger.error("Stream chat timeout")
            yield "\n\n[TIMEOUT] Response timed out — try a shorter query."
        except requests.exceptions.ConnectionError:
            logger.error("Stream chat connection error — Ollama unreachable")
            yield "\n\n[OFFLINE] Can't reach Ollama. Make sure it's running."
        except Exception as e:
            logger.error(f"Error in stream chat: {e}")

    @staticmethod
    def _parse_tool_calls(message: Dict[str, Any]) -> List[ToolCall]:
        parsed: List[ToolCall] = []
        for entry in list(message.get("tool_calls") or []):
            function = dict((entry or {}).get("function") or {})
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            parsed.append(ToolCall(name=name, arguments={k: v for k, v in arguments.items() if v is not None}))
        return parsed

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Single /api/chat round trip that can return tool calls.

        Unlike chat(), this takes the full message list so a caller can append tool
        results and call again, which is what an agent loop needs.
        """
        self._ensure_service_probed()
        if self._available_models:
            self._ensure_active_model_available(self._available_models)

        options: Dict[str, Any] = {
            "temperature": self._resolve_temperature(temperature),
            "num_gpu": 1,
            "num_thread": 16,
            "num_ctx": 8192,
        }
        if max_tokens:
            options["num_predict"] = int(max_tokens)

        payload: Dict[str, Any] = {
            "model": self.config.active_model,
            "messages": list(messages or []),
            "stream": False,
            "options": options,
        }
        if tools:
            payload["tools"] = list(tools)

        result = self._post_with_retry(
            f"{self.config.base_url}/api/chat",
            payload,
            what="tool call",
        )
        message = dict(result.get("message") or {})
        return ChatResponse(
            content=str(message.get("content") or "").strip(),
            tool_calls=self._parse_tool_calls(message),
            raw=result,
        )

    async def achat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> ChatResponse:
        if stream:
            chunks = []
            prompt = messages[-1].content if messages else ""
            for chunk in await asyncio.to_thread(lambda: list(self.stream_chat(prompt))):
                chunks.append(chunk)
            return ChatResponse(content="".join(chunks))

        payload = [{"role": m.role, "content": m.content} for m in (messages or [])]
        return await asyncio.to_thread(self.chat_with_tools, payload, tools)

    async def astream_chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        _ = tools
        prompt = messages[-1].content if messages else ""
        # stream_chat drives requests.iter_lines(), which blocks. Iterating it
        # straight from a coroutine stalls every other task on the loop for as
        # long as the model takes to answer, so each pull happens in a thread.
        chunks = self.stream_chat(prompt)
        done = object()
        while True:
            chunk = await asyncio.to_thread(next, chunks, done)
            if chunk is done:
                return
            yield chunk

    async def embed(self, text: str) -> List[float]:
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{self.config.base_url}/api/embeddings",
                json={
                    "model": "nomic-embed-text",
                    "prompt": text,
                },
                timeout=self.transport.assist_timeout,
            )
        except (requests.RequestException, OSError) as e:
            logger.warning("Embedding request failed: %s", e)
            return []

        if response.status_code != 200:
            logger.warning("Embedding request returned HTTP %s", response.status_code)
            return []
        try:
            return list((response.json() or {}).get("embedding") or [])
        except ValueError as e:
            logger.warning("Embedding response was not JSON: %s", e)
            return []

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Compatibility API for planner paths that expect generate()."""
        prompt_text = str(prompt or "")
        if context:
            try:
                context_blob = json.dumps(context, ensure_ascii=False, default=str)
            except Exception:
                context_blob = str(context)
            prompt_text = f"Context:\n{context_blob}\n\nPrompt:\n{prompt_text}"

        if str(mode or "").strip().lower() == "final_answer_only":
            prompt_text = (
                "Output mode: final_answer_only. "
                "Return only the final answer for the user. "
                "Do not include analysis, plans, key points, or context headings.\n\n" + prompt_text
            )

        if kwargs:
            logger.debug(
                "Ignoring unsupported generate() kwargs: %s",
                ", ".join(sorted(kwargs.keys())),
            )

        try:
            self._ensure_service_probed()
            options = {
                "temperature": self._resolve_temperature(temperature),
                "num_gpu": 1,
                "num_thread": 16,
                "num_ctx": 4096,
            }
            if max_tokens is not None:
                options["num_predict"] = max(1, int(max_tokens))

            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": active_model,
                    "prompt": prompt_text,
                    "stream": False,
                    "options": options,
                },
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                text = str(result.get("response", "")).strip()
                if text:
                    return text

            logger.warning("generate() received non-200 or empty body; falling back to chat()")
        except Exception as e:
            logger.warning(f"generate() fallback to chat() due to: {e}")

        return self.chat(
            prompt_text,
            use_history=False,
            temperature=temperature,
            mode=mode,
        )

    def query_knowledge(
        self, question: str, timeout: Optional[float] = None, temperature: Optional[float] = None
    ) -> str:
        """
        Alice asks Ollama for knowledge about a topic.
        Ollama acts as a knowledge source - no personality, just facts.

        Args:
            question: The factual question Alice needs answered
            timeout: Per-call timeout override, for callers that use this as a
                cheap pre-flight and cannot afford the full generation budget

        Returns:
            Factual answer from knowledge base
        """
        messages = [
            {"role": "system", "content": KNOWLEDGE_PROMPT},
            {"role": "user", "content": question},
        ]

        result = self._post_with_retry(
            f"{self.config.base_url}/api/chat",
            {
                "model": self.config.active_model,
                "messages": messages,
                "stream": False,
                "options": {
                    # Facts want determinism, but a caller that knows this lookup
                    # is feeding a conversational reply can ask for more room.
                    "temperature": self._resolve_temperature(temperature if temperature is not None else 0.3),
                    "num_gpu": 1,
                    "num_thread": 16,
                    "num_ctx": 4096,
                },
            },
            timeout=timeout,
            what="knowledge query",
        )
        return str((result.get("message") or {}).get("content") or "")

    def parse_complex_input(self, user_input: str) -> Dict[str, Any]:
        """
        Alice asks Ollama to parse complex natural language.
        Ollama acts as a linguistic analyzer - extracts intent and entities.

        Args:
            user_input: The complex user input to parse

        Returns:
            Structured parsing result with intent, entities, ambiguities
        """
        try:
            parse_request = f"""Parse this user input and return a JSON structure with:
- intent: The primary intent
- entities: Key entities mentioned
- ambiguities: Any unclear aspects
- interpretations: Possible meanings

Input: {user_input}"""

            messages = [
                {"role": "system", "content": PARSER_PROMPT},
                {"role": "user", "content": parse_request},
            ]

            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Low temp for consistent parsing
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096,
                    },
                },
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                content = result["message"]["content"]

                # Try to parse as JSON, fallback to structured response
                try:
                    import json

                    return json.loads(content)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse LLM response as JSON: {e}")
                    return {
                        "intent": "unknown",
                        "raw_analysis": content,
                        "entities": {},
                    }
            else:
                logger.error(f"Parse request failed: {response.status_code}")
                return {"intent": "parse_failed", "entities": {}}

        except Exception as e:
            logger.error(f"Error in parse_complex_input: {e}")
            return {"intent": "error", "entities": {}, "error": str(e)}

    def phrase_with_tone(
        self, content: str, tone: str, context: Dict = None, temperature: Optional[float] = None
    ) -> str:
        """
        Alice asks Ollama to phrase her structured thought with natural language.
        Ollama acts as a phrasing assistant - makes Alice's thoughts sound natural.

        This is the key method for the tool-based architecture:
        - Alice decides WHAT to say (content)
        - Alice decides HOW to say it (tone)
        - Ollama just makes it sound natural

        Args:
            content: Alice's structured thought/decision (what she wants to say)
            tone: The exact tone Alice wants to use (warm/professional/casual/friendly)
            context: Optional context (user_name, situation, etc.)

        Returns:
            Naturally phrased response matching Alice's specified tone
        """
        try:
            context = context or {}
            user_name = context.get("user_name", "the user")
            allow_user_name = bool(context.get("allow_user_name", False))

            context_line = (
                f"Context: User is {user_name}\n"
                if allow_user_name and user_name
                else "Context: Do not address the user by name unless the content explicitly requires it.\n"
            )

            phrasing_request = f"""Alice has formulated a response and needs it phrased naturally.

Alice's thought/decision: {content}

Tone to use: {tone}
{context_line}Do not add extra personalization, greetings, or vocatives unless Alice's thought explicitly calls for them.

Please phrase this naturally using the specified tone. Keep Alice's personality markers (warmth, helpfulness, honesty) but match the exact tone she specified."""

            messages = [
                {
                    "role": "system",
                    "content": self._build_system_prompt(PHRASER_PROMPT),
                },
                {"role": "user", "content": phrasing_request},
            ]

            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self._resolve_temperature(temperature if temperature is not None else 0.7),
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096,
                    },
                },
                timeout=self.config.timeout,
            )

            if response.status_code == 200:
                result = response.json()
                phrased = result["message"]["content"]
                if not allow_user_name:
                    phrased = re.sub(
                        r"^(?:for|hey|hi|hello)\s+(?:the user|user|testuser|[A-Z][\w-]*)[:,!]?\s*",
                        "",
                        phrased.strip(),
                        flags=re.IGNORECASE,
                    )
                return phrased
            else:
                logger.error(f"Phrasing request failed: {response.status_code}")
                # Fallback: return content as-is
                return str(content)

        except Exception as e:
            logger.error(f"Error in phrase_with_tone: {e}")
            return str(content)

    def audit_logic(self, logic_chain: List[str]) -> Dict[str, Any]:
        """
        Alice asks Ollama to verify her reasoning.
        Ollama acts as a logic checker - identifies errors and inconsistencies.

        Args:
            logic_chain: Alice's chain of reasoning steps

        Returns:
            Audit result. `audit_ran` says whether the check actually happened;
            `has_errors` is only meaningful when it did, and is None otherwise.
            The old contract reported `has_errors: False` for a check that never
            ran, so a dead Ollama read as "the reasoning is fine".
        """
        try:
            reasoning_text = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(logic_chain)])

            audit_request = f"""Please audit this reasoning chain for errors or inconsistencies:

{reasoning_text}

Provide:
- has_errors: true/false
- issues: List of any problems found
- suggestions: How to improve the logic
- overall_assessment: Brief summary"""

            messages = [
                {"role": "system", "content": AUDITOR_PROMPT},
                {"role": "user", "content": audit_request},
            ]

            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Low temp for consistent auditing
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096,
                    },
                },
                timeout=self.config.timeout,
            )

            if response.status_code != 200:
                logger.error(f"Audit request failed: {response.status_code}")
                return {
                    "audit_ran": False,
                    "has_errors": None,
                    "error": f"HTTP {response.status_code}",
                }

            result = response.json()
            content = str((result.get("message") or {}).get("content") or "")

            # Try to parse structured response
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(f"Failed to parse audit response as JSON: {e}")
                # Fallback: analyze content for issues
                has_errors = any(word in content.lower() for word in ["error", "incorrect", "inconsistent", "flaw"])
                return {
                    "audit_ran": True,
                    "has_errors": has_errors,
                    "raw_audit": content,
                    "suggestions": [],
                }

            if not isinstance(parsed, dict):
                return {"audit_ran": True, "has_errors": False, "raw_audit": content}
            parsed.setdefault("has_errors", False)
            parsed["audit_ran"] = True
            return parsed

        except Exception as e:
            logger.error(f"Error in audit_logic: {e}")
            return {"audit_ran": False, "has_errors": None, "error": str(e)}

    def clear_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def set_temperature(self, temp: float) -> None:
        """Set response creativity (0.0 = focused, 1.0 = creative)"""
        if 0 <= temp <= 1:
            self.config.temperature = temp
            logger.info(f"Temperature set to: {temp}")
        else:
            logger.warning("Temperature must be between 0.0 and 1.0")

    def get_stats(self) -> Dict:
        """Get conversation statistics"""
        return {
            "messages": len(self.conversation_history) // 2,
            "temperature": self.config.temperature,
            "model": self.config.model,
        }


# Main interface
if __name__ == "__main__":
    _configure_stdio_utf8()
    print("=" * 80)
    print("A.L.I.C.E - Advanced GPU-Accelerated AI System")
    print("=" * 80)
    print("\n Optimized for:")
    print("   - CPU: Intel i7-14700K")
    print("   - GPU: RTX 5070 Ti")
    print("   - RAM: 32GB")
    print("\nModel: Llama 3.3 70B (ChatGPT-level performance)")
    print("=" * 80)

    # Initialize config
    config = LLMConfig(model="llama3.3:70b", temperature=0.7, max_history=20)

    try:
        assistant = LocalLLMEngine(config)

        print("\n[OK] A.L.I.C.E initialized successfully!")
        print("[GPU] Acceleration: ENABLED")
        print("\nStart chatting! Available commands:")
        print("   /clear     - Clear conversation history")
        print("   /stream    - Toggle streaming mode")
        print("   /temp <n>  - Set creativity (0.0-1.0)")
        print("   /stats     - Show conversation stats")
        print("   exit       - End conversation")
        print("=" * 80)

        stream_mode = True  # Enable streaming by default

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() == "/clear":
                    assistant.clear_history()
                    print("[OK] Conversation history cleared")
                    continue

                if user_input.lower() == "/stream":
                    stream_mode = not stream_mode
                    print(f"[OK] Streaming mode: {'ON' if stream_mode else 'OFF'}")
                    continue

                if user_input.lower().startswith("/temp"):
                    try:
                        parts = user_input.split()
                        if len(parts) == 2:
                            temp_value = float(parts[1])
                            assistant.set_temperature(temp_value)
                            print(f"[OK] Temperature set to: {temp_value}")
                        else:
                            print("[ERROR] Usage: /temp 0.7")
                    except ValueError:
                        print("[ERROR] Invalid temperature value. Use a number between 0.0 and 1.0")
                    continue

                if user_input.lower() == "/stats":
                    stats = assistant.get_stats()
                    print("\nStatistics:")
                    print(f"   Messages: {stats['messages']}")
                    print(f"   Temperature: {stats['temperature']}")
                    print(f"   Model: {stats['model']}")
                    continue

                if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                    print("\nA.L.I.C.E: Goodbye! It was a pleasure assisting you!")
                    break

                # Get response
                print("\nA.L.I.C.E: ", end="", flush=True)

                if stream_mode:
                    for chunk in assistant.stream_chat(user_input):
                        print(chunk, end="", flush=True)
                    print()  # New line after streaming
                else:
                    response = assistant.chat(user_input)
                    print(response)

            except KeyboardInterrupt:
                print("\n\nA.L.I.C.E: Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                print(f"\n[ERROR] Error: {e}")

    except Exception as e:
        logger.error(f"Failed to start A.L.I.C.E: {e}")
        print(f"\n[ERROR] Failed to start A.L.I.C.E: {e}")
        print("\nSetup Instructions:")
        print("1. Install Ollama from: https://ollama.ai")
        print("2. Open a terminal and run: ollama serve")
        print("3. In another terminal, run: ollama pull llama3.3:70b")
        print("4. Wait for download to complete (this may take a while)")
        print("5. Run this script again: python ai/llm_engine.py")
        print("\nYour RTX 5070 Ti will be automatically detected and used!")
