"""Single-owner turn routing policy for plugin-vs-LLM execution decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnRoutingDecision:
    should_try_plugins: bool
    force_skip_plugins: bool
    force_try_plugins: bool
    owner: str
    reason: str


class TurnRoutingPolicy:
    """Centralized ownership for runtime route selection."""

    OWNER = "executive_turn_routing_policy"

    def decide(
        self,
        *,
        executive_action: str,
        runtime_allow_tools: bool,
        runtime_preference: str,
        is_short_followup: bool,
        is_pure_conversation: bool,
        force_plugins_for_notes: bool,
    ) -> TurnRoutingDecision:
        action = str(executive_action or "").strip().lower()
        preference = str(runtime_preference or "balanced").strip().lower()

        force_skip_plugins = action in {"use_llm", "answer_direct"}
        force_try_plugins = action in {"use_plugin", "search"}

        if not runtime_allow_tools:
            force_skip_plugins = True

        if runtime_allow_tools and preference == "tool_first":
            force_try_plugins = True

        should_try_plugins = (
            force_try_plugins
            or (
                (not is_short_followup and not is_pure_conversation)
                or force_plugins_for_notes
            )
        ) and not force_skip_plugins

        if should_try_plugins:
            reason = "policy_allows_tool_path"
        elif not runtime_allow_tools:
            reason = "runtime_controls_disabled_tools"
        elif is_pure_conversation:
            reason = "pure_conversation_prefers_non_tool"
        else:
            reason = "short_followup_prefers_non_tool"

        return TurnRoutingDecision(
            should_try_plugins=bool(should_try_plugins),
            force_skip_plugins=bool(force_skip_plugins),
            force_try_plugins=bool(force_try_plugins),
            owner=self.OWNER,
            reason=reason,
        )


_turn_routing_policy: TurnRoutingPolicy | None = None


def get_turn_routing_policy() -> TurnRoutingPolicy:
    global _turn_routing_policy
    if _turn_routing_policy is None:
        _turn_routing_policy = TurnRoutingPolicy()
    return _turn_routing_policy
