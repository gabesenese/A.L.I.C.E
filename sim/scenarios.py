"""
Scenario Definitions for A.L.I.C.E Testing and Training

Defines conversation scenarios across different domains to test routing,
intent detection, and generate training data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ExpectedRoute(Enum):
    """Expected routing decision for a scenario step"""
    SELF_REFLECTION = "SELF_REFLECTION"
    CONVERSATIONAL = "CONVERSATIONAL"
    TOOL = "TOOL"
    RAG = "RAG"
    LLM_FALLBACK = "LLM_FALLBACK"
    CLARIFICATION = "CLARIFICATION"


@dataclass
class ScenarioStep:
    """A single step in a conversation scenario"""
    user_input: str
    expected_intent: str
    expected_route: ExpectedRoute
    domain: Optional[str] = None
    expected_entities: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass
class ScenarioResult:
    """Result of running a scenario step"""
    step: ScenarioStep
    actual_route: str
    actual_intent: str
    actual_response: str
    teacher_response: Optional[str] = None
    route_match: bool = False
    intent_match: bool = False
    needs_learning: bool = False
    confidence: float = 0.0


@dataclass
class Scenario:
    """A complete conversation scenario"""
    name: str
    description: str
    domain: str
    steps: List[ScenarioStep]
    tags: List[str] = field(default_factory=list)


# Email Domain Scenarios
EMAIL_SCENARIOS = [
    Scenario(
        name="List Recent Emails",
        description="User wants to see recent emails",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="show me my recent emails",
                expected_intent="list_emails",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            ),
            ScenarioStep(
                user_input="show the first one",
                expected_intent="read_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"index": 1}
            )
        ],
        tags=["email", "list", "read"]
    ),
    Scenario(
        name="Search Emails",
        description="User wants to search for specific emails",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="find emails from john",
                expected_intent="search_emails",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"query": "from:john"}
            )
        ],
        tags=["email", "search"]
    ),
    Scenario(
        name="Compose Email",
        description="User wants to compose a new email",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="compose an email to sarah",
                expected_intent="compose_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"to": "sarah"}
            )
        ],
        tags=["email", "compose"]
    ),
    Scenario(
        name="Delete Email",
        description="User wants to delete an email",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="delete the third email",
                expected_intent="delete_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"index": 3}
            )
        ],
        tags=["email", "delete"]
    )
]


# Notes Domain Scenarios
NOTES_SCENARIOS = [
    Scenario(
        name="Create Note",
        description="User wants to create a new note",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="create a note about the meeting tomorrow",
                expected_intent="create_note",
                expected_route=ExpectedRoute.TOOL,
                domain="notes",
                expected_entities={"content": "meeting tomorrow"}
            )
        ],
        tags=["notes", "create"]
    ),
    Scenario(
        name="Search Notes",
        description="User wants to search their notes",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="find my notes about project ideas",
                expected_intent="search_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes",
                expected_entities={"query": "project ideas"}
            )
        ],
        tags=["notes", "search"]
    ),
    Scenario(
        name="List All Notes",
        description="User wants to see all notes",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="show me all my notes",
                expected_intent="list_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes"
            )
        ],
        tags=["notes", "list"]
    )
]


# Weather/Time/System Scenarios
SYSTEM_SCENARIOS = [
    Scenario(
        name="Check Weather",
        description="User wants current weather",
        domain="weather",
        steps=[
            ScenarioStep(
                user_input="what's the weather like?",
                expected_intent="get_weather",
                expected_route=ExpectedRoute.TOOL,
                domain="weather"
            )
        ],
        tags=["weather", "system"]
    ),
    Scenario(
        name="Check Time",
        description="User wants current time",
        domain="time",
        steps=[
            ScenarioStep(
                user_input="what time is it?",
                expected_intent="get_time",
                expected_route=ExpectedRoute.TOOL,
                domain="time"
            )
        ],
        tags=["time", "system"]
    ),
    Scenario(
        name="System Status",
        description="User wants system information",
        domain="system",
        steps=[
            ScenarioStep(
                user_input="how's the system doing?",
                expected_intent="system_status",
                expected_route=ExpectedRoute.CONVERSATIONAL,
                domain="system"
            )
        ],
        tags=["system", "status"]
    )
]


# Conversational Scenarios
CONVERSATIONAL_SCENARIOS = [
    Scenario(
        name="Greeting",
        description="User greets Alice",
        domain="conversational",
        steps=[
            ScenarioStep(
                user_input="hi alice",
                expected_intent="greeting",
                expected_route=ExpectedRoute.CONVERSATIONAL,
                domain="conversational"
            )
        ],
        tags=["conversational", "greeting"]
    ),
    Scenario(
        name="Thanks",
        description="User thanks Alice",
        domain="conversational",
        steps=[
            ScenarioStep(
                user_input="thanks for your help",
                expected_intent="thanks",
                expected_route=ExpectedRoute.CONVERSATIONAL,
                domain="conversational"
            )
        ],
        tags=["conversational", "thanks"]
    ),
    Scenario(
        name="How Are You",
        description="User asks how Alice is doing",
        domain="conversational",
        steps=[
            ScenarioStep(
                user_input="how are you doing?",
                expected_intent="status_inquiry",
                expected_route=ExpectedRoute.CONVERSATIONAL,
                domain="conversational"
            )
        ],
        tags=["conversational", "status"]
    )
]


# Clarification Scenarios (Tricky Vague Prompts)
CLARIFICATION_SCENARIOS = [
    Scenario(
        name="Vague Sun Question",
        description="Ambiguous question about sun requiring clarification",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="tell me about the sun",
                expected_intent="vague_question",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="Should ask: weather, astronomy, or general info?"
            )
        ],
        tags=["clarification", "vague"]
    ),
    Scenario(
        name="Vague Thing Reference",
        description="User references 'that thing' without context",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="can you do that thing?",
                expected_intent="vague_request",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="Should ask what thing they're referring to"
            )
        ],
        tags=["clarification", "vague"]
    ),
    Scenario(
        name="Ambiguous Time Reference",
        description="Unclear time reference",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="schedule it for tomorrow",
                expected_intent="schedule_action",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="Should ask what to schedule and what time"
            )
        ],
        tags=["clarification", "ambiguous"]
    ),
    Scenario(
        name="Generic Question",
        description="Very generic question requiring domain clarification",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="what about yesterday?",
                expected_intent="vague_temporal_question",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="Should ask: weather, emails, calendar, or what?"
            )
        ],
        tags=["clarification", "temporal"]
    )
]


# All scenarios combined
ALL_SCENARIOS = (
    EMAIL_SCENARIOS +
    NOTES_SCENARIOS +
    SYSTEM_SCENARIOS +
    CONVERSATIONAL_SCENARIOS +
    CLARIFICATION_SCENARIOS
)


def get_scenarios_by_domain(domain: str) -> List[Scenario]:
    """Get all scenarios for a specific domain"""
    return [s for s in ALL_SCENARIOS if s.domain == domain]


def get_scenarios_by_tag(tag: str) -> List[Scenario]:
    """Get all scenarios with a specific tag"""
    return [s for s in ALL_SCENARIOS if tag in s.tags]


def get_scenario_by_name(name: str) -> Optional[Scenario]:
    """Get a specific scenario by name"""
    for scenario in ALL_SCENARIOS:
        if scenario.name == name:
            return scenario
    return None
