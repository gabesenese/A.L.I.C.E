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
                user_input="read the first email",
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


# Enhanced Email Scenarios (Delete, Compose Variations)
ENHANCED_EMAIL_SCENARIOS = [
    Scenario(
        name="Delete Email by Index",
        description="User wants to delete a specific email",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="delete the second email",
                expected_intent="delete_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"index": 2}
            )
        ],
        tags=["email", "delete"]
    ),
    Scenario(
        name="Compose with Recipient",
        description="User composes email to specific person",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="send an email to alice@example.com",
                expected_intent="compose_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"recipient": "alice@example.com"}
            )
        ],
        tags=["email", "compose"]
    ),
    Scenario(
        name="Reply to Email",
        description="User wants to reply to most recent email",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="reply to the last email",
                expected_intent="reply_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            )
        ],
        tags=["email", "reply"]
    )
]


# Enhanced Notes Scenarios (List, Search Variations)
ENHANCED_NOTES_SCENARIOS = [
    Scenario(
        name="List All Notes",
        description="User wants to see all their notes",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="show all my notes",
                expected_intent="list_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes"
            )
        ],
        tags=["notes", "list"]
    ),
    Scenario(
        name="Search Notes by Topic",
        description="User searches notes for specific topic",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="find my notes about python",
                expected_intent="search_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes",
                expected_entities={"query": "python"}
            )
        ],
        tags=["notes", "search"]
    ),
    Scenario(
        name="Delete Note",
        description="User wants to delete a note",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="delete my old todo list",
                expected_intent="delete_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes",
                notes="Should delete the todo list"
            )
        ],
        tags=["notes", "delete"]
    )
]


# Enhanced Clarification Scenarios (Multi-turn, Context)
ENHANCED_CLARIFICATION_SCENARIOS = [
    Scenario(
        name="Ambiguous Command Without Context",
        description="User asks unclear question needing domain clarification",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="what happened last week?",
                expected_intent="vague_temporal_question",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="Could be: emails, calendar, notes, or general info?"
            )
        ],
        tags=["clarification", "temporal"]
    ),
    Scenario(
        name="Pronoun Reference Without Context",
        description="User uses pronoun but no prior context",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="who is he?",
                expected_intent="vague_question",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="No prior mention of 'he' in conversation"
            )
        ],
        tags=["clarification", "pronoun"]
    ),
    Scenario(
        name="Vague Action Request",
        description="User asks to do something but what?",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="add this to the list",
                expected_intent="vague_request",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="No 'this' mentioned yet, which list?"
            )
        ],
        tags=["clarification", "action"]
    )
]


# Enhanced Weather Scenarios
ENHANCED_WEATHER_SCENARIOS = [
    Scenario(
        name="Weather for Specific Location",
        description="User asks weather for a specific place",
        domain="weather",
        steps=[
            ScenarioStep(
                user_input="what's the weather in tokyo?",
                expected_intent="get_weather",
                expected_route=ExpectedRoute.TOOL,
                domain="weather",
                expected_entities={"location": "tokyo"}
            )
        ],
        tags=["weather", "location"]
    ),
    Scenario(
        name="Weather Forecast",
        description="User asks for weather forecast",
        domain="weather",
        steps=[
            ScenarioStep(
                user_input="will it rain tomorrow?",
                expected_intent="get_weather",
                expected_route=ExpectedRoute.TOOL,
                domain="weather"
            )
        ],
        tags=["weather", "forecast"]
    )
]


# Enhanced System Scenarios
ENHANCED_SYSTEM_SCENARIOS = [
    Scenario(
        name="CPU Usage Check",
        description="User asks about system CPU usage",
        domain="system",
        steps=[
            ScenarioStep(
                user_input="what's my cpu usage?",
                expected_intent="system_status",
                expected_route=ExpectedRoute.CONVERSATIONAL,
                domain="system"
            )
        ],
        tags=["system", "cpu"]
    ),
    Scenario(
        name="Memory Status",
        description="User asks about available memory",
        domain="system",
        steps=[
            ScenarioStep(
                user_input="how much memory is available?",
                expected_intent="system_status",
                expected_route=ExpectedRoute.CONVERSATIONAL,
                domain="system"
            )
        ],
        tags=["system", "memory"]
    ),
    Scenario(
        name="Battery Status",
        description="User checks battery level",
        domain="system",
        steps=[
            ScenarioStep(
                user_input="is the battery low?",
                expected_intent="system_status",
                expected_route=ExpectedRoute.CONVERSATIONAL,
                domain="system"
            )
        ],
        tags=["system", "battery"]
    )
]


# Multi-turn Context Scenarios - Testing follow-up questions and pronoun resolution
MULTI_TURN_SCENARIOS = [
    Scenario(
        name="Email Reply Follow-up",
        description="User asks to reply to an email, then asks to send it",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="reply to the last email",
                expected_intent="reply_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            ),
            ScenarioStep(
                user_input="send it",
                expected_intent="compose_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                notes="Follow-up should understand 'it' refers to the reply"
            )
        ],
        tags=["multi-turn", "email", "pronoun-resolution"]
    ),
    Scenario(
        name="Note Delete Follow-up",
        description="User asks to create a note, then delete it",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="create a note about my meeting",
                expected_intent="create_note",
                expected_route=ExpectedRoute.TOOL,
                domain="notes"
            ),
            ScenarioStep(
                user_input="actually delete it",
                expected_intent="delete_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes",
                notes="Follow-up should understand 'it' refers to the note just created"
            )
        ],
        tags=["multi-turn", "notes", "pronoun-resolution"]
    ),
    Scenario(
        name="Search Email Then Reply",
        description="User searches for emails, then replies to one",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="search for emails from john",
                expected_intent="search_emails",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            ),
            ScenarioStep(
                user_input="reply to the first one",
                expected_intent="reply_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                notes="Follow-up should remember the search context"
            )
        ],
        tags=["multi-turn", "email", "entity-tracking"]
    ),
    Scenario(
        name="Clarification With Action",
        description="User asks vague question, then clarifies with specific action",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="can you do something with my emails?",
                expected_intent="vague_request",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification"
            ),
            ScenarioStep(
                user_input="sort them by date",
                expected_intent="vague_request",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="Follow-up clarifies the vague request with more context"
            )
        ],
        tags=["multi-turn", "clarification", "follow-up"]
    ),
    Scenario(
        name="Weather Then Schedule",
        description="User asks about weather, then wants to plan something",
        domain="weather",
        steps=[
            ScenarioStep(
                user_input="what's the weather like tomorrow?",
                expected_intent="get_weather",
                expected_route=ExpectedRoute.TOOL,
                domain="weather"
            ),
            ScenarioStep(
                user_input="schedule a picnic if it's sunny",
                expected_intent="schedule_action",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="weather",
                notes="Follow-up connects weather info to scheduling decision"
            )
        ],
        tags=["multi-turn", "weather", "conditional-action"]
    ),
    Scenario(
        name="List Then Specific Read",
        description="User lists emails, then wants to read a specific one",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="show me my emails",
                expected_intent="list_emails",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            ),
            ScenarioStep(
                user_input="read the second one",
                expected_intent="read_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                notes="Follow-up references ordinal position from previous list"
            )
        ],
        tags=["multi-turn", "email", "ordinal-reference"]
    ),
    Scenario(
        name="Create Then List Notes",
        description="User creates a note, then lists all notes",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="create a note called 'todo'",
                expected_intent="create_note",
                expected_route=ExpectedRoute.TOOL,
                domain="notes"
            ),
            ScenarioStep(
                user_input="show me all my notes",
                expected_intent="list_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes",
                notes="Follow-up should see newly created note in context"
            )
        ],
        tags=["multi-turn", "notes", "state-tracking"]
    )
]


# File Operation Scenarios (Testing advertised file capabilities)
FILE_SCENARIOS = [
    Scenario(
        name="Create File",
        description="User wants to create a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="create a file called test.txt",
                expected_intent="create_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "test.txt"}
            )
        ],
        tags=["file", "create"]
    ),
    Scenario(
        name="Read File",
        description="User wants to read a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="read the file called notes.txt",
                expected_intent="read_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "notes.txt"}
            )
        ],
        tags=["file", "read"]
    ),
    Scenario(
        name="Delete File",
        description="User wants to delete a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="delete the file test.txt",
                expected_intent="delete_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "test.txt"}
            )
        ],
        tags=["file", "delete"]
    ),
    Scenario(
        name="Move File",
        description="User wants to move/rename a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="move notes.txt to archive folder",
                expected_intent="move_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"source": "notes.txt", "destination": "archive"}
            )
        ],
        tags=["file", "move"]
    )
]


# Memory/RAG Scenarios (Testing advertised memory capabilities)
MEMORY_SCENARIOS = [
    Scenario(
        name="Remember Preference",
        description="User tells Alice to remember something",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="remember that I prefer coffee in the morning",
                expected_intent="store_preference",
                expected_route=ExpectedRoute.TOOL,
                domain="memory",
                expected_entities={"preference": "coffee", "context": "morning"}
            )
        ],
        tags=["memory", "preferences"]
    ),
    Scenario(
        name="Recall Preference",
        description="User asks Alice what she remembers",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="what do you remember about my morning routine?",
                expected_intent="recall_memory",
                expected_route=ExpectedRoute.RAG,
                domain="memory",
                expected_entities={"topic": "morning routine"}
            )
        ],
        tags=["memory", "recall", "rag"]
    ),
    Scenario(
        name="Search Memory",
        description="User searches past conversations",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="what did we talk about yesterday?",
                expected_intent="search_memory",
                expected_route=ExpectedRoute.RAG,
                domain="memory",
                expected_entities={"timeframe": "yesterday"}
            )
        ],
        tags=["memory", "search", "rag"]
    )
]


# All scenarios combined
ALL_SCENARIOS = (
    EMAIL_SCENARIOS +
    ENHANCED_EMAIL_SCENARIOS +
    NOTES_SCENARIOS +
    ENHANCED_NOTES_SCENARIOS +
    SYSTEM_SCENARIOS +
    ENHANCED_SYSTEM_SCENARIOS +
    CONVERSATIONAL_SCENARIOS +
    CLARIFICATION_SCENARIOS +
    ENHANCED_CLARIFICATION_SCENARIOS +
    ENHANCED_WEATHER_SCENARIOS +
    FILE_SCENARIOS +
    MEMORY_SCENARIOS +
    MULTI_TURN_SCENARIOS
)


def get_scenarios_by_domain(domain: str) -> List[Scenario]:
        name="Create File",
        description="User wants to create a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="create a file called test.txt",
                expected_intent="create_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "test.txt"}
            )
        ],
        tags=["file", "create"]
    ),
    Scenario(
        name="Read File",
        description="User wants to read a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="read the file called notes.txt",
                expected_intent="read_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "notes.txt"}
            )
        ],
        tags=["file", "read"]
    ),
    Scenario(
        name="Delete File",
        description="User wants to delete a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="delete the file test.txt",
                expected_intent="delete_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "test.txt"}
            )
        ],
        tags=["file", "delete"]
    ),
    Scenario(
        name="Move File",
        description="User wants to move/rename a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="move notes.txt to archive folder",
                expected_intent="move_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"source": "notes.txt", "destination": "archive"}
            )
        ],
        tags=["file", "move"]
    )
]


# Memory/RAG Scenarios (Testing advertised memory capabilities)
MEMORY_SCENARIOS = [
    Scenario(
        name="Remember Preference",
        description="User tells Alice to remember something",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="remember that I prefer coffee in the morning",
                expected_intent="store_preference",
                expected_route=ExpectedRoute.TOOL,
                domain="memory",
                expected_entities={"preference": "coffee", "context": "morning"}
            )
        ],
        tags=["memory", "preferences"]
    ),
    Scenario(
        name="Recall Preference",
        description="User asks Alice what she remembers",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="what do you remember about my morning routine?",
                expected_intent="recall_memory",
                expected_route=ExpectedRoute.RAG,
                domain="memory",
                expected_entities={"topic": "morning routine"}
            )
        ],
        tags=["memory", "recall", "rag"]
    ),
    Scenario(
        name="Search Memory",
        description="User searches past conversations",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="what did we talk about yesterday?",
                expected_intent="search_memory",
                expected_route=ExpectedRoute.RAG,
                domain="memory",
                expected_entities={"timeframe": "yesterday"}
            )
        ],
        tags=["memory", "search", "rag"]
    )
]


def get_scenarios_by_domain(domain: str) -> List[Scenario]:
    """Get all scenarios for a specific domain"""
    return [s for s in ALL_SCENARIOS if s.domain == domain]


def get_scenarios_by_tag(tag: str) -> List[Scenario]:
    """Get all scenarios with a specific tag"""
    return [s for s in ALL_SCENARIOS if tag in s.tags]


# Multi-turn Context Scenarios - Testing follow-up questions and pronoun resolution
MULTI_TURN_SCENARIOS = [
    Scenario(
        name="Email Reply Follow-up",
        description="User asks to reply to an email, then asks to send it",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="reply to the last email",
                expected_intent="reply_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            ),
            ScenarioStep(
                user_input="send it",
                expected_intent="compose_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                notes="Follow-up should understand 'it' refers to the reply"
            )
        ],
        tags=["multi-turn", "email", "pronoun-resolution"]
    ),
    Scenario(
        name="Note Delete Follow-up",
        description="User asks to create a note, then delete it",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="create a note about my meeting",
                expected_intent="create_note",
                expected_route=ExpectedRoute.TOOL,
                domain="notes"
            ),
            ScenarioStep(
                user_input="actually delete it",
                expected_intent="delete_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes",
                notes="Follow-up should understand 'it' refers to the note just created"
            )
        ],
        tags=["multi-turn", "notes", "pronoun-resolution"]
    ),
    Scenario(
        name="Search Email Then Reply",
        description="User searches for emails, then replies to one",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="search for emails from john",
                expected_intent="search_emails",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            ),
            ScenarioStep(
                user_input="reply to the first one",
                expected_intent="reply_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                notes="Follow-up should remember the search context"
            )
        ],
        tags=["multi-turn", "email", "entity-tracking"]
    ),
    Scenario(
        name="Clarification With Action",
        description="User asks vague question, then clarifies with specific action",
        domain="clarification",
        steps=[
            ScenarioStep(
                user_input="can you do something with my emails?",
                expected_intent="vague_request",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification"
            ),
            ScenarioStep(
                user_input="sort them by date",
                expected_intent="vague_request",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="clarification",
                notes="Follow-up clarifies the vague request with more context"
            )
        ],
        tags=["multi-turn", "clarification", "follow-up"]
    ),
    Scenario(
        name="Weather Then Schedule",
        description="User asks about weather, then wants to plan something",
        domain="weather",
        steps=[
            ScenarioStep(
                user_input="what's the weather like tomorrow?",
                expected_intent="get_weather",
                expected_route=ExpectedRoute.TOOL,
                domain="weather"
            ),
            ScenarioStep(
                user_input="schedule a picnic if it's sunny",
                expected_intent="schedule_action",
                expected_route=ExpectedRoute.CLARIFICATION,
                domain="weather",
                notes="Follow-up connects weather info to scheduling decision"
            )
        ],
        tags=["multi-turn", "weather", "conditional-action"]
    ),
    Scenario(
        name="List Then Specific Read",
        description="User lists emails, then wants to read a specific one",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="show me my emails",
                expected_intent="list_emails",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            ),
            ScenarioStep(
                user_input="read the second one",
                expected_intent="read_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                notes="Follow-up references ordinal position from previous list"
            )
        ],
        tags=["multi-turn", "email", "ordinal-reference"]
    ),
    Scenario(
        name="Create Then List Notes",
        description="User creates a note, then lists all notes",
        domain="notes",
        steps=[
            ScenarioStep(
                user_input="create a note called 'todo'",
                expected_intent="create_note",
                expected_route=ExpectedRoute.TOOL,
                domain="notes"
            ),
            ScenarioStep(
                user_input="show me all my notes",
                expected_intent="list_notes",
                expected_route=ExpectedRoute.TOOL,
                domain="notes",
                notes="Follow-up should see newly created note in context"
            )
        ],
        tags=["multi-turn", "notes", "state-tracking"]
    )
]


def get_scenario_by_name(name: str) -> Optional[Scenario]:
    """Get a specific scenario by name"""
    for scenario in ALL_SCENARIOS:
        if scenario.name == name:
            return scenario
    return None
