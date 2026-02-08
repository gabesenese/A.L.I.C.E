"""
Scenario Definitions for A.L.I.C.E Testing and Training

Defines conversation scenarios across different domains to test routing,
intent detection, and generate training data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
from pathlib import Path


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
        name="Weekly Forecast",
        description="User wants this week's weather forecast",
        domain="weather",
        steps=[
            ScenarioStep(
                user_input="what is this week forecast ?",
                expected_intent="get_weather_forecast",
                expected_route=ExpectedRoute.TOOL,
                domain="weather",
                expected_entities={"time_range": "week"}
            )
        ],
        tags=["weather", "forecast"]
    ),
    Scenario(
        name="Weekend Forecast",
        description="User wants weekend weather forecast",
        domain="weather",
        steps=[
            ScenarioStep(
                user_input="how will the weather be this weekend?",
                expected_intent="get_weather_forecast",
                expected_route=ExpectedRoute.TOOL,
                domain="weather",
                expected_entities={"time_range": "weekend"}
            )
        ],
        tags=["weather", "forecast", "weekend"]
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
    ),
    Scenario(
        name="Reply to Email Variant 1",
        description="User wants to reply using different wording",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="respond to the most recent email",
                expected_intent="reply_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            )
        ],
        tags=["email", "reply"]
    ),
    Scenario(
        name="Reply to Specific Email",
        description="User wants to reply to a specific email",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="reply to john's email",
                expected_intent="reply_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"sender": "john"}
            )
        ],
        tags=["email", "reply", "context"]
    ),
    Scenario(
        name="Forward Email",
        description="User wants to forward an email",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="forward this email to sarah",
                expected_intent="forward_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"recipient": "sarah"}
            )
        ],
        tags=["email", "forward"]
    ),
    Scenario(
        name="Mark Email as Read",
        description="User wants to mark email as read",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="mark all emails as read",
                expected_intent="mark_read",
                expected_route=ExpectedRoute.TOOL,
                domain="email"
            )
        ],
        tags=["email", "manage"]
    ),
    Scenario(
        name="Archive Email",
        description="User wants to archive emails",
        domain="email",
        steps=[
            ScenarioStep(
                user_input="archive the first email",
                expected_intent="archive_email",
                expected_route=ExpectedRoute.TOOL,
                domain="email",
                expected_entities={"index": 1}
            )
        ],
        tags=["email", "archive"]
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

# Time Scenarios (Testing time intent recognition)
TIME_SCENARIOS = [
    Scenario(
        name="Ask Current Time",
        description="User asks what time it is",
        domain="time",
        steps=[
            ScenarioStep(
                user_input="what time is it?",
                expected_intent="get_time",
                expected_route=ExpectedRoute.TOOL,
                domain="time"
            )
        ],
        tags=["time", "current"]
    ),
    Scenario(
        name="Ask Time Variant 1",
        description="User asks for the time differently",
        domain="time",
        steps=[
            ScenarioStep(
                user_input="tell me the current time",
                expected_intent="get_time",
                expected_route=ExpectedRoute.TOOL,
                domain="time"
            )
        ],
        tags=["time", "current"]
    ),
    Scenario(
        name="Ask Time Variant 2",
        description="User asks what's the time",
        domain="time",
        steps=[
            ScenarioStep(
                user_input="what's the time right now?",
                expected_intent="get_time",
                expected_route=ExpectedRoute.TOOL,
                domain="time"
            )
        ],
        tags=["time", "current"]
    ),
    Scenario(
        name="Ask Date",
        description="User asks for today's date",
        domain="time",
        steps=[
            ScenarioStep(
                user_input="what's today's date?",
                expected_intent="get_date",
                expected_route=ExpectedRoute.TOOL,
                domain="time"
            )
        ],
        tags=["time", "date"]
    ),
    Scenario(
        name="Ask Date Variant",
        description="User asks what day it is",
        domain="time",
        steps=[
            ScenarioStep(
                user_input="what day is it today?",
                expected_intent="get_date",
                expected_route=ExpectedRoute.TOOL,
                domain="time"
            )
        ],
        tags=["time", "date"]
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
        name="Create File Variant 1",
        description="User wants to make a new file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="make a new file named data.json",
                expected_intent="create_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "data.json"}
            )
        ],
        tags=["file", "create"]
    ),
    Scenario(
        name="Create File Variant 2",
        description="User wants to create a file with different wording",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="can you create report.pdf for me",
                expected_intent="create_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "report.pdf"}
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
        name="Read File Variant 1",
        description="User wants to open a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="open config.yaml",
                expected_intent="read_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "config.yaml"}
            )
        ],
        tags=["file", "read"]
    ),
    Scenario(
        name="Read File Variant 2",
        description="User wants to see file contents",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="show me the contents of readme.md",
                expected_intent="read_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "readme.md"}
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
        name="Delete File Variant 1",
        description="User wants to remove a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="remove old_data.csv",
                expected_intent="delete_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"filename": "old_data.csv"}
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
    ),
    Scenario(
        name="Rename File",
        description="User wants to rename a file",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="rename document.txt to final_report.txt",
                expected_intent="move_file",
                expected_route=ExpectedRoute.TOOL,
                domain="file",
                expected_entities={"source": "document.txt", "destination": "final_report.txt"}
            )
        ],
        tags=["file", "move", "rename"]
    ),
    Scenario(
        name="List Files",
        description="User wants to see files in directory",
        domain="file",
        steps=[
            ScenarioStep(
                user_input="list all files in this directory",
                expected_intent="list_files",
                expected_route=ExpectedRoute.TOOL,
                domain="file"
            )
        ],
        tags=["file", "list"]
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
        name="Remember Preference Variant 1",
        description="User wants Alice to save a preference",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="remember I like working out at 6am",
                expected_intent="store_preference",
                expected_route=ExpectedRoute.TOOL,
                domain="memory",
                expected_entities={"preference": "working out", "context": "6am"}
            )
        ],
        tags=["memory", "preferences"]
    ),
    Scenario(
        name="Remember Fact",
        description="User tells Alice to remember a fact",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="remember my birthday is March 15th",
                expected_intent="store_preference",
                expected_route=ExpectedRoute.TOOL,
                domain="memory",
                expected_entities={"preference": "birthday", "context": "March 15th"}
            )
        ],
        tags=["memory", "facts"]
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
        name="Recall Preference Variant 1",
        description="User asks what Alice knows",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="what do you know about my preferences?",
                expected_intent="recall_memory",
                expected_route=ExpectedRoute.RAG,
                domain="memory",
                expected_entities={"topic": "preferences"}
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
    ),
    Scenario(
        name="Search Memory Variant 1",
        description="User searches for past topic",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="find our conversation about the project",
                expected_intent="search_memory",
                expected_route=ExpectedRoute.RAG,
                domain="memory",
                expected_entities={"topic": "project"}
            )
        ],
        tags=["memory", "search", "rag"]
    ),
    Scenario(
        name="Search Memory Variant 2",
        description="User asks about previous discussion",
        domain="memory",
        steps=[
            ScenarioStep(
                user_input="did we discuss the budget last week?",
                expected_intent="search_memory",
                expected_route=ExpectedRoute.RAG,
                domain="memory",
                expected_entities={"topic": "budget", "timeframe": "last week"}
            )
        ],
        tags=["memory", "search", "rag"]
    )
]


def _load_generated_scenarios() -> List[Scenario]:
    """Load Ollama-generated scenarios from scenarios/auto_generated.json."""
    generated_file = Path(__file__).resolve().parents[1] / "auto_generated.json"
    if not generated_file.exists():
        return []

    try:
        with open(generated_file, 'r', encoding='utf-8', errors='ignore') as f:
            payload = json.load(f)
        scenario_defs = payload.get('scenarios', []) if isinstance(payload, dict) else payload
    except Exception:
        return []

    scenarios = []
    for s in scenario_defs:
        try:
            steps = []
            for step in s.get('steps', []):
                expected_route = ExpectedRoute(step.get('expected_route', 'TOOL'))
                steps.append(ScenarioStep(
                    user_input=step.get('user_input', ''),
                    expected_intent=step.get('expected_intent', ''),
                    expected_route=expected_route,
                    domain=s.get('domain'),
                    expected_entities=step.get('expected_entities', {}),
                    notes=step.get('notes', '')
                ))

            scenarios.append(Scenario(
                name=s.get('name', 'Generated Scenario'),
                description=s.get('description', ''),
                domain=s.get('domain', 'unknown'),
                steps=steps,
                tags=s.get('tags', ['generated'])
            ))
        except Exception:
            continue

    return scenarios


# Generated scenarios
GENERATED_SCENARIOS = _load_generated_scenarios()

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
    TIME_SCENARIOS +
    FILE_SCENARIOS +
    MEMORY_SCENARIOS +
    MULTI_TURN_SCENARIOS +
    GENERATED_SCENARIOS
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
