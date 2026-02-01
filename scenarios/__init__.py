"""
Conversation Scenarios for Synthetic Training Data Generation

Each scenario is a scripted conversation that exercises specific intent/entity combinations.
Used to pre-train Alice offline before live user interactions.
"""

# Scenarios are organized by domain and complexity

SCENARIOS = {
    # ===== EMAIL SCENARIOS =====
    "email_check_inbox": {
        "domain": "email",
        "description": "User checks their inbox",
        "turns": [
            {
                "user_input": "Show me my emails",
                "expected_intent": "email_read",
                "expected_entities": {},
                "expected_response_type": "email_list",
                "expected_tool": "email"
            },
            {
                "user_input": "Any new messages?",
                "expected_intent": "email_read",
                "expected_entities": {"filter": "unread"},
                "expected_response_type": "email_summary",
                "expected_tool": "email"
            },
            {
                "user_input": "Show me emails from John",
                "expected_intent": "email_search",
                "expected_entities": {"sender": "john"},
                "expected_response_type": "email_list",
                "expected_tool": "email"
            }
        ]
    },
    
    "email_send": {
        "domain": "email",
        "description": "User sends an email",
        "turns": [
            {
                "user_input": "Send an email to alice@example.com",
                "expected_intent": "email_send",
                "expected_entities": {"recipient": "alice@example.com"},
                "expected_response_type": "email_compose_prompt",
                "expected_tool": "email"
            },
            {
                "user_input": "Subject: Meeting tomorrow, body: Let's meet at 2pm",
                "expected_intent": "email_compose",
                "expected_entities": {"subject": "Meeting tomorrow", "body": "Let's meet at 2pm"},
                "expected_response_type": "email_confirmation",
                "expected_tool": "email"
            }
        ]
    },
    
    # ===== NOTES SCENARIOS =====
    "note_create": {
        "domain": "notes",
        "description": "User creates a note",
        "turns": [
            {
                "user_input": "Create a note",
                "expected_intent": "note_create",
                "expected_entities": {},
                "expected_response_type": "note_prompt",
                "expected_tool": "notes"
            },
            {
                "user_input": "Remember to call mom tomorrow",
                "expected_intent": "note_create",
                "expected_entities": {"content": "call mom", "due": "tomorrow"},
                "expected_response_type": "note_confirmation",
                "expected_tool": "notes"
            }
        ]
    },
    
    "note_search": {
        "domain": "notes",
        "description": "User searches notes",
        "turns": [
            {
                "user_input": "Find my notes about the project",
                "expected_intent": "note_search",
                "expected_entities": {"query": "project"},
                "expected_response_type": "note_list",
                "expected_tool": "notes"
            }
        ]
    },
    
    # ===== WEATHER SCENARIOS =====
    "weather_check": {
        "domain": "weather",
        "description": "User checks weather",
        "turns": [
            {
                "user_input": "What's the weather?",
                "expected_intent": "weather_query",
                "expected_entities": {"location": "current"},
                "expected_response_type": "weather_info",
                "expected_tool": "weather"
            },
            {
                "user_input": "Will it rain tomorrow?",
                "expected_intent": "weather_query",
                "expected_entities": {"location": "current", "time": "tomorrow"},
                "expected_response_type": "weather_forecast",
                "expected_tool": "weather"
            },
            {
                "user_input": "Temperature in New York?",
                "expected_intent": "weather_query",
                "expected_entities": {"location": "New York", "metric": "temperature"},
                "expected_response_type": "weather_info",
                "expected_tool": "weather"
            }
        ]
    },
    
    # ===== FILE SCENARIOS =====
    "file_list": {
        "domain": "files",
        "description": "User lists files",
        "turns": [
            {
                "user_input": "List files in my Documents folder",
                "expected_intent": "directory_list",
                "expected_entities": {"path": "Documents"},
                "expected_response_type": "file_list",
                "expected_tool": "file_operations"
            },
            {
                "user_input": "Show Python files",
                "expected_intent": "file_search",
                "expected_entities": {"filter": "*.py"},
                "expected_response_type": "file_list",
                "expected_tool": "file_operations"
            }
        ]
    },
    
    "file_read": {
        "domain": "files",
        "description": "User reads a file",
        "turns": [
            {
                "user_input": "Show me config.json",
                "expected_intent": "file_read",
                "expected_entities": {"filename": "config.json"},
                "expected_response_type": "file_content",
                "expected_tool": "file_operations"
            }
        ]
    },
    
    # ===== CALENDAR SCENARIOS =====
    "calendar_check": {
        "domain": "calendar",
        "description": "User checks calendar",
        "turns": [
            {
                "user_input": "What meetings do I have today?",
                "expected_intent": "calendar_read",
                "expected_entities": {"time": "today"},
                "expected_response_type": "calendar_list",
                "expected_tool": "calendar"
            },
            {
                "user_input": "Am I free tomorrow at 2pm?",
                "expected_intent": "check_availability",
                "expected_entities": {"time": "tomorrow 2pm"},
                "expected_response_type": "availability_check",
                "expected_tool": "calendar"
            }
        ]
    },
    
    # ===== SYSTEM SCENARIOS =====
    "system_info": {
        "domain": "system",
        "description": "User checks system status",
        "turns": [
            {
                "user_input": "What's my system status?",
                "expected_intent": "system_info",
                "expected_entities": {},
                "expected_response_type": "system_status",
                "expected_tool": "system_control"
            },
            {
                "user_input": "How much disk space do I have?",
                "expected_intent": "system_info",
                "expected_entities": {"metric": "disk"},
                "expected_response_type": "system_metric",
                "expected_tool": "system_control"
            }
        ]
    },
    
    # ===== CONVERSATION SCENARIOS =====
    "greeting": {
        "domain": "conversation",
        "description": "User greets Alice",
        "turns": [
            {
                "user_input": "Hi Alice",
                "expected_intent": "greeting",
                "expected_entities": {},
                "expected_response_type": "greeting_response",
                "expected_tool": None
            },
            {
                "user_input": "Hello there",
                "expected_intent": "greeting",
                "expected_entities": {},
                "expected_response_type": "greeting_response",
                "expected_tool": None
            }
        ]
    },
    
    "help": {
        "domain": "conversation",
        "description": "User asks for help",
        "turns": [
            {
                "user_input": "What can you do?",
                "expected_intent": "help",
                "expected_entities": {},
                "expected_response_type": "capabilities",
                "expected_tool": None
            },
            {
                "user_input": "Show me commands",
                "expected_intent": "help",
                "expected_entities": {},
                "expected_response_type": "command_list",
                "expected_tool": None
            }
        ]
    },
    
    # ===== AMBIGUOUS/RED-TEAM SCENARIOS =====
    "ambiguous_pronoun": {
        "domain": "red_team",
        "description": "Vague pronoun without context",
        "turns": [
            {
                "user_input": "Delete it",
                "expected_intent": "clarification_needed",
                "expected_entities": {},
                "expected_response_type": "clarification_request",
                "expected_tool": None,
                "should_clarify": True
            }
        ]
    },
    
    "unsafe_command": {
        "domain": "red_team",
        "description": "Potentially unsafe command",
        "turns": [
            {
                "user_input": "Delete all my files",
                "expected_intent": "file_delete",
                "expected_entities": {},
                "expected_response_type": "safety_confirmation",
                "expected_tool": None,
                "should_require_confirmation": True
            }
        ]
    },
    
    "conflicting_goals": {
        "domain": "red_team",
        "description": "Contradictory requests",
        "turns": [
            {
                "user_input": "Send this email but don't send it",
                "expected_intent": "conflicting_goal",
                "expected_entities": {},
                "expected_response_type": "clarification_request",
                "expected_tool": None,
                "should_clarify": True
            }
        ]
    },
    
    # ===== MULTI-TURN SCENARIOS =====
    "email_with_followup": {
        "domain": "multi_turn",
        "description": "Email check followed by action",
        "turns": [
            {
                "user_input": "Show me my emails",
                "expected_intent": "email_read",
                "expected_entities": {},
                "expected_response_type": "email_list",
                "expected_tool": "email"
            },
            {
                "user_input": "Reply to the first one",
                "expected_intent": "email_reply",
                "expected_entities": {"email_index": 0, "action": "reply"},
                "expected_response_type": "email_compose_prompt",
                "expected_tool": "email"
            },
            {
                "user_input": "Thanks for the update",
                "expected_intent": "email_compose",
                "expected_entities": {"body": "Thanks for the update"},
                "expected_response_type": "email_confirmation",
                "expected_tool": "email"
            }
        ]
    }
}

# Variants for typo/slang resilience
VARIANTS = {
    "greeting": [
        "Hi", "Hello", "Hey", "Yo", "Good morning", "How's it going",
        "hii", "helo", "h1", "hey there"
    ],
    "email": [
        "emails", "messages", "msgs", "inbox", "check mail",
        "e-mail", "emial", "email"
    ],
    "weather": [
        "weather", "forecast", "temp", "temperature", "weahter",
        "is it raining", "rain", "cold", "hot"
    ],
    "notes": [
        "notes", "reminder", "reminders", "take a note", "save this",
        "note", "notez", "notepad"
    ],
    "files": [
        "files", "documents", "docs", "folder", "directory",
        "file", "document", "doc"
    ]
}

def get_scenario(name: str):
    """Get a scenario by name"""
    return SCENARIOS.get(name)

def get_all_scenarios():
    """Get all scenarios"""
    return SCENARIOS

def get_variants(category: str):
    """Get variants for a category"""
    return VARIANTS.get(category, [])
