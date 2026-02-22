"""
Seed Response Templates for A.L.I.C.E
======================================
Creates initial response templates to teach Alice how to formulate responses.

Run this once to give Alice examples of how to phrase different types of actions.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.core.response_formulator import get_response_formulator


def seed_templates():
    """Seed initial response templates"""
    formulator = get_response_formulator()

    print("Seeding response templates for Alice...")
    print("=" * 60)

    # Notes plugin templates
    formulator.add_template(
        action="create_note",
        example_data={"title": "Shopping List", "tags": ["personal"], "note_type": "general"},
        example_phrasings=[
            "I created a note called 'Shopping List' for you.",
            "Done. I added a new note titled 'Shopping List'.",
            "Your note 'Shopping List' is ready.",
        ],
        formulation_rules=[
            "Mention the note title",
            "Be concise", 
            "Confirm the action"
        ]
    )

    formulator.add_template(
        action="delete_notes",
        example_data={"count": 3, "archived": True, "permanent": False},
        example_phrasings=[
            "I archived 3 notes. They're not permanently deleted.",
            "Done. I moved 3 notes to your archive.",
            "Archived 3 notes for you. You can restore them anytime.",
        ],
        formulation_rules=[
            "State the count",
            "Clarify archived vs deleted",
            "Mention restoration possibility"
        ]
    )

    formulator.add_template(
        action="search_notes",
        example_data={"count": 5, "query": "meeting", "found": True},
        example_phrasings=[
            "I found 5 notes about 'meeting'.",
            "Found 5 notes matching 'meeting'.",
            "I located 5 notes for you about 'meeting'.",
        ],
        formulation_rules=[
            "State result count",
            "Reference the search query",
            "Use 'found' for success, 'didn't find' for no results"
        ]
    )

    formulator.add_template(
        action="search_notes_empty",
        example_data={"count": 0, "query": "vacation", "found": False},
        example_phrasings=[
            "I didn't find any notes about 'vacation'.",
            "No notes found matching 'vacation'.",
            "I couldn't find notes for 'vacation'.",
        ],
        formulation_rules=[
            "Be clear no results were found",
            "Reference the search query",
            "Don't apologize unnecessarily"
        ]
    )

    formulator.add_template(
        action="set_priority",
        example_data={"note_title": "meeting", "priority": "high", "note_id": "abc123"},
        example_phrasings=[
            "Set priority to high for 'meeting'.",
            "I marked 'meeting' as high priority.",
            "Done. 'meeting' is now high priority.",
        ],
        formulation_rules=[
            "State the new priority level",
            "Mention the note title",
            "Confirm the action was completed"
        ]
    )

    # Weather plugin templates
    formulator.add_template(
        action="weather_current",
        example_data={"location": "Seattle", "temp": 65, "condition": "Cloudy", "unit": "C"},
        example_phrasings=[
            "It's 65°C and cloudy in Seattle right now.",
            "Current weather in Seattle: 65°C, cloudy conditions.",
            "In Seattle, it's 65 degrees and cloudy.",
        ],
        formulation_rules=[
            "State temperature with unit",
            "Mention condition",
            "Include location",
            "Use 'current' or 'right now' for present tense"
        ]
    )

    formulator.add_template(
        action="weather_forecast",
        example_data={"location": "Portland", "high": 72, "low": 55, "condition": "Rainy", "day": "tomorrow"},
        example_phrasings=[
            "Tomorrow in Portland: high of 72°F, low of 55°F, with rain.",
            "Portland tomorrow will be rainy with temps between 55 and 72 degrees.",
            "For tomorrow in Portland, expect rain with highs around 72°F.",
        ],
        formulation_rules=[
            "Specify when (today/tomorrow/etc)",
            "Include location",
            "State high and low temps",
            "Mention condition"
        ]
    )

    # Time plugin templates
    formulator.add_template(
        action="current_time",
        example_data={"time": "3:45 PM", "timezone": "PST", "date": "February 12, 2026"},
        example_phrasings=[
            "It's 3:45 PM PST.",
            "The time is 3:45 PM.",
            "Right now it's 3:45 in the afternoon.",
        ],
        formulation_rules=[
            "State the time clearly",
            "Include timezone if relevant",
            "Use natural language (afternoon vs PM when appropriate)"
        ]
    )

    formulator.add_template(
        action="current_date",
        example_data={"date": "February 12, 2026", "day_of_week": "Thursday"},
        example_phrasings=[
            "Today is Thursday, February 12, 2026.",
            "It's Thursday, February 12th.",
            "Today's date is February 12, 2026 - a Thursday.",
        ],
        formulation_rules=[
            "Include day of week",
            "State month, day, and year",
            "Use 'today is' for present tense"
        ]
    )

    # Search plugin templates
    formulator.add_template(
        action="web_search",
        example_data={"query": "Python tutorials", "results_count": 10, "top_result": "Learn Python - Official Docs"},
        example_phrasings=[
            "I found 10 results for 'Python tutorials'. The top result is 'Learn Python - Official Docs'.",
            "Here are 10 results for 'Python tutorials', starting with 'Learn Python - Official Docs'.",
            "I searched for 'Python tutorials' and found 10 matches. First up: 'Learn Python - Official Docs'.",
        ],
        formulation_rules=[
            "Reference the search query",
            "State result count",
            "Mention top result if relevant",
            "Use active voice"
        ]
    )

    # File operations templates
    formulator.add_template(
        action="list_files",
        example_data={"path": "/home/user/documents", "count": 15, "file_types": ["pdf", "docx"]},
        example_phrasings=[
            "I found 15 files in your documents folder.",
            "There are 15 files in /home/user/documents.",
            "Your documents folder has 15 files, including PDFs and Word documents.",
        ],
        formulation_rules=[
            "State the count",
            "Reference the location",
            "Mention file types if relevant"
        ]
    )

    formulator.add_template(
        action="file_operation_error",
        example_data={"operation": "delete", "file": "report.pdf", "error": "File not found"},
        example_phrasings=[
            "I couldn't delete report.pdf - the file wasn't found.",
            "report.pdf doesn't exist, so I couldn't remove it.",
            "I can't delete report.pdf because it's not there.",
        ],
        formulation_rules=[
            "State what failed",
            "Explain why briefly",
            "Be clear but not overly technical"
        ]
    )

    # Generic success/failure templates
    formulator.add_template(
        action="operation_success",
        example_data={"operation": "update", "target": "settings"},
        example_phrasings=[
            "Done. I updated your settings.",
            "Settings updated successfully.",
            "I've updated your settings.",
        ],
        formulation_rules=[
            "Confirm completion",
            "Reference what was done",
            "Be concise"
        ]
    )

    formulator.add_template(
        action="operation_failure",
        example_data={"operation": "save", "target": "configuration", "reason": "Permission denied"},
        example_phrasings=[
            "I couldn't save the configuration - permission was denied.",
            "Save failed due to a permission issue.",
            "I don't have permission to save that configuration.",
        ],
        formulation_rules=[
            "State what failed",
            "Briefly explain why",
            "Don't over-apologize"
        ]
    )

    formulator.add_template(
        action="count_items",
        example_data={"item_type": "notes", "count": 7, "filter": "work"},
        example_phrasings=[
            "You have 7 work notes.",
            "I found 7 notes tagged with 'work'.",
            "There are 7 work-related notes.",
        ],
        formulation_rules=[
            "State the count clearly",
            "Mention the filter/category if any",
            "Use natural language"
        ]
    )

    print(f"\n Seeded {len(formulator.templates)} response templates")
    print("\nAlice will now learn to formulate these types of responses.")
    print("After seeing 3 examples of each type, she'll formulate independently.")
    print("\nResponse templates saved to: data/response_templates/")


if __name__ == "__main__":
    seed_templates()
