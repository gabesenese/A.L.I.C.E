"""
Ollama Teaching Specification
Defines what Alice must master per domain
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TeachingVector:
    """What Alice must learn in a domain"""
    domain: str
    skill: str
    description: str
    test_template: str  # Template for generating test queries
    success_criteria: List[str]


TEACHING_VECTORS = {
    "weather": [
        TeachingVector(
            domain="weather",
            skill="forecast_interpretation",
            description="Understand and communicate multi-day forecasts",
            test_template="What's the weather forecast for {day}? Will it be {condition}?",
            success_criteria=[
                "Correctly identifies high/low temps",
                "Mentions precipitation chance",
                "Recommends appropriate clothing",
                "Notes any warnings (frost, heat, etc.)"
            ]
        ),
        TeachingVector(
            domain="weather",
            skill="contextual_response",
            description="Provide context-aware weather responses",
            test_template="Should I bring an umbrella {timeframe}?",
            success_criteria=[
                "Checks actual weather data",
                "Answers yes/no decisively",
                "Explains reasoning",
                "Suggests alternatives if needed"
            ]
        )
    ],
    "email": [
        TeachingVector(
            domain="email",
            skill="summary_accuracy",
            description="Accurately summarize email threads",
            test_template="Summarize my latest emails about {topic}",
            success_criteria=[
                "Includes sender names",
                "Captures main points",
                "No fabricated details",
                "Preserves tone/urgency"
            ]
        ),
        TeachingVector(
            domain="email",
            skill="composition_clarity",
            description="Draft clear, professional emails",
            test_template="Draft an email to {recipient} about {subject}",
            success_criteria=[
                "Professional tone",
                "Clear subject line",
                "Logical flow",
                "Appropriate length",
                "No grammar errors"
            ]
        )
    ],
    "code": [
        TeachingVector(
            domain="code",
            skill="analysis_depth",
            description="Analyze code for issues and improvements",
            test_template="Analyze this code for {aspect}: {code_snippet}",
            success_criteria=[
                "Identifies logic errors",
                "Notes performance issues",
                "Suggests improvements",
                "Explains reasoning",
                "Code example syntax correct"
            ]
        ),
        TeachingVector(
            domain="code",
            skill="explanation_clarity",
            description="Explain code in plain English",
            test_template="Explain what this does: {code_snippet}",
            success_criteria=[
                "Clear English explanation",
                "Correct terminology",
                "Step-by-step logic",
                "Mentions edge cases",
                "Beginner-understandable"
            ]
        )
    ],
    "calendar": [
        TeachingVector(
            domain="calendar",
            skill="scheduling_logic",
            description="Intelligently handle scheduling requests",
            test_template="Schedule {event} on {date_spec} with {duration}",
            success_criteria=[
                "Finds available time slots",
                "Avoids conflicts",
                "Respects constraints",
                "Confirms before scheduling",
                "Suggests alternatives if blocked"
            ]
        )
    ],
    "notes": [
        TeachingVector(
            domain="notes",
            skill="organization",
            description="Organize and categorize notes logically",
            test_template="Create a note about {topic} with tags {tags}",
            success_criteria=[
                "Clear title",
                "Logical structure",
                "Appropriate tags",
                "Easy to retrieve",
                "Metadata complete"
            ]
        )
    ],
    "reasoning": [
        TeachingVector(
            domain="reasoning",
            skill="multi_step_logic",
            description="Execute multi-step reasoning",
            test_template="If {condition1} and {condition2}, what should I do about {goal}?",
            success_criteria=[
                "Identifies all constraints",
                "Chains logic correctly",
                "Explains assumptions",
                "Reaches justified conclusion",
                "Considers alternatives"
            ]
        )
    ],
    "conversation": [
        TeachingVector(
            domain="conversation",
            skill="context_awareness",
            description="Maintain conversation context across turns",
            test_template="Previous: {prior_context}. Now user says: {new_input}",
            success_criteria=[
                "References prior messages",
                "Maintains topic",
                "Answers new questions",
                "Connects ideas logically",
                "No contradictions"
            ]
        )
    ]
}


def get_domain_vectors(domain: str) -> List[TeachingVector]:
    """Get all teaching vectors for a domain"""
    return TEACHING_VECTORS.get(domain, [])


def get_all_vectors() -> Dict[str, List[TeachingVector]]:
    """Get all teaching vectors"""
    return TEACHING_VECTORS


def get_skill_template(domain: str, skill: str) -> str:
    """Get test template for specific skill"""
    vectors = get_domain_vectors(domain)
    for v in vectors:
        if v.skill == skill:
            return v.test_template
    return None


def get_success_criteria(domain: str, skill: str) -> List[str]:
    """Get success criteria for specific skill"""
    vectors = get_domain_vectors(domain)
    for v in vectors:
        if v.skill == skill:
            return v.success_criteria
    return []
