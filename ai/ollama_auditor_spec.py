"""
Ollama Auditor Specification
Multi-dimensional grading rubric for Alice responses
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ScoringDimension(Enum):
    """Audit scoring dimensions"""
    ACCURACY = "accuracy"  # Factual correctness
    RELEVANCE = "relevance"  # Answer matches question
    CLARITY = "clarity"  # Easy to understand
    COMPLETENESS = "completeness"  # Covers all aspects
    TONE = "tone"  # Appropriate voice/formality
    TIMELINESS = "timeliness"  # Fast enough
    REASONING = "reasoning"  # Logic is sound
    ACTIONABILITY = "actionability"  # User can act on it


@dataclass
class DimensionRubric:
    """Scoring rubric for one dimension"""
    dimension: ScoringDimension
    description: str
    scale: int  # Usually 1-5
    indicators: Dict[int, str]  # Score -> what this looks like


# Domain-specific audit rubrics
AUDIT_DIMENSIONS = {
    "weather": {
        ScoringDimension.ACCURACY: DimensionRubric(
            dimension=ScoringDimension.ACCURACY,
            description="Weather data matches actual forecasts",
            scale=5,
            indicators={
                1: "Wrong temperature/condition entirely",
                2: "Off by 5+ degrees or wrong condition",
                3: "Minor temp variance (±2°), condition correct",
                4: "Exact match with forecast",
                5: "Perfect match + mentions confidence level"
            }
        ),
        ScoringDimension.CLARITY: DimensionRubric(
            dimension=ScoringDimension.CLARITY,
            description="Weather info presented clearly",
            scale=5,
            indicators={
                1: "Confusing, hard to parse",
                2: "Clear but missing context",
                3: "Clear main info, some ambiguity",
                4: "Very clear, well-organized",
                5: "Crystal clear + proactive guidance"
            }
        ),
        ScoringDimension.ACTIONABILITY: DimensionRubric(
            dimension=ScoringDimension.ACTIONABILITY,
            description="User can make decisions from response",
            scale=5,
            indicators={
                1: "No actionable info",
                2: "Basic info, user has to infer",
                3: "Implied actions, somewhat clear",
                4: "Clear recommendation (bring umbrella)",
                5: "Clear rec + why + alternatives"
            }
        )
    },
    "email": {
        ScoringDimension.ACCURACY: DimensionRubric(
            dimension=ScoringDimension.ACCURACY,
            description="Email content summarized correctly",
            scale=5,
            indicators={
                1: "Fabricated details or missed key points",
                2: "Mixed accuracy, some invented parts",
                3: "Mostly accurate, minor omissions",
                4: "Accurate with all key points",
                5: "Perfect accuracy + tone/urgency preserved"
            }
        ),
        ScoringDimension.COMPLETENESS: DimensionRubric(
            dimension=ScoringDimension.COMPLETENESS,
            description="Summary covers email completely",
            scale=5,
            indicators={
                1: "Only 1-2 sentences, bare minimum",
                2: "Hits main point but misses context",
                3: "Main points + some details",
                4: "Comprehensive coverage",
                5: "Complete + metadata (sender, date, importance)"
            }
        ),
        ScoringDimension.TONE: DimensionRubric(
            dimension=ScoringDimension.TONE,
            description="Response tone matches email urgency",
            scale=5,
            indicators={
                1: "Completely wrong tone",
                2: "Off-tone, misses urgency",
                3: "Generally appropriate tone",
                4: "Tone matches email urgency well",
                5: "Perfect tone + conveys importance"
            }
        )
    },
    "code": {
        ScoringDimension.ACCURACY: DimensionRubric(
            dimension=ScoringDimension.ACCURACY,
            description="Code analysis is technically correct",
            scale=5,
            indicators={
                1: "Incorrect analysis or false claims",
                2: "Some correct points but major errors",
                3: "Mostly correct, minor inaccuracies",
                4: "Correct analysis with valid reasoning",
                5: "Perfect accuracy + edge cases covered"
            }
        ),
        ScoringDimension.DEPTH: DimensionRubric(
            dimension=ScoringDimension.REASONING,
            description="Analysis goes deep enough",
            scale=5,
            indicators={
                1: "Superficial, surface-level only",
                2: "Basic analysis, missing depth",
                3: "Adequate depth, some missed nuances",
                4: "Good depth, covers main issues",
                5: "Deep analysis + performance + style notes"
            }
        ),
        ScoringDimension.CLARITY: DimensionRubric(
            dimension=ScoringDimension.CLARITY,
            description="Explanation is understandable",
            scale=5,
            indicators={
                1: "Jargon-heavy, hard to follow",
                2: "Unclear with some jargon",
                3: "Mostly clear, some jargon",
                4: "Clear explanation for most developers",
                5: "Crystal clear + beginner-friendly"
            }
        )
    },
    "conversation": {
        ScoringDimension.ACCURACY: DimensionRubric(
            dimension=ScoringDimension.ACCURACY,
            description="Response is factually correct",
            scale=5,
            indicators={
                1: "Factually incorrect",
                2: "Mixed accuracy",
                3: "Mostly correct",
                4: "Accurate with minimal errors",
                5: "Perfect accuracy"
            }
        ),
        ScoringDimension.RELEVANCE: DimensionRubric(
            dimension=ScoringDimension.RELEVANCE,
            description="Response answers the actual question",
            scale=5,
            indicators={
                1: "Completely off-topic",
                2: "Tangentially related",
                3: "Somewhat relevant, misses nuance",
                4: "Directly answers question",
                5: "Perfect match + anticipates follow-ups"
            }
        ),
        ScoringDimension.REASONING: DimensionRubric(
            dimension=ScoringDimension.REASONING,
            description="Logic chain is sound",
            scale=5,
            indicators={
                1: "Logical fallacies present",
                2: "Some reasoning gaps",
                3: "Generally sound, some assumptions unclear",
                4: "Sound logic, well explained",
                5: "Impeccable logic + acknowledges assumptions"
            }
        )
    }
}


def get_domain_dimensions(domain: str) -> Dict[ScoringDimension, DimensionRubric]:
    """Get all dimensions for a domain"""
    return AUDIT_DIMENSIONS.get(domain, {})


def score_dimension(
    dimension: ScoringDimension,
    rubric: DimensionRubric,
    score: int
) -> Tuple[int, str]:
    """
    Get indicator text for a given score
    Returns: (score, indicator_text)
    """
    if score < 1 or score > rubric.scale:
        score = max(1, min(rubric.scale, score))
    return (score, rubric.indicators.get(score, "Unknown"))


def get_all_dimensions() -> Dict[str, Dict[ScoringDimension, DimensionRubric]]:
    """Get all audit dimensions"""
    return AUDIT_DIMENSIONS
