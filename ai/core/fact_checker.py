"""
Fact Checker
=============
Validates factual claims before Alice makes them.
Prevents hallucination by verifying claims against actual data.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FactChecker:
    """
    Fact checking system that validates claims before they're spoken.

    Prevents:
    - Claiming features that don't exist in code
    - Making up file contents
    - Inventing data that wasn't observed
    - Over-promising capabilities
    """

    def __init__(self):
        self.confidence_threshold = 0.7
        self.uncertain_markers = [
            "I think",
            "probably",
            "might",
            "maybe",
            "possibly",
            "likely",
            "could be"
        ]

    def check_code_claim(
        self,
        claim: str,
        actual_analysis: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a claim about code is factual.

        Args:
            claim: The claim being made
            actual_analysis: Actual AST analysis of the code

        Returns:
            (is_factual, correction_if_false)
        """
        claim_lower = claim.lower()

        # Check for claimed features that don't exist
        feature_claims = {
            'graph': ['graph', 'node', 'edge', 'vertex'],
            'database': ['database', 'sql', 'query', 'table'],
            'network': ['network', 'socket', 'connection', 'request'],
            'encryption': ['encrypt', 'decrypt', 'cipher', 'hash'],
            'ai': ['neural', 'machine learning', 'deep learning', 'model']
        }

        for feature, keywords in feature_claims.items():
            if any(kw in claim_lower for kw in keywords):
                # Check if this feature actually exists in imports or function names
                imports_str = ' '.join(actual_analysis.get('imports', [])).lower()
                functions_str = ' '.join(
                    f['name'] for f in actual_analysis.get('functions', [])
                ).lower()
                classes_str = ' '.join(
                    c['name'] for c in actual_analysis.get('classes', [])
                ).lower()

                all_code = imports_str + functions_str + classes_str

                # If claim mentions feature but code doesn't have related keywords
                if not any(kw in all_code for kw in keywords):
                    correction = f"Note: The claim mentions {feature}, but I don't see evidence of that in the actual code."
                    return False, correction

        return True, None

    def check_file_existence_claim(
        self,
        claimed_file: str,
        available_files: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a claimed file actually exists.

        Args:
            claimed_file: File path being claimed
            available_files: List of actual available files

        Returns:
            (exists, correction_if_false)
        """
        # Normalize paths
        claimed_normalized = Path(claimed_file).as_posix().lower()

        for actual_file in available_files:
            actual_normalized = Path(actual_file).as_posix().lower()
            if claimed_normalized in actual_normalized or actual_normalized in claimed_normalized:
                return True, None

        correction = f"File '{claimed_file}' not found in codebase."
        return False, correction

    def detect_uncertainty(self, response: str) -> float:
        """
        Detect uncertainty level in a response.

        Args:
            response: The response text

        Returns:
            Uncertainty score (0.0 = certain, 1.0 = very uncertain)
        """
        response_lower = response.lower()

        uncertainty_count = sum(
            1 for marker in self.uncertain_markers
            if marker in response_lower
        )

        # Normalize to 0-1
        return min(1.0, uncertainty_count * 0.2)

    def flag_unverified_claims(self, response: str) -> str:
        """
        Add disclaimers to unverified claims.

        Args:
            response: The response text

        Returns:
            Modified response with disclaimers if needed
        """
        uncertainty = self.detect_uncertainty(response)

        if uncertainty > 0.5:
            # High uncertainty already indicated
            return response

        # Check for definitive claims without sources
        has_definitive_claim = any(phrase in response.lower() for phrase in [
            'uses a',
            'employs',
            'implements',
            'contains',
            'has a',
            'includes a'
        ])

        has_source_citation = any(marker in response for marker in [
            'line',
            'function',
            'class',
            'import',
            '```'
        ])

        if has_definitive_claim and not has_source_citation:
            # Add disclaimer
            return response + "\n\n(Note: This is based on general patterns, not verified against the actual code)"

        return response

    def require_evidence(
        self,
        claim_type: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if sufficient evidence exists for a claim type.

        Args:
            claim_type: Type of claim ('code_feature', 'file_content', 'data_value')
            evidence: Evidence dict with relevant data

        Returns:
            True if evidence is sufficient
        """
        if not evidence:
            return False

        if claim_type == 'code_feature':
            # Need actual AST analysis
            return 'functions' in evidence or 'classes' in evidence

        elif claim_type == 'file_content':
            # Need actual file content
            return 'content' in evidence and len(evidence['content']) > 0

        elif claim_type == 'data_value':
            # Need actual data
            return 'value' in evidence

        return False


# Global singleton
_fact_checker = None


def get_fact_checker() -> FactChecker:
    """Get global fact checker instance"""
    global _fact_checker
    if _fact_checker is None:
        _fact_checker = FactChecker()
    return _fact_checker
