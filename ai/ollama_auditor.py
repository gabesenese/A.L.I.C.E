"""
Ollama Auditor - Multi-Dimensional Response Grader
Grades Alice responses across defined quality dimensions
"""

import json
import logging
from typing import Dict, Any, Tuple
from ai.ollama_auditor_spec import AUDIT_DIMENSIONS, ScoringDimension, DimensionRubric
from ai.llm_engine import LocalLLMEngine

logger = logging.getLogger(__name__)


class AuditScore:
    """Score for a single response across all dimensions"""
    
    def __init__(self, domain: str, query: str, response: str):
        self.domain = domain
        self.query = query
        self.response = response
        self.scores: Dict[ScoringDimension, int] = {}
        self.reasoning: Dict[ScoringDimension, str] = {}
        self.timestamp = None
        self.overall_score = 0.0
    
    def add_dimension_score(
        self,
        dimension: ScoringDimension,
        score: int,
        reasoning: str
    ):
        """Add a dimension score with reasoning"""
        self.scores[dimension] = score
        self.reasoning[dimension] = reasoning
    
    def calculate_overall(self) -> float:
        """Calculate overall score (average)"""
        if not self.scores:
            return 0.0
        total = sum(self.scores.values())
        self.overall_score = total / len(self.scores)
        return self.overall_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage"""
        return {
            'domain': self.domain,
            'query': self.query,
            'response': self.response,
            'scores': {d.value: s for d, s in self.scores.items()},
            'reasoning': {d.value: r for d, r in self.reasoning.items()},
            'overall_score': self.overall_score,
            'timestamp': self.timestamp
        }


class OllamaAuditor:
    """Grades Alice responses across defined dimensions"""
    
    def __init__(self, llm: LocalLLMEngine):
        self.llm = llm
        self.audit_history = []
    
    def audit_response(
        self,
        domain: str,
        query: str,
        response: str
    ) -> AuditScore:
        """
        Audit a single response from Alice
        
        Args:
            domain: Domain of the query (weather, email, code, etc.)
            query: The user's question
            response: Alice's response to grade
        
        Returns:
            AuditScore with ratings across all dimensions
        """
        score = AuditScore(domain, query, response)
        
        # Get rubric for domain
        rubrics = AUDIT_DIMENSIONS.get(domain, {})
        if not rubrics:
            logger.warning(f"No audit rubrics for domain: {domain}")
            return score
        
        # Score each dimension
        for dimension, rubric in rubrics.items():
            dim_score, reasoning = self._score_dimension(
                dimension, rubric, query, response
            )
            score.add_dimension_score(dimension, dim_score, reasoning)
        
        # Calculate overall
        score.calculate_overall()
        
        # Store in history
        self.audit_history.append(score)
        
        logger.info(
            f"Audited {domain}: {score.overall_score:.1f}/5.0 "
            f"({len(score.scores)} dimensions)"
        )
        
        return score
    
    def _score_dimension(
        self,
        dimension: ScoringDimension,
        rubric: DimensionRubric,
        query: str,
        response: str
    ) -> Tuple[int, str]:
        """
        Use Ollama to score a single dimension
        
        Returns:
            (score, reasoning)
        """
        prompt = f"""You are grading an AI assistant's response.

DIMENSION: {dimension.value}
{rubric.description}

SCORING SCALE (1-5):
"""
        
        # Add indicator text
        for score_level in range(1, rubric.scale + 1):
            indicator = rubric.indicators.get(score_level, "")
            prompt += f"{score_level}: {indicator}\n"
        
        prompt += f"""
QUERY: {query}
RESPONSE: {response}

Score this response on the {dimension.value} dimension.
Output JSON: {{"score": <1-5>, "reasoning": "<explanation>"}}
"""
        
        try:
            response_text = self.llm.query(prompt, max_tokens=200)
            
            # Parse JSON
            import re
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                score = int(result.get('score', 3))
                reasoning = result.get('reasoning', 'Unknown')
                
                # Clamp to valid range
                score = max(1, min(rubric.scale, score))
                
                return score, reasoning
        
        except Exception as e:
            logger.error(f"Error scoring {dimension.value}: {e}")
        
        # Default fallback
        return 3, "Unable to score"
    
    def audit_batch(
        self,
        domain: str,
        test_cases: list
    ) -> list:
        """
        Audit multiple responses
        
        Args:
            domain: Domain for all test cases
            test_cases: List of (query, response) tuples
        
        Returns:
            List of AuditScore objects
        """
        scores = []
        for query, response in test_cases:
            score = self.audit_response(domain, query, response)
            scores.append(score)
        
        return scores
    
    def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """Get summary stats for domain audits"""
        domain_audits = [a for a in self.audit_history if a.domain == domain]
        
        if not domain_audits:
            return {'domain': domain, 'count': 0}
        
        scores = [a.overall_score for a in domain_audits]
        avg_score = sum(scores) / len(scores)
        
        # Per-dimension stats
        dimension_stats = {}
        for dimension in ScoringDimension:
            dim_scores = [
                a.scores.get(dimension, 0)
                for a in domain_audits
                if dimension in a.scores
            ]
            if dim_scores:
                dimension_stats[dimension.value] = {
                    'avg': sum(dim_scores) / len(dim_scores),
                    'min': min(dim_scores),
                    'max': max(dim_scores)
                }
        
        return {
            'domain': domain,
            'count': len(domain_audits),
            'avg_score': avg_score,
            'min_score': min(scores),
            'max_score': max(scores),
            'dimension_stats': dimension_stats
        }
    
    def clear_history(self):
        """Clear audit history"""
        self.audit_history.clear()


def create_auditor(llm: LocalLLMEngine) -> OllamaAuditor:
    """Factory to create auditor"""
    return OllamaAuditor(llm)
