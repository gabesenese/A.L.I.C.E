"""
Ollama Teacher - Synthetic Query Generator
Generates diverse test queries to teach Alice per domain
"""

import json
import logging
from typing import List, Dict, Any
from ai.ollama_teaching_spec import TEACHING_VECTORS, TeachingVector
from ai.llm_engine import LocalLLMEngine

logger = logging.getLogger(__name__)


class OllamaTeacher:
    """Generates synthetic test queries for each teaching vector"""
    
    def __init__(self, llm: LocalLLMEngine):
        self.llm = llm
        self.queries_cache = {}
    
    def generate_test_queries(
        self,
        domain: str,
        skill: str,
        count: int = 5,
        use_cache: bool = True
    ) -> List[str]:
        """
        Generate diverse test queries for a domain/skill
        
        Args:
            domain: Domain name (weather, email, code, etc.)
            skill: Skill within domain
            count: How many queries to generate
            use_cache: Use cached queries if available
        
        Returns:
            List of test query strings
        """
        cache_key = f"{domain}:{skill}"
        if use_cache and cache_key in self.queries_cache:
            cached = self.queries_cache[cache_key]
            return cached[:count]
        
        # Find teaching vector
        vector = self._get_teaching_vector(domain, skill)
        if not vector:
            logger.warning(f"No teaching vector for {domain}:{skill}")
            return []
        
        # Generate queries using Ollama
        queries = self._generate_from_template(vector, count)
        
        # Cache them
        self.queries_cache[cache_key] = queries
        
        return queries
    
    def _get_teaching_vector(self, domain: str, skill: str) -> TeachingVector:
        """Get teaching vector for domain/skill"""
        for vector in TEACHING_VECTORS.get(domain, []):
            if vector.skill == skill:
                return vector
        return None
    
    def _generate_from_template(self, vector: TeachingVector, count: int) -> List[str]:
        """Generate queries by filling in the template via Ollama"""
        prompt = f"""You are generating {count} diverse test queries to teach an AI assistant about: {vector.description}

Template: {vector.test_template}

Generate {count} realistic, varied test queries. Make them:
- Different from each other
- Cover the skill well
- Be realistic for a user to ask
- Vary in complexity

Output as JSON array of strings, one query per line.
Example: ["query1", "query2", ...]
"""
        
        try:
            response = self.llm.query(prompt, max_tokens=500)
            
            # Parse JSON response
            # Try to extract JSON array from response
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                queries = json.loads(match.group())
                return queries[:count]
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
        
        # Fallback: generate from template manually
        return self._fallback_generation(vector, count)
    
    def _fallback_generation(self, vector: TeachingVector, count: int) -> List[str]:
        """Fallback simple query generation"""
        queries = []
        template = vector.test_template
        
        # For weather
        if vector.domain == "weather":
            days = ["today", "tomorrow", "next monday", "this weekend", "next week"]
            conditions = ["rain", "snow", "clear", "cloudy", "warm"]
            for i in range(count):
                day = days[i % len(days)]
                condition = conditions[i % len(conditions)]
                q = template.format(day=day, condition=condition)
                queries.append(q)
        
        # For email
        elif vector.domain == "email":
            recipients = ["John", "Sarah", "the team", "my manager"]
            subjects = ["project update", "meeting notes", "question about timeline", "status report"]
            for i in range(count):
                recipient = recipients[i % len(recipients)]
                subject = subjects[i % len(subjects)]
                q = template.format(recipient=recipient, subject=subject)
                queries.append(q)
        
        # For code
        elif vector.domain == "code":
            aspects = ["performance", "bugs", "readability", "security"]
            snippets = ["def foo(x): return x*2", "if a and b:", "for i in range(10)"]
            for i in range(count):
                aspect = aspects[i % len(aspects)]
                snippet = snippets[i % len(snippets)]
                q = template.format(aspect=aspect, code_snippet=snippet)
                queries.append(q)
        
        else:
            # Generic fallback
            for i in range(count):
                queries.append(f"{template} (variant {i+1})")
        
        return queries[:count]
    
    def generate_all_domains(self, count_per_skill: int = 3) -> Dict[str, Dict[str, List[str]]]:
        """
        Generate test queries for all domains and skills
        
        Returns:
            {domain: {skill: [queries]}}
        """
        all_queries = {}
        
        for domain, vectors in TEACHING_VECTORS.items():
            all_queries[domain] = {}
            for vector in vectors:
                queries = self.generate_test_queries(domain, vector.skill, count_per_skill)
                all_queries[domain][vector.skill] = queries
                logger.info(f"Generated {len(queries)} queries for {domain}:{vector.skill}")
        
        return all_queries
    
    def clear_cache(self):
        """Clear query cache"""
        self.queries_cache.clear()


def create_teacher(llm: LocalLLMEngine) -> OllamaTeacher:
    """Factory to create teacher"""
    return OllamaTeacher(llm)
