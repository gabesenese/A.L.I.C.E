"""
Synthetic Corpus Generator: Auto-generates training pairs from LLM.
Creates variants with typos, slang, partial commands for robustness.
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
from datetime import datetime


class SyntheticCorpusGenerator:
    def __init__(self, output_path: str = "data/training/synthetic_corpus.jsonl",
                 llm_engine=None):
        """
        Initialize synthetic corpus generator.
        
        Args:
            output_path: Path to save generated corpus
            llm_engine: LLM engine for generating responses
        """
        self.output_path = output_path
        self.llm_engine = llm_engine

        # Base templates for common intents
        self.base_templates = {
            "greeting": [
                "Hi Alice",
                "Hello",
                "Hey there",
                "Good morning",
                "How's it going",
            ],
            "weather": [
                "What's the weather",
                "Tell me the weather",
                "Weather forecast",
                "Is it going to rain",
                "How's the weather today",
            ],
            "email": [
                "Show my emails",
                "Any new messages",
                "Send an email",
                "Check my inbox",
                "List emails",
            ],
            "notes": [
                "Take a note",
                "Create a note",
                "Save this",
                "Remember this",
                "Add a reminder",
            ],
            "calendar": [
                "Show my calendar",
                "What meetings do I have",
                "My schedule today",
                "Am I free tomorrow",
                "List my events",
            ],
            "help": [
                "What can you do",
                "Show commands",
                "Help",
                "What's available",
                "Tell me your capabilities",
            ],
            "farewell": [
                "Bye",
                "Goodbye",
                "See you later",
                "Exit",
                "Talk to you later",
            ]
        }

        # Common typos and slang variations
        self.typo_map = {
            "the": ["teh", "th"],
            "weather": ["weahter", "weathr"],
            "message": ["msg", "messge"],
            "email": ["emai", "emial"],
            "calendar": ["cal", "calender"],
            "reminder": ["rmnd", "remind"],
            "today": ["2day", "tdy"],
            "tomorrow": ["tmrw", "tom"],
        }

        self.slang_variations = {
            "hello": ["hey", "yo", "sup"],
            "thanks": ["thx", "thnx", "ty"],
            "please": ["pls", "plz"],
            "can you": ["can u", "u can"],
            "what is": ["whats", "wuts"],
            "how is": ["hows", "how's"],
        }

    def add_typo(self, text: str, typo_probability: float = 0.3) -> str:
        """Add random typos to text."""
        if random.random() > typo_probability:
            return text

        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in self.typo_map and random.random() < 0.5:
                words[i] = random.choice(self.typo_map[word.lower()])

        return " ".join(words)

    def add_slang(self, text: str, slang_probability: float = 0.4) -> str:
        """Add slang variations to text."""
        if random.random() > slang_probability:
            return text

        for standard, slang_options in self.slang_variations.items():
            if standard in text.lower():
                variant = random.choice(slang_options)
                text = text.replace(standard, variant, 1)
                break

        return text

    def truncate_command(self, text: str, truncate_probability: float = 0.2) -> str:
        """Create partial/incomplete command variants."""
        if random.random() > truncate_probability:
            return text

        words = text.split()
        if len(words) > 2:
            # Remove 1-2 words randomly
            keep_count = random.randint(len(words) - 2, len(words) - 1)
            return " ".join(words[:keep_count])

        return text

    def capitalize_variation(self, text: str) -> str:
        """Create capitalization variations."""
        variations = [
            text.lower(),
            text.upper(),
            text.capitalize(),
            " ".join(word.capitalize() for word in text.split())
        ]
        return random.choice(variations)

    def generate_variants(self, base_input: str, count: int = 5) -> List[str]:
        """Generate N variants of a base input."""
        variants = {base_input}  # Include original

        while len(variants) < count:
            variant = base_input
            variant = self.add_typo(variant)
            variant = self.add_slang(variant)
            variant = self.truncate_command(variant)
            variant = self.capitalize_variation(variant)
            variants.add(variant)

        return list(variants)[:count]

    def generate_training_pairs(self, intent: str, base_input: str,
                               response: str, num_variants: int = 3) -> List[Dict[str, Any]]:
        """
        Generate training pairs from a base input-response.
        
        Args:
            intent: The intent category
            base_input: The original user input
            response: The ideal response
            num_variants: Number of variants to generate
            
        Returns:
            List of training pairs
        """
        pairs = []
        variants = self.generate_variants(base_input, num_variants)

        for variant_input in variants:
            pair = {
                "user_input": variant_input,
                "intent": intent,
                "entities": [],
                "llm_response": response,
                "category": "synthetic",
                "quality_score": 0.85 + random.uniform(-0.05, 0.1),  # 0.8-0.95
                "generated": True,
                "timestamp": datetime.now().isoformat()
            }
            pairs.append(pair)

        return pairs

    def generate_corpus(self, templates: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        """
        Generate synthetic corpus from templates.
        
        Args:
            templates: Optional custom templates. Uses self.base_templates if None.
            
        Returns:
            List of training pairs
        """
        if templates is None:
            templates = self.base_templates

        corpus = []
        response_templates = {
            "greeting": "Hello! How can I help you today?",
            "weather": "Let me check the weather for you.",
            "email": "I'll check your email.",
            "notes": "I'll save that for you.",
            "calendar": "Let me check your calendar.",
            "help": "I can help with weather, emails, notes, calendar, and more.",
            "farewell": "Goodbye! See you later.",
        }

        for intent, inputs in templates.items():
            response = response_templates.get(intent, "How can I help?")
            for base_input in inputs:
                variants = self.generate_variants(base_input, num_variants=3)
                for variant in variants:
                    pair = {
                        "user_input": variant,
                        "intent": intent,
                        "entities": self._extract_entities(variant),
                        "llm_response": response,
                        "category": "synthetic",
                        "quality_score": 0.82 + random.uniform(-0.05, 0.1),
                        "generated": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    corpus.append(pair)

        return corpus

    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction (can be enhanced)."""
        entities = []
        entity_keywords = {
            "person": ["john", "jane", "alice", "bob", "mom", "dad", "friend"],
            "place": ["office", "home", "meeting", "room"],
            "time": ["today", "tomorrow", "monday", "tuesday"],
        }

        text_lower = text.lower()
        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities.append(keyword)

        return entities

    def save_corpus(self, corpus: List[Dict[str, Any]]):
        """Save corpus to JSONL file."""
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        with open(self.output_path, 'a') as f:
            for item in corpus:
                f.write(json.dumps(item) + '\n')

    def load_corpus(self) -> List[Dict[str, Any]]:
        """Load corpus from JSONL file."""
        corpus = []
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                for line in f:
                    if line.strip():
                        corpus.append(json.loads(line))
        return corpus

    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about generated corpus."""
        corpus = self.load_corpus()

        intents = {}
        for item in corpus:
            intent = item.get("intent", "unknown")
            intents[intent] = intents.get(intent, 0) + 1

        return {
            "total_pairs": len(corpus),
            "intents": intents,
            "synthetic_count": sum(1 for item in corpus if item.get("generated")),
            "avg_quality": sum(item.get("quality_score", 0.8) for item in corpus) / max(len(corpus), 1)
        }
