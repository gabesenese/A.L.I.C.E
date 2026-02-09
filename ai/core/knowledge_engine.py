"""
Alice's Knowledge Engine
========================
Alice's own intelligence - learns from every interaction.
NOT an Ollama wrapper - Alice IS the AI.

Components:
- Knowledge Graph: Stores facts, entities, relationships
- Semantic Learning: Learns patterns, concepts, meanings
- Confidence System: Knows when she knows, asks when she doesn't
- Progressive Independence: Learns from Ollama, becomes independent
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class Entity:
    """Represents a learned entity (person, place, concept, etc.)"""

    def __init__(self, name: str, entity_type: str):
        self.name = name
        self.type = entity_type
        self.attributes = {}
        self.aliases = set()
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.confidence = 0.5
        self.mention_count = 0

    def to_dict(self):
        return {
            'name': self.name,
            'type': self.type,
            'attributes': self.attributes,
            'aliases': list(self.aliases),
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'confidence': self.confidence,
            'mention_count': self.mention_count
        }

    @staticmethod
    def from_dict(data):
        entity = Entity(data['name'], data['type'])
        entity.attributes = data['attributes']
        entity.aliases = set(data['aliases'])
        entity.first_seen = datetime.fromisoformat(data['first_seen'])
        entity.last_seen = datetime.fromisoformat(data['last_seen'])
        entity.confidence = data['confidence']
        entity.mention_count = data['mention_count']
        return entity


class Relationship:
    """Represents a learned relationship between entities"""

    def __init__(self, subject: str, predicate: str, object: str):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.confidence = 0.5
        self.evidence_count = 1
        self.first_seen = datetime.now()
        self.last_confirmed = datetime.now()

    def to_dict(self):
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence,
            'evidence_count': self.evidence_count,
            'first_seen': self.first_seen.isoformat(),
            'last_confirmed': self.last_confirmed.isoformat()
        }

    @staticmethod
    def from_dict(data):
        rel = Relationship(data['subject'], data['predicate'], data['object'])
        rel.confidence = data['confidence']
        rel.evidence_count = data['evidence_count']
        rel.first_seen = datetime.fromisoformat(data['first_seen'])
        rel.last_confirmed = datetime.fromisoformat(data['last_confirmed'])
        return rel


class ConceptLearning:
    """Learns semantic concepts and patterns from interactions"""

    def __init__(self):
        self.concepts = {}  # concept_name -> {examples, patterns, confidence}
        self.word_associations = defaultdict(lambda: defaultdict(int))
        self.question_patterns = []  # Learned question-answer patterns

    def learn_concept(self, concept: str, example: str, context: Dict[str, Any]):
        """Learn a new concept or strengthen existing one"""
        if concept not in self.concepts:
            self.concepts[concept] = {
                'examples': [],
                'patterns': [],
                'confidence': 0.3,
                'first_learned': datetime.now().isoformat()
            }

        self.concepts[concept]['examples'].append({
            'text': example,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })

        # Increase confidence with more examples
        self.concepts[concept]['confidence'] = min(
            0.95,
            0.3 + (len(self.concepts[concept]['examples']) * 0.1)
        )

    def learn_word_association(self, word1: str, word2: str):
        """Learn that two words are related (co-occur frequently)"""
        self.word_associations[word1.lower()][word2.lower()] += 1
        self.word_associations[word2.lower()][word1.lower()] += 1

    def get_related_words(self, word: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """Get words most associated with this word"""
        word = word.lower()
        if word not in self.word_associations:
            return []

        associations = self.word_associations[word]
        return sorted(associations.items(), key=lambda x: x[1], reverse=True)[:top_n]


class KnowledgeEngine:
    """
    Alice's Knowledge Engine - Her Own Intelligence

    This is NOT a wrapper around Ollama.
    Alice learns, remembers, and reasons on her own.
    """

    def __init__(self, storage_path: str = "data/knowledge"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Core knowledge structures
        self.entities = {}  # name -> Entity
        self.relationships = []  # List of Relationship objects
        self.concepts = ConceptLearning()

        # Response patterns Alice has learned
        self.learned_responses = defaultdict(list)  # intent -> [responses]

        # Confidence tracking
        self.topic_confidence = defaultdict(float)  # topic -> confidence (0-1)

        # Load existing knowledge
        self._load_knowledge()

        logger.info("[KnowledgeEngine] Alice's knowledge engine initialized")
        logger.info(f"   Entities: {len(self.entities)}")
        logger.info(f"   Relationships: {len(self.relationships)}")
        logger.info(f"   Concepts: {len(self.concepts.concepts)}")

    def learn_from_interaction(
        self,
        user_input: str,
        alice_response: str,
        intent: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """
        Alice learns from EVERY interaction.
        This is where Alice builds her intelligence.
        """
        # Extract and learn entities
        self._extract_and_learn_entities(user_input, alice_response, entities)

        # Learn relationships between entities
        self._learn_relationships(user_input, alice_response, entities)

        # Learn semantic patterns
        self._learn_semantic_patterns(user_input, alice_response, intent)

        # Learn response strategies
        self._learn_response_strategy(user_input, alice_response, intent, context)

        # Update confidence in topics
        self._update_topic_confidence(intent, context)

        # Save learned knowledge
        self._save_knowledge()

    def _extract_and_learn_entities(self, user_input: str, response: str, entities: Dict[str, Any]):
        """Extract entities from conversation and learn about them"""
        combined_text = f"{user_input} {response}"

        # Learn from provided entities
        for entity_type, entity_value in entities.items():
            if isinstance(entity_value, str) and entity_value:
                self._add_or_update_entity(entity_value, entity_type)

        # Simple pattern matching for common entity types
        # Person names (capitalized words)
        names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', combined_text)
        for name in names:
            if name not in ['Alice', 'A.L.I.C.E', 'I', 'You']:
                self._add_or_update_entity(name, 'person')

        # Locations (common patterns)
        loc_patterns = [
            r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]
        for pattern in loc_patterns:
            locations = re.findall(pattern, combined_text)
            for loc in locations:
                self._add_or_update_entity(loc, 'location')

    def _add_or_update_entity(self, name: str, entity_type: str):
        """Add new entity or update existing one"""
        if name in self.entities:
            entity = self.entities[name]
            entity.last_seen = datetime.now()
            entity.mention_count += 1
            entity.confidence = min(0.99, entity.confidence + 0.05)
        else:
            entity = Entity(name, entity_type)
            self.entities[name] = entity
            logger.debug(f"[Learning] New entity: {name} ({entity_type})")

    def _learn_relationships(self, user_input: str, response: str, entities: Dict[str, Any]):
        """Learn relationships between entities"""
        # Simple relationship extraction
        # Pattern: X is Y, X created Y, X lives in Y, etc.

        relationship_patterns = [
            (r'(.+?)\s+is\s+(.+)', 'is'),
            (r'(.+?)\s+created\s+(.+)', 'created'),
            (r'(.+?)\s+lives\s+in\s+(.+)', 'lives_in'),
            (r'(.+?)\s+works\s+at\s+(.+)', 'works_at'),
            (r'(.+?)\'s\s+(.+)', 'has'),
        ]

        combined = f"{user_input} {response}"
        for pattern, predicate in relationship_patterns:
            matches = re.findall(pattern, combined, re.IGNORECASE)
            for subject, obj in matches:
                subject = subject.strip()
                obj = obj.strip()

                # Add relationship
                self._add_relationship(subject, predicate, obj)

    def _add_relationship(self, subject: str, predicate: str, obj: str):
        """Add or strengthen a relationship"""
        # Check if relationship exists
        for rel in self.relationships:
            if (rel.subject.lower() == subject.lower() and
                rel.predicate == predicate and
                rel.object.lower() == obj.lower()):
                # Strengthen existing relationship
                rel.evidence_count += 1
                rel.confidence = min(0.99, rel.confidence + 0.1)
                rel.last_confirmed = datetime.now()
                logger.debug(f"[Learning] Strengthened: {subject} {predicate} {obj} (confidence: {rel.confidence:.2f})")
                return

        # New relationship
        rel = Relationship(subject, predicate, obj)
        self.relationships.append(rel)
        logger.debug(f"[Learning] New relationship: {subject} {predicate} {obj}")

    def _learn_semantic_patterns(self, user_input: str, response: str, intent: str):
        """Learn semantic patterns from the conversation"""
        # Learn word associations
        words = re.findall(r'\b\w+\b', f"{user_input} {response}".lower())
        for i in range(len(words) - 1):
            self.concepts.learn_word_association(words[i], words[i + 1])

        # Learn intent patterns
        if intent and response:
            self.concepts.learn_concept(
                concept=intent,
                example=user_input,
                context={'response': response[:100]}
            )

    def _learn_response_strategy(self, user_input: str, response: str, intent: str, context: Dict[str, Any]):
        """Learn what kinds of responses work for different intents"""
        if intent and response:
            # Store successful response patterns
            self.learned_responses[intent].append({
                'user_input': user_input,
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'context_keys': list(context.keys())
            })

            # Keep only recent examples (last 50 per intent)
            if len(self.learned_responses[intent]) > 50:
                self.learned_responses[intent] = self.learned_responses[intent][-50:]

    def _update_topic_confidence(self, intent: str, context: Dict[str, Any]):
        """Update Alice's confidence in different topics"""
        if intent:
            topic = intent.split(':')[0] if ':' in intent else intent

            # Increase confidence with each interaction
            current = self.topic_confidence[topic]
            self.topic_confidence[topic] = min(0.95, current + 0.02)

    def can_answer_independently(self, question: str, intent: str) -> Tuple[bool, float]:
        """
        Check if Alice can answer this question from her own knowledge.
        Returns (can_answer, confidence)
        """
        # Check entities mentioned in question
        words = question.lower().split()
        relevant_entities = [e for e in self.entities.values() if e.name.lower() in question.lower()]

        # Check if we have learned responses for this intent
        has_responses = intent in self.learned_responses and len(self.learned_responses[intent]) > 2

        # Check topic confidence
        topic = intent.split(':')[0] if ':' in intent else intent
        topic_conf = self.topic_confidence.get(topic, 0.0)

        # Calculate overall confidence
        entity_conf = sum(e.confidence for e in relevant_entities) / max(len(relevant_entities), 1) if relevant_entities else 0.0
        response_conf = 0.5 if has_responses else 0.0

        overall_conf = (entity_conf * 0.3 + response_conf * 0.3 + topic_conf * 0.4)

        # Can answer if confidence > 0.6
        can_answer = overall_conf > 0.6

        return can_answer, overall_conf

    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get what Alice knows about an entity"""
        entity = self.entities.get(entity_name)
        if not entity:
            # Try case-insensitive search
            for name, ent in self.entities.items():
                if name.lower() == entity_name.lower():
                    entity = ent
                    break

        if entity:
            return entity.to_dict()
        return None

    def get_relationships_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get all relationships involving an entity"""
        entity_lower = entity_name.lower()
        relevant_rels = []

        for rel in self.relationships:
            if (rel.subject.lower() == entity_lower or
                rel.object.lower() == entity_lower):
                relevant_rels.append(rel.to_dict())

        return relevant_rels

    def _save_knowledge(self):
        """Save Alice's knowledge to disk"""
        try:
            # Save entities
            entities_path = self.storage_path / "entities.json"
            with open(entities_path, 'w') as f:
                entities_data = {name: entity.to_dict() for name, entity in self.entities.items()}
                json.dump(entities_data, f, indent=2)

            # Save relationships
            rels_path = self.storage_path / "relationships.json"
            with open(rels_path, 'w') as f:
                rels_data = [rel.to_dict() for rel in self.relationships]
                json.dump(rels_data, f, indent=2)

            # Save concepts and patterns
            concepts_path = self.storage_path / "concepts.json"
            with open(concepts_path, 'w') as f:
                json.dump({
                    'concepts': self.concepts.concepts,
                    'word_associations': {k: dict(v) for k, v in self.concepts.word_associations.items()},
                    'learned_responses': dict(self.learned_responses),
                    'topic_confidence': dict(self.topic_confidence)
                }, f, indent=2)

        except Exception as e:
            logger.error(f"[KnowledgeEngine] Error saving knowledge: {e}")

    def _load_knowledge(self):
        """Load Alice's existing knowledge"""
        try:
            # Load entities
            entities_path = self.storage_path / "entities.json"
            if entities_path.exists():
                with open(entities_path) as f:
                    entities_data = json.load(f)
                    self.entities = {name: Entity.from_dict(data) for name, data in entities_data.items()}

            # Load relationships
            rels_path = self.storage_path / "relationships.json"
            if rels_path.exists():
                with open(rels_path) as f:
                    rels_data = json.load(f)
                    self.relationships = [Relationship.from_dict(data) for data in rels_data]

            # Load concepts
            concepts_path = self.storage_path / "concepts.json"
            if concepts_path.exists():
                with open(concepts_path) as f:
                    data = json.load(f)
                    self.concepts.concepts = data.get('concepts', {})
                    self.concepts.word_associations = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get('word_associations', {}).items()}
                    )
                    self.learned_responses = defaultdict(list, data.get('learned_responses', {}))
                    self.topic_confidence = defaultdict(float, data.get('topic_confidence', {}))

        except Exception as e:
            logger.error(f"[KnowledgeEngine] Error loading knowledge: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about Alice's knowledge"""
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'total_concepts': len(self.concepts.concepts),
            'learned_intents': len(self.learned_responses),
            'topics_confident_in': len([t for t, c in self.topic_confidence.items() if c > 0.7]),
            'top_entities': sorted(
                [(e.name, e.mention_count) for e in self.entities.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
