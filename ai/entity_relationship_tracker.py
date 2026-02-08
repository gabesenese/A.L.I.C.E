"""
Entity Relationship Tracker for A.L.I.C.E
Tracks relationships between entities mentioned in conversations
"""

import json
import logging
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EntityRelationship:
    """Represents a relationship between two entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: str
    timestamp: datetime
    source: str = "conversation"  # conversation, document, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRelationship':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Entity:
    """Represents an entity with metadata"""
    name: str
    entity_type: str
    aliases: Set[str]
    first_mentioned: datetime
    last_mentioned: datetime
    mention_count: int
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['aliases'] = list(self.aliases)
        data['first_mentioned'] = self.first_mentioned.isoformat()
        data['last_mentioned'] = self.last_mentioned.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create from dictionary"""
        data['aliases'] = set(data['aliases'])
        data['first_mentioned'] = datetime.fromisoformat(data['first_mentioned'])
        data['last_mentioned'] = datetime.fromisoformat(data['last_mentioned'])
        return cls(**data)


class RelationshipType:
    """Common relationship types"""
    # Personal relationships
    FAMILY = "family"
    FRIEND = "friend"
    COLLEAGUE = "colleague"
    SPOUSE = "spouse"
    PARENT = "parent"
    CHILD = "child"
    SIBLING = "sibling"
    
    # Professional relationships
    WORKS_AT = "works_at"
    MANAGES = "manages"
    REPORTS_TO = "reports_to"
    COLLABORATES_WITH = "collaborates_with"
    
    # Location relationships
    LIVES_IN = "lives_in"
    WORKS_IN = "works_in"
    BORN_IN = "born_in"
    VISITED = "visited"
    
    # Ownership/possession
    OWNS = "owns"
    HAS = "has"
    BELONGS_TO = "belongs_to"
    
    # Social relationships
    KNOWS = "knows"
    LIKES = "likes"
    DISLIKES = "dislikes"
    FOLLOWS = "follows"
    
    # Educational
    STUDIED_AT = "studied_at"
    TEACHES = "teaches"
    LEARNS_FROM = "learns_from"
    
    # General associations
    ASSOCIATED_WITH = "associated_with"
    PART_OF = "part_of"
    MEMBER_OF = "member_of"
    SIMILAR_TO = "similar_to"
    
    # Temporal
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    
    @classmethod
    def all_types(cls) -> List[str]:
        """Get all relationship types"""
        return [
            cls.FAMILY, cls.FRIEND, cls.COLLEAGUE, cls.SPOUSE, cls.PARENT, cls.CHILD, cls.SIBLING,
            cls.WORKS_AT, cls.MANAGES, cls.REPORTS_TO, cls.COLLABORATES_WITH,
            cls.LIVES_IN, cls.WORKS_IN, cls.BORN_IN, cls.VISITED,
            cls.OWNS, cls.HAS, cls.BELONGS_TO,
            cls.KNOWS, cls.LIKES, cls.DISLIKES, cls.FOLLOWS,
            cls.STUDIED_AT, cls.TEACHES, cls.LEARNS_FROM,
            cls.ASSOCIATED_WITH, cls.PART_OF, cls.MEMBER_OF, cls.SIMILAR_TO,
            cls.BEFORE, cls.AFTER, cls.DURING
        ]


class EntityRelationshipTracker:
    """
    Tracks and manages relationships between entities
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Storage files
        self.entities_file = self.data_dir / "entities.json"
        self.relationships_file = self.data_dir / "relationships.json"
        
        # In-memory storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[EntityRelationship] = []
        
        # Relationship extraction patterns
        self.relationship_patterns = self._build_relationship_patterns()
        
        # Load existing data
        self._load_data()
        
        logger.info(f"Entity Relationship Tracker initialized with {len(self.entities)} entities and {len(self.relationships)} relationships")
    
    def _build_relationship_patterns(self) -> Dict[str, List[Tuple[str, float]]]:
        """Build patterns for extracting relationships from text"""
        return {
            # Family relationships
            RelationshipType.PARENT: [
                (r'(\w+)(?:\s+is)?\s+(?:my|his|her)\s+(?:father|dad|mother|mom|parent)', 0.9),
                (r'(\w+)(?:\s+and\s+\w+)?\s+(?:are|is)\s+(?:my|his|her)\s+parents?', 0.9),
            ],
            RelationshipType.CHILD: [
                (r'(?:my|his|her)\s+(?:son|daughter|child|kid)\s+(\w+)', 0.9),
                (r'(\w+)\s+is\s+(?:my|his|her)\s+(?:son|daughter|child)', 0.9),
            ],
            RelationshipType.SIBLING: [
                (r'(?:my|his|her)\s+(?:brother|sister|sibling)\s+(\w+)', 0.9),
                (r'(\w+)\s+is\s+(?:my|his|her)\s+(?:brother|sister)', 0.9),
            ],
            RelationshipType.SPOUSE: [
                (r'(?:my|his|her)\s+(?:wife|husband|spouse)\s+(\w+)', 0.9),
                (r'(\w+)\s+is\s+(?:my|his|her)\s+(?:wife|husband)', 0.9),
                (r'(?:I|he|she)\s+(?:am|is)\s+married\s+to\s+(\w+)', 0.8),
            ],
            
            # Professional relationships  
            RelationshipType.WORKS_AT: [
                (r'(\w+)\s+works?\s+(?:at|for)\s+(\w+(?:\s+\w+)*)', 0.8),
                (r'(\w+)(?:\s+is)?\s+(?:an?\s+)?(?:employee|worker)\s+at\s+(\w+(?:\s+\w+)*)', 0.8),
                (r"(\w+)(?:'s)?\s+job\s+(?:is\s+)?at\s+(\w+(?:\s+\w+)*)", 0.7),
            ],
            RelationshipType.COLLEAGUE: [
                (r'(?:my|his|her)\s+colleague\s+(\w+)', 0.8),
                (r'(\w+)\s+is\s+(?:my|his|her)\s+colleague', 0.8),
                (r'(\w+)\s+and\s+(\w+)\s+work\s+together', 0.7),
            ],
            RelationshipType.MANAGES: [
                (r'(\w+)\s+manages?\s+(\w+)', 0.8),
                (r"(\w+)\s+is\s+(\w+)(?:'s)?\s+(?:boss|manager|supervisor)", 0.8),
            ],
            
            # Location relationships
            RelationshipType.LIVES_IN: [
                (r'(\w+)\s+lives?\s+in\s+(\w+(?:\s+\w+)*)', 0.8),
                (r'(\w+)(?:\s+is)?\s+from\s+(\w+(?:\s+\w+)*)', 0.7),
                (r"(\w+)(?:'s)?\s+(?:home|house)\s+(?:is\s+)?in\s+(\w+(?:\s+\w+)*)", 0.8),
            ],
            RelationshipType.BORN_IN: [
                (r'(\w+)\s+was\s+born\s+in\s+(\w+(?:\s+\w+)*)', 0.9),
                (r'(\w+)(?:\s+is)?\s+(?:a\s+)?native\s+of\s+(\w+(?:\s+\w+)*)', 0.8),
            ],
            
            # Social relationships
            RelationshipType.FRIEND: [
                (r'(?:my|his|her)\s+friend\s+(\w+)', 0.8),
                (r'(\w+)\s+is\s+(?:my|his|her)\s+friend', 0.8),
                (r'(\w+)\s+and\s+(\w+)\s+are\s+friends', 0.8),
            ],
            RelationshipType.KNOWS: [
                (r'(\w+)\s+knows?\s+(\w+)', 0.6),
                (r'(\w+)\s+(?:is\s+)?familiar\s+with\s+(\w+)', 0.6),
                (r'(\w+)\s+(?:has\s+)?met\s+(\w+)', 0.7),
            ],
            
            # Possession/ownership
            RelationshipType.OWNS: [
                (r'(\w+)\s+owns?\s+(?:a\s+)?(\w+(?:\s+\w+)*)', 0.8),
                (r"(\w+)(?:'s)?\s+(\w+(?:\s+\w+)*)", 0.5),  # Possessive form
                (r'(\w+)\s+has\s+(?:a\s+)?(\w+(?:\s+\w+)*)', 0.6),
            ],
            
            # Educational
            RelationshipType.STUDIED_AT: [
                (r'(\w+)\s+studied\s+at\s+(\w+(?:\s+\w+)*)', 0.8),
                (r'(\w+)\s+went\s+to\s+(\w+(?:\s+\w+)*)', 0.7),
                (r'(\w+)\s+graduated\s+from\s+(\w+(?:\s+\w+)*)', 0.8),
                (r'(\w+)(?:\s+is)?\s+(?:a\s+)?(?:student|graduate)\s+(?:of\s+)?(?:at\s+)?(\w+(?:\s+\w+)*)', 0.7),
            ],
        }
    
    def _load_data(self):
        """Load entities and relationships from storage"""
        # Load entities
        if self.entities_file.exists():
            try:
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)
                self.entities = {
                    name: Entity.from_dict(data) 
                    for name, data in entities_data.items()
                }
                logger.info(f"📂 Loaded {len(self.entities)} entities")
            except Exception as e:
                logger.error(f"Error loading entities: {e}")
        
        # Load relationships
        if self.relationships_file.exists():
            try:
                with open(self.relationships_file, 'r', encoding='utf-8') as f:
                    relationships_data = json.load(f)
                self.relationships = [
                    EntityRelationship.from_dict(data) 
                    for data in relationships_data
                ]
                logger.info(f"📂 Loaded {len(self.relationships)} relationships")
            except Exception as e:
                logger.error(f"Error loading relationships: {e}")
    
    def _save_data(self):
        """Save entities and relationships to storage"""
        try:
            # Save entities
            entities_data = {
                name: entity.to_dict() 
                for name, entity in self.entities.items()
            }
            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_data, f, indent=2, ensure_ascii=False)
            
            # Save relationships
            relationships_data = [
                relationship.to_dict() 
                for relationship in self.relationships
            ]
            with open(self.relationships_file, 'w', encoding='utf-8') as f:
                json.dump(relationships_data, f, indent=2, ensure_ascii=False)
            
            logger.debug("💾 Saved entity relationship data")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def add_entity(self, name: str, entity_type: str, confidence: float = 1.0, 
                   metadata: Dict[str, Any] = None) -> Entity:
        """Add or update an entity"""
        name_lower = name.lower()
        now = datetime.now()
        
        if name_lower in self.entities:
            # Update existing entity
            entity = self.entities[name_lower]
            entity.last_mentioned = now
            entity.mention_count += 1
            entity.confidence = max(entity.confidence, confidence)
            if metadata:
                entity.metadata.update(metadata)
        else:
            # Create new entity
            entity = Entity(
                name=name,
                entity_type=entity_type,
                aliases=set(),
                first_mentioned=now,
                last_mentioned=now,
                mention_count=1,
                confidence=confidence,
                metadata=metadata or {}
            )
            self.entities[name_lower] = entity
        
        self._save_data()
        return entity
    
    def add_relationship(self, source: str, target: str, relationship_type: str, 
                        confidence: float, context: str, source_type: str = "conversation"):
        """Add a relationship between two entities"""
        # Ensure entities exist
        self.add_entity(source, "unknown", confidence)
        self.add_entity(target, "unknown", confidence)
        
        # Check if relationship already exists
        existing = self._find_relationship(source, target, relationship_type)
        if existing:
            # Update confidence and context
            existing.confidence = max(existing.confidence, confidence)
            existing.context = context  # Update with latest context
            existing.timestamp = datetime.now()
        else:
            # Create new relationship
            relationship = EntityRelationship(
                source_entity=source.lower(),
                target_entity=target.lower(),
                relationship_type=relationship_type,
                confidence=confidence,
                context=context,
                timestamp=datetime.now(),
                source=source_type
            )
            self.relationships.append(relationship)
        
        self._save_data()
        logger.info(f"Added relationship: {source} {relationship_type} {target} (confidence: {confidence:.2f})")
    
    def _find_relationship(self, source: str, target: str, relationship_type: str) -> Optional[EntityRelationship]:
        """Find existing relationship"""
        source_lower = source.lower()
        target_lower = target.lower()
        
        for rel in self.relationships:
            if (rel.source_entity == source_lower and 
                rel.target_entity == target_lower and 
                rel.relationship_type == relationship_type):
                return rel
        return None
    
    def extract_relationships_from_text(self, text: str) -> List[EntityRelationship]:
        """Extract relationships from text using pattern matching"""
        extracted = []
        text_lower = text.lower()
        
        for relationship_type, patterns in self.relationship_patterns.items():
            for pattern, confidence in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        source = groups[0].strip()
                        target = groups[1].strip()
                        
                        # Filter out common words and short matches
                        if (len(source) > 2 and len(target) > 2 and 
                            source not in ['the', 'and', 'or', 'but', 'my', 'his', 'her'] and
                            target not in ['the', 'and', 'or', 'but', 'my', 'his', 'her']):
                            
                            relationship = EntityRelationship(
                                source_entity=source,
                                target_entity=target,
                                relationship_type=relationship_type,
                                confidence=confidence,
                                context=text,
                                timestamp=datetime.now(),
                                source="pattern_extraction"
                            )
                            extracted.append(relationship)
        
        return extracted
    
    def process_text(self, text: str) -> List[EntityRelationship]:
        """Process text and extract/store relationships"""
        extracted = self.extract_relationships_from_text(text)
        
        for relationship in extracted:
            self.add_relationship(
                relationship.source_entity,
                relationship.target_entity,
                relationship.relationship_type,
                relationship.confidence,
                relationship.context
            )
        
        return extracted
    
    def get_entity_relationships(self, entity_name: str) -> List[EntityRelationship]:
        """Get all relationships for an entity"""
        entity_lower = entity_name.lower()
        return [
            rel for rel in self.relationships
            if rel.source_entity == entity_lower or rel.target_entity == entity_lower
        ]
    
    def get_relationships_by_type(self, relationship_type: str) -> List[EntityRelationship]:
        """Get all relationships of a specific type"""
        return [
            rel for rel in self.relationships
            if rel.relationship_type == relationship_type
        ]
    
    def find_path_between_entities(self, source: str, target: str, max_depth: int = 3) -> List[List[EntityRelationship]]:
        """Find relationship paths between two entities"""
        source_lower = source.lower()
        target_lower = target.lower()
        
        if source_lower == target_lower:
            return [[]]
        
        visited = set()
        paths = []
        
        def dfs(current: str, target: str, path: List[EntityRelationship], depth: int):
            if depth > max_depth or current in visited:
                return
            
            visited.add(current)
            
            if current == target and path:
                paths.append(path.copy())
                visited.remove(current)
                return
            
            # Find connected entities
            for rel in self.relationships:
                next_entity = None
                current_rel = None
                
                if rel.source_entity == current:
                    next_entity = rel.target_entity
                    current_rel = rel
                elif rel.target_entity == current:
                    next_entity = rel.source_entity
                    # Create reverse relationship for path
                    current_rel = EntityRelationship(
                        source_entity=rel.target_entity,
                        target_entity=rel.source_entity,
                        relationship_type=f"inverse_{rel.relationship_type}",
                        confidence=rel.confidence,
                        context=rel.context,
                        timestamp=rel.timestamp,
                        source=rel.source
                    )
                
                if next_entity and current_rel:
                    path.append(current_rel)
                    dfs(next_entity, target, path, depth + 1)
                    path.pop()
            
            visited.remove(current)
        
        dfs(source_lower, target_lower, [], 0)
        return paths
    
    def get_entity_network(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get entity network (entities connected within max_depth)"""
        entity_lower = entity_name.lower()
        visited = set()
        network = {
            'center': entity_name,
            'entities': {},
            'relationships': []
        }
        
        def explore(current: str, depth: int):
            if depth > max_depth or current in visited:
                return
            
            visited.add(current)
            
            # Add entity to network
            if current in self.entities:
                network['entities'][current] = self.entities[current].to_dict()
            
            # Find connected entities
            for rel in self.relationships:
                if rel.source_entity == current or rel.target_entity == current:
                    network['relationships'].append(rel.to_dict())
                    
                    # Explore connected entity
                    next_entity = (rel.target_entity if rel.source_entity == current 
                                 else rel.source_entity)
                    explore(next_entity, depth + 1)
        
        explore(entity_lower, 0)
        return network
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get relationship tracking statistics"""
        relationship_types = defaultdict(int)
        entity_types = defaultdict(int)
        
        for rel in self.relationships:
            relationship_types[rel.relationship_type] += 1
        
        for entity in self.entities.values():
            entity_types[entity.entity_type] += 1
        
        return {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'relationship_types': dict(relationship_types),
            'entity_types': dict(entity_types),
            'most_connected_entities': self._get_most_connected_entities(5),
            'recent_relationships': self._get_recent_relationships(10)
        }
    
    def _get_most_connected_entities(self, limit: int) -> List[Tuple[str, int]]:
        """Get entities with most relationships"""
        entity_counts = defaultdict(int)
        
        for rel in self.relationships:
            entity_counts[rel.source_entity] += 1
            entity_counts[rel.target_entity] += 1
        
        return sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def _get_recent_relationships(self, limit: int) -> List[Dict[str, Any]]:
        """Get most recent relationships"""
        sorted_relationships = sorted(self.relationships, key=lambda x: x.timestamp, reverse=True)
        return [rel.to_dict() for rel in sorted_relationships[:limit]]
    
    def query_relationships(self, query: str) -> List[EntityRelationship]:
        """Query relationships with natural language"""
        query_lower = query.lower()
        results = []
        
        # Simple keyword-based querying
        for rel in self.relationships:
            if (query_lower in rel.source_entity.lower() or 
                query_lower in rel.target_entity.lower() or 
                query_lower in rel.relationship_type.lower() or 
                query_lower in rel.context.lower()):
                results.append(rel)
        
        return results