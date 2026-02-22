"""
A.L.I.C.E Notes Plugin

This plugin provides comprehensive note-taking capabilities including:
- Create, edit, and delete notes
- Organize notes with tags
- Search notes by keyword, tags, date range
- Semantic search using RAG system
- Markdown support for rich formatting
- Auto-save functionality

Supports commands like:
- "Create a note about project ideas"
- "Add a note: Buy groceries tomorrow #personal"
- "Make a note: Finish report by Friday #work #urgent"
- "Search my notes for meeting with John"
- "Find notes for #work"
- "Show me notes from this week"
- "List all my notes"
- "Show all tags"

Notes are stored in: data/notes/notes.json

Tags: Add tags to your notes using #tagname (e.g., #work, #personal, #urgent)
Note Types: general, todo, idea, meeting (auto-detected or can be specified)
"""

import os
import sys
import json
import logging
import uuid
from typing import (
    Dict, List, Optional, Any, Tuple,
    Final, ClassVar, Protocol, runtime_checkable,
)
from typing import TypedDict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import re

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import the proper plugin interface
from ai.plugins.plugin_system import PluginInterface

@dataclass
class Note:
    """Represents a note"""
    id: str
    title: str
    content: str
    tags: List[str]
    created_at: str
    updated_at: str
    note_type: str = "general"  # general, todo, idea, meeting, etc.
    pinned: bool = False
    archived: bool = False
    due_date: Optional[str] = None
    category: str = "general"  # general, work, personal, project, etc.
    priority: str = "medium"  # low, medium, high, urgent
    reminder: Optional[str] = None
    checklist_items: Optional[List[Dict[str, Any]]] = None  # [{"text": "item", "checked": False}]
    related_notes: Optional[List[str]] = None  # List of note IDs
    versions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Transient fields (not saved to disk)
    _relevance_cache: Dict[str, float] = field(default_factory=dict, repr=False, compare=False)
    _tokenized_title: Optional[List[str]] = field(default=None, repr=False, compare=False)
    _tokenized_content: Optional[List[str]] = field(default=None, repr=False, compare=False)
    
    def to_dict(self) -> Dict:
        """Convert note to dictionary"""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict) -> 'Note':
        """Create note from dictionary with backward compatibility"""
        # Set defaults for new fields if not present
        defaults = {
            'pinned': False,
            'archived': False,
            'due_date': None,
            'category': 'general',
            'priority': 'medium',
            'reminder': None,
            'checklist_items': None,
            'related_notes': None,
            'versions': [],
        }
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        
        # Remove transient fields if they somehow got saved
        data.pop('_relevance_cache', None)
        data.pop('_tokenized_title', None)
        data.pop('_tokenized_content', None)
        
        return Note(**data)
    
    def __str__(self) -> str:
        tags_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        pin_str = " " if self.pinned else ""
        priority_icons = {"low": "", "medium": "", "high": "", "urgent": ""}
        priority_str = f"{priority_icons.get(self.priority, '')} " if self.priority != "medium" else ""
        return f"{pin_str}{priority_str}{self.title}{tags_str}\n{self.content}"
    
    def matches_keyword(self, keyword: str) -> bool:
        """Check if note contains keyword"""
        keyword_lower = keyword.lower()
        return (keyword_lower in self.title.lower() or 
                keyword_lower in self.content.lower() or
                any(keyword_lower in tag.lower() for tag in self.tags) or
                keyword_lower in self.category.lower())
    
    def has_tag(self, tag: str) -> bool:
        """Check if note has specific tag"""
        return tag.lower() in [t.lower() for t in self.tags]
    
    def get_relevance_score(self, keyword: str) -> float:
        """Calculate relevance score for fuzzy search with caching and optimization"""
        # Check cache first
        cache_key = keyword.lower()
        if cache_key in self._relevance_cache:
            return self._relevance_cache[cache_key]
        
        keyword_lower = keyword.lower()
        score = 0.0
        
        # Pre-tokenize if not already done (lazy initialization)
        if self._tokenized_title is None:
            self._tokenized_title = self.title.lower().split()
        if self._tokenized_content is None:
            self._tokenized_content = self.content.lower().split()
        
        # Title match (highest weight) - early exit for exact matches
        title_lower = self.title.lower()
        if keyword_lower == title_lower:
            score = 15.0  # Exact match
            if self.pinned:
                score *= 1.2
            if self.priority == "urgent":
                score *= 1.15
            elif self.priority == "high":
                score *= 1.1
            self._relevance_cache[cache_key] = score
            return score
        
        if keyword_lower in title_lower:
            score += 10.0
        
        # Content match
        if keyword_lower in self.content.lower():
            score += 5.0
        
        # Tag match
        if any(keyword_lower in tag.lower() for tag in self.tags):
            score += 7.0
        
        # Category match
        if keyword_lower in self.category.lower():
            score += 3.0
        
        # Early exit if strong match found
        if score >= 15.0:
            if self.pinned:
                score *= 1.2
            if self.priority == "urgent":
                score *= 1.15
            elif self.priority == "high":
                score *= 1.1
            self._relevance_cache[cache_key] = score
            return score
        
        # Partial word matching using pre-tokenized lists
        for word in self._tokenized_title:
            if keyword_lower in word or word in keyword_lower:
                score += 2.0
        
        # Limit content word checking to first 50 words for performance
        for word in self._tokenized_content[:50]:
            if keyword_lower in word or word in keyword_lower:
                score += 0.5
        
        # Boost for pinned and high priority
        if self.pinned:
            score *= 1.2
        if self.priority == "urgent":
            score *= 1.15
        elif self.priority == "high":
            score *= 1.1
        
        # Cache the result
        self._relevance_cache[cache_key] = score
        return score


# ---------------------------------------------------------------------------
# Module-level constants  (Final → cannot be accidentally overridden)
# ---------------------------------------------------------------------------
DESTRUCTIVE_ACTIONS: Final[frozenset] = frozenset({
    "delete_note", "archive_note", "delete_all_notes",
})
_BULK_DESTRUCTIVE_ACTIONS: Final[frozenset] = frozenset({"delete_all_notes"})
_UPCOMING_REMINDER_HOURS: Final[int] = 48
_CONFIRMATION_KEYWORD: Final[str] = "confirm"
_MAX_CONTENT_SNIPPET_CHARS: Final[int] = 160


# ---------------------------------------------------------------------------
# Protocol: any class that can supply note snippets to the LLM context builder
# ---------------------------------------------------------------------------
@runtime_checkable
class NoteContextProvider(Protocol):
    """Contract for objects that can yield note context snippets for the LLM."""
    def get_note_context_snippet(self, query: str, max_chars: int = 600) -> str: ...


# ---------------------------------------------------------------------------
# TypedDict: structured note search parameters
# ---------------------------------------------------------------------------
class NoteSearchQuery(TypedDict, total=False):
    """Type-safe bag of note search parameters."""
    keyword: str
    tags: List[str]
    content_query: str
    include_archived: bool
    limit: int


# ---------------------------------------------------------------------------
# Immutable value object for full-text content search results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContentSearchResult:
    """Immutable scored result from full-text body search."""
    note: Note
    score: float
    matched_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note_id": self.note.id,
            "title": self.note.title,
            "tags": self.note.tags,
            "score": round(self.score, 3),
            "matched_snippet": self.matched_snippet,
            "updated_at": self.note.updated_at,
            "priority": self.note.priority,
        }


class NotesManager:
    """Handles note storage and retrieval"""
    
    def __init__(self, notes_dir: str = "data/notes"):
        self.notes_dir = notes_dir
        self.notes_file = os.path.join(notes_dir, "notes.json")
        self.notes: Dict[str, Note] = {}
        self._ensure_directory()
        self._load_notes()
        
        self.NOTE_TEMPLATES: Dict[str, Dict[str, Any]] = {
            "meeting": {
                "title_prefix": "Meeting",
                "category": "work",
                "note_type": "meeting",
                "template": "Attendees:\nAgenda:\nDecisions:\nAction Items:\n",
                "tags": ["meeting"],
            },
            "project": {
                "title_prefix": "Project",
                "category": "project",
                "note_type": "general",
                "template": "Objective:\nMilestones:\nRisks:\nNext Steps:\n",
                "tags": ["project"],
            },
            "daily": {
                "title_prefix": "Daily",
                "category": "personal",
                "note_type": "todo",
                "template": "Top Priorities:\n1.\n2.\n3.\nNotes:\n",
                "tags": ["daily"],
            },
        }
    
    def _ensure_directory(self):
        """Ensure notes directory exists"""
        os.makedirs(self.notes_dir, exist_ok=True)
    
    def _load_notes(self):
        """Load notes from JSON file with validation and error recovery"""
        try:
            if os.path.exists(self.notes_file):
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Validate and load notes with UUID checking
                loaded_notes = {}
                corrupted_count = 0
                
                for note_id, note_data in data.items():
                    try:
                        # Validate UUID format in note ID
                        if '_' in note_id:
                            uuid_part = note_id.split('_')[-1]
                            try:
                                # Validate it's hex
                                int(uuid_part, 16)
                            except ValueError:
                                logger.warning(f"Invalid UUID in note ID: {note_id}")
                                corrupted_count += 1
                                continue
                        
                        note = Note.from_dict(note_data)
                        loaded_notes[note_id] = note
                    except (ValueError, TypeError, KeyError) as e:
                        logger.warning(f"Skipping corrupted note {note_id}: {e}")
                        corrupted_count += 1
                
                self.notes = loaded_notes
                logger.info(f"Loaded {len(self.notes)} notes from storage" + 
                           (f" ({corrupted_count} corrupted entries skipped)" if corrupted_count else ""))
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error loading notes: {e}. Attempting backup recovery...")
            self._recover_from_backup()
        except FileNotFoundError:
            logger.info("No existing notes file found, starting fresh")
            self.notes = {}
        except Exception as e:
            logger.error(f"Unexpected error loading notes: {e}")
            self.notes = {}
    
    def _save_notes(self):
        """Save notes to JSON file with atomic writes and backup"""
        temp_file = self.notes_file + ".tmp"
        backup_file = self.notes_file + ".backup"
        
        try:
            # Prepare data
            data = {
                note_id: note.to_dict()
                for note_id, note in self.notes.items()
            }
            
            # Write to temporary file first (atomic write pattern)
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Create backup of existing file before replacing
            if os.path.exists(self.notes_file):
                try:
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                    os.rename(self.notes_file, backup_file)
                except OSError as e:
                    logger.warning(f"Could not create backup: {e}")
            
            # Move temp file to actual location
            os.rename(temp_file, self.notes_file)
            
            # Save human-readable text file
            self._save_readable_notes()
            
            logger.info(f"Saved {len(self.notes)} notes to storage")
        except json.JSONEncodeError as e:
            logger.error(f"JSON encode error saving notes: {e}")
        except OSError as e:
            logger.error(f"File system error saving notes: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving notes: {e}")
        finally:
            # Clean up temp file if it still exists
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
    
    def _save_readable_notes(self):
        """Save notes to a human-readable text file"""
        try:
            text_file = os.path.join(self.notes_dir, "notes.txt")
            
            # Get all notes sorted (pinned first, then by update time)
            all_notes = self.get_all_notes(sort_by="updated_at", reverse=True, include_archived=True)
            
            with open(text_file, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*80 + "\n")
                f.write(" " * 30 + "MY NOTES\n")
                f.write("="*80 + "\n")
                f.write(f"Total Notes: {len([n for n in all_notes if not n.archived])}\n")
                f.write(f"Archived: {len([n for n in all_notes if n.archived])}\n")
                f.write(f"Last Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n")
                f.write("="*80 + "\n\n")
                
                # Separate active and archived notes
                active_notes = [n for n in all_notes if not n.archived]
                archived_notes = [n for n in all_notes if n.archived]
                
                # Write active notes
                if active_notes:
                    f.write("\n" + "─"*80 + "\n")
                    f.write(" " * 32 + "ACTIVE NOTES\n")
                    f.write("─"*80 + "\n\n")
                    
                    for note in active_notes:
                        self._write_note_to_file(f, note)
                
                # Write archived notes
                if archived_notes:
                    f.write("\n\n" + "─"*80 + "\n")
                    f.write(" " * 30 + "ARCHIVED NOTES\n")
                    f.write("─"*80 + "\n\n")
                    
                    for note in archived_notes:
                        self._write_note_to_file(f, note)
                
                # Write footer
                f.write("\n" + "="*80 + "\n")
                f.write(" " * 25 + "END OF NOTES\n")
                f.write("="*80 + "\n")
            
            logger.info(f"Saved readable notes to {text_file}")
        except Exception as e:
            logger.error(f"Error saving readable notes: {e}")
    
    def _recover_from_backup(self):
        """Attempt to recover notes from backup file"""
        backup_file = self.notes_file + ".backup"
        if os.path.exists(backup_file):
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.notes = {
                        note_id: Note.from_dict(note_data)
                        for note_id, note_data in data.items()
                    }
                logger.info(f"Recovered {len(self.notes)} notes from backup")
                # Save the recovered data to main file
                self._save_notes()
            except Exception as e:
                logger.error(f"Backup recovery failed: {e}")
                self.notes = {}
        else:
            logger.warning("No backup file available")
            self.notes = {}
    
    def _write_note_to_file(self, f, note: Note):
        """Write a single note to file with proper formatting"""
        # Status indicators
        status = []
        if note.pinned:
            status.append(" PINNED")
        if note.priority == "urgent":
            status.append(" URGENT")
        elif note.priority == "high":
            status.append(" HIGH PRIORITY")
        elif note.priority == "low":
            status.append(" LOW PRIORITY")
        
        # Header
        f.write("┌" + "─"*78 + "┐\n")
        f.write(f"│ {note.title[:74]:<74} │\n")
        
        # Status line
        if status:
            status_str = " • ".join(status)
            f.write(f"│ {status_str[:74]:<74} │\n")
        
        # Metadata
        f.write("├" + "─"*78 + "┤\n")
        
        # Type and Category
        type_cat = f"Type: {note.note_type.title()} | Category: {note.category.title()}"
        f.write(f"│ {type_cat[:74]:<74} │\n")
        
        # Dates
        created = datetime.fromisoformat(note.created_at).strftime("%b %d, %Y %I:%M %p")
        updated = datetime.fromisoformat(note.updated_at).strftime("%b %d, %Y %I:%M %p")
        f.write(f"│ Created: {created:<62} │\n")
        f.write(f"│ Updated: {updated:<62} │\n")
        
        # Due date if exists
        if note.due_date:
            f.write(f"│ Due: {note.due_date:<66} │\n")
        
        # Tags
        if note.tags:
            tags_str = ", ".join([f"#{tag}" for tag in note.tags])
            f.write(f"│ Tags: {tags_str[:69]:<69} │\n")
        
        # Content
        f.write("├" + "─"*78 + "┤\n")
        
        # Split content into lines and wrap if necessary
        content_lines = note.content.split('\n')
        for line in content_lines:
            if len(line) <= 74:
                f.write(f"│ {line:<74} │\n")
            else:
                # Wrap long lines
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 74:
                        current_line += (word + " ")
                    else:
                        f.write(f"│ {current_line:<74} │\n")
                        current_line = word + " "
                if current_line:
                    f.write(f"│ {current_line:<74} │\n")
        
        # Checklist items if exists
        if note.checklist_items:
            f.write("│ " + " "*74 + " │\n")
            f.write(f"│ {'Checklist:':<74} │\n")
            for item in note.checklist_items:
                checkbox = "[x]" if item.get("checked", False) else "[ ]"
                item_text = item.get("text", "")
                f.write(f"│   {checkbox} {item_text[:70]:<70} │\n")
        
        # Footer
        f.write("└" + "─"*78 + "┘\n\n")
    
    def _generate_id(self) -> str:
        """Generate unique note ID"""
        while True:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            suffix = uuid.uuid4().hex[:8]
            note_id = f"note_{timestamp}_{suffix}"
            if note_id not in self.notes:
                return note_id

    def _save_note_version(self, note: Note) -> None:
        """Capture current note state for history before a mutating operation."""
        snapshot = note.to_dict()
        snapshot.pop("versions", None)
        snapshot.pop("_relevance_cache", None)
        snapshot.pop("_tokenized_title", None)
        snapshot.pop("_tokenized_content", None)
        snapshot["version_timestamp"] = datetime.now().isoformat()

        note.versions.append(snapshot)
        if len(note.versions) > 20:
            note.versions = note.versions[-20:]

    def list_templates(self) -> List[str]:
        """Return available note template names."""
        return sorted(list(self.NOTE_TEMPLATES.keys()))

    def create_note_from_template(
        self,
        template_name: str,
        title: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> Optional[Note]:
        """Create a new note from a built-in template."""
        template = self.NOTE_TEMPLATES.get(template_name.lower())
        if not template:
            return None

        content = template["template"]
        if variables:
            for key, value in variables.items():
                content = content.replace("{" + key + "}", value)

        final_title = title or f"{template['title_prefix']} {datetime.now().strftime('%Y-%m-%d')}"
        return self.create_note(
            title=final_title,
            content=content,
            tags=list(template.get("tags", [])),
            note_type=template.get("note_type", "general"),
            category=template.get("category", "general"),
        )

    def get_note_versions(self, note_id: str) -> List[Dict[str, Any]]:
        """Return version history for a note."""
        note = self.notes.get(note_id)
        if not note:
            return []
        return note.versions

    def restore_note_version(self, note_id: str, version_index: int) -> bool:
        """Restore note fields from a previous version index."""
        note = self.notes.get(note_id)
        if not note or version_index < 0 or version_index >= len(note.versions):
            return False

        self._save_note_version(note)
        version = note.versions[version_index]

        note.title = version.get("title", note.title)
        note.content = version.get("content", note.content)
        note.tags = version.get("tags", note.tags)
        note.note_type = version.get("note_type", note.note_type)
        note.category = version.get("category", note.category)
        note.priority = version.get("priority", note.priority)
        note.due_date = version.get("due_date", note.due_date)
        note.checklist_items = version.get("checklist_items", note.checklist_items)
        note.related_notes = version.get("related_notes", note.related_notes)
        note.updated_at = datetime.now().isoformat()
        self._save_notes()
        return True

    def semantic_search_notes(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.40,
    ) -> List[Dict[str, Any]]:
        """Semantic search across note title + content using embeddings."""
        try:
            from ai.memory.embedding_manager import get_embedding_manager

            embedding_manager = get_embedding_manager()
            query_vec = embedding_manager.create_embedding(query)

            scored: List[Dict[str, Any]] = []
            for note in self.notes.values():
                if note.archived:
                    continue
                text = f"{note.title}\n{note.content}"
                note_vec = embedding_manager.create_embedding(text)
                similarity = embedding_manager.calculate_similarity(query_vec, note_vec)
                if similarity >= min_similarity:
                    scored.append(
                        {
                            "note_id": note.id,
                            "title": note.title,
                            "similarity": float(similarity),
                            "snippet": (note.content or "")[:220],
                            "tags": note.tags,
                        }
                    )

            scored.sort(key=lambda x: x["similarity"], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.debug(f"Semantic search unavailable, falling back to keyword search: {e}")
            fallback_notes = self.fuzzy_search(query)[:top_k]
            return [
                {
                    "note_id": n.id,
                    "title": n.title,
                    "similarity": n.get_relevance_score(query),
                    "snippet": (n.content or "")[:220],
                    "tags": n.tags,
                }
                for n in fallback_notes
            ]

    def get_proactive_insights(self) -> Dict[str, Any]:
        """Generate actionable note insights (reminders, stale items, meeting prep)."""
        active_notes = [n for n in self.notes.values() if not n.archived]
        now = datetime.now()

        upcoming = self.get_upcoming_reminders(hours=72)
        overdue = self.get_overdue_notes()
        stale = [
            n
            for n in active_notes
            if (now - datetime.fromisoformat(n.updated_at)).days >= 14
        ]

        meeting_candidates = [
            n
            for n in active_notes
            if n.note_type == "meeting" or "meeting" in [t.lower() for t in n.tags]
        ]

        return {
            "upcoming_reminders": [
                {"note_id": n.id, "title": n.title, "due": n.reminder or n.due_date}
                for n in upcoming[:8]
            ],
            "overdue": [
                {"note_id": n.id, "title": n.title, "due": n.due_date}
                for n in overdue[:8]
            ],
            "stale_notes": [
                {"note_id": n.id, "title": n.title, "updated_at": n.updated_at}
                for n in stale[:8]
            ],
            "meeting_prep": [
                {
                    "note_id": n.id,
                    "title": n.title,
                    "summary_hint": (n.content or "")[:160],
                }
                for n in meeting_candidates[:6]
            ],
        }
    
    def create_note(self, title: str, content: str, tags: List[str] = None, 
                   note_type: str = "general", category: str = "general",
                   priority: str = "medium", due_date: str = None,
                   checklist_items: List[Dict[str, Any]] = None) -> Note:
        """Create a new note"""
        note_id = self._generate_id()
        now = datetime.now().isoformat()
        
        note = Note(
            id=note_id,
            title=title,
            content=content,
            tags=tags or [],
            created_at=now,
            updated_at=now,
            note_type=note_type,
            category=category,
            priority=priority,
            due_date=due_date,
            checklist_items=checklist_items
        )
        
        self.notes[note_id] = note
        self._save_notes()
        logger.info(f"Created note: {title}")
        return note
    
    def get_note(self, note_id: str) -> Optional[Note]:
        """Get a note by ID"""
        return self.notes.get(note_id)
    
    def update_note(self, note_id: str, title: str = None, content: str = None,
                   tags: List[str] = None) -> Optional[Note]:
        """Update an existing note"""
        note = self.notes.get(note_id)
        if not note:
            return None

        self._save_note_version(note)
        
        if title is not None:
            note.title = title
        if content is not None:
            note.content = content
        if tags is not None:
            note.tags = tags
        
        note.updated_at = datetime.now().isoformat()
        self._save_notes()
        logger.info(f"Updated note: {note_id}")
        return note
    
    def delete_note(self, note_id: str) -> bool:
        """Delete a note"""
        if note_id in self.notes:
            del self.notes[note_id]
            self._save_notes()
            logger.info(f"Deleted note: {note_id}")
            return True
        return False
    
    def get_all_notes(self, sort_by: str = "updated_at", reverse: bool = True, 
                     include_archived: bool = False) -> List[Note]:
        """Get all notes sorted by specified field"""
        if include_archived:
            notes_list = list(self.notes.values())
        else:
            notes_list = [note for note in self.notes.values() if not note.archived]
        
        # Sort with pinned notes first
        notes_list.sort(key=lambda x: (
            not x.pinned,  # Pinned first (False < True)
            -1 if reverse else 1 * (getattr(x, sort_by) or "")
        ), reverse=False)
        
        if reverse and sort_by != "pinned":
            # Re-sort non-pinned notes
            pinned = [n for n in notes_list if n.pinned]
            unpinned = [n for n in notes_list if not n.pinned]
            unpinned.sort(key=lambda x: getattr(x, sort_by) or "", reverse=reverse)
            notes_list = pinned + unpinned
        
        return notes_list
    
    def search_by_keyword(self, keyword: str) -> List[Note]:
        """Search notes by keyword in title, content, or tags"""
        results = [note for note in self.notes.values() if note.matches_keyword(keyword)]
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results
    
    def search_by_tag(self, tag: str) -> List[Note]:
        """Search notes by tag"""
        results = [note for note in self.notes.values() if note.has_tag(tag)]
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results
    
    def search_by_date_range(self, start_date: datetime = None, 
                            end_date: datetime = None) -> List[Note]:
        """Search notes by date range"""
        results = []
        for note in self.notes.values():
            note_date = datetime.fromisoformat(note.updated_at)
            
            if start_date and note_date < start_date:
                continue
            if end_date and note_date > end_date:
                continue
            
            results.append(note)
        
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results
    
    def search_by_type(self, note_type: str) -> List[Note]:
        """Search notes by type"""
        results = [note for note in self.notes.values() if note.note_type == note_type]
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results
    
    def get_all_tags(self) -> List[str]:
        """Get all unique tags used in notes"""
        tags = set()
        for note in self.notes.values():
            tags.update(note.tags)
        return sorted(list(tags))
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories used in notes"""
        categories = set()
        for note in self.notes.values():
            if note.category:
                categories.add(note.category)
        return sorted(list(categories))
    
    def pin_note(self, note_id: str) -> bool:
        """Pin a note"""
        note = self.notes.get(note_id)
        if note:
            self._save_note_version(note)
            note.pinned = True
            note.updated_at = datetime.now().isoformat()
            self._save_notes()
            logger.info(f"Pinned note: {note_id}")
            return True
        return False
    
    def unpin_note(self, note_id: str) -> bool:
        """Unpin a note"""
        note = self.notes.get(note_id)
        if note:
            self._save_note_version(note)
            note.pinned = False
            note.updated_at = datetime.now().isoformat()
            self._save_notes()
            logger.info(f"Unpinned note: {note_id}")
            return True
        return False
    
    def archive_note(self, note_id: str) -> bool:
        """Archive a note"""
        note = self.notes.get(note_id)
        if note:
            self._save_note_version(note)
            note.archived = True
            note.updated_at = datetime.now().isoformat()
            self._save_notes()
            logger.info(f"Archived note: {note_id}")
            return True
        return False
    
    def unarchive_note(self, note_id: str) -> bool:
        """Unarchive a note"""
        note = self.notes.get(note_id)
        if note:
            self._save_note_version(note)
            note.archived = False
            note.updated_at = datetime.now().isoformat()
            self._save_notes()
            logger.info(f"Unarchived note: {note_id}")
            return True
        return False
    
    def set_due_date(self, note_id: str, due_date: str) -> bool:
        """Set due date for a note"""
        note = self.notes.get(note_id)
        if note:
            self._save_note_version(note)
            note.due_date = due_date
            note.updated_at = datetime.now().isoformat()
            self._save_notes()
            logger.info(f"Set due date for note {note_id}: {due_date}")
            return True
        return False
    
    def set_category(self, note_id: str, category: str) -> bool:
        """Set category for a note"""
        note = self.notes.get(note_id)
        if note:
            self._save_note_version(note)
            note.category = category
            note.updated_at = datetime.now().isoformat()
            self._save_notes()
            logger.info(f"Set category for note {note_id}: {category}")
            return True
        return False
    
    def set_priority(self, note_id: str, priority: str) -> bool:
        """Set priority for a note"""
        note = self.notes.get(note_id)
        if note and priority in ["low", "medium", "high", "urgent"]:
            self._save_note_version(note)
            note.priority = priority
            note.updated_at = datetime.now().isoformat()
            self._save_notes()
            logger.info(f"Set priority for note {note_id}: {priority}")
            return True
        return False
    
    def link_notes(self, note_id: str, related_note_id: str) -> bool:
        """Link two notes together"""
        note = self.notes.get(note_id)
        related_note = self.notes.get(related_note_id)
        
        if note and related_note:
            if note.related_notes is None:
                note.related_notes = []
            if related_note.related_notes is None:
                related_note.related_notes = []
            
            # Add bidirectional link
            if related_note_id not in note.related_notes:
                note.related_notes.append(related_note_id)
            if note_id not in related_note.related_notes:
                related_note.related_notes.append(note_id)
            
            note.updated_at = datetime.now().isoformat()
            related_note.updated_at = datetime.now().isoformat()
            self._save_notes()
            logger.info(f"Linked notes: {note_id} <-> {related_note_id}")
            return True
        return False
    
    def fuzzy_search(self, keyword: str, include_archived: bool = False) -> List[Note]:
        """Search notes with relevance scoring"""
        results = []
        for note in self.notes.values():
            if not include_archived and note.archived:
                continue
            score = note.get_relevance_score(keyword)
            if score > 0:
                results.append((note, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        return [note for note, score in results]
    
    def search_by_category(self, category: str) -> List[Note]:
        """Search notes by category"""
        results = [note for note in self.notes.values() 
                  if note.category.lower() == category.lower() and not note.archived]
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results
    
    def get_overdue_notes(self) -> List[Note]:
        """Get notes with past due dates"""
        now = datetime.now()
        results = []
        for note in self.notes.values():
            if note.due_date and not note.archived:
                try:
                    due = datetime.fromisoformat(note.due_date)
                    if due < now:
                        results.append(note)
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse due date '{note.due_date}' for note '{note.title}': {e}")
                    pass
        results.sort(key=lambda x: x.due_date)
        return results
    
    def find_by_title(self, title_query: str) -> List[Note]:
        """Find notes by title (for editing/deletion)"""
        query_lower = title_query.lower()
        results = []
        for note in self.notes.values():
            if not note.archived and query_lower in note.title.lower():
                results.append(note)
        # Sort by best match (shortest title = more specific match)
        results.sort(key=lambda x: len(x.title))
        return results

    # ------------------------------------------------------------------
    # Full-text content search  (feature: body search with scoring)
    # ------------------------------------------------------------------
    def search_by_content(
        self,
        query: str,
        *,
        include_archived: bool = False,
        limit: int = 20,
    ) -> List["ContentSearchResult"]:
        """Search the body text of all notes, returning scored results.

        Scoring weights:
        - exact phrase in content × 10
        - each individual token match (title bonus ×3, body ×1)
        - recency boost: notes updated in the last 7 days get +1
        """
        query_lower = query.lower()
        tokens = [t for t in re.findall(r"[a-z0-9']+", query_lower) if len(t) > 2]
        now = datetime.now()
        results: List[ContentSearchResult] = []

        for note in self.notes.values():
            if not include_archived and note.archived:
                continue

            score = 0.0
            body_lower = (note.content or "").lower()
            title_lower = note.title.lower()

            # Exact phrase bonus — title match outranks body match
            if query_lower in title_lower:
                score += 15.0
            if query_lower in body_lower:
                score += 10.0

            # Token-level matching
            for tok in tokens:
                if tok in title_lower:
                    score += 3.0
                if tok in body_lower:
                    score += 1.0

            if score == 0.0:
                continue

            # Recency boost
            try:
                updated = datetime.fromisoformat(note.updated_at)
                if (now - updated).days <= 7:
                    score += 1.0
            except (ValueError, TypeError):
                pass

            # Build a contextual snippet around the first match
            idx = body_lower.find(query_lower)
            if idx == -1 and tokens:
                idx = body_lower.find(tokens[0])
            if idx >= 0:
                start = max(0, idx - 40)
                end = min(len(note.content), idx + _MAX_CONTENT_SNIPPET_CHARS)
                snippet = ("..." if start > 0 else "") + note.content[start:end].strip()
                if end < len(note.content):
                    snippet += "..."
            else:
                snippet = note.content[:_MAX_CONTENT_SNIPPET_CHARS]

            results.append(ContentSearchResult(note=note, score=score, matched_snippet=snippet))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Append content  (feature: non-destructive append to note body)
    # ------------------------------------------------------------------
    def append_note_content(self, note_id: str, text: str) -> Optional[Note]:
        """Append *text* as a new line to a note's content without replacing it."""
        note = self.notes.get(note_id)
        if not note:
            return None
        separator = "\n" if note.content else ""
        note.content = (note.content or "").rstrip() + separator + text
        note.updated_at = datetime.now().isoformat()
        self._save_notes()
        logger.info(f"Appended content to note: {note_id}")
        return note

    # ------------------------------------------------------------------
    # Partial field patch  (feature: update individual fields without
    # touching content — e.g. just change tags or due_date)
    # ------------------------------------------------------------------
    def patch_note_fields(self, note_id: str, patches: Dict[str, Any]) -> Optional[Note]:
        """Apply *patches* dict to allowed mutable fields of a note.

        Allowed keys: title, content, tags, note_type, category, priority,
                      due_date, reminder, pinned, archived.
        """
        _ALLOWED: Final[frozenset] = frozenset({
            "title", "content", "tags", "note_type", "category",
            "priority", "due_date", "reminder", "pinned", "archived",
        })
        note = self.notes.get(note_id)
        if not note:
            return None
        for key, value in patches.items():
            if key in _ALLOWED:
                setattr(note, key, value)
        note.updated_at = datetime.now().isoformat()
        self._save_notes()
        logger.info(f"Patched note {note_id}: {list(patches.keys())}")
        return note

    # ------------------------------------------------------------------
    # Upcoming reminders  (feature: surface notes due within N hours)
    # ------------------------------------------------------------------
    def get_upcoming_reminders(self, hours: int = _UPCOMING_REMINDER_HOURS) -> List[Note]:
        """Return active notes whose due_date or reminder falls within *hours* from now."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        results: List[Note] = []
        for note in self.notes.values():
            if note.archived:
                continue
            for dt_field in (note.due_date, note.reminder):
                if not dt_field:
                    continue
                try:
                    dt = datetime.fromisoformat(dt_field)
                    if now <= dt <= cutoff:
                        results.append(note)
                        break
                except (ValueError, TypeError):
                    pass
        results.sort(key=lambda n: n.due_date or n.reminder or "")
        return results

    # ------------------------------------------------------------------
    # Note linking  (ensure bidirectional, idempotent)
    # ------------------------------------------------------------------
    def link_notes_by_ids(self, note_id_a: str, note_id_b: str) -> bool:
        """Create a bidirectional link between two notes; idempotent."""
        note_a = self.notes.get(note_id_a)
        note_b = self.notes.get(note_id_b)
        if not note_a or not note_b or note_id_a == note_id_b:
            return False
        note_a.related_notes = note_a.related_notes or []
        note_b.related_notes = note_b.related_notes or []
        changed = False
        if note_id_b not in note_a.related_notes:
            note_a.related_notes.append(note_id_b)
            changed = True
        if note_id_a not in note_b.related_notes:
            note_b.related_notes.append(note_id_a)
            changed = True
        if changed:
            now_iso = datetime.now().isoformat()
            note_a.updated_at = now_iso
            note_b.updated_at = now_iso
            self._save_notes()
        return True
    
    def auto_link_related_notes(self, note_id: str, min_shared_keywords: int = 2) -> List[str]:
        """Automatically link a note to other notes with shared keywords/tags.
        
        Returns list of note IDs that were linked.
        """
        note = self.notes.get(note_id)
        if not note:
            return []
        
        # Extract keywords from the note (title + tags + key content words)
        note_keywords = set()
        
        # Add title words (3+ chars)
        note_keywords.update(w.lower() for w in note.title.split() if len(w) >= 3)
        
        # Add tags
        note_keywords.update(t.lower() for t in note.tags)
        
        # Add category
        note_keywords.add(note.category.lower())
        
        # Find other notes with shared keywords
        linked_note_ids = []
        for other_id, other_note in self.notes.items():
            if other_id == note_id or other_note.archived:
                continue
            
            # Skip if already linked
            if note.related_notes and other_id in note.related_notes:
                continue
            
            # Build other note's keywords
            other_keywords = set()
            other_keywords.update(w.lower() for w in other_note.title.split() if len(w) >= 3)
            other_keywords.update(t.lower() for t in other_note.tags)
            other_keywords.add(other_note.category.lower())
            
            # Check for shared keywords
            shared = note_keywords & other_keywords
            if len(shared) >= min_shared_keywords:
                if self.link_notes_by_ids(note_id, other_id):
                    linked_note_ids.append(other_id)
        
        return linked_note_ids
    
    def suggest_tags_via_llm(self, note: Note, max_tags: int = 5) -> List[str]:
        """Use local LLM to suggest tags for a note.
        
        Returns list of suggested tags.
        """
        try:
            # Try to import LLM interface
            from ai.core.llm_engine import LocalLLMEngine
            
            llm = LocalLLMEngine()
            
            prompt = f"""Analyze this note and suggest {max_tags} relevant tags. 
Return ONLY the tags as a comma-separated list, no explanation.

Title: {note.title}
Content: {note.content[:500]}

Tags (comma-separated):"""
            
            response = llm.generate_response(prompt, system_prompt="You are a helpful tag generator.")
            
            # Parse response
            suggested_tags = [
                tag.strip().lower().replace('#', '').replace(' ', '_')
                for tag in response.strip().split(',')
                if tag.strip()
            ]
            
            return suggested_tags[:max_tags]
        except Exception as e:
            logger.debug(f"LLM auto-tagging unavailable: {e}")
            return []
    
    def extract_action_items(self, content: str) -> List[Dict[str, Any]]:
        """Extract potential checklist items from note content.
        
        Returns list of checklist items.
        """
        items = []
        lines = content.split('\n')
        
        # Patterns that suggest action items
        action_patterns = [
            r'^\s*[-*]\s+(.+)$',  # Bullet points
            r'^\s*\d+[\\.)]\s+(.+)$',  # Numbered lists
            r'^\s*\[[ x]\]\s+(.+)$',  # Existing checkboxes
            r'^(?:TODO|TASK|ACTION):\s*(.+)$',  # Explicit TODO markers
            r'^(?:Call|Email|Send|Buy|Finish|Complete|Review|Check)\s+(.+)$',  # Action verbs
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in action_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    text = match.group(1).strip()
                    # Check if already checked   
                    is_checked = '[x]' in line.lower()
                    items.append({
                        "text": text,
                        "checked": is_checked
                    })
                    break
        
        return items
    
    def update_checklist_item(self, note_id: str, item_index: int, checked: bool) -> bool:
        """Update the checked status of a checklist item.
        
        Returns True if successful.
        """
        note = self.notes.get(note_id)
        if not note or not note.checklist_items:
            return False
        
        if 0 <= item_index < len(note.checklist_items):
            self._save_note_version(note)
            note.checklist_items[item_index]["checked"] = checked
            note.updated_at = datetime.now().isoformat()
            self._save_notes()
            return True
        
        return False
    
    def get_tag_analytics(self) -> Dict[str, Any]:
        """Get comprehensive tag usage analytics.
        
        Returns dictionary with tag statistics.
        """
        tag_counts = {}
        tag_notes = {}
        category_counts = {}
        priority_counts = {"low": 0, "medium": 0, "high": 0, "urgent": 0}
        type_counts = {}
        
        active_notes = [n for n in self.notes.values() if not n.archived]
        
        for note in active_notes:
            # Count tags
            for tag in note.tags:
                tag_lower = tag.lower()
                tag_counts[tag_lower] = tag_counts.get(tag_lower, 0) + 1
                if tag_lower not in tag_notes:
                    tag_notes[tag_lower] = []
                tag_notes[tag_lower].append(note.id)
            
            # Count categories
            category_counts[note.category] = category_counts.get(note.category, 0) + 1
            
            # Count priorities
            priority_counts[note.priority] = priority_counts.get(note.priority, 0) + 1
            
            # Count types
            type_counts[note.note_type] = type_counts.get(note.note_type, 0) + 1
        
        # Calculate aging notes (not updated in 30+ days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        aging_notes = [
            note.id for note in active_notes
            if note.updated_at < thirty_days_ago
        ]
        
        # Find most productive tags (tags with most completed checklist items)
        productive_tags = {}
        for tag, note_ids in tag_notes.items():
            completed_items = 0
            total_items = 0
            for note_id in note_ids:
                note = self.notes.get(note_id)
                if note and note.checklist_items:
                    for item in note.checklist_items:
                        total_items += 1
                        if item.get("checked", False):
                            completed_items += 1
            if total_items > 0:
                productive_tags[tag] = {
                    "completion_rate": completed_items / total_items,
                    "completed": completed_items,
                    "total": total_items
                }
        
        return {
            "total_notes": len(active_notes),
            "archived_notes": len([n for n in self.notes.values() if n.archived]),
            "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)),
            "top_tags": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "category_distribution": category_counts,
            "priority_distribution": priority_counts,
            "type_distribution": type_counts,
            "aging_notes_count": len(aging_notes),
            "aging_note_ids": aging_notes,
            "productive_tags": dict(sorted(productive_tags.items(), 
                                           key=lambda x: x[1]["completion_rate"], 
                                           reverse=True)[:5]),
            "total_tags_used": len(tag_counts)
        }


class NotesPlugin(PluginInterface):
    """Plugin for managing notes with search capabilities"""
    
    def __init__(self):
        super().__init__()
        self.manager = NotesManager()
        self.name = "Notes Plugin"
        self.version = "2.0.0"
        self.description = "Advanced note management with pinning, archiving, priorities, and fuzzy search"
        self.enabled = True
        self.capabilities = [
            "create_notes", "search_notes", "list_notes", "tag_management",
            "pin_notes", "archive_notes", "categorize_notes", "priority_management",
            "fuzzy_search", "note_linking", "due_dates"
        ]
        self.commands = [
            "create note", "add note", "new note", "make note",
            "search notes", "find notes", "show notes", "list notes",
            "show note content", "read note", "summarize note",
            "delete note", "remove note", "archive note",
            "edit note", "update note",
            "pin note", "unpin note",
            "set priority", "mark as urgent",
            "categorize note", "set category",
            "show all tags", "list tags",
            "show overdue notes"
        ]
        # Track last accessed note for context-aware operations
        self.last_note_id = None
        self.last_note_title = None
        self.last_note_result_ids: List[str] = []
        self.last_resolution_path = "none"
        self.pending_disambiguation: Optional[Dict[str, Any]] = None

        # Destructive-action confirmation gate
        self.pending_confirmation: Optional[Dict[str, Any]] = None

        # Sliding window of recent conversation turns for "create from context" feature
        self._conversation_context: List[Dict[str, str]] = []  # [{"role": "user"|"assistant", "text": "..."}]

        # Learning and telemetry state (no hardcoded response text path)
        self.learning_state_path = Path("data/notes/notes_learning_state.json")
        self.telemetry_log_path = Path("data/analytics/notes_plugin_telemetry.jsonl")
        self._action_token_weights: Dict[str, Dict[str, float]] = {}
        self._note_selection_weights: Dict[str, float] = {}
        self._load_learning_state()

    def _load_learning_state(self) -> None:
        """Load note action/selection learning weights for reranking."""
        try:
            self.learning_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.learning_state_path.exists():
                self._action_token_weights = {}
                self._note_selection_weights = {}
                return

            with open(self.learning_state_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._action_token_weights = data.get('action_token_weights', {}) or {}
            self._note_selection_weights = data.get('note_selection_weights', {}) or {}
        except Exception as e:
            logger.debug(f"[NotesLearning] Failed to load learning state: {e}")
            self._action_token_weights = {}
            self._note_selection_weights = {}

    def _save_learning_state(self) -> None:
        """Persist learning weights for future ranking decisions."""
        try:
            payload = {
                'action_token_weights': self._action_token_weights,
                'note_selection_weights': self._note_selection_weights,
                'updated_at': datetime.now().isoformat(),
            }
            with open(self.learning_state_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug(f"[NotesLearning] Failed to save learning state: {e}")

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if len(t) > 2]

    def _extract_ordinal_index(self, command: str) -> Optional[int]:
        """Extract 0-based ordinal index from utterances like first/2nd/number 3."""
        words_map = {
            'first': 0, 'second': 1, 'third': 2, 'fourth': 3, 'fifth': 4,
            'sixth': 5, 'seventh': 6, 'eighth': 7, 'ninth': 8, 'tenth': 9,
        }
        lower = command.lower()
        for word, idx in words_map.items():
            if re.search(rf"\b{word}\b", lower):
                return idx

        m = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", lower)
        if m:
            n = int(m.group(1))
            if n >= 1:
                return n - 1
        return None

    def _rank_note_candidates(self, command: str, notes: List[Note]) -> List[Note]:
        """Rank note candidates using token overlap + learned selection weights + recency."""
        cmd_tokens = set(self._tokenize(command))

        def score(note: Note) -> float:
            title_tokens = set(self._tokenize(note.title))
            content_tokens = set(self._tokenize(note.content[:120])) if note.content else set()
            overlap = len(cmd_tokens & title_tokens) * 3.0 + len(cmd_tokens & content_tokens) * 1.0
            learned = sum(self._note_selection_weights.get(tok, 0.0) for tok in title_tokens)
            recency = 0.0
            try:
                recency = datetime.fromisoformat(note.updated_at).timestamp() / 1e10
            except Exception:
                pass
            return overlap + learned + recency

        ranked = sorted(notes, key=score, reverse=True)
        return ranked

    def _normalize_title_query(self, query: str) -> str:
        """Normalize extracted title query for more robust note matching."""
        cleaned = (query or "").strip().strip('"\'').strip('?.!,')
        cleaned = re.sub(
            r"^(?:what(?:'s|\s+is)?\s+(?:in|inside)(?:\s+of)?\s+(?:the\s+)?)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^(?:show\s+(?:me\s+)?)?(?:the\s+)?(?:content|contents)\s+(?:of\s+)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:of\s+)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(?:the|my|this|that)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:note|notes|list|lists)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:for\s+me|please|right\s+now)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _resolve_note_reference(self, command: str, *, include_archived: bool = False) -> Dict[str, Any]:
        """Resolve note reference via context, ordinals, title match, or fallback with disambiguation."""
        lower = command.lower()
        all_notes = self.manager.get_all_notes(include_archived=include_archived)

        # 1) Ordinal from last list result: "second note", "3rd"
        ordinal_index = self._extract_ordinal_index(command)
        if ordinal_index is not None and self.last_note_result_ids:
            if 0 <= ordinal_index < len(self.last_note_result_ids):
                note = self.manager.get_note(self.last_note_result_ids[ordinal_index])
                if note:
                    return {"status": "resolved", "note": note, "resolution_path": "result_set_ordinal"}

        # 2) Pronoun/context references
        if any(token in lower for token in ['last', 'this', 'that', 'it', 'the note']) and self.last_note_id:
            note = self.manager.get_note(self.last_note_id)
            if note:
                return {"status": "resolved", "note": note, "resolution_path": "last_note_context"}

        # 2b) Follow-up fallback: resolve pronouns against last result set.
        if any(token in lower for token in ['last', 'this', 'that', 'it', 'the note']) and self.last_note_result_ids:
            if len(self.last_note_result_ids) == 1:
                note = self.manager.get_note(self.last_note_result_ids[0])
                if note:
                    return {"status": "resolved", "note": note, "resolution_path": "last_result_single"}
            first_result = self.manager.get_note(self.last_note_result_ids[0])
            if first_result:
                return {"status": "resolved", "note": first_result, "resolution_path": "last_result_first"}

        # 2c) Last known title fallback for pronoun follow-ups.
        if any(token in lower for token in ['last', 'this', 'that', 'it', 'the note']) and self.last_note_title:
            title_matches = self.manager.find_by_title(self.last_note_title)
            if len(title_matches) == 1:
                return {"status": "resolved", "note": title_matches[0], "resolution_path": "last_title_context"}

        # 3) Title extraction and matching
        title_patterns = [
            r"(?:what(?:'s|\s+is)?\s+(?:in|inside)(?:\s+of)?\s+(?:the\s+)?)?(.+?)\s+notes?\b",
            r'(?:called|titled|named|about)\s+(.+)$',
            r'(?:note|list)\s+(?:called|titled|named)\s+(.+)$',
            r'(?:the\s+)?(.+?)\s+(?:notes?|lists?)\b',
            r'(?:delete|remove|edit|update|archive|unarchive|pin|unpin|categorize|category|priority|title|name)\s+(?:the\s+)?(.+)$',
        ]
        candidates: List[Note] = []
        matched_query = None
        for pattern in title_patterns:
            m = re.search(pattern, command, re.IGNORECASE)
            if not m:
                continue
            matched_query = self._normalize_title_query(m.group(1))
            if matched_query:
                candidates = self.manager.find_by_title(matched_query)
                if not candidates:
                    contextual_notes = [
                        self.manager.get_note(note_id)
                        for note_id in self.last_note_result_ids
                        if note_id
                    ]
                    contextual_candidates = [
                        note for note in contextual_notes
                        if note and not note.archived and matched_query.lower() in note.title.lower()
                    ]
                    if contextual_candidates:
                        candidates = contextual_candidates
                if candidates:
                    break

        # 4) If no explicit query and only one note exists, use it
        if not candidates and len(all_notes) == 1:
            return {"status": "resolved", "note": all_notes[0], "resolution_path": "single_note_fallback"}

        # 5) Ranking/disambiguation
        if len(candidates) == 1:
            return {"status": "resolved", "note": candidates[0], "resolution_path": "title_exact_or_partial"}
        if len(candidates) > 1:
            ranked = self._rank_note_candidates(command, candidates)
            return {
                "status": "ambiguous",
                "candidates": ranked[:5],
                "resolution_path": "title_ambiguous_ranked",
                "query": matched_query,
            }

        return {"status": "unresolved", "resolution_path": "no_match"}

    def _extract_disambiguation_selection_index(
        self,
        command: str,
        candidate_count: int,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        """Extract candidate index for disambiguation follow-ups by number, tag, or title hint."""
        if not command or candidate_count <= 0:
            return None

        lower = command.lower().strip()
        number_match = re.fullmatch(r"(?:option\s*)?(\d+)", lower)
        if number_match:
            selected = int(number_match.group(1)) - 1
            return selected

        if 'last' in lower and candidate_count > 0:
            return candidate_count - 1

        ordinal_idx = self._extract_ordinal_index(lower)
        if ordinal_idx is not None:
            return ordinal_idx

        if candidates:
            tag_match = re.search(r"(?:tag(?:ged)?\s+|#)([a-z0-9_-]+)", lower)
            if tag_match:
                wanted_tag = tag_match.group(1).lower()
                matched_indices = []
                for idx, candidate in enumerate(candidates):
                    tags = [str(tag).lower() for tag in (candidate.get('tags') or [])]
                    if wanted_tag in tags:
                        matched_indices.append(idx)
                if len(matched_indices) == 1:
                    return matched_indices[0]

            ignore_tokens = {
                'the', 'this', 'that', 'one', 'option', 'note', 'list', 'pick', 'choose',
                'select', 'title', 'tag', 'tagged', 'with', 'please', 'from', 'for'
            }
            command_tokens = {tok for tok in self._tokenize(lower) if tok not in ignore_tokens}
            if command_tokens:
                best_idx = None
                best_score = 0
                tie = False
                for idx, candidate in enumerate(candidates):
                    title_tokens = set(self._tokenize(str(candidate.get('title', ''))))
                    tag_tokens = set(self._tokenize(' '.join(str(tag) for tag in (candidate.get('tags') or []))))
                    candidate_tokens = title_tokens | tag_tokens
                    overlap = len(command_tokens & candidate_tokens)
                    if overlap > best_score:
                        best_score = overlap
                        best_idx = idx
                        tie = False
                    elif overlap == best_score and overlap > 0:
                        tie = True

                if best_idx is not None and best_score > 0 and not tie:
                    return best_idx

        return None

    def _extract_edit_content(self, command: str) -> Optional[str]:
        content_patterns = [
            r'to\s+say\s+(.+)',
            r'to:\s*(.+)',
            r':\s*(.+)',
            r'with\s+(.+)',
        ]
        for pattern in content_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_priority(self, command: str) -> str:
        lower = command.lower()
        if 'urgent' in lower:
            return 'urgent'
        if 'high' in lower:
            return 'high'
        if 'low' in lower:
            return 'low'
        if 'medium' in lower:
            return 'medium'
        return 'medium'

    def _extract_category(self, command: str) -> str:
        category_match = re.search(r'(?:category|categorize)\s+(?:as|to)\s+(\w+)', command, re.IGNORECASE)
        if category_match:
            return category_match.group(1).lower()
        return 'general'

    def _build_note_content_result(self, note: Note, resolution_path: str) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "get_note_content",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "content": note.content,
                "tags": note.tags,
                "note_type": note.note_type,
                "category": note.category,
                "priority": note.priority,
                "created_at": note.created_at,
                "updated_at": note.updated_at,
                "diagnostics": {
                    "resolution_path": resolution_path,
                },
            },
            "formulate": True,
        }

    def _build_note_summary_result(self, note: Note, resolution_path: str) -> Dict[str, Any]:
        text = (note.content or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        key_points: List[str] = []
        action_items: List[str] = []
        dates: List[str] = []

        date_pattern = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2})\b", re.IGNORECASE)
        action_pattern = re.compile(r"^(?:[-*]|\d+[.)]|\[ ?\]|todo\b|action\b|call\b|email\b|buy\b|send\b|finish\b)", re.IGNORECASE)

        for line in lines:
            normalized = re.sub(r"^[-*\d\[\]()\.\s]+", "", line).strip()
            if normalized and len(key_points) < 8:
                key_points.append(normalized[:220])

            if action_pattern.search(line):
                cleaned = re.sub(r"^[-*\d\[\]()\.\s]+", "", line).strip()
                if cleaned:
                    action_items.append(cleaned[:220])

            for match in date_pattern.findall(line):
                if match not in dates:
                    dates.append(match)

        if not key_points and text:
            key_points = [text[:220]]

        return {
            "success": True,
            "action": "summarize_note",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "summary": {
                    "overview": key_points[:3],
                    "key_points": key_points,
                    "action_items": action_items,
                    "dates": dates,
                    "line_count": len(lines),
                },
                "tags": note.tags,
                "note_type": note.note_type,
                "category": note.category,
                "priority": note.priority,
                "created_at": note.created_at,
                "updated_at": note.updated_at,
                "diagnostics": {
                    "resolution_path": resolution_path,
                },
            },
            "formulate": True,
        }

    def _execute_selected_disambiguation(self, action: str, note: Note, original_command: str) -> Dict[str, Any]:
        """Execute a disambiguated action against the selected note."""
        resolution_path = "disambiguation_selection"
        self.last_note_id = note.id
        self.last_note_title = note.title

        if action == "get_note_title":
            return {
                "success": True,
                "action": "get_note_title",
                "data": {
                    "note_id": note.id,
                    "title": note.title,
                    "diagnostics": {"resolution_path": resolution_path},
                },
                "formulate": True,
            }

        if action == "get_note_content":
            return self._build_note_content_result(note, resolution_path)

        if action == "summarize_note":
            return self._build_note_summary_result(note, resolution_path)

        if action == "delete_note":
            success = self.manager.archive_note(note.id)
            if success:
                return {
                    "success": True,
                    "action": "delete_note",
                    "data": {
                        "note_id": note.id,
                        "note_title": note.title,
                        "archived": True,
                        "permanent": False,
                        "restorable": True,
                        "diagnostics": {"resolution_path": resolution_path},
                    },
                    "formulate": True,
                }
            return {
                "success": False,
                "action": "delete_note",
                "data": {
                    "error": "archive_failed",
                    "message_code": "notes:archive_failed",
                    "note_id": note.id,
                    "diagnostics": {"resolution_path": resolution_path},
                },
                "formulate": True,
            }

        if action in ("pin_note", "unpin_note"):
            success = self.manager.unpin_note(note.id) if action == "unpin_note" else self.manager.pin_note(note.id)
            if success:
                return {
                    "success": True,
                    "action": action,
                    "data": {
                        "note_id": note.id,
                        "note_title": note.title,
                        "pinned": action == "pin_note",
                        "diagnostics": {"resolution_path": resolution_path},
                    },
                    "formulate": True,
                }

        if action in ("archive_note", "unarchive_note"):
            success = self.manager.unarchive_note(note.id) if action == "unarchive_note" else self.manager.archive_note(note.id)
            if success:
                return {
                    "success": True,
                    "action": action,
                    "data": {
                        "note_id": note.id,
                        "note_title": note.title,
                        "archived": action == "archive_note",
                        "diagnostics": {"resolution_path": resolution_path},
                    },
                    "formulate": True,
                }

        if action == "set_priority":
            priority = self._extract_priority(original_command)
            if self.manager.set_priority(note.id, priority):
                return {
                    "success": True,
                    "action": "set_priority",
                    "data": {
                        "note_id": note.id,
                        "note_title": note.title,
                        "priority": priority,
                        "diagnostics": {"resolution_path": resolution_path},
                    },
                    "formulate": True,
                }

        if action == "set_category":
            category = self._extract_category(original_command)
            if self.manager.set_category(note.id, category):
                return {
                    "success": True,
                    "action": "set_category",
                    "data": {
                        "note_id": note.id,
                        "note_title": note.title,
                        "category": category,
                        "diagnostics": {"resolution_path": resolution_path},
                    },
                    "formulate": True,
                }

        if action == "edit_note":
            new_content = self._extract_edit_content(original_command)
            if not new_content:
                return {
                    "success": False,
                    "action": "edit_note",
                    "data": {
                        "error": "missing_new_content",
                        "message_code": "notes:edit_missing_content",
                        "note_id": note.id,
                        "note_title": note.title,
                        "diagnostics": {"resolution_path": resolution_path},
                    },
                    "formulate": True,
                }
            self.manager.update_note(note.id, content=new_content)
            return {
                "success": True,
                "action": "edit_note",
                "data": {
                    "note_id": note.id,
                    "note_title": note.title,
                    "new_content": new_content,
                    "diagnostics": {"resolution_path": resolution_path},
                },
                "formulate": True,
            }

        return {
            "success": True,
            "action": "select_note_candidate",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "next_action": action,
                "message_code": "notes:selection_applied",
                "diagnostics": {"resolution_path": resolution_path},
            },
            "formulate": True,
        }

    def _resolve_pending_disambiguation(self, command: str) -> Optional[Dict[str, Any]]:
        """Resolve pending disambiguation if user provides a candidate selection."""
        pending = self.pending_disambiguation
        if not pending:
            return None

        candidates = pending.get("candidates", [])
        action = pending.get("action", "unknown")
        original_command = pending.get("command", "")
        idx = self._extract_disambiguation_selection_index(command, len(candidates), candidates=candidates)
        if idx is None:
            return None

        if idx < 0 or idx >= len(candidates):
            return {
                "success": False,
                "action": action,
                "data": {
                    "error": "invalid_disambiguation_selection",
                    "message_code": "notes:selection_out_of_range",
                    "selected": idx + 1,
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "requires_selection": True,
                    "diagnostics": {"resolution_path": "disambiguation_selection_invalid"},
                },
                "formulate": True,
            }

        selected = candidates[idx]
        note_id = selected.get("note_id")
        note = self.manager.get_note(note_id) if note_id else None
        if not note:
            self.pending_disambiguation = None
            return {
                "success": False,
                "action": action,
                "data": {
                    "error": "selected_note_not_found",
                    "message_code": "notes:selected_note_not_found",
                    "diagnostics": {"resolution_path": "disambiguation_selection_missing_note"},
                },
                "formulate": True,
            }

        result = self._execute_selected_disambiguation(action, note, original_command)
        self.pending_disambiguation = None
        return result

    def _make_disambiguation_result(self, action: str, candidates: List[Note], message_code: str) -> Dict[str, Any]:
        candidate_payload = [
            {
                "option": index + 1,
                "note_id": note.id,
                "title": note.title,
                "updated_at": note.updated_at,
                "tags": note.tags,
            }
            for index, note in enumerate(candidates)
        ]
        self.pending_disambiguation = {
            "action": action,
            "candidates": candidate_payload,
            "command": "",
            "created_at": datetime.now().isoformat(),
        }
        return {
            "success": False,
            "action": action,
            "data": {
                "error": "note_ambiguous",
                "message_code": message_code,
                "candidates": candidate_payload,
                "requires_selection": True,
                "selection_hint": "Reply with a number (e.g., 1), or name/tag hint (e.g., weekend one, tagged work)",
            },
            "formulate": True,
        }

    def _record_learning_outcome(self, command: str, action: str, success: bool, resolution_path: str) -> None:
        """Update token weights for action routing and note selection from outcomes."""
        tokens = self._tokenize(command)
        if not tokens or not action:
            return

        action_weights = self._action_token_weights.setdefault(action, {})
        delta = 0.15 if success else -0.08
        for token in tokens[:20]:
            action_weights[token] = float(action_weights.get(token, 0.0) + delta)
            action_weights[token] = max(-2.0, min(2.0, action_weights[token]))

        if success and resolution_path in ("title_exact_or_partial", "result_set_ordinal", "last_note_context"):
            for token in tokens[:20]:
                self._note_selection_weights[token] = float(self._note_selection_weights.get(token, 0.0) + 0.05)
                self._note_selection_weights[token] = max(-2.0, min(2.0, self._note_selection_weights[token]))

        self._save_learning_state()

    def _update_context_from_result(self, result: Dict[str, Any]) -> None:
        """Keep note context memory synchronized with latest structured plugin result."""
        data = result.get('data', {}) if isinstance(result, dict) else {}
        note_id = data.get('note_id')
        note_title = data.get('note_title') or data.get('title')
        if note_id:
            self.last_note_id = note_id
        if note_title:
            self.last_note_title = note_title

        listed_notes = data.get('notes') if isinstance(data, dict) else None
        if isinstance(listed_notes, list):
            self.last_note_result_ids = [n.get('note_id') for n in listed_notes if isinstance(n, dict) and n.get('note_id')]

        diagnostics = data.get('diagnostics', {}) if isinstance(data, dict) else {}
        if diagnostics.get('resolution_path'):
            self.last_resolution_path = diagnostics['resolution_path']

    def _log_telemetry(self, *, command: str, intent: str, result: Dict[str, Any], resolution_path: str) -> None:
        """Emit notes-plugin telemetry for mining and reranking improvements."""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "intent": intent,
                "command": command[:200],
                "action": result.get('action'),
                "success": bool(result.get('success', False)),
                "resolution_path": resolution_path,
                "error": (result.get('data') or {}).get('error'),
                "message_code": (result.get('data') or {}).get('message_code'),
            }
            with open(self.telemetry_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
            # ---------------------------------------------------------------
            # Feature #6: Feedback loop — bridge telemetry outcome to the
            # AdaptiveContextSelector so notes context is learned over time.
            # ---------------------------------------------------------------
            self._bridge_telemetry_to_context_selector(entry)
        except Exception as e:
            logger.debug(f"[NotesTelemetry] failed: {e}")

    # ------------------------------------------------------------------
    # Feature #6: Feedback loop unification
    # Bridge successful/failed operations back to AdaptiveContextSelector
    # so the "notes" context type gains learned relevance weights.
    # ------------------------------------------------------------------
    def _bridge_telemetry_to_context_selector(self, telemetry_entry: Dict[str, Any]) -> None:
        """Push telemetry outcome into AdaptiveContextSelector as feedback."""
        try:
            from ai.memory.adaptive_context_selector import get_context_selector
            selector = get_context_selector()
            action = telemetry_entry.get("action") or "unknown"
            success = bool(telemetry_entry.get("success", False))
            # Map success → 4-star, failure → 2-star (conservative nudge)
            rating = 4 if success else 2
            # Use a deterministic selection_id derived from the telemetry timestamp+action
            import hashlib
            sid = hashlib.md5(
                f"{telemetry_entry.get('timestamp','')}-{action}".encode()
            ).hexdigest()[:16]
            selector.record_feedback(
                selection_id=sid,
                context_type="notes",
                user_input=telemetry_entry.get("command", "")[:120],
                intent=telemetry_entry.get("intent") or action,
                success=success,
                rating=rating,
            )
        except Exception as e:
            logger.debug(f"[NotesTelemetryBridge] skipped feedback: {e}")

    # ------------------------------------------------------------------
    # Feature #2: NoteContextProvider implementation
    # Satisfies the Protocol → can be registered with _build_llm_context
    # ------------------------------------------------------------------
    def get_note_context_snippet(self, query: str, max_chars: int = 600) -> str:
        """Return a compact context snippet of the most relevant notes for *query*.

        Implements ``NoteContextProvider`` protocol so the LLM context builder
        can inject live note data without knowing plugin internals.
        """
        if not query or not self.manager.notes:
            return ""

        # Use full-text search for the best matches
        results = self.manager.search_by_content(query, limit=5)
        if not results:
            # Fall back to fuzzy title/tag search
            fuzzy = self.manager.fuzzy_search(query)
            if not fuzzy:
                return ""
            results = [
                ContentSearchResult(
                    note=n,
                    score=n.get_relevance_score(query),
                    matched_snippet=n.content[:_MAX_CONTENT_SNIPPET_CHARS],
                )
                for n in fuzzy[:5]
            ]

        lines = ["Relevant notes:"]
        total = 0
        for r in results:
            line = f"- [{r.note.title}] {r.matched_snippet[:120]}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Feature #4: Destructive-action confirmation gate
    # ------------------------------------------------------------------
    def _check_pending_confirmation(self, command: str) -> Optional[Dict[str, Any]]:
        """If there is a pending confirmation request, check if the user confirmed.

        Returns:
            The confirmed action result if user said "confirm", a cancellation
            result if the user said "cancel/no", or *None* if nothing is pending.
        """
        pending = self.pending_confirmation
        if not pending:
            return None

        lower = command.lower().strip()
        # Accept explicit confirmation keywords
        confirmed = any(
            w in lower
            for w in (_CONFIRMATION_KEYWORD, "yes", "do it", "go ahead", "proceed")
        )
        cancelled = any(w in lower for w in ("cancel", "no", "abort", "stop", "nevermind"))

        if confirmed:
            self.pending_confirmation = None
            # Re-dispatch stored action
            stored_command = pending.get("command", "")
            stored_action = pending.get("action", "")
            logger.info(f"[ConfirmationGate] confirmed {stored_action} for: {stored_command[:60]}")
            return self._dispatch_confirmed_destructive(stored_action, pending)

        if cancelled:
            self.pending_confirmation = None
            return {
                "success": True,
                "action": "action_cancelled",
                "data": {
                    "message_code": "notes:action_cancelled",
                    "cancelled_action": pending.get("action"),
                    "note_title": pending.get("note_title"),
                },
                "formulate": True,
            }

        return None  # Still waiting — not a yes/no → fall through to normal routing

    def _dispatch_confirmed_destructive(self, action: str, pending: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the stored destructive action after explicit user confirmation."""
        note_id = pending.get("note_id")
        note_title = pending.get("note_title", "")

        if action == "delete_all_notes":
            return self._execute_confirmed_delete_all()

        note = self.manager.get_note(note_id) if note_id else None
        if not note:
            return {
                "success": False,
                "action": action,
                "data": {
                    "error": "note_not_found_after_confirmation",
                    "message_code": "notes:confirm_note_missing",
                    "note_title": note_title,
                },
                "formulate": True,
            }

        if action in ("delete_note", "archive_note"):
            if self.manager.archive_note(note.id):
                return {
                    "success": True,
                    "action": action,
                    "data": {
                        "note_id": note.id,
                        "note_title": note.title,
                        "archived": True,
                        "permanent": False,
                        "restorable": True,
                        "diagnostics": {"resolution_path": "confirmed_destructive"},
                    },
                    "formulate": True,
                }

        return {
            "success": False,
            "action": action,
            "data": {
                "error": "confirmation_dispatch_failed",
                "message_code": "notes:confirm_dispatch_error",
            },
            "formulate": True,
        }

    def _build_confirmation_request(
        self, action: str, note_id: str, note_title: str, command: str
    ) -> Dict[str, Any]:
        """Gate a destructive action behind an explicit confirmation request."""
        self.pending_confirmation = {
            "action": action,
            "note_id": note_id,
            "note_title": note_title,
            "command": command,
            "created_at": datetime.now().isoformat(),
        }
        return {
            "success": False,
            "action": "requires_confirmation",
            "data": {
                "error": "requires_confirmation",
                "message_code": "notes:confirm_required",
                "pending_action": action,
                "note_title": note_title,
                "prompt": f'This will archive "{note_title}". Reply "confirm" to proceed or "cancel" to abort.',
            },
            "formulate": True,
        }

    # ------------------------------------------------------------------
    # Feature #5: Append-mode note update
    # ------------------------------------------------------------------
    def _append_note(self, command: str) -> Dict[str, Any]:
        """Append text to an existing note without replacing its content."""
        # Extract text to append
        append_patterns = [
            r'(?:append|add to|attach to)\s+(?:note\s+)?(.+?)\s*:\s*(.+)',
            r'(?:append|add to)\s+(?:note\s+)?(.+)',
        ]
        text_to_append: Optional[str] = None
        explicit_title: Optional[str] = None

        for pat in append_patterns:
            m = re.search(pat, command, re.IGNORECASE)
            if m:
                if m.lastindex and m.lastindex >= 2:
                    explicit_title = self._normalize_title_query(m.group(1))
                    text_to_append = m.group(2).strip()
                else:
                    text_to_append = m.group(1).strip()
                break

        # Fallback: "append [text]" or "add [text] to [title]"
        if not text_to_append:
            colon_m = re.search(r':\s*(.+)$', command)
            if colon_m:
                text_to_append = colon_m.group(1).strip()

        if not text_to_append:
            return {
                "success": False,
                "action": "append_note",
                "data": {
                    "error": "missing_append_text",
                    "message_code": "notes:append_missing_text",
                },
                "formulate": True,
            }

        # Resolve target note
        if explicit_title:
            candidates = self.manager.find_by_title(explicit_title)
            if not candidates:
                return {
                    "success": False,
                    "action": "append_note",
                    "data": {
                        "error": "note_not_found",
                        "message_code": "notes:append_target_not_found",
                        "query": explicit_title,
                    },
                    "formulate": True,
                }
            if len(candidates) > 1:
                return self._make_disambiguation_result(
                    "append_note", candidates, "notes:append_ambiguous"
                )
            note = candidates[0]
        elif self.last_note_id:
            note = self.manager.get_note(self.last_note_id)
        else:
            note = None

        if not note:
            return {
                "success": False,
                "action": "append_note",
                "data": {
                    "error": "no_target_note",
                    "message_code": "notes:append_no_target",
                },
                "formulate": True,
            }

        updated = self.manager.append_note_content(note.id, text_to_append)
        if not updated:
            return {
                "success": False,
                "action": "append_note",
                "data": {"error": "append_failed", "message_code": "notes:append_failed"},
                "formulate": True,
            }

        self.last_note_id = note.id
        self.last_note_title = note.title
        return {
            "success": True,
            "action": "append_note",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "appended_text": text_to_append,
                "new_length": len(updated.content or ""),
            },
            "formulate": True,
        }

    # ------------------------------------------------------------------
    # Feature #3: Create note from conversation context
    # ------------------------------------------------------------------
    def record_conversation_turn(self, role: str, text: str) -> None:
        """Keep a sliding window of recent turns (max 10) for "save this" feature."""
        self._conversation_context.append({"role": role, "text": text})
        if len(self._conversation_context) > 10:
            self._conversation_context = self._conversation_context[-10:]

    def _create_note_from_context(self, command: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a note from the recent conversation context.

        Triggers: "save this", "remember that", "make a note of this",
                  "note of what I just said", etc.
        """
        # Assemble source text from recent conversation turns
        source_parts: List[str] = []
        for turn in reversed(self._conversation_context[-6:]):
            if turn.get("role") == "assistant":
                source_parts.insert(0, f"A: {turn['text']}")
            else:
                source_parts.insert(0, f"U: {turn['text']}")

        # Also check the context dict passed from main app
        if context and isinstance(context, dict):
            ctx_text = context.get("last_assistant_response") or context.get("last_response", "")
            if ctx_text and not any(ctx_text in p for p in source_parts):
                source_parts.insert(0, f"A: {ctx_text}")

        if not source_parts:
            return {
                "success": False,
                "action": "create_note_from_context",
                "data": {
                    "error": "no_context_available",
                    "message_code": "notes:context_create_no_context",
                },
                "formulate": True,
            }

        content = "\n".join(source_parts)
        # Extract optional title override from command
        title_m = re.search(
            r'(?:title|call it|name it|titled?)\s+["\']?(.+?)["\']?$', command, re.IGNORECASE
        )
        if title_m:
            title = title_m.group(1).strip()
        else:
            # Auto-title: first 8 words of user's last turn
            user_turns = [t["text"] for t in self._conversation_context if t.get("role") == "user"]
            last_user = user_turns[-1] if user_turns else ""
            words = last_user.split()
            title = " ".join(words[:8])
            if len(words) > 8:
                title += "..."
            if not title:
                title = f"Context note {datetime.now().strftime('%b %d %H:%M')}"

        tags = re.findall(r'#(\w+)', command)
        note = self.manager.create_note(
            title=title,
            content=content,
            tags=tags,
            note_type="general",
        )
        self.last_note_id = note.id
        self.last_note_title = note.title
        return {
            "success": True,
            "action": "create_note_from_context",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "tags": tags,
                "source_turns": len(source_parts),
            },
            "formulate": True,
        }

    # ------------------------------------------------------------------
    # Feature #9: Note linking (improved)
    # ------------------------------------------------------------------
    def _link_notes(self, command: str) -> Dict[str, Any]:
        """Link two notes by extracting two title hints from the command."""
        # Pattern: "link [A] to [B]" / "link [A] and [B]" / "relate [A] to [B]"
        link_patterns = [
            r'(?:link|relate|connect)\s+(.+?)\s+(?:to|with|and)\s+(.+)',
        ]
        title_a: Optional[str] = None
        title_b: Optional[str] = None
        for pat in link_patterns:
            m = re.search(pat, command, re.IGNORECASE)
            if m:
                title_a = self._normalize_title_query(m.group(1))
                title_b = self._normalize_title_query(m.group(2))
                break

        if not title_a or not title_b:
            return {
                "success": False,
                "action": "link_notes",
                "data": {
                    "error": "insufficient_link_arguments",
                    "message_code": "notes:link_requires_two_titles",
                },
                "formulate": True,
            }

        candidates_a = self.manager.find_by_title(title_a)
        candidates_b = self.manager.find_by_title(title_b)

        if not candidates_a:
            return {
                "success": False, "action": "link_notes",
                "data": {"error": "note_a_not_found", "message_code": "notes:link_note_a_missing", "query": title_a},
                "formulate": True,
            }
        if not candidates_b:
            return {
                "success": False, "action": "link_notes",
                "data": {"error": "note_b_not_found", "message_code": "notes:link_note_b_missing", "query": title_b},
                "formulate": True,
            }

        note_a = candidates_a[0]
        note_b = candidates_b[0]

        if self.manager.link_notes_by_ids(note_a.id, note_b.id):
            return {
                "success": True,
                "action": "link_notes",
                "data": {
                    "note_a_id": note_a.id,
                    "note_a_title": note_a.title,
                    "note_b_id": note_b.id,
                    "note_b_title": note_b.title,
                    "bidirectional": True,
                },
                "formulate": True,
            }

        return {
            "success": False, "action": "link_notes",
            "data": {"error": "link_failed", "message_code": "notes:link_failed"},
            "formulate": True,
        }

    # ------------------------------------------------------------------
    # Feature #1: Content search action
    # ------------------------------------------------------------------
    def _search_notes_by_content(self, query: str) -> Dict[str, Any]:
        """Full-text search inside note bodies, return scored results."""
        results = self.manager.search_by_content(query, limit=15)
        if not results:
            return {
                "success": True,
                "action": "search_notes_content",
                "data": {"query": query, "count": 0, "found": False, "results": []},
                "formulate": True,
            }

        self.last_note_result_ids = [r.note.id for r in results]
        if results:
            self.last_note_id = results[0].note.id
            self.last_note_title = results[0].note.title

        return {
            "success": True,
            "action": "search_notes_content",
            "data": {
                "query": query,
                "count": len(results),
                "found": True,
                "results": [r.to_dict() for r in results],
            },
            "formulate": True,
        }

    def initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            logger.info("Initializing Notes Plugin")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Notes Plugin: {e}")
            return False
    
    def shutdown(self):
        """Cleanup when plugin is disabled"""
        logger.info("Shutting down Notes Plugin")
    
    def get_name(self) -> str:
        return self.name
    
    def get_description(self) -> str:
        return self.description
    
    def get_commands(self) -> List[str]:
        return self.commands
    
    def can_handle(self, intent: str = None, entities: Dict = None, command: str = None) -> bool:
        """Check if this plugin can handle the command"""
        # Explicit intent strings - always handle these
        notes_intents = ['notes:create', 'notes:append', 'notes:update', 'notes:list',
                         'notes:search', 'notes:delete', 'notes:edit', 'notes:read']
        if intent and intent.lower() in notes_intents:
            return True

        # Handle both old and new interface
        if command is None and intent:
            command = intent

        if not command:
            return False

        command_lower = command.lower()

        # Do not hijack code/file introspection prompts such as
        # "what does notes_plugin.py do". Those are better handled by
        # general conversation/code explanation logic outside Notes.
        if re.search(r'\b[\w\-]+\.(?:py|js|ts|tsx|jsx|md|json|ya?ml|toml|ini|cfg)\b', command_lower) and re.search(
            r'\b(?:what\s+does|tell\s+me\s+what|explain|describe|how\s+does)\b',
            command_lower,
        ):
            return False
        
        # Keywords that indicate note-related commands
        note_keywords = ['note', 'notes', 'write down', 'remember this', 'jot down']
        action_keywords = [
            'create', 'add', 'new', 'make', 'search', 'find', 'show',
            'list', 'delete', 'remove', 'edit', 'update', 'read', 'open',
            'summary', 'summarize', 'append', 'link', 'connect', 'relate',
            'stats', 'analytics', 'check', 'uncheck', 'toggle', 'suggest',
            'semantic', 'template', 'version', 'restore', 'proactive', 'insights',
        ]

        # Question patterns about notes
        question_patterns = [
            r'how many notes',
            r'do (?:i|we|you) have.*notes?',
            r'(?:do|did)\s+(?:i|we|you)\s+(?:not\s+)?have.*notes?',
            r'have any notes?',
            r'any notes?',
            r'count.*notes?',
            r'number of notes',
            r'what.*\bnotes?\b',
            r'show.*\bnotes?\b',
            r'list.*\bnotes?\b',
        ]

        # Pending confirmation gate — always intercept "confirm" / "cancel" / "yes" / "no"
        if self.pending_confirmation and any(
            w in command_lower
            for w in (_CONFIRMATION_KEYWORD, "yes", "cancel", "no", "abort", "do it", "proceed")
        ):
            return True

        # Check for question patterns
        if self.pending_disambiguation and self._extract_disambiguation_selection_index(
            command_lower,
            len(self.pending_disambiguation.get('candidates', [])),
            candidates=self.pending_disambiguation.get('candidates', []),
        ) is not None:
            return True

        if any(re.search(pattern, command_lower) for pattern in question_patterns):
            return True

        # "save/remember this conversation" patterns
        if re.search(
            r'(?:save|remember|note down|capture|keep)\s+(?:this|that|what i (?:just )?said|this conversation)',
            command_lower,
        ):
            return True
        
        # Check for "add X to list/note" pattern (context-aware update)
        if re.search(r'add\s+.+\s+to\s+(?:the\s+)?(?:list|note)', command_lower):
            return True
        
        # "delete/remove the X list" (e.g. "delete the grocery list") — notes are often called "lists"
        if any(w in command_lower for w in ['delete', 'remove']) and 'list' in command_lower:
            return True

        # Append to note
        if re.search(r'\bappend\b', command_lower) and any(
            w in command_lower for w in ['note', 'to', 'it']
        ):
            return True
        
        # Context-aware follow-up: when we already have a note in context,
        # intercept short pronoun questions like "what is it about?", "what's in it?"
        if self.last_note_id and re.search(
            r'\b(?:it|this|that|the note)\b',
            command_lower,
        ) and re.search(
            r'\babout\b|\bin\b|\binside\b|\bcontains\b|\bsay\b|\bcontent\b|\bcontents\b|\bwhat\b|\bread\b|\bshow\b|\bopen\b',
            command_lower,
        ):
            return True

        # Check if command contains note-related keywords
        has_note_keyword = any(keyword in command_lower for keyword in note_keywords)
        has_action_keyword = any(keyword in command_lower for keyword in action_keywords)
        
        return has_note_keyword or (has_action_keyword and 'note' in command_lower)
    
    def execute(self, intent: str = None, query: str = None, entities: Dict = None, 
                context: Dict = None, command: str = None, **kwargs) -> Dict[str, Any]:
        """Execute a notes command"""
        # Handle both old and new interface
        if command is None:
            command = query or intent
        
        if not command:
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": "no_command",
                    "message_code": "notes:no_command"
                }
            }
        
        try:
            command_lower = command.lower()
            resolution_path = "rule_router"

            # ── Confirmation gate (highest priority) ──────────────────────────
            confirmation_result = self._check_pending_confirmation(command)
            if confirmation_result is not None:
                result = confirmation_result
                resolution_path = "confirmation_gate"
            else:
                result = None
 
            # ── Disambiguation resolution ──────────────────────────────────────
            if result is None:
                pending_resolution = self._resolve_pending_disambiguation(command)
                if pending_resolution is not None:
                    result = pending_resolution
                    resolution_path = "disambiguation_selection"
            
            # Count notes (check first for "how many notes")
            if result is None and re.search(r'how many|count|number of', command_lower) and 'note' in command_lower:
                result = self._count_notes(command)

            # Create note FROM conversation context ("save this", "remember that", etc.)
            elif result is None and re.search(
                r'(?:save|remember|note down|capture|keep)\s+(?:this|that|what i (?:just )?said|this conversation)',
                command_lower
            ):
                result = self._create_note_from_context(command, context)

            # Append to existing note
            elif result is None and re.search(r'\bappend\b', command_lower):
                result = self._append_note(command)
            
            # Add to list/note (context-aware update)
            elif result is None and re.search(r'add\s+.+\s+to\s+(?:the\s+)?(?:list|note)', command_lower):
                result = self._add_to_note(command)

            # Ask for note title/name (context-aware follow-up)
            elif result is None and re.search(r"(?:what(?:'s|\s+is)?\s+the\s+title|title\s+of\s+(?:the\s+)?note|name\s+of\s+(?:the\s+)?note)", command_lower):
                result = self._get_note_title(command)

            # Summarize note content
            elif result is None and (
                (any(word in command_lower for word in ['summarize', 'summary', 'tldr', 'highlights'])
                 and any(word in command_lower for word in ['note', 'it', 'this', 'that', 'one']))
                or re.search(r'what\s+(?:is|are)\s+\S+\s+\S*\s*note\b.*\babout\b', command_lower)
                or re.search(r'(?:tell me|explain)\s+(?:me\s+)?about\s+(?:my|the|this)?\s*\S*\s*note', command_lower)
                # Context follow-up: "what is it about?" when last_note_id is set
                or (self.last_note_id and re.search(r'\b(?:it|this|that)\b.*\babout\b|\babout\b.*\b(?:it|this|that)\b', command_lower)
                    and not re.search(r'\bnote\b', command_lower))
            ):
                result = self._summarize_note_content(command)

            # Read/show full note content
            elif result is None and (
                any(word in command_lower for word in ['read', 'open', 'full content', 'show content'])
                or re.search(r'what(?:\s+is|\s*\'s)\s+in\b', command_lower)
                or re.search(r'what(?:\s+is|\s*\'s)\s+inside(?:\s+of)?\b', command_lower)
                or re.search(r'(?:anything|something|what)\s+in\s+(?:the\s+)?(?:note|it|this|that)', command_lower)
                or re.search(r'(?:wanna|want to)\s+know.*(?:note|in|inside)', command_lower)
                or re.search(r'is there\s+(?:anything|something)\s+in', command_lower)
                or (re.search(r'\bcontents\b', command_lower)
                    and any(w in command_lower for w in ['note', 'it', 'this', 'that', 'one']))
                # Context follow-up: "what's in it?" / "anything in it?" when note context exists
                or (self.last_note_id and re.search(
                    r'\b(?:in|inside|contains?)\b.*\b(?:it|this|that)\b|\b(?:it|this|that)\b.*\b(?:in|inside|contains?)\b',
                    command_lower,
                ) and not re.search(r'\bnote\b', command_lower))
            ) and any(word in command_lower for word in ['note', 'notes', 'it', 'this', 'that', 'one', 'anything', 'something', 'know']):
                result = self._get_note_content(command)
            
            # Show tags (check first before list/show notes)
            elif result is None and 'tag' in command_lower and any(word in command_lower for word in ['show', 'list', 'all']):
                result = self._show_tags()
            
            # Create note
            elif result is None and any(word in command_lower for word in ['create', 'add', 'new', 'make', 'write', 'jot']) and 'note' in command_lower:
                result = self._create_note(command)

            # Content / body search (full-text)
            elif result is None and any(word in command_lower for word in ['search', 'find']) and any(
                word in command_lower for word in ['content', 'body', 'contains', 'inside', 'text of']
            ):
                query_m = re.search(
                    r'(?:search|find)\s+(?:note\s+)?(?:content|body|text)(?:\s+(?:for|containing|with))?\s+(.+)',
                    command, re.IGNORECASE,
                )
                query = query_m.group(1).strip() if query_m else command
                result = self._search_notes_by_content(query)

            # Semantic search (conceptual similarity)
            elif result is None and re.search(r'(semantic|smart|meaning|concept).*(search|find)|(search|find).*(semantic|similar)', command_lower):
                result = self._semantic_search_notes(command)
            
            # Search notes
            elif result is None and any(word in command_lower for word in ['search', 'find']):
                result = self._search_notes(command)
            
            # Check for specific note by name (before list route)
            # "do i have a note called X?", "is there a note named X?", etc.
            elif result is None and re.search(
                r'(?:do|did|is there)\s+(?:i|we|you|there)?\s*(?:have|has)?\s*(?:a|an|the)?\s*note\s+(?:called|named|titled|about)\s+',
                command_lower,
            ):
                result = self._search_notes(command)
            
            # Delete/remove note (check before list — "delete the grocery list" must not match list)
            elif result is None and any(word in command_lower for word in ['delete', 'remove']):
                result = self._delete_note(command)
            
            # List/show notes (also handle "do i/we have notes?", "any notes?")
            elif result is None and (
                any(word in command_lower for word in ['list', 'show', 'all notes', 'my notes'])
                or re.search(r'(?:do|did)\s+(?:i|we|you)\s+(?:not\s+)?have.*notes?', command_lower)
                or re.search(r'have any notes?|any notes?', command_lower)
            ):
                result = self._list_notes(command)
            
            # Edit/update note
            elif result is None and any(word in command_lower for word in ['edit', 'update']):
                result = self._edit_note(command)
            
            # Pin/unpin note
            elif result is None and 'pin' in command_lower:
                result = self._pin_unpin_note(command)
            
            # List archived notes (check before archive/unarchive action)
            elif result is None and (
                any(word in command_lower for word in ['archived notes', 'show archived', 'list archived'])
                or re.search(r'what.*archived|archived.*what|show.*archived|list.*archived', command_lower)
            ):
                result = self._list_archived_notes()
            
            # Archive/unarchive action
            elif result is None and ('archive' in command_lower or 'unarchive' in command_lower):
                result = self._archive_unarchive_note(command)
            
            # Set priority
            elif result is None and ('priority' in command_lower or any(word in command_lower for word in ['urgent', 'urgency', 'important'])):
                result = self._set_priority(command)
            
            # Set category
            elif result is None and ('category' in command_lower or 'categorize' in command_lower):
                result = self._set_category(command)
            
            # Link notes
            elif result is None and ('link' in command_lower or 'relate' in command_lower or 'connect' in command_lower):
                result = self._link_notes(command)

            # Tag analytics / stats
            elif result is None and re.search(r'(?:tag|note).*(?:stats?|analytics?|analysis)', command_lower):
                result = self._get_tag_analytics()
            
            # Auto-link note to related notes
            elif result is None and re.search(r'auto.*link|find.*related|link.*similar', command_lower):
                result = self._auto_link_note(command)

            # Proactive note intelligence (reminders, stale notes, meeting prep)
            elif result is None and re.search(r'proactive|insights|meeting prep|what should i (?:do|review)', command_lower):
                result = self._show_proactive_insights()

            # List note templates
            elif result is None and re.search(r'(list|show).*(template)|(template).*(list|available)', command_lower):
                result = self._list_note_templates()

            # Create note from template
            elif result is None and re.search(r'(create|make|new).*(template)|(template).*(note)', command_lower):
                result = self._create_note_from_template(command)

            # Show note version history
            elif result is None and re.search(r'(show|list|view).*(version|history)|(version history)', command_lower):
                result = self._show_note_versions(command)

            # Restore note version
            elif result is None and re.search(r'(restore|revert).*(version)', command_lower):
                result = self._restore_note_version(command)
            
            # Check/uncheck checklist items
            elif result is None and re.search(r'\\b(check|uncheck|toggle)\\b.*\\b(item|checklist|task)\\b|\\b(item|task)\\s+\\d+', command_lower):
                result = self._check_checklist_item(command)
            
            # Suggest tags for a note
            elif result is None and re.search(r'suggest.*tags?|tag.*suggest', command_lower):
                result = self._suggest_tags_for_note(command)
            
            # Show overdue notes
            elif result is None and 'overdue' in command_lower:
                result = self._show_overdue_notes()
            
            elif result is None:
                result = {
                    "success": False,
                    "action": "notes_unknown_intent",
                    "data": {
                        "error": "unsupported_notes_command",
                        "message_code": "notes:unsupported_command",
                        "command": command,
                    },
                    "formulate": True,
                }
            
            # Ensure response field exists for new interface
            if 'response' not in result:
                result['response'] = result.get('message', '')

            if result.get('data', {}).get('error') == 'note_ambiguous' and self.pending_disambiguation:
                self.pending_disambiguation['command'] = command

            data = result.setdefault('data', {}) if isinstance(result, dict) else {}
            diagnostics = data.setdefault('diagnostics', {}) if isinstance(data, dict) else {}
            if isinstance(diagnostics, dict) and not diagnostics.get('resolution_path'):
                diagnostics['resolution_path'] = resolution_path

            self._update_context_from_result(result)
            action = result.get('action', 'notes_unknown_action')
            success = bool(result.get('success', False))
            resolved_path = diagnostics.get('resolution_path', resolution_path)
            self._record_learning_outcome(command, action, success, resolved_path)
            self._log_telemetry(command=command, intent=intent or "", result=result, resolution_path=resolved_path)
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing notes command: {e}")
            return {
                "success": False,
                "action": "notes_execution_error",
                "response": "",
                "data": {
                    "error": "notes_execution_error",
                    "message_code": "notes:execution_error",
                },
                "formulate": True,
            }
    
    def _create_note(self, command: str) -> Dict[str, Any]:
        """Create a new note from command"""
        # Extract title and content
        # Patterns to handle various note creation commands
        
        content = None
        title = None
        
        # Pattern 1: "create a note and call it X"
        match = re.search(r'(?:create|add|make|new)\s+(?:a\s+)?note\s+and\s+call\s+it\s+(.+)', command, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            content = title  # Use title as content for now
        
        # Pattern 2: "create a note called X"
        if not content:
            match = re.search(r'(?:create|add|make|new)\s+(?:a\s+)?note\s+called\s+(.+)', command, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                content = title
        
        # Pattern 3: "create a note about X" or "create note: X"
        if not content:
            patterns = [
                r'(?:create|add|make|new)\s+(?:a\s+)?note\s+about\s+(.+)',
                r'(?:create|add|make|new)\s+(?:a\s+)?note:\s*(.+)',
                r'(?:create|add|make|new)\s+(?:a\s+)?note\s+(.+)',
                r'(?:write down|jot down|remember)\s+(.+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    break
        
        if not content:
            return {
                "success": False,
                "action": "create_note",
                "data": {
                    "error": "missing_note_content",
                    "message_code": "notes:create_missing_content",
                },
                "formulate": True,
            }
        
        # Extract tags if present (words starting with #)
        tags = re.findall(r'#(\w+)', content)
        # Remove tags from content (keep the text clean)
        content_without_tags = re.sub(r'\s*#\w+\s*', ' ', content).strip()
        
        # If we already have a title (from "call it X" or "called X"), use it
        # Otherwise generate title from first line or first few words
        if not title:
            title_parts = content_without_tags.split('\n', 1)
            if len(title_parts) > 1:
                title = title_parts[0][:50]  # First line, max 50 chars
                content_text = title_parts[1]
            else:
                words = content_without_tags.split()
                title = ' '.join(words[:7])  # First 7 words
                if len(words) > 7:
                    title += '...'
                content_text = content_without_tags
        else:
            # We have a title, use cleaned content as the content text
            content_text = content_without_tags
        
        # Detect note type
        note_type = "general"
        if any(word in command.lower() for word in ['todo', 'task', 'to-do']):
            note_type = "todo"
        elif any(word in command.lower() for word in ['idea', 'brainstorm']):
            note_type = "idea"
        elif any(word in command.lower() for word in ['meeting']):
            note_type = "meeting"
        
        # Detect priority
        priority = "medium"
        if 'urgent' in command.lower() or '!!!' in content:
            priority = "urgent"
        elif 'important' in command.lower() or 'high priority' in command.lower():
            priority = "high"
        elif 'low priority' in command.lower():
            priority = "low"
        
        # Detect category from tags or content
        category = "general"
        if any(tag in ['work', 'job', 'office'] for tag in tags):
            category = "work"
        elif any(tag in ['personal', 'home', 'family'] for tag in tags):
            category = "personal"
        elif any(tag in ['project', 'dev', 'code'] for tag in tags):
            category = "project"
        
        # Create the note
        note = self.manager.create_note(
            title=title,
            content=content_text,
            tags=tags,
            note_type=note_type,
            category=category,
            priority=priority
        )
        
        # Track last created note for context
        self.last_note_id = note.id
        self.last_note_title = note.title

        return {
            "success": True,
            "action": "create_note",
            "data": {
                "title": note.title,
                "note_type": note.note_type,
                "tags": tags,
                "category": category,
                "priority": priority,
                "note_id": note.id
            },
            "formulate": True
        }
    
    def _add_to_note(self, command: str) -> Dict[str, Any]:
        """Add content to an existing note (context-aware)"""
        # Extract what to add: "add X to the list/note"
        match = re.search(r'add\s+(.+?)\s+to\s+(?:the\s+)?(?:list|note)', command, re.IGNORECASE)
        
        if not match:
            return {
                "success": False,
                "action": "add_to_note",
                "data": {
                    "error": "missing_add_content",
                    "message_code": "notes:add_missing_content",
                },
                "formulate": True,
            }
        
        item_to_add = match.group(1).strip()
        
        # Try to determine which note to update
        note_id = None
        note_title = None
        
        # First check if there's a specific note mentioned
        # Pattern: "add X to [note name] list/note"
        note_match = re.search(r'to\s+(?:the\s+)?(.+?)\s+(?:list|note)', command, re.IGNORECASE)
        if note_match:
            potential_title = note_match.group(1).strip()
            # Search for a note with this title
            matching_notes = [n for n in self.manager.notes.values() 
                            if potential_title.lower() in n.title.lower()]
            if matching_notes:
                note_id = matching_notes[0].id
                note_title = matching_notes[0].title
        
        # If no specific note found, use last created/accessed note
        if not note_id and self.last_note_id:
            note_id = self.last_note_id
            note_title = self.last_note_title
        
        # If still no note, return error
        if not note_id:
            return {
                "success": False,
                "action": "add_to_note",
                "data": {
                    "error": "target_note_not_found",
                    "message_code": "notes:add_target_not_found",
                },
                "formulate": True,
            }
        
        # Get the note and append content
        note = self.manager.get_note(note_id)
        if not note:
            return {
                "success": False,
                "action": "add_to_note",
                "data": {
                    "error": "target_note_missing",
                    "message_code": "notes:add_target_missing",
                    "note_title": note_title,
                },
                "formulate": True,
            }
        
        # Append the item to the content
        # If content is empty or same as title, start fresh
        if not note.content or note.content == note.title:
            new_content = f"- {item_to_add}"
        else:
            # Append as a new list item
            new_content = note.content.strip() + f"\n- {item_to_add}"
        
        # Update the note
        self.manager.update_note(note_id, content=new_content)

        # Keep context current so follow-up commands know which note we just touched
        self.last_note_id = note_id
        self.last_note_title = note.title

        return {
            "success": True,
            "action": "add_to_note",
            "data": {
                "title": note.title,
                "note_id": note_id,
                "item_added": item_to_add,
                "new_content": new_content
            },
            "formulate": True
        }
    
    def _search_notes(self, command: str) -> Dict[str, Any]:
        """Search notes based on command"""
        # Extract search query
        patterns = [
            r'(?:search|find)\s+(?:my\s+)?notes?\s+(?:for|about|containing)\s+(.+)',
            r'(?:search|find)\s+(?:my\s+)?notes?\s+(.+)',
        ]
        
        query = None
        for pattern in patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                break
        
        if not query:
            return {
                "success": False,
                "action": "search_notes",
                "data": {
                    "error": "missing_search_query",
                    "message_code": "notes:search_missing_query",
                },
                "formulate": True,
            }
        
        # Check for date range keywords
        results: List[Note] = []
        if any(word in command.lower() for word in ['today', 'yesterday', 'this week', 'last week', 'this month']):
            results = self._search_by_date_keywords(command)
        
        # Check for tag search
        elif query.startswith('#') or 'tagged' in command.lower():
            tag = query.lstrip('#')
            results = self.manager.search_by_tag(tag)
        
        # Default: fuzzy search with relevance scoring
        else:
            results = self.manager.fuzzy_search(query)

        # Feature #1: If no fuzzy results, fall back to full-text content search
        if not results:
            content_results = self.manager.search_by_content(query, limit=10)
            if content_results:
                self.last_note_result_ids = [r.note.id for r in content_results]
                return {
                    "success": True,
                    "action": "search_notes_content",
                    "data": {
                        "query": query,
                        "count": len(content_results),
                        "found": True,
                        "fallback": True,
                        "results": [r.to_dict() for r in content_results],
                    },
                    "formulate": True,
                }
            return {
                "success": True,
                "action": "search_notes_empty",
                "data": {"query": query, "count": 0, "found": False},
                "formulate": True,
            }

        self.last_note_result_ids = [n.id for n in results[:10]]
        return {
            "success": True,
            "action": "search_notes",
            "data": {
                "query": query,
                "count": len(results),
                "found": True,
                "results": [
                    {"note_id": n.id, "title": n.title, "tags": n.tags, "updated_at": n.updated_at}
                    for n in results[:10]
                ],
            },
            "formulate": True,
        }
    
    def _search_by_date_keywords(self, command: str) -> List[Note]:
        """Search by date keywords like 'today', 'last week', etc."""
        now = datetime.now()
        start_date = None
        end_date = now
        
        if 'today' in command.lower():
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif 'yesterday' in command.lower():
            start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif 'this week' in command.lower():
            start_date = now - timedelta(days=now.weekday())
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif 'last week' in command.lower():
            start_date = now - timedelta(days=now.weekday() + 7)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now - timedelta(days=now.weekday())
        elif 'this month' in command.lower():
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        return self.manager.search_by_date_range(start_date, end_date)

    def _extract_list_limit(self, command: str, default_limit: int = 200) -> int:
        """Extract explicit list limit from command text; default to large list for full visibility."""
        match = re.search(r'\b(?:top|last|first|show)\s+(\d+)\b', command.lower())
        if match:
            try:
                return max(1, min(int(match.group(1)), 500))
            except Exception:
                return default_limit
        return default_limit
    
    def _list_notes(self, command: str) -> Dict[str, Any]:
        """List all notes or filter by type"""
        note_type = None
        if 'todo' in command.lower() or 'task' in command.lower():
            note_type = "todo"
        elif 'idea' in command.lower():
            note_type = "idea"
        elif 'meeting' in command.lower():
            note_type = "meeting"
        
        if note_type:
            notes = self.manager.search_by_type(note_type)
        else:
            notes = self.manager.get_all_notes()

        list_limit = self._extract_list_limit(command)
        
        if not notes:
            return {
                "success": True,
                "action": "list_notes",
                "data": {
                    "count": 0,
                    "has_notes": False,
                    "note_type": note_type,
                    "notes": [],
                },
                "formulate": True,
            }
        
        notes_payload = []
        for note in notes[:list_limit]:
            preview = ""
            if note.content and note.content != note.title:
                preview = note.content[:140].replace('\n', ' ')
                if len(note.content) > 140:
                    preview += '...'
            notes_payload.append({
                "note_id": note.id,
                "title": note.title,
                "tags": note.tags,
                "note_type": note.note_type,
                "category": note.category,
                "priority": note.priority,
                "updated_at": note.updated_at,
                "created_at": note.created_at,
                "content_length": len(note.content or ""),
                "preview": preview,
            })

        # Track first listed note for context-aware follow-ups like "what is the title of the note?"
        first_note = notes[0]
        self.last_note_id = first_note.id
        self.last_note_title = first_note.title
        self.last_note_result_ids = [n.get('note_id') for n in notes_payload if n.get('note_id')]

        # ── Feature #7: Surface upcoming reminders and overdue notes ─────────
        upcoming = self.manager.get_upcoming_reminders()
        overdue = self.manager.get_overdue_notes()
        overdue_count = len(overdue)
        upcoming_payload = [
            {
                "note_id": n.id,
                "title": n.title,
                "due_date": n.due_date,
                "reminder": n.reminder,
                "priority": n.priority,
            }
            for n in upcoming[:5]
        ]

        return {
            "success": True,
            "action": "list_notes",
            "data": {
                "count": len(notes),
                "shown": len(notes_payload),
                "has_more": len(notes) > list_limit,
                "limit": list_limit,
                "note_type": note_type,
                "notes": notes_payload,
                "overdue_count": overdue_count,
                "upcoming_reminders": upcoming_payload,
            },
            "formulate": True,
        }

    def _get_note_content(self, command: str) -> Dict[str, Any]:
        """Return full content of a referenced note with clean structured metadata."""
        resolution = self._resolve_note_reference(command, include_archived=True)
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "get_note_content",
                resolution.get("candidates", []),
                "notes:content_ambiguous",
            )

        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            return {
                "success": False,
                "action": "get_note_content",
                "data": {
                    "error": "note_not_identified",
                    "message_code": "notes:content_target_not_found",
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "no_match")
                    },
                },
                "formulate": True,
            }

        self.last_note_id = note.id
        self.last_note_title = note.title

        return {
            "success": True,
            "action": "get_note_content",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "content": note.content,
                "tags": note.tags,
                "note_type": note.note_type,
                "category": note.category,
                "priority": note.priority,
                "created_at": note.created_at,
                "updated_at": note.updated_at,
                "diagnostics": {
                    "resolution_path": resolution.get("resolution_path", "resolved")
                },
            },
            "formulate": True,
        }

    def _summarize_note_content(self, command: str) -> Dict[str, Any]:
        """Generate structured summary of a note for clean downstream formatting."""
        content_res = self._get_note_content(command)
        if not content_res.get("success"):
            content_res["action"] = "summarize_note"
            return content_res

        data = content_res.get("data", {})
        text = (data.get("content") or "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        key_points: List[str] = []
        action_items: List[str] = []
        dates: List[str] = []

        date_pattern = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2})\b", re.IGNORECASE)
        action_pattern = re.compile(r"^(?:[-*]|\d+[.)]|\[ ?\]|todo\b|action\b|call\b|email\b|buy\b|send\b|finish\b)", re.IGNORECASE)

        for line in lines:
            normalized = re.sub(r"^[-*\d\[\]()\.\s]+", "", line).strip()
            if normalized and len(key_points) < 8:
                key_points.append(normalized[:220])

            if action_pattern.search(line):
                cleaned = re.sub(r"^[-*\d\[\]()\.\s]+", "", line).strip()
                if cleaned:
                    action_items.append(cleaned[:220])

            for match in date_pattern.findall(line):
                if match not in dates:
                    dates.append(match)

        if not key_points and text:
            key_points = [text[:220]]

        content_res["action"] = "summarize_note"
        content_res["data"] = {
            "note_id": data.get("note_id"),
            "note_title": data.get("note_title"),
            "summary": {
                "overview": key_points[:3],
                "key_points": key_points,
                "action_items": action_items,
                "dates": dates,
                "line_count": len(lines),
            },
            "tags": data.get("tags", []),
            "note_type": data.get("note_type"),
            "category": data.get("category"),
            "priority": data.get("priority"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "diagnostics": data.get("diagnostics", {}),
        }
        content_res["formulate"] = True
        return content_res

    def _get_note_title(self, command: str) -> Dict[str, Any]:
        """Return the title of a referenced note using context or explicit title mention."""
        resolution = self._resolve_note_reference(command)
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "get_note_title",
                resolution.get("candidates", []),
                "notes:title_ambiguous",
            )

        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            return {
                "success": False,
                "action": "get_note_title",
                "data": {
                    "error": "note_not_identified",
                    "message_code": "notes:title_target_not_found",
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "no_match")
                    },
                },
                "formulate": True,
            }

        self.last_note_id = note.id
        self.last_note_title = note.title

        return {
            "success": True,
            "action": "get_note_title",
            "data": {
                "note_id": note.id,
                "title": note.title,
                "diagnostics": {
                    "resolution_path": resolution.get("resolution_path", "resolved")
                },
            },
            "formulate": True
        }
    
    def _list_archived_notes(self) -> Dict[str, Any]:
        """List all archived notes"""
        archived_notes = [note for note in self.manager.notes.values() if note.archived]
        
        if not archived_notes:
            return {
                "success": True,
                "action": "list_archived_notes",
                "data": {
                    "count": 0,
                    "notes": [],
                },
                "formulate": True,
            }
        
        # Sort by updated date (most recent first)
        archived_notes.sort(key=lambda x: x.updated_at, reverse=True)
        
        archived_payload = []
        for note in archived_notes[:20]:
            preview = ""
            if note.content and note.content != note.title:
                preview = note.content[:100].replace('\n', ' ')
                if len(note.content) > 100:
                    preview += '...'
            archived_payload.append({
                "note_id": note.id,
                "title": note.title,
                "tags": note.tags,
                "note_type": note.note_type,
                "updated_at": note.updated_at,
                "preview": preview,
            })
        
        return {
            "success": True,
            "action": "list_archived_notes",
            "data": {
                "count": len(archived_notes),
                "shown": len(archived_payload),
                "has_more": len(archived_notes) > 20,
                "limit": 20,
                "notes": archived_payload,
            },
            "formulate": True,
        }
    
    def _delete_note(self, command: str) -> Dict[str, Any]:
        """Delete/archive a note"""
        command_lower = command.lower()

        # Pattern 0: "delete all notes" or "delete any notes" - bulk delete
        if re.search(r'delete\s+(?:all|any|every)\s+(?:my\s+)?notes?', command_lower):
            return self._delete_all_notes()

        resolution = self._resolve_note_reference(command)
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "delete_note",
                resolution.get("candidates", []),
                "notes:delete_ambiguous",
            )

        note_to_delete = resolution.get("note") if resolution.get("status") == "resolved" else None
        note_id = note_to_delete.id if note_to_delete else None
        
        if not note_to_delete:
            return {
                "success": False,
                "action": "delete_note",
                "data": {
                    "error": "note_not_identified",
                    "message_code": "notes:delete_target_not_found",
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "no_match")
                    },
                },
                "formulate": True,
            }
        
        # Archive instead of permanently deleting (safer)
        if self.manager.archive_note(note_id):
            return {
                "success": True,
                "action": "delete_note",
                "data": {
                    "note_id": note_id,
                    "note_title": note_to_delete.title,
                    "archived": True,
                    "permanent": False,
                    "restorable": True,
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "resolved")
                    },
                },
                "formulate": True,
            }
        else:
            return {
                "success": False,
                "action": "delete_note",
                "data": {
                    "error": "archive_failed",
                    "message_code": "notes:archive_failed",
                    "note_id": note_id,
                    "note_title": note_to_delete.title,
                },
                "formulate": True,
            }

    def _delete_all_notes(self) -> Dict[str, Any]:
        """Delete/archive all active notes — requires explicit confirmation (confidence guard)."""
        active_notes = [note for note in self.manager.notes.values() if not note.archived]

        if not active_notes:
            return {
                "success": True,
                "action": "delete_notes_empty",
                "data": {"count": 0, "had_notes": False},
                "formulate": True,
            }

        count = len(active_notes)
        # ── Feature #4: Confidence guard for bulk destructive action ──────────
        # Never archive everything silently. Gate behind user confirmation.
        return self._build_confirmation_request(
            action="delete_all_notes",
            note_id="",
            note_title=f"ALL {count} active notes",
            command="delete_all_notes",
        )

    def _execute_confirmed_delete_all(self) -> Dict[str, Any]:
        """Actually archive all notes — only called after confirmation."""
        active_notes = [note for note in self.manager.notes.values() if not note.archived]
        count = len(active_notes)
        for note in active_notes:
            self.manager.archive_note(note.id)
        return {
            "success": True,
            "action": "delete_notes",
            "data": {"count": count, "archived": True, "permanent": False, "restorable": True},
            "formulate": True,
        }


    def _edit_note(self, command: str) -> Dict[str, Any]:
        """Edit a note"""
        resolution = self._resolve_note_reference(command)
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "edit_note",
                resolution.get("candidates", []),
                "notes:edit_ambiguous",
            )

        note_to_edit = resolution.get("note") if resolution.get("status") == "resolved" else None
        note_id = note_to_edit.id if note_to_edit else None
        new_content = None

        # Extract new content regardless of resolution path
        content_patterns = [
            r'to\s+say\s+(.+)',
            r'to:\s*(.+)',
            r':\s*(.+)',
            r'with\s+(.+)',
        ]
        for pattern in content_patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                new_content = match.group(1).strip()
                break
        
        if not note_to_edit:
            return {
                "success": False,
                "action": "edit_note",
                "data": {
                    "error": "note_not_identified",
                    "message_code": "notes:edit_target_not_found",
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "no_match")
                    },
                },
                "formulate": True,
            }
        
        if not new_content:
            return {
                "success": False,
                "action": "edit_note",
                "data": {
                    "error": "missing_new_content",
                    "message_code": "notes:edit_missing_content",
                    "note_id": note_id,
                    "note_title": note_to_edit.title,
                },
                "formulate": True,
            }
        
        # Update the note
        self.manager.update_note(note_id, content=new_content)
        
        return {
            "success": True,
            "action": "edit_note",
            "data": {
                "note_id": note_id,
                "note_title": note_to_edit.title,
                "new_content": new_content,
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "resolved")
                    },
            },
            "formulate": True,
        }
    
    def _pin_unpin_note(self, command: str) -> Dict[str, Any]:
        """Pin or unpin a note"""
        is_unpin = 'unpin' in command.lower()
        resolution = self._resolve_note_reference(command)
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "unpin_note" if is_unpin else "pin_note",
                resolution.get("candidates", []),
                "notes:pin_ambiguous",
            )

        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            return {
                "success": False,
                "action": "unpin_note" if is_unpin else "pin_note",
                "data": {
                    "error": "note_not_identified",
                    "message_code": "notes:pin_target_not_found",
                },
                "formulate": True,
            }
        note_id = note.id
        
        if is_unpin:
            success = self.manager.unpin_note(note_id)
            action = "unpin_note"
        else:
            success = self.manager.pin_note(note_id)
            action = "pin_note"
        
        if success:
            return {
                "success": True,
                "action": action,
                "data": {
                    "note_id": note_id,
                    "note_title": note.title,
                    "pinned": not is_unpin,
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "resolved")
                    },
                },
                "formulate": True,
            }
        return {
            "success": False,
            "action": action,
            "data": {
                "error": "pin_state_update_failed",
                "message_code": "notes:pin_update_failed",
                "note_id": note_id,
                "note_title": note.title,
            },
            "formulate": True,
        }
    
    def _archive_unarchive_note(self, command: str) -> Dict[str, Any]:
        """Archive or unarchive a note"""
        is_unarchive = 'unarchive' in command.lower()
        resolution = self._resolve_note_reference(command, include_archived=True)
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "unarchive_note" if is_unarchive else "archive_note",
                resolution.get("candidates", []),
                "notes:archive_ambiguous",
            )

        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            return {
                "success": False,
                "action": "unarchive_note" if is_unarchive else "archive_note",
                "data": {
                    "error": "note_not_identified",
                    "message_code": "notes:archive_target_not_found",
                },
                "formulate": True,
            }
        note_id = note.id
        
        if is_unarchive:
            success = self.manager.unarchive_note(note_id)
            action = "unarchive_note"
        else:
            success = self.manager.archive_note(note_id)
            action = "archive_note"
        
        if success:
            return {
                "success": True,
                "action": action,
                "data": {
                    "note_id": note_id,
                    "note_title": note.title,
                    "archived": not is_unarchive,
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "resolved")
                    },
                },
                "formulate": True,
            }
        return {
            "success": False,
            "action": action,
            "data": {
                "error": "archive_state_update_failed",
                "message_code": "notes:archive_update_failed",
                "note_id": note_id,
                "note_title": note.title,
            },
            "formulate": True,
        }
    
    def _set_priority(self, command: str) -> Dict[str, Any]:
        """Set priority for a note"""
        # Determine priority level
        priority = "medium"
        if 'urgent' in command.lower():
            priority = "urgent"
        elif 'high' in command.lower():
            priority = "high"
        elif 'low' in command.lower():
            priority = "low"
        elif 'medium' in command.lower():
            priority = "medium"
        
        resolution = self._resolve_note_reference(command)
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "set_priority",
                resolution.get("candidates", []),
                "notes:priority_ambiguous",
            )
        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        note_id = note.id if note else None
        
        if not note:
            return {
                "success": False,
                "action": "set_priority",
                "data": {
                    "error": "note_not_identified",
                    "message_code": "notes:priority_target_not_found",
                },
                "formulate": True,
            }

        if self.manager.set_priority(note_id, priority):
            return {
                "success": True,
                "action": "set_priority",
                "data": {
                    "note_title": note.title,
                    "priority": priority,
                    "note_id": note_id,
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "resolved")
                    },
                },
                "formulate": True
            }
        return {
            "success": False,
            "action": "set_priority",
            "data": {
                "error": "priority_update_failed",
                "message_code": "notes:priority_update_failed",
                "note_id": note_id,
            },
            "formulate": True,
        }
    
    def _set_category(self, command: str) -> Dict[str, Any]:
        """Set category for a note"""
        # Extract category
        category = "general"
        category_match = re.search(r'(?:category|categorize)\s+(?:as|to)\s+(\w+)', command, re.IGNORECASE)
        if category_match:
            category = category_match.group(1).lower()
        
        resolution = self._resolve_note_reference(command)
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "set_category",
                resolution.get("candidates", []),
                "notes:category_ambiguous",
            )
        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        note_id = note.id if note else None
        
        if not note:
            return {
                "success": False,
                "action": "set_category",
                "data": {
                    "error": "note_not_identified",
                    "message_code": "notes:category_target_not_found",
                    "category": category,
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "no_match")
                    },
                },
                "formulate": True,
            }
        
        if self.manager.set_category(note_id, category):
            return {
                "success": True,
                "action": "set_category",
                "data": {
                    "note_id": note_id,
                    "note_title": note.title,
                    "category": category,
                    "diagnostics": {
                        "resolution_path": resolution.get("resolution_path", "resolved")
                    },
                },
                "formulate": True,
            }
        return {
            "success": False,
            "action": "set_category",
            "data": {
                "error": "set_category_failed",
                "message_code": "notes:category_update_failed",
                "note_id": note_id,
                "category": category,
            },
            "formulate": True,
        }
    
    def _show_overdue_notes(self) -> Dict[str, Any]:
        """Show notes with overdue dates"""
        overdue = self.manager.get_overdue_notes()
        
        if not overdue:
            return {
                "success": True,
                "action": "show_overdue_notes",
                "data": {
                    "count": 0,
                    "notes": [],
                },
                "formulate": True,
            }

        overdue_payload = []
        for note in overdue:
            overdue_payload.append({
                "note_id": note.id,
                "title": note.title,
                "tags": note.tags,
                "due_date": note.due_date,
                "priority": note.priority,
            })

        return {
            "success": True,
            "action": "show_overdue_notes",
            "data": {
                "count": len(overdue_payload),
                "notes": overdue_payload,
            },
            "formulate": True,
        }
    
    def _count_notes(self, command: str) -> Dict[str, Any]:
        """Return count of notes based on query"""
        all_notes = self.manager.get_all_notes(include_archived=False)
        total_count = len(all_notes)
        
        # Check for specific filters
        archived = len([n for n in self.manager.notes.values() if n.archived])
        todos = len([n for n in all_notes if n.note_type == "todo"])
        ideas = len([n for n in all_notes if n.note_type == "idea"])
        meetings = len([n for n in all_notes if n.note_type == "meeting"])
        pinned = len([n for n in all_notes if n.pinned])
        
        # Build response
        message = f" **Note Statistics**\n\n"
        message += f"   Total active notes: {total_count}\n"
        
        if todos > 0:
            message += f"    To-do notes: {todos}\n"
        if ideas > 0:
            message += f"    Ideas: {ideas}\n"
        if meetings > 0:
            message += f"    Meeting notes: {meetings}\n"
        if pinned > 0:
            message += f"    Pinned: {pinned}\n"
        if archived > 0:
            message += f"    Archived: {archived}\n"
        
        # Add tag count
        all_tags = self.manager.get_all_tags()

        return {
            "success": True,
            "action": "count_notes",
            "data": {
                "total": total_count,
                "todos": todos,
                "ideas": ideas,
                "meetings": meetings,
                "pinned": pinned,
                "archived": archived,
                "tags_count": len(all_tags)
            },
            "formulate": True
        }
    
    def _get_tag_analytics(self) -> Dict[str, Any]:
        """Show comprehensive tag analytics and statistics"""
        analytics = self.manager.get_tag_analytics()
        
        return {
            "success": True,
            "action": "tag_analytics",
            "data": analytics,
            "formulate": True
        }
    
    def _auto_link_note(self, command: str) -> Dict[str, Any]:
        """Automatically link a note to related notes based on shared keywords"""
        resolution = self._resolve_note_reference(command)
        
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "auto_link_note",
                resolution.get("candidates", []),
                "notes:auto_link_ambiguous",
            )
        
        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            return {
                "success": False,
                "action": "auto_link_note",
                "data": {"error": "note_not_found"},
                "formulate": True
            }
        
        linked_ids = self.manager.auto_link_related_notes(note.id)
        
        return {
            "success": True,
            "action": "auto_link_note",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "linked_count": len(linked_ids),
                "linked_note_ids": linked_ids
            },
            "formulate": True
        }
    
    def _check_checklist_item(self, command: str) -> Dict[str, Any]:
        """Check or uncheck a checklist item in a note"""
        # Extract which item to toggle (by number)
        item_match = re.search(r'(check|uncheck|toggle)\s+(?:item\s+)?(\d+)', command, re.IGNORECASE)
        if not item_match:
            return {
                "success": False,
                "action": "check_checklist_item",
                "data": {"error": "no_item_number"},
                "formulate": True
            }
        
        action = item_match.group(1).lower()
        item_num = int(item_match.group(2)) - 1  # Convert to 0-based index
        
        # Resolve which note
        resolution = self._resolve_note_reference(command)
        
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "check_checklist_item",
                resolution.get("candidates", []),
                "notes:checklist_ambiguous",
            )
        
        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            # Try last_note_id
            if self.last_note_id:
                note = self.manager.get_note(self.last_note_id)
        
        if not note:
            return {
                "success": False,
                "action": "check_checklist_item",
                "data": {"error": "note_not_found"},
                "formulate": True
            }
        
        if not note.checklist_items:
            return {
                "success": False,
                "action": "check_checklist_item",
                "data": {"error": "no_checklist_items", "note_title": note.title},
                "formulate": True
            }
        
        # Determine checked state
        if action == "toggle":
            current_state = note.checklist_items[item_num].get("checked", False)
            checked = not current_state
        else:
            checked = (action == "check")
        
        success = self.manager.update_checklist_item(note.id, item_num, checked)
        
        if success:
            return {
                "success": True,
                "action": "check_checklist_item",
                "data": {
                    "note_id": note.id,
                    "note_title": note.title,
                    "item_index": item_num,
                    "item_text": note.checklist_items[item_num]["text"],
                    "checked": checked
                },
                "formulate": True
            }
        else:
            return {
                "success": False,
                "action": "check_checklist_item",
                "data": {"error": "invalid_item_index"},
                "formulate": True
            }
    
    def _suggest_tags_for_note(self, command: str) -> Dict[str, Any]:
        """Use LLM to suggest tags for a note"""
        resolution = self._resolve_note_reference(command)
        
        if resolution.get("status") == "ambiguous":
            return self._make_disambiguation_result(
                "suggest_tags",
                resolution.get("candidates", []),
                "notes:suggest_tags_ambiguous",
            )
        
        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            return {
                "success": False,
                "action": "suggest_tags",
                "data": {"error": "note_not_found"},
                "formulate": True
            }
        
        suggested_tags = self.manager.suggest_tags_via_llm(note)
        
        return {
            "success": True,
            "action": "suggest_tags",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "current_tags": note.tags,
                "suggested_tags": suggested_tags
            },
            "formulate": True
        }

    def _semantic_search_notes(self, command: str) -> Dict[str, Any]:
        """Run semantic (embedding-based) search across notes."""
        match = re.search(
            r'(?:semantic|smart|meaning|concept).*(?:search|find)\s+(?:for\s+)?(.+)$',
            command,
            re.IGNORECASE,
        )
        query = match.group(1).strip() if match else command
        query = re.sub(r'^(?:semantic|smart|meaning|concept)\s+(?:search|find)\s*', '', query, flags=re.IGNORECASE).strip()

        if not query:
            return {
                "success": False,
                "action": "semantic_search_notes",
                "data": {"error": "missing_query"},
                "formulate": True,
            }

        results = self.manager.semantic_search_notes(query=query, top_k=8)
        return {
            "success": True,
            "action": "semantic_search_notes",
            "data": {
                "query": query,
                "count": len(results),
                "results": results,
            },
            "formulate": True,
        }

    def _show_proactive_insights(self) -> Dict[str, Any]:
        """Return proactive reminders and planning insights from notes."""
        insights = self.manager.get_proactive_insights()
        return {
            "success": True,
            "action": "proactive_notes_insights",
            "data": insights,
            "formulate": True,
        }

    def _list_note_templates(self) -> Dict[str, Any]:
        """List available note templates."""
        templates = self.manager.list_templates()
        return {
            "success": True,
            "action": "list_note_templates",
            "data": {
                "count": len(templates),
                "templates": templates,
            },
            "formulate": True,
        }

    def _create_note_from_template(self, command: str) -> Dict[str, Any]:
        """Create note from template command."""
        template_names = self.manager.list_templates()
        template_name = next((name for name in template_names if name in command.lower()), None)
        if not template_name:
            return {
                "success": False,
                "action": "create_note_from_template",
                "data": {"error": "template_not_identified", "templates": template_names},
                "formulate": True,
            }

        title_match = re.search(r'(?:called|named|title)\s+(.+)$', command, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else None
        created = self.manager.create_note_from_template(template_name=template_name, title=title)

        if not created:
            return {
                "success": False,
                "action": "create_note_from_template",
                "data": {"error": "template_creation_failed", "template": template_name},
                "formulate": True,
            }

        return {
            "success": True,
            "action": "create_note_from_template",
            "data": {
                "template": template_name,
                "note_id": created.id,
                "note_title": created.title,
            },
            "formulate": True,
        }

    def _show_note_versions(self, command: str) -> Dict[str, Any]:
        """Show version history for a note."""
        resolution = self._resolve_note_reference(command, include_archived=True)
        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            return {
                "success": False,
                "action": "show_note_versions",
                "data": {"error": "note_not_found"},
                "formulate": True,
            }

        versions = self.manager.get_note_versions(note.id)
        return {
            "success": True,
            "action": "show_note_versions",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "count": len(versions),
                "versions": versions,
            },
            "formulate": True,
        }

    def _restore_note_version(self, command: str) -> Dict[str, Any]:
        """Restore a note to a selected version index."""
        index_match = re.search(r'(?:version\s+)?(\d+)', command, re.IGNORECASE)
        if not index_match:
            return {
                "success": False,
                "action": "restore_note_version",
                "data": {"error": "missing_version_index"},
                "formulate": True,
            }

        version_index = max(0, int(index_match.group(1)) - 1)
        resolution = self._resolve_note_reference(command, include_archived=True)
        note = resolution.get("note") if resolution.get("status") == "resolved" else None
        if not note:
            return {
                "success": False,
                "action": "restore_note_version",
                "data": {"error": "note_not_found"},
                "formulate": True,
            }

        restored = self.manager.restore_note_version(note.id, version_index)
        return {
            "success": restored,
            "action": "restore_note_version",
            "data": {
                "note_id": note.id,
                "note_title": note.title,
                "version_index": version_index,
                "restored": restored,
            },
            "formulate": True,
        }
    
    def _show_tags(self) -> Dict[str, Any]:
        """Show all tags used in notes"""
        tags = self.manager.get_all_tags()
        
        if not tags:
            return {
                "success": True,
                "action": "show_tags",
                "data": {
                    "count": 0,
                    "tags": [],
                    "tag_counts": {},
                },
                "formulate": True,
            }
        
        # Count notes per tag
        tag_counts = {}
        for tag in tags:
            notes_with_tag = self.manager.search_by_tag(tag)
            tag_counts[tag] = len(notes_with_tag)
        
        # Sort by count (most used first)
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "success": True,
            "action": "show_tags",
            "data": {
                "count": len(tags),
                "tags": tags,
                "tag_counts": tag_counts,
                "sorted_tags": [{"tag": tag, "count": count} for tag, count in sorted_tags],
            },
            "formulate": True,
        }


# For standalone testing
if __name__ == "__main__":
    plugin = NotesPlugin()
    plugin.initialize()
    
    # Test commands
    print("Testing Notes Plugin\n")
    print("=" * 60)
    
    # Create some test notes
    print("\n1. Creating notes...")
    result = plugin.execute(command="Create note about Project Alpha meeting tomorrow #work #important")
    print(f"   {result['message']}")
    
    result = plugin.execute(command="Add note: Buy groceries - milk, eggs, bread #personal")
    print(f"   {result['message']}")
    
    result = plugin.execute(command="Make a note implement user authentication feature #todo #dev")
    print(f"   {result['message']}")
    
    # List notes
    print("\n2. Listing all notes...")
    result = plugin.execute(command="List all notes")
    print(f"   {result['message']}")
    
    # Search by keyword
    print("\n3. Searching notes...")
    result = plugin.execute(command="Search notes for work")
    print(f"   {result['message']}")
    
    # Search by tag
    print("\n4. Searching by tag...")
    result = plugin.execute(command="Search notes for #personal")
    print(f"   {result['message']}")
    
    # Tags
    print("\n5. Showing all tags...")
    result = plugin.execute(command="Show all tags")
    print(f"   {result['message']}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
