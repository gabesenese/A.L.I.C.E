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
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
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
            'related_notes': None
        }
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        return Note(**data)
    
    def __str__(self) -> str:
        tags_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        pin_str = "📌 " if self.pinned else ""
        priority_icons = {"low": "🔵", "medium": "🟡", "high": "🟠", "urgent": "🔴"}
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
        """Calculate relevance score for fuzzy search"""
        keyword_lower = keyword.lower()
        score = 0.0
        
        # Title match (highest weight)
        if keyword_lower in self.title.lower():
            score += 10.0
            # Bonus for exact match
            if keyword_lower == self.title.lower():
                score += 5.0
        
        # Content match
        if keyword_lower in self.content.lower():
            score += 5.0
        
        # Tag match
        if any(keyword_lower in tag.lower() for tag in self.tags):
            score += 7.0
        
        # Category match
        if keyword_lower in self.category.lower():
            score += 3.0
        
        # Partial word matching
        title_words = self.title.lower().split()
        content_words = self.content.lower().split()
        for word in title_words:
            if keyword_lower in word or word in keyword_lower:
                score += 2.0
        for word in content_words[:50]:  # Check first 50 words
            if keyword_lower in word or word in keyword_lower:
                score += 0.5
        
        # Boost for pinned and high priority
        if self.pinned:
            score *= 1.2
        if self.priority == "urgent":
            score *= 1.15
        elif self.priority == "high":
            score *= 1.1
        
        return score


class NotesManager:
    """Handles note storage and retrieval"""
    
    def __init__(self, notes_dir: str = "data/notes"):
        self.notes_dir = notes_dir
        self.notes_file = os.path.join(notes_dir, "notes.json")
        self.notes: Dict[str, Note] = {}
        self._ensure_directory()
        self._load_notes()
    
    def _ensure_directory(self):
        """Ensure notes directory exists"""
        os.makedirs(self.notes_dir, exist_ok=True)
    
    def _load_notes(self):
        """Load notes from JSON file"""
        try:
            if os.path.exists(self.notes_file):
                with open(self.notes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.notes = {
                        note_id: Note.from_dict(note_data)
                        for note_id, note_data in data.items()
                    }
                logger.info(f"Loaded {len(self.notes)} notes from storage")
        except Exception as e:
            logger.error(f"Error loading notes: {e}")
            self.notes = {}
    
    def _save_notes(self):
        """Save notes to JSON file and human-readable text file"""
        try:
            # Save JSON data file
            data = {
                note_id: note.to_dict()
                for note_id, note in self.notes.items()
            }
            with open(self.notes_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save human-readable text file
            self._save_readable_notes()
            
            logger.info(f"Saved {len(self.notes)} notes to storage")
        except Exception as e:
            logger.error(f"Error saving notes: {e}")
    
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
    
    def _write_note_to_file(self, f, note: Note):
        """Write a single note to file with proper formatting"""
        # Status indicators
        status = []
        if note.pinned:
            status.append("📌 PINNED")
        if note.priority == "urgent":
            status.append("🔴 URGENT")
        elif note.priority == "high":
            status.append("🟠 HIGH PRIORITY")
        elif note.priority == "low":
            status.append("🔵 LOW PRIORITY")
        
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
                checkbox = "☑" if item.get("checked", False) else "☐"
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
        cleaned = re.sub(r"^(?:the|my|this|that)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?:note|list)\b", "", cleaned, flags=re.IGNORECASE)
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

        # 3) Title extraction and matching
        title_patterns = [
            r'(?:called|titled|named|about)\s+(.+)$',
            r'(?:note|list)\s+(?:called|titled|named)\s+(.+)$',
            r'(?:the\s+)?(.+?)\s+(?:note|list)\b',
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

    def _make_disambiguation_result(self, action: str, candidates: List[Note], message_code: str) -> Dict[str, Any]:
        return {
            "success": False,
            "action": action,
            "data": {
                "error": "note_ambiguous",
                "message_code": message_code,
                "candidates": [
                    {
                        "note_id": n.id,
                        "title": n.title,
                        "updated_at": n.updated_at,
                        "tags": n.tags,
                    }
                    for n in candidates
                ],
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
        except Exception as e:
            logger.debug(f"[NotesTelemetry] failed: {e}")
    
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
        
        # Keywords that indicate note-related commands
        note_keywords = ['note', 'notes', 'write down', 'remember this', 'jot down']
        action_keywords = ['create', 'add', 'new', 'make', 'search', 'find', 'show', 
                          'list', 'delete', 'remove', 'edit', 'update', 'read', 'open', 'summary', 'summarize']
        
        # Question patterns about notes
        question_patterns = [
            r'how many notes',
            r'do i have.*notes?',
            r'count.*notes?',
            r'number of notes',
            r'what.*notes',
            r'show.*notes',
            r'list.*notes'
        ]
        
        # Check for question patterns
        if any(re.search(pattern, command_lower) for pattern in question_patterns):
            return True
        
        # Check for "add X to list/note" pattern (context-aware update)
        if re.search(r'add\s+.+\s+to\s+(?:the\s+)?(?:list|note)', command_lower):
            return True
        
        # "delete/remove the X list" (e.g. "delete the grocery list") — notes are often called "lists"
        if any(w in command_lower for w in ['delete', 'remove']) and 'list' in command_lower:
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
            
            # Count notes (check first for "how many notes")
            if re.search(r'how many|count|number of', command_lower) and 'note' in command_lower:
                result = self._count_notes(command)
            
            # Add to list/note (context-aware update)
            elif re.search(r'add\s+.+\s+to\s+(?:the\s+)?(?:list|note)', command_lower):
                result = self._add_to_note(command)

            # Ask for note title/name (context-aware follow-up)
            elif re.search(r"(?:what(?:'s|\s+is)?\s+the\s+title|title\s+of\s+(?:the\s+)?note|name\s+of\s+(?:the\s+)?note)", command_lower):
                result = self._get_note_title(command)

            # Summarize note content
            elif any(word in command_lower for word in ['summarize', 'summary', 'tldr', 'highlights']) and any(word in command_lower for word in ['note', 'it', 'this', 'that', 'one']):
                result = self._summarize_note_content(command)

            # Read/show full note content
            elif any(word in command_lower for word in ['read', 'open', 'full content', 'show content', 'what is in']) and any(word in command_lower for word in ['note', 'it', 'this', 'that', 'one']):
                result = self._get_note_content(command)
            
            # Show tags (check first before list/show notes)
            elif 'tag' in command_lower and any(word in command_lower for word in ['show', 'list', 'all']):
                result = self._show_tags()
            
            # Create note
            elif any(word in command_lower for word in ['create', 'add', 'new', 'make', 'write', 'jot']) and 'note' in command_lower:
                result = self._create_note(command)
            
            # Search notes
            elif any(word in command_lower for word in ['search', 'find']):
                result = self._search_notes(command)
            
            # Delete/remove note (check before list — "delete the grocery list" must not match list)
            elif any(word in command_lower for word in ['delete', 'remove']):
                result = self._delete_note(command)
            
            # List/show notes (also handle "do i have notes")
            elif any(word in command_lower for word in ['list', 'show', 'all notes', 'my notes', 'do i have']):
                result = self._list_notes(command)
            
            # Edit/update note
            elif any(word in command_lower for word in ['edit', 'update']):
                result = self._edit_note(command)
            
            # Pin/unpin note
            elif 'pin' in command_lower:
                result = self._pin_unpin_note(command)
            
            # List archived notes (check before archive/unarchive action)
            elif (any(word in command_lower for word in ['archived notes', 'show archived', 'list archived']) or
                  re.search(r'what.*archived|archived.*what|show.*archived|list.*archived', command_lower)):
                result = self._list_archived_notes()
            
            # Archive/unarchive action
            elif 'archive' in command_lower or 'unarchive' in command_lower:
                result = self._archive_unarchive_note(command)
            
            # Set priority
            elif 'priority' in command_lower or any(word in command_lower for word in ['urgent', 'urgency', 'important']):
                result = self._set_priority(command)
            
            # Set category
            elif 'category' in command_lower or 'categorize' in command_lower:
                result = self._set_category(command)
            
            # Link notes
            elif 'link' in command_lower or 'relate' in command_lower:
                result = self._link_notes(command)
            
            # Show overdue notes
            elif 'overdue' in command_lower:
                result = self._show_overdue_notes()
            
            else:
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
        results = []
        if any(word in command.lower() for word in ['today', 'yesterday', 'this week', 'last week', 'this month']):
            results = self._search_by_date_keywords(command)
        
        # Check for tag search
        elif query.startswith('#') or 'tagged' in command.lower():
            tag = query.lstrip('#')
            results = self.manager.search_by_tag(tag)
        
        # Default: fuzzy search with relevance scoring
        else:
            results = self.manager.fuzzy_search(query)
        
        if not results:
            return {
                "success": True,
                "action": "search_notes_empty",
                "data": {
                    "query": query,
                    "count": 0,
                    "found": False
                },
                "formulate": True
            }

        return {
            "success": True,
            "action": "search_notes",
            "data": {
                "query": query,
                "count": len(results),
                "found": True,
                "results": [{"title": n.title, "tags": n.tags} for n in results[:10]]
            },
            "formulate": True
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
        """Delete/archive all active notes"""
        active_notes = [note for note in self.manager.notes.values() if not note.archived]

        if not active_notes:
            return {
                "success": True,
                "action": "delete_notes_empty",
                "data": {
                    "count": 0,
                    "had_notes": False
                },
                "formulate": True
            }

        count = len(active_notes)

        # Archive all active notes
        for note in active_notes:
            self.manager.archive_note(note.id)

        return {
            "success": True,
            "action": "delete_notes",
            "data": {
                "count": count,
                "archived": True,
                "permanent": False,
                "restorable": True
            },
            "formulate": True
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
    
    def _link_notes(self, command: str) -> Dict[str, Any]:
        """Link related notes together"""
        # This is a simplified version - could be enhanced
        return {
            "success": False,
            "action": "link_notes",
            "data": {
                "error": "insufficient_link_arguments",
                "message_code": "notes:link_requires_two_titles",
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
        message = f"📊 **Note Statistics**\n\n"
        message += f"   Total active notes: {total_count}\n"
        
        if todos > 0:
            message += f"   ✅ To-do notes: {todos}\n"
        if ideas > 0:
            message += f"   💡 Ideas: {ideas}\n"
        if meetings > 0:
            message += f"   👥 Meeting notes: {meetings}\n"
        if pinned > 0:
            message += f"   📌 Pinned: {pinned}\n"
        if archived > 0:
            message += f"   📦 Archived: {archived}\n"
        
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
