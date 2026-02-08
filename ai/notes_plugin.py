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
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import re

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import the proper plugin interface
from ai.plugin_system import PluginInterface

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
        return f"{pin_str}{priority_str}📝 {self.title}{tags_str}\n{self.content}"
    
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
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"note_{timestamp}"
    
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
                except:
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
        # Handle both old and new interface
        if command is None and intent:
            command = intent
        
        if not command:
            return False
        
        command_lower = command.lower()
        
        # Keywords that indicate note-related commands
        note_keywords = ['note', 'notes', 'write down', 'remember this', 'jot down']
        action_keywords = ['create', 'add', 'new', 'make', 'search', 'find', 'show', 
                          'list', 'delete', 'remove', 'edit', 'update']
        
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
            
            # Count notes (check first for "how many notes")
            if re.search(r'how many|count|number of', command_lower) and 'note' in command_lower:
                result = self._count_notes(command)
            
            # Add to list/note (context-aware update)
            elif re.search(r'add\s+.+\s+to\s+(?:the\s+)?(?:list|note)', command_lower):
                result = self._add_to_note(command)
            
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
            elif 'priority' in command_lower or any(word in command_lower for word in ['urgent', 'important']):
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
                    "message": "I'm not sure what you want to do with notes. Try 'create note', 'search notes', or 'list notes'."
                }
            
            # Ensure response field exists for new interface
            if 'response' not in result:
                result['response'] = result.get('message', '')
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing notes command: {e}")
            return {
                "success": False,
                "response": f"Error: {str(e)}",
                "message": f"Error: {str(e)}"
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
                "message": "I couldn't understand what you want to write in the note. Try: 'Create note about project ideas', 'Create a note called shopping list', or 'Add note: Buy groceries'"
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
            "message": f"= Note created: {note.title}",
            "note": note.to_dict()
        }
    
    def _add_to_note(self, command: str) -> Dict[str, Any]:
        """Add content to an existing note (context-aware)"""
        # Extract what to add: "add X to the list/note"
        match = re.search(r'add\s+(.+?)\s+to\s+(?:the\s+)?(?:list|note)', command, re.IGNORECASE)
        
        if not match:
            return {
                "success": False,
                "message": "I couldn't understand what you want to add. Try: 'add eggs to the list'"
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
                "message": "I'm not sure which note to add to. Try creating a note first or specify: 'add eggs to grocery list'"
            }
        
        # Get the note and append content
        note = self.manager.get_note(note_id)
        if not note:
            return {
                "success": False,
                "message": f"I couldn't find the note '{note_title}'"
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
        
        return {
            "success": True,
            "message": f"✅ Added '{item_to_add}' to {note.title}",
            "note_id": note_id
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
                "message": "What would you like to search for? Try: 'Search notes for meeting'"
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
                "message": f"No notes found matching '{query}'",
                "notes": []
            }
        
        # Format results
        notes_text = f"Found {len(results)} note(s):\n\n"
        for i, note in enumerate(results[:10], 1):  # Show max 10 results
            tags_str = f" #{' #'.join(note.tags)}" if note.tags else ""
            notes_text += f"{i}. {note.title}{tags_str}\n"
            notes_text += f"   {note.content[:100]}{'...' if len(note.content) > 100 else ''}\n"
            notes_text += f"   📅 {note.updated_at[:10]}\n\n"
        
        if len(results) > 10:
            notes_text += f"... and {len(results) - 10} more notes"
        
        return {
            "success": True,
            "message": notes_text,
            "notes": [note.to_dict() for note in results]
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
        
        if not notes:
            return {
                "success": True,
                "message": "You don't have any notes yet. Create one with 'Create note about...'",
                "notes": []
            }
        
        # Format output
        notes_text = f"📚 You have {len(notes)} note(s):\n\n"
        for i, note in enumerate(notes[:20], 1):  # Show max 20
            tags_str = f" #{' #'.join(note.tags)}" if note.tags else ""
            icon = "📝" if note.note_type == "general" else "✅" if note.note_type == "todo" else "💡" if note.note_type == "idea" else "👥"
            notes_text += f"{i}. {icon} {note.title}{tags_str}\n"
            # Show content preview if different from title
            if note.content and note.content != note.title:
                preview = note.content[:100].replace('\n', ' ')
                if len(note.content) > 100:
                    preview += '...'
                notes_text += f"   {preview}\n"
            notes_text += f"   📅 {note.updated_at[:10]}\n"
        
        if len(notes) > 20:
            notes_text += f"\n... and {len(notes) - 20} more notes"
        
        return {
            "success": True,
            "message": notes_text,
            "notes": [note.to_dict() for note in notes]
        }
    
    def _list_archived_notes(self) -> Dict[str, Any]:
        """List all archived notes"""
        archived_notes = [note for note in self.manager.notes.values() if note.archived]
        
        if not archived_notes:
            return {
                "success": True,
                "message": "📦 You don't have any archived notes.",
                "notes": []
            }
        
        # Sort by updated date (most recent first)
        archived_notes.sort(key=lambda x: x.updated_at, reverse=True)
        
        # Format output
        notes_text = f"📦 You have {len(archived_notes)} archived note(s):\n\n"
        for i, note in enumerate(archived_notes[:20], 1):  # Show max 20
            tags_str = f" #{' #'.join(note.tags)}" if note.tags else ""
            icon = "📝" if note.note_type == "general" else "✅" if note.note_type == "todo" else "💡" if note.note_type == "idea" else "👥"
            notes_text += f"{i}. {icon} {note.title}{tags_str}\n"
            # Show content preview if different from title
            if note.content and note.content != note.title:
                preview = note.content[:100].replace('\n', ' ')
                if len(note.content) > 100:
                    preview += '...'
                notes_text += f"   {preview}\n"
            notes_text += f"   📅 Archived: {note.updated_at[:10]}\n"
        
        if len(archived_notes) > 20:
            notes_text += f"\n... and {len(archived_notes) - 20} more archived notes"
        
        notes_text += "\n\n💡 Tip: Use 'unarchive note called [title]' to restore a note."
        
        return {
            "success": True,
            "message": notes_text,
            "notes": [note.to_dict() for note in archived_notes]
        }
    
    def _delete_note(self, command: str) -> Dict[str, Any]:
        """Delete/archive a note"""
        # Extract note identifier from command
        note_to_delete = None
        note_id = None
        
        # Pattern 1: "delete last note"
        if 'last' in command.lower() and self.last_note_id:
            note_id = self.last_note_id
            note_to_delete = self.manager.get_note(note_id)
        
        # Pattern 2: "delete note called/titled/about X" or "delete the X" (e.g. "delete the grocery list")
        else:
            patterns = [
                r'delete\s+(?:note\s+)?(?:called|titled|named|about)\s+(.+)',
                r'delete\s+(?:the\s+)?(.+?)\s+note',
                r'remove\s+(?:note\s+)?(?:called|titled|named|about)\s+(.+)',
                r'remove\s+(?:the\s+)?(.+?)\s+note',
                r'(?:delete|remove)\s+(?:the\s+)?(.+)$',  # "delete the grocery list" -> title "grocery list"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    title_query = match.group(1).strip()
                    matching_notes = self.manager.find_by_title(title_query)
                    if matching_notes:
                        note_to_delete = matching_notes[0]
                        note_id = note_to_delete.id
                    break
        
        if not note_to_delete:
            return {
                "success": False,
                "message": "I couldn't identify which note to delete. Try: 'delete last note' or 'delete note called shopping list'"
            }
        
        # Archive instead of permanently deleting (safer)
        if self.manager.archive_note(note_id):
            return {
                "success": True,
                "message": f"✅ Archived note: {note_to_delete.title}\n(Note is archived, not permanently deleted. Use 'unarchive' to restore it)",
                "note_id": note_id
            }
        else:
            return {
                "success": False,
                "message": f"❌ Failed to archive note: {note_to_delete.title}"
            }
    
    def _edit_note(self, command: str) -> Dict[str, Any]:
        """Edit a note"""
        note_to_edit = None
        note_id = None
        new_content = None
        
        # Pattern 1: "edit last note to say X" or "update last note: X"
        if 'last' in command.lower() and self.last_note_id:
            note_id = self.last_note_id
            note_to_edit = self.manager.get_note(note_id)
            
            # Extract new content
            patterns = [
                r'to\s+say\s+(.+)',
                r'to:\s*(.+)',
                r'with\s+(.+)',
                r':\s*(.+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    new_content = match.group(1).strip()
                    break
        
        # Pattern 2: "edit note called X to say Y" or "update shopping list: add milk"
        else:
            # First extract note title
            title_patterns = [
                r'(?:edit|update)\s+(?:note\s+)?(?:called|titled|named)\s+(.+?)\s+(?:to|:|with)',
                r'(?:edit|update)\s+(?:the\s+)?(.+?)\s+note\s+(?:to|:|with)',
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    title_query = match.group(1).strip()
                    matching_notes = self.manager.find_by_title(title_query)
                    if matching_notes:
                        note_to_edit = matching_notes[0]
                        note_id = note_to_edit.id
                    break
            
            # Extract new content
            if note_to_edit:
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
                "message": "I couldn't identify which note to edit. Try: 'edit last note to say...' or 'update shopping list: add eggs'"
            }
        
        if not new_content:
            return {
                "success": False,
                "message": f"What would you like to change in '{note_to_edit.title}'? Try: 'edit {note_to_edit.title} to say: new content'"
            }
        
        # Update the note
        self.manager.update_note(note_id, content=new_content)
        
        return {
            "success": True,
            "message": f"✅ Updated note: {note_to_edit.title}",
            "note_id": note_id
        }
    
    def _show_tags(self) -> Dict[str, Any]:
        """Show all tags used in notes"""
        tags = self.manager.get_all_tags()
        
        if not tags:
            return {
                "success": True,
                "message": "No tags found in your notes. Add tags to notes using #tagname",
                "tags": []
            }
        
        tags_text = f"🏷️ All tags ({len(tags)}):\n"
        tags_text += ', '.join(f"#{tag}" for tag in tags)
        
        return {
            "success": True,
            "message": tags_text,
            "tags": tags
        }
    
    def _pin_unpin_note(self, command: str) -> Dict[str, Any]:
        """Pin or unpin a note"""
        is_unpin = 'unpin' in command.lower()
        note_id = None
        note = None
        
        # Check for "last note"
        if 'last' in command.lower() and self.last_note_id:
            note_id = self.last_note_id
            note = self.manager.get_note(note_id)
        else:
            # Try to find by title
            patterns = [
                r'(?:pin|unpin)\s+(?:note\s+)?(?:called|titled)\s+(.+)',
                r'(?:pin|unpin)\s+(?:the\s+)?(.+?)\s+note',
            ]
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    title_query = match.group(1).strip()
                    matching_notes = self.manager.find_by_title(title_query)
                    if matching_notes:
                        note = matching_notes[0]
                        note_id = note.id
                    break
        
        if not note:
            return {
                "success": False,
                "message": "I couldn't identify which note to pin/unpin. Try: 'pin last note' or 'pin note called shopping list'"
            }
        
        if is_unpin:
            success = self.manager.unpin_note(note_id)
            action = "Unpinned"
        else:
            success = self.manager.pin_note(note_id)
            action = "Pinned"
        
        if success:
            return {
                "success": True,
                "message": f"✅ {action} note: {note.title}"
            }
        return {
            "success": False,
            "message": f"❌ Failed to {action.lower()} note"
        }
    
    def _archive_unarchive_note(self, command: str) -> Dict[str, Any]:
        """Archive or unarchive a note"""
        is_unarchive = 'unarchive' in command.lower()
        note_id = None
        note = None
        
        # Check for "last note"
        if 'last' in command.lower() and self.last_note_id:
            note_id = self.last_note_id
            note = self.manager.get_note(note_id)
        else:
            # Try to find by title
            patterns = [
                r'(?:archive|unarchive)\s+(?:note\s+)?(?:called|titled)\s+(.+)',
                r'(?:archive|unarchive)\s+(?:the\s+)?(.+?)\s+note',
            ]
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    title_query = match.group(1).strip()
                    matching_notes = self.manager.find_by_title(title_query)
                    if matching_notes:
                        note = matching_notes[0]
                        note_id = note.id
                    break
        
        if not note:
            return {
                "success": False,
                "message": "I couldn't identify which note to archive/unarchive."
            }
        
        if is_unarchive:
            success = self.manager.unarchive_note(note_id)
            action = "Unarchived"
        else:
            success = self.manager.archive_note(note_id)
            action = "Archived"
        
        if success:
            return {
                "success": True,
                "message": f"✅ {action} note: {note.title}"
            }
        return {
            "success": False,
            "message": f"❌ Failed to {action.lower()} note"
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
        
        # Find the note
        note_id = None
        note = None
        
        if 'last' in command.lower() and self.last_note_id:
            note_id = self.last_note_id
            note = self.manager.get_note(note_id)
        else:
            # Extract note title
            patterns = [
                r'(?:set|mark)\s+(.+?)\s+as\s+(?:urgent|high|low|medium)',
                r'(?:priority|prioritize)\s+(.+?)\s+(?:as|to)\s+(?:urgent|high|low|medium)',
            ]
            for pattern in patterns:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    title_query = match.group(1).strip().replace('note called', '').replace('note', '').strip()
                    matching_notes = self.manager.find_by_title(title_query)
                    if matching_notes:
                        note = matching_notes[0]
                        note_id = note.id
                    break
        
        if not note:
            return {
                "success": False,
                "message": "I couldn't identify which note to set priority for."
            }
        
        if self.manager.set_priority(note_id, priority):
            priority_icons = {"low": "🔵", "medium": "🟡", "high": "🟠", "urgent": "🔴"}
            return {
                "success": True,
                "message": f"✅ Set priority to {priority_icons[priority]} {priority} for: {note.title}"
            }
        return {
            "success": False,
            "message": "❌ Failed to set priority"
        }
    
    def _set_category(self, command: str) -> Dict[str, Any]:
        """Set category for a note"""
        # Extract category
        category = "general"
        category_match = re.search(r'(?:category|categorize)\s+(?:as|to)\s+(\w+)', command, re.IGNORECASE)
        if category_match:
            category = category_match.group(1).lower()
        
        # Find the note
        note_id = None
        note = None
        
        if 'last' in command.lower() and self.last_note_id:
            note_id = self.last_note_id
            note = self.manager.get_note(note_id)
        
        if not note:
            return {
                "success": False,
                "message": "I couldn't identify which note to categorize."
            }
        
        if self.manager.set_category(note_id, category):
            return {
                "success": True,
                "message": f"✅ Set category to '{category}' for: {note.title}"
            }
        return {
            "success": False,
            "message": "❌ Failed to set category"
        }
    
    def _link_notes(self, command: str) -> Dict[str, Any]:
        """Link related notes together"""
        # This is a simplified version - could be enhanced
        return {
            "success": False,
            "message": "Note linking feature is available but requires specifying both note titles. Try: 'link note A to note B'"
        }
    
    def _show_overdue_notes(self) -> Dict[str, Any]:
        """Show notes with overdue dates"""
        overdue = self.manager.get_overdue_notes()
        
        if not overdue:
            return {
                "success": True,
                "message": "✅ No overdue notes!",
                "notes": []
            }
        
        notes_text = f"⚠️ You have {len(overdue)} overdue note(s):\n\n"
        for i, note in enumerate(overdue, 1):
            tags_str = f" #{' #'.join(note.tags)}" if note.tags else ""
            notes_text += f"{i}. {note.title}{tags_str}\n"
            notes_text += f"   Due: {note.due_date[:10]}\n\n"
        
        return {
            "success": True,
            "message": notes_text,
            "notes": [note.to_dict() for note in overdue]
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
        message += f"   📝 Total active notes: {total_count}\n"
        
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
        if all_tags:
            message += f"\n   🏷️ Total tags: {len(all_tags)}\n"
        
        return {
            "success": True,
            "message": message,
            "count": total_count,
            "stats": {
                "total": total_count,
                "todos": todos,
                "ideas": ideas,
                "meetings": meetings,
                "pinned": pinned,
                "archived": archived,
                "tags": len(all_tags)
            }
        }
    
    def _show_tags(self) -> Dict[str, Any]:
        """Show all tags used in notes"""
        tags = self.manager.get_all_tags()
        
        if not tags:
            return {
                "success": True,
                "message": "You haven't used any tags in your notes yet. Add tags with #tagname when creating notes!",
                "tags": []
            }
        
        # Count notes per tag
        tag_counts = {}
        for tag in tags:
            notes_with_tag = self.manager.search_by_tag(tag)
            tag_counts[tag] = len(notes_with_tag)
        
        # Sort by count (most used first)
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        message = f"🏷️ **All Tags** ({len(tags)} total)\n\n"
        for tag, count in sorted_tags:
            message += f"   #{tag} ({count} note{'s' if count != 1 else ''})\n"
        
        return {
            "success": True,
            "message": message,
            "tags": tags,
            "tag_counts": tag_counts
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
