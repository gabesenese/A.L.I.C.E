"""
Simple Answer Formatters for A.L.I.C.E Tools
Converts structured tool output to natural language WITHOUT using LLM
"""

from typing import Dict, Any, List, Optional
from datetime import datetime


class SimpleFormatter:
    """Base class for simple formatters"""
    
    @staticmethod
    def format(data: Any, **kwargs) -> Optional[str]:
        """Format data into natural language. Returns None if can't format."""
        raise NotImplementedError


class WeatherFormatter(SimpleFormatter):
    """Format weather data without LLM"""
    
    @staticmethod
    def format(data: Dict[str, Any], **kwargs) -> Optional[str]:
        """
        Format weather data
        
        Expected data:
        {
            'temperature': float,
            'condition': str,
            'humidity': int,
            'location': str
        }
        """
        if not isinstance(data, dict):
            return None
        
        temp = data.get('temperature')
        condition = data.get('condition', 'unknown')
        location = data.get('location', 'your location')
        humidity = data.get('humidity')
        
        parts = []
        
        # Temperature
        if temp is not None:
            parts.append(f"The temperature in {location} is {temp} degrees")
        
        # Condition
        if condition and condition != 'unknown':
            parts.append(f"with {condition}")
        
        # Humidity
        if humidity is not None:
            parts.append(f"Humidity is {humidity}%")
        
        if not parts:
            return None
        
        return ". ".join(parts) + "."


class EmailFormatter(SimpleFormatter):
    """Format email data without LLM"""
    
    @staticmethod
    def format(data: Any, **kwargs) -> Optional[str]:
        """
        Format email list or single email
        
        Expected data for list:
        [
            {'from': str, 'subject': str, 'date': str, 'unread': bool},
            ...
        ]
        """
        if isinstance(data, list):
            return EmailFormatter._format_email_list(data)
        elif isinstance(data, dict):
            return EmailFormatter._format_single_email(data)
        return None
    
    @staticmethod
    def _format_email_list(emails: List[Dict]) -> str:
        """Format list of emails"""
        if not emails:
            return "No emails found."
        
        count = len(emails)
        unread_count = sum(1 for e in emails if e.get('unread', False))
        
        lines = [f"You have {count} email(s)" + (f", {unread_count} unread" if unread_count > 0 else "") + ":"]
        
        for i, email in enumerate(emails[:10], 1):  # Limit to 10
            sender = email.get('from', 'Unknown')
            subject = email.get('subject', 'No subject')
            unread_mark = " [UNREAD]" if email.get('unread') else ""
            lines.append(f"{i}. From {sender}: {subject}{unread_mark}")
        
        if count > 10:
            lines.append(f"... and {count - 10} more")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_single_email(email: Dict) -> str:
        """Format single email"""
        sender = email.get('from', 'Unknown')
        subject = email.get('subject', 'No subject')
        date = email.get('date', 'Unknown date')
        body = email.get('body', '')
        
        lines = [
            f"From: {sender}",
            f"Subject: {subject}",
            f"Date: {date}",
            "",
            body[:500] + ("..." if len(body) > 500 else "")
        ]
        
        return "\n".join(lines)


class CalendarFormatter(SimpleFormatter):
    """Format calendar events without LLM"""
    
    @staticmethod
    def format(data: Any, **kwargs) -> Optional[str]:
        """
        Format calendar events
        
        Expected data:
        [
            {'title': str, 'start': str, 'end': str, 'location': str},
            ...
        ]
        """
        if not isinstance(data, list):
            return None
        
        if not data:
            return "No events scheduled."
        
        count = len(data)
        lines = [f"You have {count} upcoming event(s):"]
        
        for i, event in enumerate(data[:10], 1):
            title = event.get('title', 'Untitled')
            start = event.get('start', '')
            location = event.get('location', '')
            
            event_line = f"{i}. {title}"
            if start:
                event_line += f" at {start}"
            if location:
                event_line += f" ({location})"
            
            lines.append(event_line)
        
        if count > 10:
            lines.append(f"... and {count - 10} more")
        
        return "\n".join(lines)


class FileSearchFormatter(SimpleFormatter):
    """Format file search results without LLM"""
    
    @staticmethod
    def format(data: Any, **kwargs) -> Optional[str]:
        """
        Format file search results
        
        Expected data:
        [
            {'name': str, 'path': str, 'size': int, 'modified': str},
            ...
        ]
        """
        if not isinstance(data, list):
            return None
        
        if not data:
            return "No files found."
        
        count = len(data)
        lines = [f"Found {count} file(s):"]
        
        for i, file in enumerate(data[:10], 1):
            name = file.get('name', 'Unknown')
            path = file.get('path', '')
            size = file.get('size', 0)
            
            # Format size
            size_str = FileSearchFormatter._format_size(size)
            
            lines.append(f"{i}. {name} ({size_str}) - {path}")
        
        if count > 10:
            lines.append(f"... and {count - 10} more")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_size(bytes: int) -> str:
        """Format file size"""
        if bytes < 1024:
            return f"{bytes}B"
        elif bytes < 1024 * 1024:
            return f"{bytes / 1024:.1f}KB"
        elif bytes < 1024 * 1024 * 1024:
            return f"{bytes / (1024 * 1024):.1f}MB"
        else:
            return f"{bytes / (1024 * 1024 * 1024):.1f}GB"


class NotesFormatter(SimpleFormatter):
    """Format notes without LLM"""
    
    @staticmethod
    def format(data: Any, **kwargs) -> Optional[str]:
        """
        Format notes
        
        Expected data:
        [
            {'title': str, 'content': str, 'tags': list, 'created': str},
            ...
        ]
        """
        if not isinstance(data, list):
            # Single note
            if isinstance(data, dict):
                return NotesFormatter._format_single_note(data)
            return None
        
        if not data:
            return "No notes found."
        
        count = len(data)
        lines = [f"Found {count} note(s):"]
        
        for i, note in enumerate(data[:10], 1):
            title = note.get('title', 'Untitled')
            tags = note.get('tags', [])
            created = note.get('created', '')
            
            note_line = f"{i}. {title}"
            if tags:
                note_line += f" [tags: {', '.join(tags[:3])}]"
            if created:
                note_line += f" (created {created})"
            
            lines.append(note_line)
        
        if count > 10:
            lines.append(f"... and {count - 10} more")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_single_note(note: Dict) -> str:
        """Format single note"""
        title = note.get('title', 'Untitled')
        content = note.get('content', '')
        tags = note.get('tags', [])
        created = note.get('created', '')
        
        lines = [f"Note: {title}"]
        if tags:
            lines.append(f"Tags: {', '.join(tags)}")
        if created:
            lines.append(f"Created: {created}")
        lines.append("")
        lines.append(content)
        
        return "\n".join(lines)


class MusicFormatter(SimpleFormatter):
    """Format music playback info without LLM"""
    
    @staticmethod
    def format(data: Any, **kwargs) -> Optional[str]:
        """
        Format music status
        
        Expected data:
        {
            'status': str,  # 'playing', 'paused', 'stopped'
            'track': str,
            'artist': str,
            'duration': int
        }
        """
        if not isinstance(data, dict):
            return None
        
        status = data.get('status', 'unknown')
        track = data.get('track', 'Unknown track')
        artist = data.get('artist', 'Unknown artist')
        
        if status == 'playing':
            return f"Now playing: {track} by {artist}"
        elif status == 'paused':
            return f"Paused: {track} by {artist}"
        elif status == 'stopped':
            return "Music stopped."
        else:
            return f"Music status: {status}"


class RAGFormatter(SimpleFormatter):
    """Format RAG/document query results without LLM"""
    
    @staticmethod
    def format(data: Any, **kwargs) -> Optional[str]:
        """
        Format RAG retrieval results
        
        Expected data:
        [
            {'content': str, 'relevance': float, 'source': str},
            ...
        ]
        """
        if not isinstance(data, list):
            return None
        
        if not data:
            return "No relevant information found in documents."
        
        lines = ["Here's what I found in the documents:"]
        
        for i, result in enumerate(data[:3], 1):  # Top 3 results
            content = result.get('content', '')
            source = result.get('source', 'Unknown source')
            relevance = result.get('relevance', 0.0)
            
            # Truncate content
            if len(content) > 200:
                content = content[:200] + "..."
            
            lines.append(f"\n{i}. {content}")
            lines.append(f"   Source: {source} (relevance: {relevance:.1%})")
        
        return "\n".join(lines)


class FormatterRegistry:
    """Registry of all simple formatters"""
    
    FORMATTERS = {
        'weather': WeatherFormatter,
        'email': EmailFormatter,
        'calendar': CalendarFormatter,
        'file_operations': FileSearchFormatter,
        'notes': NotesFormatter,
        'music': MusicFormatter,
        'documents': RAGFormatter
    }
    
    @staticmethod
    def format(tool_name: str, data: Any, **kwargs) -> Optional[str]:
        """
        Format tool output using appropriate formatter
        
        Args:
            tool_name: Name of tool
            data: Tool output data
            **kwargs: Additional formatting options
        
        Returns:
            Formatted string or None if formatter can't handle it
        """
        formatter_class = FormatterRegistry.FORMATTERS.get(tool_name)
        if not formatter_class:
            return None
        
        try:
            return formatter_class.format(data, **kwargs)
        except Exception as e:
            import logging
            logging.error(f"Formatter error for {tool_name}: {e}")
            return None
    
    @staticmethod
    def can_format(tool_name: str) -> bool:
        """Check if tool has a simple formatter"""
        return tool_name in FormatterRegistry.FORMATTERS
