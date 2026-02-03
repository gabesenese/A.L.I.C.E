"""
Simple Answer Formatters for A.L.I.C.E Tools
Converts structured tool output to natural language WITHOUT using LLM
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


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

        if isinstance(data.get('forecast'), list):
            return WeatherFormatter._format_forecast(data, **kwargs)
        
        temp = data.get('temperature')
        condition = data.get('condition', 'unknown')
        location = data.get('location', 'your location')
        humidity = data.get('humidity')

        parts = []

        # Temperature and condition
        if temp is not None and condition and condition != 'unknown':
            parts.append(f"Weather in {location}: {condition}, {temp}°C")
        elif temp is not None:
            parts.append(f"Temperature in {location}: {temp}°C")
        elif condition and condition != 'unknown':
            parts.append(f"Weather in {location}: {condition}")

        # Humidity
        if humidity is not None:
            parts.append(f"Humidity: {humidity}%")

        if not parts:
            return None

        return "\n".join(parts)

    @staticmethod
    def _format_forecast(data: Dict[str, Any], **kwargs) -> Optional[str]:
        """Format multi-day weather forecast data."""
        forecast = data.get('forecast', [])
        location = data.get('location', 'your location')

        if not forecast:
            return None

        target_day = WeatherFormatter._extract_target_day(kwargs.get("entities"))
        days = forecast[:7]

        if target_day:
            target_date = WeatherFormatter._weekday_to_date(target_day)
            if target_date:
                for day in days:
                    if day.get('date') == target_date:
                        return WeatherFormatter._format_single_day(location, day)

        summary_lines = [f"7-day forecast for {location}:"]

        daily_lines = []
        for day in days:
            date = day.get('date')
            high = day.get('high')
            low = day.get('low')
            condition = day.get('condition', 'unknown')

            if date and high is not None and low is not None:
                daily_lines.append(f"{date}: {condition}, {low}–{high}°C")
            elif date:
                daily_lines.append(f"{date}: {condition}")

        if daily_lines:
            summary_lines.extend(daily_lines)

        return "\n".join(summary_lines)

    @staticmethod
    def _format_single_day(location: str, day: Dict[str, Any]) -> str:
        date = day.get('date')
        high = day.get('high')
        low = day.get('low')
        condition = day.get('condition', 'unknown')

        if date and high is not None and low is not None:
            return f"{location} on {date}: {condition}, {low}–{high}°C"
        if date:
            return f"{location} on {date}: {condition}"
        return f"{location}: {condition}"

    @staticmethod
    def _extract_target_day(entities: Optional[Dict[str, Any]]) -> Optional[str]:
        if not entities:
            return None
        time_entities = entities.get('TIME_RANGE')
        if not time_entities:
            return None
        # Entities may be objects or strings, handle both
        if isinstance(time_entities, str):
            # Single string, return as-is
            return time_entities.lower()
        
        for item in time_entities:
            if item is None:
                continue
            if hasattr(item, 'normalized_value') and item.normalized_value:
                return str(item.normalized_value).lower()
            if hasattr(item, 'value'):
                return str(item.value).lower()
            if isinstance(item, str):
                return item.lower()
        return None

    @staticmethod
    def _weekday_to_date(weekday: str) -> Optional[str]:
        weekdays = {
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5,
            'sunday': 6
        }
        if weekday not in weekdays:
            return None
        today = datetime.now()
        target = weekdays[weekday]
        days_ahead = (target - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        target_date = today + timedelta(days=days_ahead)
        return target_date.strftime('%Y-%m-%d')


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
            # If user requests latest N emails, limit to 5 by default
            limit = kwargs.get('limit', 5)
            return EmailFormatter._format_email_list(data, limit=limit)
        elif isinstance(data, dict):
            return EmailFormatter._format_single_email(data)
        return None
    
    @staticmethod
    def _format_email_list(emails: List[Dict], limit: int = 5) -> str:
        """Format list of emails, showing up to 'limit' emails."""
        if not emails:
            return "No emails found."

        count = len(emails)
        unread_count = sum(1 for e in emails if e.get('unread', False))

        lines = [f"Showing your latest {min(count, limit)} of {count} email(s)" + (f", {unread_count} unread" if unread_count > 0 else "") + ":"]

        for i, email in enumerate(emails[:limit], 1):
            sender = email.get('from', 'Unknown')
            subject = email.get('subject', 'No subject')
            unread_mark = " [UNREAD]" if email.get('unread') else ""
            lines.append(f"{i}. From {sender}: {subject}{unread_mark}")

        if count > limit:
            lines.append(f"... and {count - limit} more")

        return "\n".join(lines)
    
    @staticmethod
    def _format_single_email(email: Dict) -> str:
        """Format single email"""
        import html
        import re

        sender = email.get('from', 'Unknown')
        subject = email.get('subject', 'No subject')
        date = email.get('date', 'Unknown date')
        body = email.get('body', '')

        # Strip HTML if present and normalize whitespace
        if body:
            body = html.unescape(body)
            body = re.sub(r"<\s*br\s*/?>", "\n", body, flags=re.IGNORECASE)
            body = re.sub(r"</p\s*>", "\n\n", body, flags=re.IGNORECASE)
            body = re.sub(r"<[^>]+>", "", body)
            body = re.sub(r"\n{3,}", "\n\n", body)
            body = re.sub(r"[ \t]{2,}", " ", body)
            body = body.strip()

        preview = body[:500] + ("..." if len(body) > 500 else "")

        lines = [
            f"From: {sender}",
            f"Subject: {subject}",
            f"Date: {date}",
            "",
            preview if preview else "(No content)"
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
