"""
Personal Events Tracker for A.L.I.C.E
Detects and stores important personal dates: birthdays, anniversaries, etc.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PersonalEvent:
    """A personal event like birthday or anniversary"""
    event_type: str  # birthday, anniversary, deadline, etc.
    date: str  # YYYY-MM-DD format
    description: str
    person: Optional[str] = None  # Who the event is for (user, family member, etc.)
    recurring: bool = True  # Does it repeat annually
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict):
        return PersonalEvent(**data)


class PersonalEventsDetector:
    """Detects personal events from natural language"""

    # Patterns for birthday detection
    BIRTHDAY_PATTERNS = [
        r"(?:my |our )?birthday (?:is |on |in |'s )?(?:on |in )?(\w+day|\w+ \d+|\d+[a-z]{2} \w+|\w+ \d+[a-z]{2})",
   r"(\w+day|\w+ \d+|\d+[a-z]{2} \w+|\w+ \d+[a-z]{2}) (?:is |'s )?(?:my |our )?birthday",
        r"it(?:'s| is) my birthday (?:on |in )?(\w+day|\w+ \d+|\d+[a-z]{2} \w+)?",
        r"i (?:turn|am turning|will be) \d+ (?:on |in )?(\w+day|\w+ \d+)",
    ]

    # Patterns for anniversary
    ANNIVERSARY_PATTERNS = [
        r"(?:my |our )?anniversary (?:is |on |in )?(\w+day|\w+ \d+)",
        r"(\w+day|\w+ \d+) (?:is |'s )?(?:my |our )?anniversary",
    ]

    # Weekday names
    WEEKDAYS = {
        'monday': 0, 'mon': 0,
        'tuesday': 1, 'tue': 1, 'tues': 1,
        'wednesday': 2, 'wed': 2,
        'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
        'friday': 4, 'fri': 4,
        'saturday': 5, 'sat': 5,
        'sunday': 6, 'sun': 6,
    }

    # Month names
    MONTHS = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12,
    }

    def __init__(self, user_name: str = "User"):
        self.user_name = user_name

    def detect_events(self, text: str) -> List[PersonalEvent]:
        """Detect personal events from text"""
        events = []
        text_lower = text.lower().strip()

        # Try birthday patterns
        for pattern in self.BIRTHDAY_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                date_str = match.group(1) if match.lastindex else None
                date = self._parse_date(date_str) if date_str else None

                if date:
                    event = PersonalEvent(
                        event_type="birthday",
                        date=date.strftime("%Y-%m-%d"),
                        description=f"{self.user_name}'s birthday",
                        person=self.user_name,
                        recurring=True
                    )
                    events.append(event)
                    logger.info(f"Detected birthday: {date.strftime('%Y-%m-%d')}")
                break  # Only match one birthday pattern

        # Try anniversary patterns
        for pattern in self.ANNIVERSARY_PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                date = self._parse_date(date_str)

                if date:
                    event = PersonalEvent(
                        event_type="anniversary",
                        date=date.strftime("%Y-%m-%d"),
                        description=f"{self.user_name}'s anniversary",
                        person=self.user_name,
                        recurring=True
                    )
                    events.append(event)
                    logger.info(f"Detected anniversary: {date.strftime('%Y-%m-%d')}")
                break

        return events

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse a date string to datetime"""
        if not date_str:
            return None

        date_str = date_str.lower().strip()
        now = datetime.now()

        # Check if it's a weekday (e.g., "wednesday")
        if date_str in self.WEEKDAYS:
            target_weekday = self.WEEKDAYS[date_str]
            current_weekday = now.weekday()

            # Calculate days until next occurrence
            days_ahead = target_weekday - current_weekday
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7

            return now + timedelta(days=days_ahead)

        # Try "Month Day" format (e.g., "February 12")
        month_day_match = re.match(r'(\w+)\s+(\d+)(?:st|nd|rd|th)?', date_str)
        if month_day_match:
            month_str = month_day_match.group(1)
            day = int(month_day_match.group(2))

            if month_str in self.MONTHS:
                month = self.MONTHS[month_str]
                try:
                    # Try current year first
                    date = datetime(now.year, month, day)
                    # If date has passed, use next year
                    if date < now:
                        date = datetime(now.year + 1, month, day)
                    return date
                except ValueError:
                    logger.warning(f"Invalid date: {month}/{day}")
                    return None

        # Try "Day Month" format (e.g., "12th February")
        day_month_match = re.match(r'(\d+)(?:st|nd|rd|th)?\s+(\w+)', date_str)
        if day_month_match:
            day = int(day_month_match.group(1))
            month_str = day_month_match.group(2)

            if month_str in self.MONTHS:
                month = self.MONTHS[month_str]
                try:
                    date = datetime(now.year, month, day)
                    if date < now:
                        date = datetime(now.year + 1, month, day)
                    return date
                except ValueError:
                    logger.warning(f"Invalid date: {day}/{month}")
                    return None

        return None


class PersonalEventsStorage:
    """Stores and retrieves personal events"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            project_root = Path(__file__).resolve().parents[1]
            data_dir = project_root / "data" / "context"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.data_dir / "personal_events.json"

        self.events: List[PersonalEvent] = []
        self._load_events()

    def _load_events(self):
        """Load events from storage"""
        if not self.events_file.exists():
            return

        try:
            with open(self.events_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.events = [PersonalEvent.from_dict(e) for e in data]
            logger.info(f"Loaded {len(self.events)} personal events")
        except Exception as e:
            logger.error(f"Failed to load personal events: {e}")

    def _save_events(self):
        """Save events to storage"""
        try:
            with open(self.events_file, 'w', encoding='utf-8') as f:
                json.dump([e.to_dict() for e in self.events], f, indent=2)
            logger.info(f"Saved {len(self.events)} personal events")
        except Exception as e:
            logger.error(f"Failed to save personal events: {e}")

    def add_event(self, event: PersonalEvent) -> bool:
        """Add a new event"""
        # Check for duplicates
        for existing in self.events:
            if (existing.event_type == event.event_type and
                existing.date == event.date and
                existing.person == event.person):
                logger.info(f"Event already exists: {event.description}")
                return False

        self.events.append(event)
        self._save_events()
        logger.info(f"Added event: {event.description} on {event.date}")
        return True

    def get_upcoming_events(self, days: int = 30) -> List[PersonalEvent]:
        """Get upcoming events within the next N days"""
        now = datetime.now()
        upcoming = []

        for event in self.events:
            event_date = datetime.strptime(event.date, "%Y-%m-%d")
            days_until = (event_date - now).days

            if 0 <= days_until <= days:
                upcoming.append(event)

        return sorted(upcoming, key=lambda e: e.date)

    def get_event_on_date(self, date: datetime) -> List[PersonalEvent]:
        """Get events on a specific date"""
        date_str = date.strftime("%Y-%m-%d")
        return [e for e in self.events if e.date == date_str]

    def delete_event(self, event_type: str, date: str) -> bool:
        """Delete an event"""
        initial_count = len(self.events)
        self.events = [e for e in self.events
                      if not (e.event_type == event_type and e.date == date)]

        if len(self.events) < initial_count:
            self._save_events()
            return True
        return False
