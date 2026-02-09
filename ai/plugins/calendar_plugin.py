"""
Calendar Plugin for A.L.I.C.E

This plugin provides calendar integration capabilities, focusing on Google Calendar
with future support for Outlook and other calendar providers.

Features:
- View calendar events (today, tomorrow, this week, specific dates)
- Create new calendar events with natural language parsing
- Update existing events
- Delete events
- Search for events
- Set reminders and notifications
- Handle recurring events

Author: A.L.I.C.E Development Team
Date: January 2026
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import re
from dataclasses import dataclass

# Google Calendar API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_AVAILABLE = True
except ImportError:
    print("Warning: Google Calendar dependencies not available. Run: pip install google-auth google-auth-oauthlib google-api-python-client")
    GOOGLE_AVAILABLE = False

from ai.plugins.plugin_system import PluginInterface
import logging

logger = logging.getLogger(__name__)

# Google Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar']

@dataclass
class CalendarEvent:
    """Represents a calendar event"""
    id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: str = ""
    attendees: List[str] = None
    reminder_minutes: int = 15
    recurring: bool = False
    calendar_id: str = "primary"

class CalendarPlugin(PluginInterface):
    """Calendar plugin for A.L.I.C.E"""
    
    def __init__(self):
        super().__init__()
        self.name = "CalendarPlugin"
        self.version = "1.0.0"
        self.description = "Manage calendar events with natural language"
        self.capabilities = [
            "calendar_view", "calendar_create", "calendar_update", 
            "calendar_delete", "calendar_search", "schedule_management"
        ]
        
        # Credentials and service
        self.credentials_file = "config/cred/calendar_credentials.json"
        self.token_file = "config/cred/calendar_token.pickle"
        self.service = None
        self.user_timezone = "America/Toronto"  # Default, can be configured
        
        # Natural language patterns
        self.time_patterns = {
            'time_12h': r'(\d{1,2}):?(\d{0,2})\s*(am|pm)',
            'time_24h': r'(\d{1,2}):(\d{2})',
            'relative_time': r'(in|after)\s+(\d+)\s+(minute|hour|day|week|month)s?',
            'named_times': {
                'morning': '09:00', 'afternoon': '14:00', 'evening': '18:00',
                'night': '20:00', 'noon': '12:00', 'midnight': '00:00'
            }
        }
        
        self.date_patterns = {
            'today': 0, 'tomorrow': 1, 'yesterday': -1,
            'next week': 7, 'next month': 30,
            'monday': 1, 'tuesday': 2, 'wednesday': 3, 'thursday': 4,
            'friday': 5, 'saturday': 6, 'sunday': 0
        }

    def initialize(self) -> bool:
        """Initialize the calendar plugin"""
        if not GOOGLE_AVAILABLE:
            logger.error("Google Calendar API not available")
            return False
        
        try:
            self.service = self._authenticate()
            if self.service:
                logger.info("📅 Calendar plugin initialized successfully")
                return True
            else:
                logger.warning("📅 Calendar plugin initialized without authentication")
                return True  # Still allow plugin to load for setup
        except Exception as e:
            logger.error(f"Failed to initialize calendar plugin: {e}")
            return False

    def can_handle(self, intent: str, entities: Dict) -> bool:
        """Check if this plugin can handle the request"""
        calendar_intents = [
            "calendar", "schedule", "appointment", "meeting", 
            "event", "reminder", "booking"
        ]
        
        calendar_keywords = [
            "calendar", "event", "meeting", "appointment", "schedule",
            "book", "reserve", "plan", "remind", "reminder"
        ]
        
        # Check intent
        if intent in calendar_intents:
            return True
        
        # Check for calendar keywords in entities
        if isinstance(entities, dict):
            text = str(entities).lower()
            # Exclude if it's a note command
            if 'note' in text or 'notes' in text:
                return False
            return any(keyword in text for keyword in calendar_keywords)
        
        return False

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        """Execute calendar operations"""
        try:
            query_lower = query.lower()
            
            # Authentication check
            if not self.service:
                return self._handle_no_auth()
            
            # View calendar events
            if any(word in query_lower for word in ['show', 'view', 'list', 'what', 'check']):
                return self._view_events(query, entities)
            
            # Create calendar event
            elif any(word in query_lower for word in ['create', 'add', 'schedule', 'book', 'set']):
                return self._create_event(query, entities)
            
            # Update calendar event
            elif any(word in query_lower for word in ['update', 'change', 'modify', 'reschedule']):
                return self._update_event(query, entities)
            
            # Delete calendar event
            elif any(word in query_lower for word in ['delete', 'remove', 'cancel']):
                return self._delete_event(query, entities)
            
            # Search calendar events
            elif any(word in query_lower for word in ['find', 'search', 'look for']):
                return self._search_events(query, entities)
            
            else:
                return {
                    'success': False,
                    'response': "I can help you with calendar operations. Try 'show my calendar', 'create a meeting', or 'schedule an appointment'.",
                    'data': {}
                }
                
        except Exception as e:
            logger.error(f"Calendar plugin error: {e}")
            return {
                'success': False,
                'response': f"Sorry, I encountered an error with the calendar: {str(e)}",
                'data': {'error': str(e)}
            }

    def _authenticate(self) -> Optional[Any]:
        """Authenticate with Google Calendar API"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, start OAuth flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    logger.warning("Calendar credentials file not found. Calendar features disabled.")
                    return None
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build service
        try:
            service = build('calendar', 'v3', credentials=creds)
            return service
        except Exception as e:
            logger.error(f"Failed to build calendar service: {e}")
            return None

    def _handle_no_auth(self) -> Dict:
        """Handle case when calendar is not authenticated"""
        return {
            'success': False,
            'response': "Calendar is not set up yet. Please add your Google Calendar credentials to enable calendar features.",
            'data': {
                'setup_required': True,
                'instructions': "Add calendar_credentials.json to the config/cred folder and restart A.L.I.C.E"
            }
        }

    def _view_events(self, query: str, entities: Dict) -> Dict:
        """View calendar events"""
        try:
            # Parse time range from query
            start_time, end_time = self._parse_time_range(query)
            
            # Get events from Google Calendar
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start_time.isoformat(),
                timeMax=end_time.isoformat(),
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                time_desc = self._get_time_description(query)
                return {
                    'success': True,
                    'response': f"No events found {time_desc}.",
                    'data': {'events': [], 'count': 0}
                }
            
            # Format events for response
            response_lines = []
            time_desc = self._get_time_description(query)
            response_lines.append(f"📅 Your calendar {time_desc}:")
            
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))
                title = event.get('summary', 'Untitled Event')
                
                # Parse datetime
                if 'T' in start:  # datetime with time
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    time_str = f"{start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')}"
                else:  # all-day event
                    start_dt = datetime.fromisoformat(start)
                    time_str = "All day"
                
                # Format event
                date_str = start_dt.strftime('%a, %b %d')
                response_lines.append(f"   • {title} - {date_str} at {time_str}")
                
                # Add location if available
                if event.get('location'):
                    response_lines.append(f"     📍 {event['location']}")
            
            return {
                'success': True,
                'response': '\n'.join(response_lines),
                'data': {
                    'events': events,
                    'count': len(events),
                    'time_range': {'start': start_time.isoformat(), 'end': end_time.isoformat()}
                }
            }
            
        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            return {
                'success': False,
                'response': "Sorry, I couldn't access your calendar. Please check your permissions.",
                'data': {'error': str(e)}
            }

    def _create_event(self, query: str, entities: Dict) -> Dict:
        """Create a new calendar event"""
        try:
            # Parse event details from query
            event_details = self._parse_event_details(query)
            
            if not event_details['title']:
                return {
                    'success': False,
                    'response': "I need more details to create the event. Please specify what the event is for.",
                    'data': {}
                }
            
            # Create event structure
            event = {
                'summary': event_details['title'],
                'description': event_details.get('description', ''),
                'start': {
                    'dateTime': event_details['start_time'].isoformat(),
                    'timeZone': self.user_timezone,
                },
                'end': {
                    'dateTime': event_details['end_time'].isoformat(),
                    'timeZone': self.user_timezone,
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': event_details.get('reminder_minutes', 15)},
                        {'method': 'popup', 'minutes': event_details.get('reminder_minutes', 15)},
                    ],
                },
            }
            
            # Add location if provided
            if event_details.get('location'):
                event['location'] = event_details['location']
            
            # Add attendees if provided
            if event_details.get('attendees'):
                event['attendees'] = [{'email': email} for email in event_details['attendees']]
            
            # Create the event
            created_event = self.service.events().insert(
                calendarId='primary', 
                body=event
            ).execute()
            
            # Format response
            start_dt = event_details['start_time']
            date_str = start_dt.strftime('%A, %B %d, %Y')
            time_str = start_dt.strftime('%I:%M %p')
            
            response = f"✅ Created event '{event_details['title']}' for {date_str} at {time_str}"
            
            if event_details.get('location'):
                response += f" at {event_details['location']}"
            
            return {
                'success': True,
                'response': response,
                'data': {
                    'event_id': created_event['id'],
                    'event_details': event_details,
                    'calendar_link': created_event.get('htmlLink')
                }
            }
            
        except HttpError as e:
            logger.error(f"Failed to create calendar event: {e}")
            return {
                'success': False,
                'response': "Sorry, I couldn't create the calendar event. Please check your permissions.",
                'data': {'error': str(e)}
            }
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            return {
                'success': False,
                'response': f"Sorry, I had trouble creating the event: {str(e)}",
                'data': {'error': str(e)}
            }

    def _update_event(self, query: str, entities: Dict) -> Dict:
        """Update an existing calendar event"""
        # Implementation for updating events
        return {
            'success': False,
            'response': "Event updating feature coming soon! For now, you can delete and recreate the event.",
            'data': {}
        }

    def _delete_event(self, query: str, entities: Dict) -> Dict:
        """Delete a calendar event"""
        # Implementation for deleting events
        return {
            'success': False,
            'response': "Event deletion feature coming soon! You can delete events directly from your calendar app.",
            'data': {}
        }

    def _search_events(self, query: str, entities: Dict) -> Dict:
        """Search for calendar events"""
        try:
            # Extract search term
            search_terms = self._extract_search_terms(query)
            
            if not search_terms:
                return {
                    'success': False,
                    'response': "What would you like me to search for in your calendar?",
                    'data': {}
                }
            
            # Search in next 30 days
            start_time = datetime.now()
            end_time = start_time + timedelta(days=30)
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start_time.isoformat(),
                timeMax=end_time.isoformat(),
                q=' '.join(search_terms),
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            if not events:
                return {
                    'success': True,
                    'response': f"No events found matching '{' '.join(search_terms)}'.",
                    'data': {'events': [], 'search_terms': search_terms}
                }
            
            # Format search results
            response_lines = [f"📅 Found {len(events)} event(s) matching '{' '.join(search_terms)}':"]
            
            for event in events[:5]:  # Limit to 5 results
                start = event['start'].get('dateTime', event['start'].get('date'))
                title = event.get('summary', 'Untitled Event')
                
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    date_time_str = start_dt.strftime('%a, %b %d at %I:%M %p')
                else:
                    start_dt = datetime.fromisoformat(start)
                    date_time_str = start_dt.strftime('%a, %b %d (All day)')
                
                response_lines.append(f"   • {title} - {date_time_str}")
            
            return {
                'success': True,
                'response': '\n'.join(response_lines),
                'data': {'events': events, 'search_terms': search_terms}
            }
            
        except Exception as e:
            logger.error(f"Calendar search error: {e}")
            return {
                'success': False,
                'response': f"Sorry, I couldn't search your calendar: {str(e)}",
                'data': {'error': str(e)}
            }

    def _parse_time_range(self, query: str) -> Tuple[datetime, datetime]:
        """Parse time range from natural language query"""
        now = datetime.now()
        query_lower = query.lower()
        
        # Today
        if 'today' in query_lower:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        
        # Tomorrow
        elif 'tomorrow' in query_lower:
            start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        
        # This week
        elif 'week' in query_lower or 'this week' in query_lower:
            days_since_monday = now.weekday()
            start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
        
        # Next 7 days
        elif 'next' in query_lower and ('week' in query_lower or 'days' in query_lower):
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
        
        # This month
        elif 'month' in query_lower:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=31)
        
        # Default: today
        else:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        
        return start, end

    def _parse_event_details(self, query: str) -> Dict:
        """Parse event details from natural language"""
        details = {
            'title': '',
            'description': '',
            'start_time': None,
            'end_time': None,
            'location': '',
            'attendees': [],
            'reminder_minutes': 15
        }
        
        # Extract title (basic implementation)
        # Look for patterns like "create meeting about X" or "schedule X"
        title_patterns = [
            r'(?:create|schedule|book|set up|add)\s+(?:a\s+)?(?:meeting|appointment|event)?\s+(?:about|for|with)?\s+(.+?)(?:\s+(?:at|on|for|tomorrow|today))',
            r'(?:create|schedule|book|set up|add)\s+(.+?)(?:\s+(?:at|on|for|tomorrow|today))',
            r'(?:meeting|appointment|event)\s+(?:about|for|with)?\s+(.+?)(?:\s+(?:at|on|for|tomorrow|today))'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                details['title'] = match.group(1).strip()
                break
        
        # If no title found, use a generic one
        if not details['title']:
            details['title'] = 'Meeting'
        
        # Parse time
        start_time, end_time = self._parse_event_time(query)
        details['start_time'] = start_time
        details['end_time'] = end_time
        
        # Extract location
        location_patterns = [
            r'at\s+([^,\n]+?)(?:\s+(?:on|at|tomorrow|today|$))',
            r'in\s+([^,\n]+?)(?:\s+(?:on|at|tomorrow|today|$))',
            r'location\s+([^,\n]+)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                # Filter out time-related words
                if not any(time_word in location.lower() for time_word in ['am', 'pm', 'morning', 'afternoon', 'evening']):
                    details['location'] = location
                    break
        
        return details

    def _parse_event_time(self, query: str) -> Tuple[datetime, datetime]:
        """Parse event time from natural language"""
        now = datetime.now()
        
        # Default: 1 hour from now
        default_start = now + timedelta(hours=1)
        default_end = default_start + timedelta(hours=1)
        
        # Look for specific times
        time_match = re.search(r'(\d{1,2}):?(\d{0,2})\s*(am|pm)?', query, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            ampm = time_match.group(3).lower() if time_match.group(3) else None
            
            # Convert to 24-hour format
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            
            # Determine date
            event_date = now.date()
            if 'tomorrow' in query.lower():
                event_date = (now + timedelta(days=1)).date()
            elif 'today' in query.lower():
                event_date = now.date()
            # Add more date parsing as needed
            
            start_time = datetime.combine(event_date, datetime.min.time().replace(hour=hour, minute=minute))
            end_time = start_time + timedelta(hours=1)
            
            return start_time, end_time
        
        # Look for relative times
        relative_match = re.search(r'in\s+(\d+)\s+(minute|hour)s?', query, re.IGNORECASE)
        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2).lower()
            
            if unit == 'minute':
                start_time = now + timedelta(minutes=amount)
            else:  # hour
                start_time = now + timedelta(hours=amount)
            
            end_time = start_time + timedelta(hours=1)
            return start_time, end_time
        
        # Check for named times
        for name, time_str in self.time_patterns['named_times'].items():
            if name in query.lower():
                hour, minute = map(int, time_str.split(':'))
                event_date = now.date()
                
                if 'tomorrow' in query.lower():
                    event_date = (now + timedelta(days=1)).date()
                
                start_time = datetime.combine(event_date, datetime.min.time().replace(hour=hour, minute=minute))
                end_time = start_time + timedelta(hours=1)
                return start_time, end_time
        
        return default_start, default_end

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from query"""
        # Remove common search words
        stop_words = ['find', 'search', 'look', 'for', 'show', 'me', 'my', 'calendar', 'events']
        words = query.lower().split()
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return search_terms

    def _get_time_description(self, query: str) -> str:
        """Get human-readable time description"""
        query_lower = query.lower()
        
        if 'today' in query_lower:
            return 'for today'
        elif 'tomorrow' in query_lower:
            return 'for tomorrow'
        elif 'week' in query_lower:
            return 'for this week'
        elif 'month' in query_lower:
            return 'for this month'
        else:
            return 'for today'

    def get_status(self) -> Dict[str, Any]:
        """Get plugin status"""
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'authenticated': self.service is not None,
            'google_api_available': GOOGLE_AVAILABLE
        }

    def shutdown(self):
        """Shutdown the calendar plugin"""
        logger.info("📅 Calendar plugin shutdown complete")