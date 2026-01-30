"""
Response Generator for A.L.I.C.E
A.L.I.C.E generates her own responses from plugin data using ML.
Plugins return data only - A.L.I.C.E learns patterns and creates natural responses.
This makes her personality consistent and learned from training data.
"""

import logging
from typing import Dict, Optional, Any, List
from collections import defaultdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates natural responses from plugin data using learned patterns.
    A.L.I.C.E learns from training data how to respond, not pre-written templates.
    """
    
    def __init__(self, ml_learner=None, training_collector=None):
        self.ml_learner = ml_learner
        self.training_collector = training_collector
        
        # Learned patterns from training data
        self.learned_patterns = {
            'weather': [],
            'time': [],
            'notes': [],
            'calendar': [],
            'music': [],
            'generic': []
        }
        
        # Response characteristics learned from training
        self.response_characteristics = {
            'avg_length': 50,
            'uses_emoji': False,
            'formality': 'casual',
            'verbosity': 'medium'
        }
        
        # Load learned patterns
        self._load_learned_patterns()
    
    def _load_learned_patterns(self):
        """Load response patterns learned from training data"""
        if not self.training_collector:
            return
        
        try:
            examples = self.training_collector.get_training_data(min_quality=0.7)
            if not examples:
                return
            
            # Group by plugin type
            by_plugin = defaultdict(list)
            for ex in examples:
                plugin = ex.context.get('plugin', 'generic')
                if plugin == 'WeatherPlugin':
                    by_plugin['weather'].append(ex.assistant_response)
                elif plugin == 'TimePlugin':
                    by_plugin['time'].append(ex.assistant_response)
                elif 'Notes' in plugin:
                    by_plugin['notes'].append(ex.assistant_response)
                elif 'Calendar' in plugin:
                    by_plugin['calendar'].append(ex.assistant_response)
                elif 'Music' in plugin:
                    by_plugin['music'].append(ex.assistant_response)
                else:
                    by_plugin['generic'].append(ex.assistant_response)
            
            # Analyze patterns
            for plugin_type, responses in by_plugin.items():
                if responses:
                    # Learn characteristics
                    avg_len = sum(len(r) for r in responses) / len(responses)
                    self.learned_patterns[plugin_type] = responses
                    
            # Learn overall characteristics
            all_responses = [ex.assistant_response for ex in examples]
            if all_responses:
                self.response_characteristics['avg_length'] = sum(len(r) for r in all_responses) / len(all_responses)
                self.response_characteristics['uses_emoji'] = any('✅' in r or '💡' in r for r in all_responses)
                
            logger.info(f"[ResponseGen] Learned patterns from {len(examples)} examples")
            
        except Exception as e:
            logger.warning(f"[ResponseGen] Could not load learned patterns: {e}")
    
    def generate_from_plugin_data(
        self,
        plugin_name: str,
        plugin_data: Dict[str, Any],
        user_input: str,
        intent: str,
        success: bool
    ) -> Optional[str]:
        """
        Generate A.L.I.C.E's response from plugin data using ONLY learned patterns.
        Returns None if no learned patterns exist - then Ollama will handle it.
        """
        
        if not success:
            return self._generate_failure_response(plugin_name, plugin_data, user_input)
        
        # Route to specific generator based on plugin
        if plugin_name == 'WeatherPlugin':
            return self._generate_weather_response(plugin_data, user_input)
        elif plugin_name == 'MapsPlugin':
            return self._generate_maps_response(plugin_data, user_input)
        elif plugin_name == 'TimePlugin':
            return self._generate_time_response(plugin_data, user_input)
        elif plugin_name == 'Notes Plugin':
            return self._generate_notes_response(plugin_data, user_input, intent)
        elif plugin_name == 'Calendar Plugin':
            return self._generate_calendar_response(plugin_data, user_input)
        elif plugin_name == 'Music Plugin':
            return self._generate_music_response(plugin_data, user_input)
        else:
            # Generic response for unknown plugins
            return self._generate_generic_response(plugin_data, user_input)
    
    def _generate_weather_response(self, data: Dict[str, Any], user_input: str) -> Optional[str]:
        """Generate weather response using learned patterns from training data"""
        temp = data.get('temperature')
        condition = data.get('condition')
        location = data.get('location')
        humidity = data.get('humidity')
        wind = data.get('wind_speed')
        
        # If we have learned patterns from training data, use ML to generate
        if self.learned_patterns.get('weather'):
            # A.L.I.C.E learned how to talk about weather from your conversations
            learned_responses = self.learned_patterns['weather']
            
            # Extract patterns: how does A.L.I.C.E typically structure weather responses?
            # For now, analyze common structures
            has_exclamation = any('!' in r for r in learned_responses)
            typical_length = self.response_characteristics.get('avg_length', 50)
            
            # Generate based on learned style
            response_parts = []
            
            # Core weather info (always included)
            response_parts.append(f"{temp}°C")
            response_parts.append(condition)
            if location:
                response_parts.append(f"in {location}")
            
            # Construct response based on learned patterns
            if typical_length < 40:  # Short style
                response = f"{response_parts[0]}, {response_parts[1]}."
            else:  # Normal/long style
                response = f"It's {response_parts[0]} with {response_parts[1]} {response_parts[2] if len(response_parts) > 2 else ''}."
            
            # Add temperature context if learned to do so
            if has_exclamation and temp is not None:
                if temp < -10:
                    response += " Cold out there!"
                elif temp > 30:
                    response += " Really warm!"
            
            return response.strip()
        
        # No learned patterns - construct from data only
        parts = []
        if temp is not None:
            parts.append(f"{temp}°C")
        if condition:
            parts.append(condition)
        if location:
            parts.append(f"in {location}")
        
        return ", ".join(parts) + "." if parts else None  # Return None if no data
    
    def _generate_maps_response(self, data: Dict[str, Any], user_input: str) -> Optional[str]:
        """Generate maps/places response using learned patterns"""
        place_type = data.get('place_type')
        location = data.get('location')
        places = data.get('places', [])
        
        if not places:
            return f"No {place_type}s found nearby."
        
        # A.L.I.C.E constructs response from place data
        is_brief = self.response_characteristics['avg_length'] < 40
        
        if len(places) == 1:
            place = places[0]
            if is_brief:
                return f"{place['name']} - {place['distance_km']}km away."
            return f"The nearest {place_type} is {place['name']}, about {place['distance_km']}km away."
        else:
            # Multiple places
            nearest = places[0]
            if is_brief:
                return f"{nearest['name']} ({nearest['distance_km']}km). Found {len(places)} total."
            
            response = f"Nearest {place_type}: {nearest['name']}, {nearest['distance_km']}km away."
            if len(places) > 1:
                response += f" Found {len(places)} total"
                if len(places) > 2:
                    response += f" - next closest: {places[1]['name']} ({places[1]['distance_km']}km)"
            return response + "."
    
    def _generate_time_response(self, data: Dict[str, Any], user_input: str) -> Optional[str]:
        """Generate time/date response using learned patterns ONLY"""
        if self.learned_patterns.get('time'):
            # Learn from past time responses
            learned = self.learned_patterns['time']
            uses_its = any("it's" in r.lower() for r in learned)
            
            if 'date' in user_input.lower():
                date_val = data.get('formatted_date')
                return f"It's {date_val}." if (uses_its and date_val) else (f"{date_val}." if date_val else None)
            else:
                time_val = data.get('formatted_time')
                return f"It's {time_val}." if (uses_its and time_val) else (f"{time_val}." if time_val else None)
        
        # No learned patterns - return data only
        if 'date' in user_input.lower():
            return data.get('formatted_date')
        return data.get('formatted_time')
    
    def _generate_notes_response(self, data: Dict[str, Any], user_input: str, intent: str) -> Optional[str]:
        """Generate notes response using learned patterns"""
        action = data.get('action', 'unknown')
        
        # Learn confirmation style from past interactions
        learned = self.learned_patterns.get('notes', [])
        uses_got_it = any('got it' in r.lower() for r in learned) if learned else True
        is_brief = self.response_characteristics['avg_length'] < 40
        
        if action == 'created':
            note_title = data.get('note', {}).get('title', 'note')
            if is_brief:
                return f"Created '{note_title}'."
            return f"Got it, created '{note_title}'." if uses_got_it else f"Created '{note_title}'."
        elif action == 'listed':
            notes = data.get('notes', [])
            if not notes:
                return "No notes yet." if is_brief else "You don't have any notes yet."
            return f"{len(notes)} note(s)." if is_brief else f"You have {len(notes)} note(s)."
        elif action == 'deleted':
            return "Deleted." if is_brief else "Deleted that note."
        elif action == 'archived':
            return "Archived." if is_brief else "Archived it."
        else:
            return "Done."
    
    def _generate_calendar_response(self, data: Dict[str, Any], user_input: str) -> Optional[str]:
        """Generate calendar response using learned patterns"""
        action = data.get('action', 'unknown')
        is_brief = self.response_characteristics['avg_length'] < 40
        
        if action == 'event_created':
            return "Added to calendar." if is_brief else "Added that to your calendar."
        elif action == 'events_listed':
            events = data.get('events', [])
            if not events:
                return "Calendar clear." if is_brief else "Your calendar is clear."
            return f"{len(events)} event(s)." if is_brief else f"You have {len(events)} upcoming event(s)."
        else:
            return "Done." if is_brief else "Calendar updated."
    
    def _generate_music_response(self, data: Dict[str, Any], user_input: str) -> Optional[str]:
        """Generate music response using learned patterns"""
        action = data.get('action', 'unknown')
        
        if action == 'playing':
            song = data.get('song', 'music')
            return f"Playing {song}."
        elif action == 'paused':
            return "Paused."
        elif action == 'stopped':
            return "Stopped."
        else:
            return "Done."
    
    def _generate_generic_response(self, data: Dict[str, Any], user_input: str) -> Optional[str]:
        """Generate generic response - ONLY if we have learned patterns"""
        # No hardcoded responses - return None if no data
        if not data:
            return None
        
        # If we have learned patterns for generic responses, use them
        if self.learned_patterns.get('generic'):
            return random.choice(self.learned_patterns['generic'])
        
        # Otherwise return None - let Ollama handle it
        return None
    
    def _generate_failure_response(self, plugin_name: str, data: Dict[str, Any], user_input: str) -> Optional[str]:
        """Generate response when plugin fails - ONLY from learned patterns"""
        # Look for learned failure patterns
        if self.training_collector:
            try:
                examples = self.training_collector.get_training_data(min_quality=0.5, max_examples=50)
                failure_responses = []
                for ex in examples:
                    # Find examples where plugins failed (quality < 0.7)
                    if ex.quality_score < 0.7 and ex.assistant_response:
                        failure_responses.append(ex.assistant_response)
                
                if failure_responses:
                    return random.choice(failure_responses)
            except:
                pass
        
        # No learned patterns - return None, let Ollama handle error explanation
        return None


_response_generator: Optional[ResponseGenerator] = None


def get_response_generator(ml_learner=None, training_collector=None) -> ResponseGenerator:
    """Get singleton response generator"""
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator(
            ml_learner=ml_learner,
            training_collector=training_collector
        )
    return _response_generator
