"""
Advanced Conversation Summarization for A.L.I.C.E
Provides intelligent conversation context compression and summarization
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class ConversationSummary:
    """Represents a summarized conversation segment"""
    timestamp: str
    duration_minutes: int
    main_topics: List[str]
    key_points: List[str]
    user_requests: List[str]
    assistant_actions: List[str]
    entities_mentioned: List[str]
    sentiment_trend: str
    summary_text: str
    original_turn_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConversationSummarizer:
    """
    Intelligent conversation summarization system
    
    Features:
    - Automatic conversation segmentation
    - Topic extraction and clustering
    - Key point identification
    - Context preservation across sessions
    - Memory-efficient summarization
    """
    
    def __init__(self, llm_engine=None, llm_gateway=None, data_dir: str = "data/context"):
        self.llm_engine = llm_engine  # Kept for backward compatibility
        self.llm_gateway = llm_gateway  # Preferred: use gateway for policy enforcement
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Configuration
        self.max_turns_before_summary = 20  # Summarize every 20 turns
        self.summary_history_limit = 50     # Keep last 50 summaries
        self.min_turns_for_summary = 5      # Need at least 5 turns to summarize
        
        # State
        self.current_session_turns = []
        self.conversation_summaries = []
        self.session_start_time = datetime.now()
        
        # Load existing summaries
        self._load_summary_history()
    
    def add_turn(self, user_input: str, assistant_response: str, 
                 intent: Optional[str] = None, entities: Optional[List[str]] = None,
                 sentiment: Optional[str] = None):
        """Add a new conversation turn"""
        
        turn = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": assistant_response,
            "intent": intent,
            "entities": entities or [],
            "sentiment": sentiment
        }
        
        self.current_session_turns.append(turn)
        
        # Auto-summarize when we have enough turns
        if len(self.current_session_turns) >= self.max_turns_before_summary:
            self._auto_summarize()
    
    def get_context_summary(self, max_summaries: int = 3) -> str:
        """Get a condensed context summary for the LLM"""
        
        context_parts = []
        
        # Recent summaries for historical context
        if self.conversation_summaries:
            recent_summaries = self.conversation_summaries[-max_summaries:]
            for summary in recent_summaries:
                context_parts.append(f"Previous discussion: {summary.summary_text}")
        
        # Current session highlights
        if len(self.current_session_turns) >= 3:
            current_summary = self._generate_quick_summary(self.current_session_turns[-5:])
            context_parts.append(f"Recent context: {current_summary}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def get_detailed_context(self) -> Dict[str, Any]:
        """Get detailed conversation context for advanced processing"""
        
        session_duration = datetime.now() - self.session_start_time
        
        # Aggregate topics and entities from all summaries
        all_topics = []
        all_entities = []
        key_points = []
        
        for summary in self.conversation_summaries[-10:]:  # Last 10 summaries
            all_topics.extend(summary.main_topics)
            all_entities.extend(summary.entities_mentioned)
            key_points.extend(summary.key_points)
        
        # Add current session data
        for turn in self.current_session_turns:
            if turn["entities"]:
                all_entities.extend(turn["entities"])
        
        return {
            "session_duration_minutes": int(session_duration.total_seconds() / 60),
            "total_turns_current_session": len(self.current_session_turns),
            "total_historical_summaries": len(self.conversation_summaries),
            "frequent_topics": list(set(all_topics))[-10:],  # Top 10 recent topics
            "mentioned_entities": list(set(all_entities))[-15:],  # Top 15 entities
            "key_insights": key_points[-8:],  # Top 8 key points
            "conversation_style": self._analyze_conversation_style()
        }
    
    def get_conversation_summary(self) -> str:
        """Get a formatted summary of the conversation"""
        if not self.current_session_turns and not self.conversation_summaries:
            return None  # Let LLM generate "no history" message
        
        summary_parts = []
        
        # Add historical summaries
        if self.conversation_summaries:
            summary_parts.append("Previous Conversation Summaries:")
            for i, conv_summary in enumerate(self.conversation_summaries[-3:], 1):
                summary_parts.append(f"  {i}. {conv_summary.summary}")
            summary_parts.append("")
        
        # Add current session summary if there are turns
        if self.current_session_turns:
            # Generate current summary
            current_summary = self._create_summary(
                self.current_session_turns[-10:]  # Last 10 turns
            )
            if current_summary:
                summary_parts.append("Current Session:")
                summary_parts.append(f"  {current_summary.summary}")
                summary_parts.append("")
        
        # Add session stats
        session_duration = datetime.now() - self.session_start_time
        summary_parts.append(f"Session Duration: {int(session_duration.total_seconds() / 60)} minutes")
        summary_parts.append(f"Current Session Turns: {len(self.current_session_turns)}")
        
        # Add topics and entities
        context = self.get_detailed_context()
        if context.get("frequent_topics"):
            topics = context["frequent_topics"][:5]
            summary_parts.append(f"Recent Topics: {', '.join(topics)}")
        
        if context.get("mentioned_entities"):
            entities = context["mentioned_entities"][:8]
            summary_parts.append(f"Mentioned: {', '.join(entities)}")
        
        return "\n".join(summary_parts) if summary_parts else "No conversation to summarize."
    
    def force_summarize_session(self) -> Optional[ConversationSummary]:
        """Manually trigger summarization of current session"""
        if len(self.current_session_turns) >= self.min_turns_for_summary:
            return self._create_summary(self.current_session_turns)
        return None
    
    def _auto_summarize(self):
        """Automatically summarize and archive conversation segments"""
        
        if len(self.current_session_turns) < self.min_turns_for_summary:
            return
        
        # Create summary for the segment
        summary = self._create_summary(self.current_session_turns)
        
        if summary:
            self.conversation_summaries.append(summary)
            
            # Maintain summary history limit
            if len(self.conversation_summaries) > self.summary_history_limit:
                self.conversation_summaries = self.conversation_summaries[-self.summary_history_limit:]
            
            # Clear current turns (keep last 3 for continuity)
            self.current_session_turns = self.current_session_turns[-3:]
            
            # Save summaries to disk
            self._save_summary_history()
            
            logger.info(f"Created conversation summary: {summary.summary_text[:100]}...")
    
    def _create_summary(self, turns: List[Dict]) -> Optional[ConversationSummary]:
        """Create a structured summary from conversation turns"""
        
        if not turns:
            return None
        
        try:
            # Extract basic metrics
            start_time = datetime.fromisoformat(turns[0]["timestamp"])
            end_time = datetime.fromisoformat(turns[-1]["timestamp"])
            duration = int((end_time - start_time).total_seconds() / 60)
            
            # Extract topics and entities
            topics = self._extract_topics(turns)
            entities = self._extract_entities(turns)
            user_requests = self._extract_user_requests(turns)
            assistant_actions = self._extract_assistant_actions(turns)
            key_points = self._extract_key_points(turns)
            sentiment_trend = self._analyze_sentiment_trend(turns)
            
            # Generate summary text
            summary_text = self._generate_summary_text(turns, topics, key_points)
            
            return ConversationSummary(
                timestamp=start_time.isoformat(),
                duration_minutes=duration,
                main_topics=topics,
                key_points=key_points,
                user_requests=user_requests,
                assistant_actions=assistant_actions,
                entities_mentioned=entities,
                sentiment_trend=sentiment_trend,
                summary_text=summary_text,
                original_turn_count=len(turns)
            )
            
        except Exception as e:
            logger.error(f"Error creating conversation summary: {e}")
            return None
    
    def _extract_topics(self, turns: List[Dict]) -> List[str]:
        """Extract main topics from conversation turns"""
        topics = set()
        
        for turn in turns:
            # Use intent as topic if available
            if turn.get("intent"):
                intent = turn["intent"].replace("_", " ")
                topics.add(intent)
            
            # Extract topics from entities
            entities = turn.get("entities", [])
            for entity in entities:
                if isinstance(entity, str) and len(entity) > 2:
                    topics.add(entity.lower())
        
        # Also extract from text using keywords
        text_topics = self._extract_topics_from_text(turns)
        topics.update(text_topics)
        
        return list(topics)[:10]  # Top 10 topics
    
    def _extract_topics_from_text(self, turns: List[Dict]) -> List[str]:
        """Extract topics from text using keyword analysis"""
        topic_keywords = [
            "weather", "email", "calendar", "task", "file", "system", "help",
            "music", "news", "time", "date", "reminder", "note", "search",
            "document", "meeting", "appointment", "travel", "food", "shopping"
        ]
        
        found_topics = []
        all_text = ""
        
        for turn in turns:
            all_text += " " + turn.get("user_input", "") + " " + turn.get("assistant_response", "")
        
        all_text = all_text.lower()
        
        for keyword in topic_keywords:
            if keyword in all_text:
                found_topics.append(keyword)
        
        return found_topics
    
    def _extract_entities(self, turns: List[Dict]) -> List[str]:
        """Extract mentioned entities from conversation"""
        entities = set()
        
        for turn in turns:
            turn_entities = turn.get("entities", [])
            if turn_entities:
                entities.update(turn_entities)
        
        return list(entities)[:15]  # Top 15 entities
    
    def _extract_user_requests(self, turns: List[Dict]) -> List[str]:
        """Extract user requests and questions"""
        requests = []
        
        for turn in turns:
            user_input = turn.get("user_input", "")
            if user_input and len(user_input) > 10:
                # Truncate long requests
                if len(user_input) > 100:
                    user_input = user_input[:97] + "..."
                requests.append(user_input)
        
        return requests[-5:]  # Last 5 requests
    
    def _extract_assistant_actions(self, turns: List[Dict]) -> List[str]:
        """Extract key assistant actions and responses"""
        actions = []
        
        for turn in turns:
            response = turn.get("assistant_response", "")
            intent = turn.get("intent", "")
            
            if intent:
                actions.append(f"Handled {intent.replace('_', ' ')} request")
            elif response and len(response) > 20:
                # Extract first sentence as action summary
                first_sentence = response.split('.')[0]
                if len(first_sentence) > 100:
                    first_sentence = first_sentence[:97] + "..."
                actions.append(first_sentence)
        
        return actions[-5:]  # Last 5 actions
    
    def _extract_key_points(self, turns: List[Dict]) -> List[str]:
        """Extract key points and important information"""
        key_points = []
        
        # Look for important patterns
        for turn in turns:
            user_input = turn.get("user_input", "").lower()
            response = turn.get("assistant_response", "").lower()
            
            # User provided information
            if any(phrase in user_input for phrase in ["my", "i am", "i have", "i need", "my name is"]):
                key_points.append(f"User mentioned: {turn['user_input'][:100]}")
            
            # Assistant provided help
            if any(phrase in response for phrase in ["here", "found", "completed", "saved", "created"]):
                key_points.append(f"Assistant provided: {turn['assistant_response'][:100]}")
        
        return key_points[-8:]  # Top 8 key points
    
    def _analyze_sentiment_trend(self, turns: List[Dict]) -> str:
        """Analyze overall sentiment trend of the conversation"""
        sentiments = []
        
        for turn in turns:
            sentiment = turn.get("sentiment")
            if sentiment:
                sentiments.append(sentiment)
        
        if not sentiments:
            return "neutral"
        
        # Simple sentiment aggregation
        positive_count = sentiments.count("positive")
        negative_count = sentiments.count("negative")
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _generate_summary_text(self, turns: List[Dict], topics: List[str], key_points: List[str]) -> str:
        """Generate a natural language summary"""
        
        if self.llm_gateway or self.llm_engine:
            # Use gateway (preferred) or LLM engine for better summarization
            return self._llm_summarize(turns)
        else:
            # Fallback to rule-based summarization
            return self._rule_based_summarize(turns, topics, key_points)
    
    def _llm_summarize(self, turns: List[Dict]) -> str:
        """Use LLM gateway to generate conversation summary"""
        try:
            conversation_text = ""
            for i, turn in enumerate(turns):
                conversation_text += f"User: {turn['user_input']}\\nAssistant: {turn['assistant_response']}\\n\\n"
            
            # Truncate if too long
            if len(conversation_text) > 2000:
                conversation_text = conversation_text[:2000] + "..."
            
            summary_prompt = f"""Summarize this conversation in 1-2 concise sentences, focusing on the main topics discussed and any important outcomes:

{conversation_text}

Summary:"""
            
            # Try gateway first (enforces policy)
            if self.llm_gateway:
                from ai.llm_policy import LLMCallType
                response = self.llm_gateway.request(
                    prompt=summary_prompt,
                    call_type=LLMCallType.GENERATION,
                    use_history=False,
                    user_input="conversation summary"
                )
                if response.success and response.response:
                    summary = response.response
                else:
                    # Gateway denied - use rule-based fallback
                    return self._rule_based_summarize(turns, [], [])
            else:
                # Fallback to direct LLM (legacy)
                summary = self.llm_engine.chat(summary_prompt, use_history=False)
            
            # Clean and truncate summary
            summary = summary.strip()
            if len(summary) > 200:
                summary = summary[:197] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._rule_based_summarize(turns, [], [])
    
    def _rule_based_summarize(self, turns: List[Dict], topics: List[str], key_points: List[str]) -> str:
        """Generate summary using rule-based approach"""
        
        if not turns:
            return None  # Let LLM generate empty content message
        
        # Let LLM generate context-aware summaries instead of templates
        return None
    
    def _generate_quick_summary(self, recent_turns: List[Dict]) -> str:
        """Generate quick summary of recent turns for context"""
        if not recent_turns:
            return ""
        
        topics = self._extract_topics(recent_turns)
        if topics:
            return f"discussing {', '.join(topics[:2])}"
        else:
            return f"recent conversation ({len(recent_turns)} exchanges)"
    
    def _analyze_conversation_style(self) -> str:
        """Analyze user's conversation style"""
        if not self.current_session_turns:
            return "neutral"
        
        avg_user_length = sum(len(turn.get("user_input", "")) 
                             for turn in self.current_session_turns) / len(self.current_session_turns)
        
        if avg_user_length > 100:
            return "detailed"
        elif avg_user_length < 20:
            return "concise"
        else:
            return "moderate"
    
    def _save_summary_history(self):
        """Save conversation summaries to disk"""
        try:
            summaries_path = os.path.join(self.data_dir, "conversation_summaries.json")
            summaries_data = [summary.to_dict() for summary in self.conversation_summaries]
            
            with open(summaries_path, 'w', encoding='utf-8') as f:
                json.dump(summaries_data, f, indent=2)
                
            logger.debug("Conversation summaries saved")
            
        except Exception as e:
            logger.error(f"Error saving conversation summaries: {e}")
    
    def _load_summary_history(self):
        """Load conversation summaries from disk"""
        try:
            summaries_path = os.path.join(self.data_dir, "conversation_summaries.json")
            
            if os.path.exists(summaries_path):
                with open(summaries_path, 'r', encoding='utf-8') as f:
                    summaries_data = json.load(f)
                
                self.conversation_summaries = []
                for data in summaries_data:
                    summary = ConversationSummary(**data)
                    self.conversation_summaries.append(summary)
                
                logger.info(f"Loaded {len(self.conversation_summaries)} conversation summaries")
            
        except Exception as e:
            logger.error(f"Error loading conversation summaries: {e}")
            self.conversation_summaries = []
    
    def clear_session(self):
        """Clear current session and optionally create final summary"""
        if len(self.current_session_turns) >= self.min_turns_for_summary:
            final_summary = self._create_summary(self.current_session_turns)
            if final_summary:
                self.conversation_summaries.append(final_summary)
                self._save_summary_history()
        
        self.current_session_turns = []
        self.session_start_time = datetime.now()
        
        logger.info("Conversation session cleared and summarized")