"""   
A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity
Main Orchestrator - Jarvis-like Personal Assistant

Integrates all components:
- Advanced NLP with intent detection
- LLM engine (Ollama with Llama 3.3 70B)
- Context management and personalization
- Memory system with RAG
- Plugin system for extensibility
- Voice interaction (speech-to-text, text-to-speech)
- Task execution and automation
"""

# Suppress warnings before importing other modules
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import sys
import logging
import random
import re
from typing import Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ALICE components
from ai.nlp_processor import NLPProcessor, Intent
from ai.llm_engine import LocalLLMEngine, LLMConfig
from ai.context_manager import ContextManager
from ai.advanced_context_handler import AdvancedContextHandler
from ai.memory_system import MemorySystem
from ai.conversation_summarizer import ConversationSummarizer
from ai.entity_relationship_tracker import EntityRelationshipTracker
from ai.active_learning_manager import ActiveLearningManager, CorrectionType, FeedbackType
from ai.email_plugin import GmailPlugin
from ai.plugin_system import (
    PluginManager, WeatherPlugin, TimePlugin,
    FileOperationsPlugin, SystemControlPlugin, WebSearchPlugin
)
from ai.document_plugin import DocumentPlugin
from ai.calendar_plugin import CalendarPlugin
from ai.music_plugin import MusicPlugin
from ai.notes_plugin import NotesPlugin
from ai.task_executor import TaskExecutor
from speech.speech_engine import SpeechEngine, SpeechConfig

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ALICE:
    """
    Main A.L.I.C.E system
    Jarvis-like personal assistant with advanced AI capabilities
    """
    
    def __init__(
        self,
        voice_enabled: bool = False,
        llm_model: str = "llama3.3:70b",
        user_name: str = "User"
    ):
        self.voice_enabled = voice_enabled
        self.running = False
        
        # Conversation state for context-aware operations
        self.last_email_list = []  # Store last displayed email list
        self.last_email_context = None  # Store context of last email operation
        self.pending_action = None  # Track multi-step actions (e.g., composing email)
        self.pending_data = {}  # Store data for pending actions
        
        # Enhanced conversation tracking
        self.conversation_summary = []  # Summary of recent exchanges
        self.referenced_items = {}  # Track items user has referenced (emails, files, etc.)
        self.conversation_topics = []  # Track topics discussed in this session
        
        # Active learning tracking
        self.last_user_input = ""
        self.last_assistant_response = ""
        self.last_intent = ""
        self.last_entities = {}
        self.last_nlp_result = {}
        
        # Conversation summarizer for intelligent context management
        self.summarizer = None  # Will be initialized after LLM engine
        
        # Advanced context handler
        self.advanced_context = None  # Will be initialized after NLP processor
        
        logger.info("=" * 80)
        logger.info("Initializing A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity")
        logger.info("=" * 80)
        
        # Initialize components
        try:
            # 1. NLP Processor
            logger.info("Loading NLP processor...")
            self.nlp = NLPProcessor()
            
            # 1.5. Advanced Context Handler (after NLP for embeddings)
            logger.info("Loading advanced context handler...")
            try:
                # Try to use the same embeddings model as NLP processor
                embeddings_model = getattr(self.nlp, 'embeddings_model', None)
                self.advanced_context = AdvancedContextHandler(embeddings_model)
            except Exception as e:
                logger.warning(f"Advanced context handler without embeddings: {e}")
                self.advanced_context = AdvancedContextHandler()
            
            # 2. Context Manager
            logger.info("Loading context manager...")
            self.context = ContextManager()
            self.context.user_prefs.name = user_name
            
            # 3. Memory System
            logger.info("Loading memory system...")
            self.memory = MemorySystem()
            
            # 4. LLM Engine
            logger.info("🚀 Loading LLM engine...")
            llm_config = LLMConfig(model=llm_model)
            self.llm = LocalLLMEngine(llm_config)
            
            # 4.5. Conversation Summarizer
            logger.info("Loading conversation summarizer...")
            self.summarizer = ConversationSummarizer(llm_engine=self.llm)
            
            # 4.6. Entity Relationship Tracker
            logger.info("Loading entity relationship tracker...")
            self.relationship_tracker = EntityRelationshipTracker()
            
            # 4.7. Active Learning Manager
            logger.info("Loading active learning system...")
            self.learning_manager = ActiveLearningManager()
            
            # 5. Plugin System
            logger.info("Loading plugins...")
            self.plugins = PluginManager()
            self._register_plugins()
            
            # 6. Task Executor
            logger.info(" Loading task executor...")
            self.executor = TaskExecutor(safe_mode=True)
            
            # 7. Speech Engine (optional)
            self.speech = None
            if voice_enabled: 
                logger.info("Loading speech engine...")
                speech_config = SpeechConfig(wake_words=["alice", "hey alice", "ok alice"])
                self.speech = SpeechEngine(speech_config)
            
            # 8. Gmail Plugin
            logger.info("Loading Gmail integration...")
            try:
                self.gmail = GmailPlugin()
                if self.gmail.service:
                    logger.info(f"[OK] Gmail connected: {self.gmail.user_email}")
                else:
                    logger.warning("[WARNING] Gmail not configured - run setup")
                    self.gmail = None
            except Exception as e:
                logger.warning(f"[WARNING] Gmail not available: {e}")
                self.gmail = None
            
            logger.info("=" * 80)
            logger.info("[OK] A.L.I.C.E initialized successfully!")
            logger.info("=" * 80)
            
            # Store system capabilities in context
            self.context.update_system_status("capabilities", self.plugins.get_capabilities())
            self.context.update_system_status("voice_enabled", voice_enabled)
            self.context.update_system_status("llm_model", llm_model)
            
            # Load previous conversation state if available
            self._load_conversation_state()
            
        except Exception as e:
            logger.error(f"[ERROR] Initialization failed: {e}")
            raise
    
    def _register_plugins(self):
        """Register all available plugins"""
        # Register NotesPlugin early to ensure it handles note commands before calendar
        self.plugins.register_plugin(NotesPlugin())
        self.plugins.register_plugin(WeatherPlugin())
        self.plugins.register_plugin(TimePlugin())
        self.plugins.register_plugin(FileOperationsPlugin())
        self.plugins.register_plugin(SystemControlPlugin())
        self.plugins.register_plugin(WebSearchPlugin())
        self.plugins.register_plugin(DocumentPlugin())
        self.plugins.register_plugin(CalendarPlugin())
        self.plugins.register_plugin(MusicPlugin())
        
        logger.info(f"[OK] Registered {len(self.plugins.plugins)} plugins")
    
    def _build_llm_context(self, user_input: str) -> str:
        """Build enhanced context for LLM using advanced context handler"""
        context_parts = []
        
        # 1. Personalization
        personalization = self.context.get_personalization_context()
        if personalization:
            context_parts.append(personalization)
        
        # 2. Advanced contextual understanding
        if self.advanced_context:
            advanced_context = self.advanced_context.get_context_for_llm(user_input)
            if advanced_context:
                context_parts.append(advanced_context)
        
        # 3. Intelligent conversation summarization
        if self.summarizer:
            conversation_context = self.summarizer.get_context_summary()
            if conversation_context:
                context_parts.append(f"Conversation context: {conversation_context}")
            
            # Get detailed context for complex interactions
            detailed_context = self.summarizer.get_detailed_context()
            if detailed_context.get("frequent_topics"):
                topics_text = ", ".join(detailed_context["frequent_topics"][:3])
                context_parts.append(f"Current session topics: {topics_text}")
        
        # 4. Recent conversation summary (fallback)
        recent_context = self._get_recent_conversation_summary()
        if recent_context and not self.advanced_context and not self.summarizer:
            context_parts.append(f"Recent discussion: {recent_context}")
        
        # 5. Active context tracking (fallback if advanced context not available)
        if not self.advanced_context:
            active_context = self._get_active_context()
            if active_context:
                context_parts.append(active_context)
        
        # 5. Relevant memories (RAG)
        memory_context = self.memory.get_context_for_llm(user_input, max_memories=5)  # Increased from 3
        if memory_context:
            context_parts.append(memory_context)
        
        # 6. System capabilities
        capabilities = self.plugins.get_capabilities()
        if capabilities:
            context_parts.append(f"Available capabilities: {', '.join(capabilities[:10])}")
        
        return "\n\n".join(context_parts)
    
    def process_input(self, user_input: str, use_voice: bool = False) -> str:
        """
        Process user input through the complete pipeline
        
        Args:
            user_input: User's message
            use_voice: Speak the response
            
        Returns:
            Assistant's response
        """
        try:
            logger.info(f"User: {user_input}")
            
            # 1. NLP Processing
            nlp_result = self.nlp.process(user_input)
            intent = nlp_result['intent']
            entities = nlp_result['entities']
            sentiment = nlp_result['sentiment']
            
            # Store for active learning
            self.last_user_input = user_input
            self.last_nlp_result = nlp_result.copy()
            self.last_intent = intent
            self.last_entities = entities
            
            # 1.5. Apply Active Learning improvements
            improved_nlp_result = self.learning_manager.apply_learning(user_input, nlp_result)
            if improved_nlp_result != nlp_result:
                logger.info("Active learning improved NLP result")
                intent = improved_nlp_result['intent']
                entities = improved_nlp_result['entities']
                sentiment = improved_nlp_result.get('sentiment', sentiment)
            
            # Store sentiment for use in conversation summarizer
            self._last_sentiment = sentiment['category'] if sentiment else None
            
            logger.info(f"Intent: {intent}, Sentiment: {sentiment['category']}")
            
            # 2. Advanced Context Processing (GLOBAL - for all interactions)
            context_resolved_input = user_input
            if self.advanced_context:
                # Process the turn and get any resolved references
                turn = self.advanced_context.process_turn(
                    user_input=user_input,
                    assistant_response="",  # Will be filled later
                    intent=intent,
                    entities=entities
                )
                
                # Apply any coreference resolutions to improve understanding
                if turn.entities_resolved:
                    for reference, entity_id in turn.entities_resolved.items():
                        entity = self.advanced_context.entities.get(entity_id)
                        if entity:
                            # Create more explicit version of input for better processing
                            entity_desc = f"{entity.entity_type} '{entity.data.get('subject', entity.data.get('name', entity_id))}'"
                            context_resolved_input = user_input.replace(reference, entity_desc)
                            logger.info(f"Resolved '{reference}' to {entity_desc}")
            
            # Use the context-resolved input for further processing
            user_input_processed = context_resolved_input
            
            # 3. General Entity Detection (for non-email interactions)
            if self.advanced_context and intent != "email":
                self._detect_general_entities(user_input, intent, entities)
            
            # Handle pending multi-step actions
            if self.pending_action == "compose_email":
                import re
                
                # Waiting for recipient email
                if 'recipient' not in self.pending_data:
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
                    if email_match:
                        self.pending_data['recipient'] = email_match.group(0)
                        return f"Email to: {self.pending_data['recipient']}\n\nWhat's the subject?"
                    else:
                        return "I need a valid email address. Please provide one (e.g., john@example.com)"
                
                # Waiting for subject
                elif 'subject' not in self.pending_data:
                    self.pending_data['subject'] = user_input.strip()
                    return f"Subject: {self.pending_data['subject']}\n\nWhat's the message?"
                
                # Waiting for message body
                elif 'body' not in self.pending_data:
                    self.pending_data['body'] = user_input.strip()
                    
                    # Confirm before sending
                    confirmation = f"Ready to send:\n\n"
                    confirmation += f"To: {self.pending_data['recipient']}\n"
                    confirmation += f"Subject: {self.pending_data['subject']}\n"
                    confirmation += f"Message:\n{self.pending_data['body']}\n\n"
                    confirmation += "Send this email? (yes/no)"
                    
                    self.pending_data['awaiting_confirmation'] = True
                    return confirmation
                
                # Waiting for confirmation
                elif self.pending_data.get('awaiting_confirmation'):
                    if user_input.lower() in ['yes', 'y', 'send', 'confirm']:
                        if self.gmail and self.gmail.send_email(
                            self.pending_data['recipient'],
                            self.pending_data['subject'],
                            self.pending_data['body']
                        ):
                            result = f"Email sent to {self.pending_data['recipient']}!"
                            # Clear pending action
                            self.pending_action = None
                            self.pending_data = {}
                            return result
                        else:
                            self.pending_action = None
                            self.pending_data = {}
                            return "Failed to send email. Please try again."
                    else:
                        # Cancel
                        self.pending_action = None
                        self.pending_data = {}
                        return "Email cancelled."
            
            # Handle reply confirmation
            elif self.pending_action == "confirm_reply":
                if user_input.lower() in ['yes', 'y', 'send', 'confirm']:
                    if self.gmail and self.gmail.reply_to_email(
                        self.pending_data['email_id'],
                        self.pending_data['reply_body']
                    ):
                        result = f"Reply sent!"
                        # Clear pending action
                        self.pending_action = None
                        self.pending_data = {}
                        return result
                    else:
                        self.pending_action = None
                        self.pending_data = {}
                        return "Failed to send reply. Please try again."
                else:
                    # Cancel
                    self.pending_action = None
                    self.pending_data = {}
                    return "Reply cancelled."
            
            # Check for numbered email references first (context-aware, intent-agnostic)
            import re
            number_match = re.search(r'\b(\d+)(st|nd|rd|th)?\b', user_input.lower())
            if number_match and self.last_email_list and any(word in user_input.lower() for word in ['delete', 'remove', 'read', 'open', 'archive', 'mark']):
                email_num = int(number_match.group(1))
                
                if 1 <= email_num <= len(self.last_email_list):
                    email = self.last_email_list[email_num - 1]
                    
                    # Check if already deleted
                    if email is None:
                        return f"Email #{email_num} was already deleted."
                    
                    query_lower = user_input.lower()
                    
                    # Perform action based on keywords
                    if 'delete' in query_lower or 'remove' in query_lower:
                        if self.gmail and self.gmail.delete_email(email['id']):
                            self.last_email_list[email_num - 1] = None  # Mark as deleted, don't shift list
                            return f"Deleted email #{email_num}: '{email['subject']}'"
                        else:
                            return "Failed to delete the email. Please try again."
                    
                    elif 'archive' in query_lower:
                        if self.gmail and self.gmail.archive_email(email['id']):
                            self.last_email_list[email_num - 1] = None  # Mark as archived
                            return f"Archived email #{email_num}: '{email['subject']}'"
                        else:
                            return "Failed to archive the email."
                    
                    elif 'read' in query_lower or 'open' in query_lower:
                        if self.gmail:
                            content = self.gmail.get_email_content(email['id'])
                            
                            response = f"Email #{email_num}:\n\n"
                            response += f"From: {email['from']}\n"
                            response += f"Subject: {email['subject']}\n"
                            response += f"Date: {email['date']}\n\n"
                            
                            if content:
                                if len(content) > 500:
                                    response += content[:500] + "...\n\n(Content truncated)"
                                else:
                                    response += content
                            
                            return response
                    
                    elif 'mark' in query_lower:
                        if self.gmail:
                            if 'unread' in query_lower:
                                if self.gmail.mark_as_unread(email['id']):
                                    return f"Marked email #{email_num} as unread"
                            else:
                                if self.gmail.mark_as_read(email['id']):
                                    return f"Marked email #{email_num} as read"
                else:
                    return f"I only showed you {len(self.last_email_list)} emails. Please choose a number between 1 and {len(self.last_email_list)}."
            
            # Handle email intents with Gmail plugin
            if intent == "email":
                if not self.gmail:
                    return "Gmail isn't set up yet. I'll need OAuth credentials to access your email. Want me to walk you through the setup?"
                
                # Parse what the user wants to do with email
                query_lower = user_input.lower()
                
                # Extract number from request (e.g., "2 latest emails", "show 3 emails")
                import re
                number_match = re.search(r'\b(\d+)\b', query_lower)
                requested_count = int(number_match.group(1)) if number_match else None
                
                # Check/List emails
                if any(word in query_lower for word in ['check', 'show', 'list', 'inbox']):
                    if 'unread' in query_lower:
                        # Count unread
                        count = self.gmail.get_unread_count()
                        return f"You have {count} unread email{'s' if count != 1 else ''}."
                    
                    # List recent emails
                    emails = self.gmail.get_recent_emails(max_results=5)
                    if not emails:
                        return "Your inbox appears empty or I couldn't fetch emails right now."
                    
                    # Store in context for numbered references
                    self.last_email_list = emails
                    self.last_email_context = "list"
                    
                    # Register emails with advanced context handler
                    if self.advanced_context:
                        for email in emails:
                            entity_id = self.advanced_context.add_entity(
                                entity_type="email",
                                data={
                                    'id': email['id'],
                                    'subject': email['subject'],
                                    'from': email['from'],
                                    'date': email['date'],
                                    'unread': email.get('unread', False)
                                },
                                aliases=[email['subject'], f"email from {email['from'].split('<')[0].strip()}"]
                            )
                    
                    response = "Here are your recent emails:\n\n"
                    for i, email in enumerate(emails, 1):
                        unread = "[UNREAD] " if email.get('unread') else ""
                        # Extract sender name without email address
                        from_field = email['from']
                        sender_name = from_field.split('<')[0].strip().strip('"') if '<' in from_field else from_field
                        # Simplify date - just the date part
                        date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                        
                        response += f"{i}. {unread}{email['subject']}\n"
                        response += f"   {sender_name} • {date_str}\n\n"
                    
                    return response.strip()
                
                # Read specific email(s) or latest
                elif any(word in query_lower for word in ['read', 'open', 'first', 'latest']):
                    # If user specified a number, get multiple emails
                    if requested_count and requested_count > 1:
                        emails = self.gmail.get_recent_emails(max_results=min(requested_count, 10))
                        if not emails:
                            return "Couldn't find any emails."
                        
                        # Store in context for potential follow-up actions
                        self.last_email_list = emails
                        self.last_email_context = f"latest_{len(emails)}_emails"
                        
                        # Register with advanced context handler
                        if self.advanced_context:
                            for email in emails:
                                entity_id = self.advanced_context.add_entity(
                                    entity_type="email",
                                    data={
                                        'id': email['id'],
                                        'subject': email['subject'],
                                        'from': email['from'],
                                        'date': email['date'],
                                        'unread': email.get('unread', False)
                                    },
                                    aliases=[
                                        email['subject'], 
                                        f"email from {email['from'].split('<')[0].strip()}"
                                    ]
                                )
                        
                        response = f"Your {len(emails)} latest emails:\n\n"
                        for i, email in enumerate(emails, 1):
                            unread = "[UNREAD] " if email.get('unread') else ""
                            from_field = email['from']
                            sender_name = from_field.split('<')[0].strip().strip('"') if '<' in from_field else from_field
                            date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                            
                            response += f"{i}. {unread}{email['subject']}\n"
                            response += f"   {sender_name} • {date_str}\n\n"
                        
                        return response.strip()
                    
                    else:
                        # Single latest email with full content
                        emails = self.gmail.get_recent_emails(max_results=1)
                        if not emails:
                            return "Couldn't find any emails."
                        
                        email = emails[0]
                        content = self.gmail.get_email_content(email['id'])
                        
                        # Store in context for potential follow-up actions (delete, archive, etc.)
                        self.last_email_list = emails
                        self.last_email_context = "latest_email"
                        
                        # Register with advanced context handler
                        if self.advanced_context:
                            entity_id = self.advanced_context.add_entity(
                                entity_type="email",
                                data={
                                    'id': email['id'],
                                    'subject': email['subject'],
                                    'from': email['from'],
                                    'date': email['date'],
                                    'content': content[:200] if content else "",  # Store snippet
                                    'unread': email.get('unread', False)
                                },
                                aliases=[
                                    "this email", "the email", "latest email", "recent email",
                                    email['subject'], 
                                    f"email from {email['from'].split('<')[0].strip()}"
                                ]
                            )
                        
                        response = f"Your latest email:\n\n"
                        response += f"From: {email['from']}\n"
                        response += f"Subject: {email['subject']}\n"
                        response += f"Date: {email['date']}\n\n"
                        
                        if content:
                            if len(content) > 500:
                                response += content[:500] + "...\n\n(Content truncated)"
                            else:
                                response += content
                        
                        return response
                
                # Search emails
                elif 'search' in query_lower or 'find' in query_lower or 'from' in query_lower:
                    # Extract search term
                    if 'from' in query_lower:
                        # Search by sender
                        words = query_lower.split()
                        try:
                            from_idx = words.index('from')
                            if from_idx + 1 < len(words):
                                sender = ' '.join(words[from_idx+1:]).strip('?!.')
                                emails = self.gmail.get_emails_by_sender(sender, max_results=5)
                                
                                if not emails:
                                    return f"No emails found from {sender}."
                                
                                # Store in context
                                self.last_email_list = emails
                                self.last_email_context = f"search_from_{sender}"
                                
                                response = f"Emails from {sender}:\n\n"
                                for i, email in enumerate(emails, 1):
                                    sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                                    date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                                    response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                                
                                return response.strip()
                        except:
                            pass
                    
                    # General search
                    search_terms = query_lower.replace('search', '').replace('find', '').replace('emails', '').replace('email', '').strip('?!. ')
                    if search_terms:
                        emails = self.gmail.search_emails(search_terms, max_results=5)
                        
                        if not emails:
                            return f"No emails found matching '{search_terms}'."
                        
                        # Store in context
                        self.last_email_list = emails
                        self.last_email_context = f"search_{search_terms}"
                        
                        response = f"Found {len(emails)} email(s):\n\n"
                        for i, email in enumerate(emails, 1):
                            sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                            date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                            response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                        
                        return response.strip()
                    else:
                        return "What would you like me to search for in your emails?"
                
                # Send/Write email
                elif any(word in query_lower for word in ['send', 'write', 'compose', 'draft']):
                    # Extract recipient and content
                    if '@' in user_input:
                        import re
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
                        if email_match:
                            self.pending_action = "compose_email"
                            self.pending_data = {'recipient': email_match.group(0)}
                            return f"Email to: {email_match.group(0)}\n\nWhat's the subject?"
                    
                    # Start email composition flow
                    self.pending_action = "compose_email"
                    self.pending_data = {}
                    return "Who would you like to send an email to? Please provide their email address."
                
                # Delete email
                elif 'delete' in query_lower or 'remove' in query_lower:
                    # Check for contextual references first
                    if any(word in query_lower for word in ['this', 'that', 'it', 'the email', 'the latest']):
                        # Advanced context resolution
                        if self.advanced_context:
                            resolved_ref = self.advanced_context.resolve_reference(user_input)
                            if resolved_ref and resolved_ref.startswith("email_"):
                                # Get entity data
                                entity = self.advanced_context.entities.get(resolved_ref)
                                if entity and entity.entity_type == "email":
                                    email_data = entity.data
                                    if self.gmail.delete_email(email_data['id']):
                                        # Remove from advanced context
                                        del self.advanced_context.entities[resolved_ref]
                                        # Clear simple context too
                                        self.last_email_list = []
                                        return f"Deleted email: '{email_data['subject']}' from {email_data.get('from', 'Unknown')}"
                                    else:
                                        return "Failed to delete the email. Please try again."
                                    del self.advanced_context.entities[resolved_ref]
                                    # Clear simple context too
                                    self.last_email_list = []
                                    return f"Deleted email: '{email_data['subject']}' from {email_data.get('from', 'Unknown')}"
                                else:
                                    return "Failed to delete the email. Please try again."
                    
                    # Fallback to simple context
                            # Single email in context (like "latest email")
                            email = self.last_email_list[0]
                            if self.gmail.delete_email(email['id']):
                                # Clear the context since we deleted it
                                self.last_email_list = []
                                return f"Deleted email: '{email['subject']}' from {email['from']}"
                            else:
                                return "Failed to delete the email. Please try again."
                        elif self.last_email_list and len(self.last_email_list) > 1:
                            # Multiple emails in context - ask for clarification
                            return "I see multiple emails from our last search. Which one would you like to delete? Use the number (e.g., 'delete email 1')"
                        else:
                            return "I don't see any emails in our current context. Please show me some emails first or be more specific about which email to delete."
                    
                    # Extract search criteria
                    search_terms = query_lower.replace('delete', '').replace('remove', '').replace('email', '').replace('that says', '').replace('about', '').strip('?!. ')
                    
                    if not search_terms:
                        return "Which email would you like to delete? Please specify (e.g., 'delete email from Amazon' or 'delete email about meeting')"
                    
                    # Search for matching emails
                    emails = self.gmail.search_emails(search_terms, max_results=5)
                    
                    if not emails:
                        return f"No emails found matching '{search_terms}'."
                    
                    if len(emails) == 1:
                        # Delete the single match
                        email = emails[0]
                        if self.gmail.delete_email(email['id']):
                            return f"Deleted email: '{email['subject']}' from {email['from']}"
                        else:
                            return "Failed to delete the email. Please try again."
                    else:
                        # Multiple matches - ask for confirmation
                        self.last_email_list = emails
                        response = f"Found {len(emails)} emails matching '{search_terms}':\n\n"
                        for i, email in enumerate(emails, 1):
                            sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                            date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                            response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                        response += "\nWhich one would you like to delete? (Tell me the number or be more specific)"
                        return response.strip()
                
                # Archive email
                elif 'archive' in query_lower:
                    # Check for contextual references first
                    if any(word in query_lower for word in ['this', 'that', 'it', 'the email', 'the latest']):
                        # User is referring to a previously shown email
                        if self.last_email_list and len(self.last_email_list) == 1:
                            # Single email in context
                            email = self.last_email_list[0]
                            if self.gmail.archive_email(email['id']):
                                self.last_email_list = []
                                return f"Archived: '{email['subject']}'"
                            else:
                                return "Failed to archive the email."
                        elif self.last_email_list and len(self.last_email_list) > 1:
                            return "I see multiple emails. Which one would you like to archive? Use the number (e.g., 'archive email 2')"
                        else:
                            return "I don't see any emails in our current context. Please show me some emails first."
                    
                    search_terms = query_lower.replace('archive', '').replace('email', '').strip('?!. ')
                    
                    if not search_terms:
                        return "Which email would you like to archive?"
                    
                    emails = self.gmail.search_emails(search_terms, max_results=5)
                    
                    if not emails:
                        return f"No emails found matching '{search_terms}'."
                    
                    if len(emails) == 1:
                        email = emails[0]
                        if self.gmail.archive_email(email['id']):
                            return f"Archived: '{email['subject']}'"
                        else:
                            return "Failed to archive the email."
                    else:
                        self.last_email_list = emails
                        response = f"Found {len(emails)} emails. Which one?\n\n"
                        for i, email in enumerate(emails, 1):
                            sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                            date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                            response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                        return response.strip()
                
                # Mark as read/unread
                elif 'mark' in query_lower:
                    if 'unread' in query_lower:
                        # Mark as unread
                        search_terms = query_lower.replace('mark', '').replace('as', '').replace('unread', '').replace('email', '').strip('?!. ')
                        
                        if search_terms:
                            emails = self.gmail.search_emails(search_terms, max_results=1)
                            if emails:
                                if self.gmail.mark_as_unread(emails[0]['id']):
                                    return f"Marked as unread: '{emails[0]['subject']}'"
                        return "Which email would you like to mark as unread?"
                    else:
                        # Mark as read
                        search_terms = query_lower.replace('mark', '').replace('as', '').replace('read', '').replace('email', '').strip('?!. ')
                        
                        if search_terms:
                            emails = self.gmail.search_emails(search_terms, max_results=1)
                            if emails:
                                if self.gmail.mark_as_read(emails[0]['id']):
                                    return f"Marked as read: '{emails[0]['subject']}'"
                        return "Which email would you like to mark as read?"
                
                # Count unread
                elif 'unread' in query_lower or 'how many' in query_lower:
                    count = self.gmail.get_unread_count()
                    return f"You have {count} unread email{'s' if count != 1 else ''}."
                
                # Emails with attachments
                elif 'attachment' in query_lower:
                    emails = self.gmail.get_emails_with_attachments(max_results=5)
                    
                    if not emails:
                        return "No recent emails with attachments found."
                    
                    self.last_email_list = emails
                    response = f"Found {len(emails)} email(s) with attachments:\n\n"
                    for i, email in enumerate(emails, 1):
                        sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                        date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                        response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                    
                    return response.strip()
                
                # Reply to email
                elif 'reply' in query_lower:
                    # Check if we have a pending reply action
                    if self.pending_action == "reply_email":
                        # User is providing the reply message
                        reply_body = user_input.strip()
                        email_id = self.pending_data.get('email_id')
                        subject = self.pending_data.get('subject', 'Unknown')
                        
                        # Show preview and ask for confirmation
                        self.pending_action = "confirm_reply"
                        self.pending_data['reply_body'] = reply_body
                        
                        return f"Reply to '{subject}':\n\n{reply_body}\n\nSend this reply? (yes/no)"
                    
                    # Look for email reference number
                    email_ref = None
                    for word in user_input.split():
                        if word.isdigit():
                            email_ref = int(word)
                            break
                    
                    if email_ref and self.last_email_list and 0 < email_ref <= len(self.last_email_list):
                        email = self.last_email_list[email_ref - 1]
                        if email:  # Check if not deleted (None marker)
                            # Start reply flow
                            self.pending_action = "reply_email"
                            self.pending_data = {'email_id': email['id'], 'subject': email['subject']}
                            return f"Replying to '{email['subject']}'. What would you like to say?"
                    elif 'latest' in query_lower or 'last' in query_lower or 'recent' in query_lower:
                        # Reply to most recent email
                        emails = self.gmail.get_recent_emails(max_results=1)
                        if emails:
                            self.pending_action = "reply_email"
                            self.pending_data = {'email_id': emails[0]['id'], 'subject': emails[0]['subject']}
                            return f"Replying to '{emails[0]['subject']}'. What would you like to say?"
                    
                    return "Which email would you like to reply to? Use the email number from the list."
                
                # Star/flag email
                elif 'star' in query_lower or 'flag' in query_lower:
                    # Look for email reference number
                    email_ref = None
                    for word in user_input.split():
                        if word.isdigit():
                            email_ref = int(word)
                            break
                    
                    if email_ref and self.last_email_list and 0 < email_ref <= len(self.last_email_list):
                        email = self.last_email_list[email_ref - 1]
                        if email:  # Check if not deleted (None marker)
                            if self.gmail.star_email(email['id']):
                                return f"Starred '{email['subject']}'"
                    
                    return "Which email would you like to star? Use the email number from the list."
                
                # Unstar/unflag email
                elif 'unstar' in query_lower or 'unflag' in query_lower or 'remove star' in query_lower:
                    # Look for email reference number
                    email_ref = None
                    for word in user_input.split():
                        if word.isdigit():
                            email_ref = int(word)
                            break
                    
                    if email_ref and self.last_email_list and 0 < email_ref <= len(self.last_email_list):
                        email = self.last_email_list[email_ref - 1]
                        if email:  # Check if not deleted (None marker)
                            if self.gmail.unstar_email(email['id']):
                                return f"Removed star from '{email['subject']}'"
                    
                    return "Which email would you like to unstar? Use the email number from the list."
                
                else:
                    return "I can help you:\n- Check emails: 'check my emails'\n- Read latest: 'read first email'\n- Search: 'find emails from Amazon'\n- Send: 'write an email'\n- Reply: 'reply to email 2'\n- Star: 'star email 3'\n- Delete: 'delete email 1'\n- Archive: 'archive email 4'\n\nWhat would you like to do?"
            
            # 2. Check if plugin can handle this
            context_summary = self.context.get_context_summary()
            plugin_result = self.plugins.execute_for_intent(
                intent, user_input, entities, context_summary
            )
            
            if plugin_result:
                # Plugin handled the request, regardless of success/failure
                response = plugin_result['response']
                plugin_name = plugin_result.get('plugin', 'Unknown')
                success = plugin_result.get('success', False)
                
                if success:
                    logger.info(f"[PLUGIN]  Handled by: {plugin_name}")
                else:
                    logger.info(f"[PLUGIN]  Failed in: {plugin_name}")
                
                # Store interaction in memory
                self._store_interaction(user_input, response, intent, entities)
                
                # Speak if voice enabled
                if use_voice and self.speech:
                    self.speech.speak(response, blocking=False)
                
                return response
            
            # 3. If plugin couldn't handle or failed, use LLM with context
            
            # Handle relationship queries before LLM generation
            if self.relationship_tracker:
                relationship_response = self._handle_relationship_query(user_input, intent, entities)
                if relationship_response:
                    return relationship_response
            
            # 3. Use LLM for general conversation
            # Build enhanced context using the processed input
            enhanced_context = self._build_llm_context(user_input_processed)
            
            # Add active learning guidance if available
            learning_guidance = improved_nlp_result.get("learning_guidance")
            if learning_guidance:
                guidance_text = self._format_learning_guidance(learning_guidance)
                enhanced_context = f"{enhanced_context}\n\n{guidance_text}"
            
            # Prepend context to conversation if available
            if enhanced_context:
                # Temporarily add context message
                self.llm.conversation_history.insert(0, {
                    "role": "system",
                    "content": f"Context: {enhanced_context}"
                })
            
            # Get LLM response using the context-resolved input
            response = self.llm.chat(user_input_processed, use_history=True)
            
            # Update the advanced context with the response
            if self.advanced_context:
                # Update the turn with the response
                turn.assistant_response = response
                
                # Extract any entities mentioned in the response and track them
                self._track_response_entities(response, intent)
            
            # Remove context message
            if enhanced_context and self.llm.conversation_history:
                if self.llm.conversation_history[0]["role"] == "system":
                    self.llm.conversation_history.pop(0)
            
            # 4. Store interaction
            self._store_interaction(user_input, response, intent, entities)
            
            # Store response for active learning
            self.last_assistant_response = response
            
            # 5. Speak if voice enabled
            if use_voice and self.speech:
                self.speech.speak(response, blocking=False)
            
            logger.info(f"A.L.I.C.E: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"[ERROR] Error processing input: {e}")
            error_response = "I apologize, but I encountered an error processing your request."
            
            if use_voice and self.speech:
                self.speech.speak(error_response, blocking=False)
            
            return error_response
    
    def _track_response_entities(self, response: str, intent: str):
        """Track entities mentioned in assistant responses"""
        if not self.advanced_context:
            return
            
        # Track files mentioned
        import re
        file_patterns = [
            r"created? (?:a )?file (?:called )?['\"]([^'\"]+)['\"]",
            r"(?:file|document) ['\"]([^'\"]+)['\"]",
            r"saved? (?:to|as) ['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in file_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                filename = match.group(1)
                self.advanced_context.add_entity(
                    entity_type="file",
                    data={"name": filename, "mentioned_in_response": True},
                    aliases=["the file", "this file", filename]
                )
        
        # Track people mentioned
        person_patterns = [
            r"(?:from|by|to) ([A-Z][a-z]+ [A-Z][a-z]+)",  # Full names
            r"(?:from|by|to) ([A-Z][a-z]+)",  # First names
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                person_name = match.group(1)
                self.advanced_context.add_entity(
                    entity_type="person",
                    data={"name": person_name, "mentioned_in_response": True},
                    aliases=["this person", person_name]
                )
        
        # Track topics mentioned
        if intent in ["weather", "time", "calculation"]:
            topic_name = intent.replace("_", " ")
            self.advanced_context.add_entity(
                entity_type="topic",
                data={"name": topic_name, "intent": intent},
                aliases=["this topic", "that", topic_name]
            )
    
    def _detect_general_entities(self, user_input: str, intent: str, entities: Dict):
        """Detect and track general entities from user input"""
        if not self.advanced_context:
            return
            
        import re
        
        # Detect file mentions
        file_patterns = [
            r"(?:file|document) (?:called |named )?['\"]([^'\"]+)['\"]",
            r"create (?:a )?file ['\"]([^'\"]+)['\"]",
            r"(?:open|read|edit) ['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in file_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            for match in matches:
                filename = match.group(1)
                self.advanced_context.add_entity(
                    entity_type="file",
                    data={"name": filename, "mentioned_by_user": True},
                    aliases=["the file", "this file", "that file", filename]
                )
        
        # Detect person mentions
        person_patterns = [
            r"(?:tell|ask|message|email|call) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:with|from|to) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            for match in matches:
                person_name = match.group(1)
                if person_name.lower() not in ["alice", "you", "me", "i"]:  # Skip self-references
                    self.advanced_context.add_entity(
                        entity_type="person",
                        data={"name": person_name, "mentioned_by_user": True},
                        aliases=["this person", "they", "them", person_name]
                    )
        
        # Detect location mentions
        location_patterns = [
            r"(?:in|at|to|from) ([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)",
            r"weather (?:in|at|for) ([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            for match in matches:
                location = match.group(1)
                # Simple validation - avoid common words
                if location.lower() not in ["the", "this", "that", "there", "here", "now", "today"]:
                    self.advanced_context.add_entity(
                        entity_type="location",
                        data={"name": location, "mentioned_by_user": True},
                        aliases=["there", "this place", "that location", location]
                    )
        
        # Detect task/topic mentions based on intent
        if intent in ["file_operation", "system_control", "weather", "time"]:
            topic_data = {
                "intent": intent,
                "user_input": user_input[:100],  # Store snippet
                "mentioned_by_user": True
            }
            
            self.advanced_context.add_entity(
                entity_type="topic",
                data=topic_data,
                aliases=["this", "that", intent.replace("_", " ")]
            )
    
    def _get_recent_conversation_summary(self) -> str:
        """Get a summary of the last few conversation exchanges"""
        if not self.conversation_summary:
            return ""
        
        # Get last 3 exchanges
        recent = self.conversation_summary[-3:]
        summary_parts = []
        
        for exchange in recent:
            # Create concise summary
            user_text = exchange['user'][:50] + "..." if len(exchange['user']) > 50 else exchange['user']
            summary_parts.append(f"User asked: {user_text}")
        
        return " | ".join(summary_parts)
    
    def _get_active_context(self) -> str:
        """Get currently active context (email lists, pending actions, etc.)"""
        context_parts = []
        
        # Track email context
        if self.last_email_list:
            num_emails = sum(1 for e in self.last_email_list if e is not None)
            if num_emails > 0:
                context_parts.append(f"User is viewing {num_emails} emails from their inbox")
        
        # Track pending actions
        if self.pending_action:
            context_parts.append(f"In progress: {self.pending_action.replace('_', ' ')}")
        
        # Track referenced items
        if self.referenced_items:
            refs = ", ".join(f"{k}: {v}" for k, v in list(self.referenced_items.items())[-3:])
            context_parts.append(f"Recently referenced: {refs}")
        
        # Track topics
        if self.conversation_topics:
            topics = ", ".join(self.conversation_topics[-3:])
            context_parts.append(f"Current topics: {topics}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _load_conversation_state(self):
        """Load conversation state from previous session"""
        import pickle
        import os
        
        state_file = "data/conversation_state.pkl"
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                    
                    # Restore conversation summary (only recent ones)
                    if 'conversation_summary' in state:
                        self.conversation_summary = state['conversation_summary'][-5:]  # Last 5 only
                    
                    # Restore topics
                    if 'conversation_topics' in state:
                        self.conversation_topics = state['conversation_topics'][-5:]
                    
                    # Restore referenced items
                    if 'referenced_items' in state:
                        self.referenced_items = state['referenced_items']
                    
                    logger.info("[OK] Previous conversation context restored")
            except Exception as e:
                logger.warning(f"[WARNING] Could not load conversation state: {e}")
        
        # Load advanced context state if available
        if self.advanced_context:
            advanced_state_file = "data/advanced_context_state.pkl"
            if os.path.exists(advanced_state_file):
                try:
                    self.advanced_context.load_state(advanced_state_file)
                    logger.info("[OK] Advanced context state restored")
                except Exception as e:
                    logger.warning(f"[WARNING] Could not load advanced context state: {e}")
    
    def _save_conversation_state(self):
        """Save conversation state for next session"""
        import pickle
        import os
        
        os.makedirs("data", exist_ok=True)
        state_file = "data/conversation_state.pkl"
        
        try:
            state = {
                'conversation_summary': self.conversation_summary[-10:],  # Keep last 10
                'conversation_topics': self.conversation_topics[-10:],
                'referenced_items': self.referenced_items,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            
            # Save advanced context state if available
            if self.advanced_context:
                advanced_state_file = "data/advanced_context_state.pkl"
                self.advanced_context.save_state(advanced_state_file)
            
            logger.info("[OK] Conversation state saved")
        except Exception as e:
            logger.warning(f"[WARNING] Could not save conversation state: {e}")
    
    def _store_interaction(
        self,
        user_input: str,
        response: str,
        intent: str,
        entities: Dict
    ):
        """Store interaction in memory and context"""
        try:
            # Process with advanced context handler
            if self.advanced_context:
                turn = self.advanced_context.process_turn(
                    user_input=user_input,
                    assistant_response=response,
                    intent=intent,
                    entities=entities
                )
            
            # Add to conversation summarizer for intelligent context management
            if self.summarizer:
                # Extract entity list from entities dict
                entity_list = []
                if entities:
                    for entity_type, entity_values in entities.items():
                        if isinstance(entity_values, list):
                            entity_list.extend(entity_values)
                        elif entity_values:
                            entity_list.append(str(entity_values))
                
                # Get sentiment from NLP result if available
                sentiment = getattr(self, '_last_sentiment', None)
                
                self.summarizer.add_turn(
                    user_input=user_input,
                    assistant_response=response,
                    intent=intent,
                    entities=entity_list,
                    sentiment=sentiment
                )
            
            # Extract and store entity relationships
            if self.relationship_tracker:
                try:
                    # Extract relationships from user input
                    relationships = self.relationship_tracker.process_text(user_input)
                    logger.debug(f"Extracted {len(relationships)} relationships from user input")
                    
                    # Also process assistant response for relationship context
                    if response and len(response) < 500:  # Only process shorter responses
                        assistant_relationships = self.relationship_tracker.process_text(response)
                        logger.debug(f"Extracted {len(assistant_relationships)} relationships from assistant response")
                except Exception as e:
                    logger.error(f"Error extracting relationships: {e}")
            
            # Add to conversation summary (keep last 10)
            self.conversation_summary.append({
                'user': user_input,
                'assistant': response[:200],  # Truncate for summary
                'intent': intent,
                'timestamp': datetime.now().isoformat()
            })
            if len(self.conversation_summary) > 10:
                self.conversation_summary.pop(0)
            
            # Track topics from intent and entities
            if intent and intent != "unknown":
                if intent not in self.conversation_topics:
                    self.conversation_topics.append(intent)
            
            # Track entities as references
            if entities:
                for entity_type, entity_values in entities.items():
                    if entity_values:
                        # Store most recent reference
                        self.referenced_items[entity_type] = entity_values[0] if isinstance(entity_values, list) else entity_values
            
            # Limit tracking size
            if len(self.conversation_topics) > 10:
                self.conversation_topics = self.conversation_topics[-10:]
            if len(self.referenced_items) > 15:
                # Keep most recent 15
                keys = list(self.referenced_items.keys())[-15:]
                self.referenced_items = {k: self.referenced_items[k] for k in keys}
            
            # Update context
            entity_list = []
            for entity_type, values in entities.items():
                entity_list.extend(values)
            
            self.context.update_conversation(user_input, response, intent, entity_list)
            
            # Store in episodic memory with enhanced metadata
            self.memory.store_memory(
                content=f"User: {user_input} | Assistant: {response}",
                memory_type="episodic",
                context={
                    "intent": intent,
                    "entities": entities,
                    "topics": self.conversation_topics[-3:],
                    "has_email_context": bool(self.last_email_list),
                    "pending_action": self.pending_action
                },
                importance=0.6,
                tags=["conversation", intent]
            )
            
        except Exception as e:
            logger.warning(f"[WARNING] Error storing interaction: {e}")
    
    def run_interactive(self):
        """Run interactive console mode"""
        self.running = True
        
        print("\n" + "=" * 80)
        print("A.L.I.C.E - Your Personal AI Assistant")
        print("=" * 80)
        print(f"\nHello {self.context.user_prefs.name}! I'm ALICE, your advanced AI assistant.")
        print("I'm here to help you with anything you need.\n")
        print("Commands:")
        print("   /help      - Show available commands")
        print("   /voice     - Toggle voice mode")
        print("   /clear     - Clear conversation history")
        print("   /memory    - Show memory statistics")
        print("   /plugins   - List available plugins")
        print("   /location  - Set or view your location")
        print("   /status    - Show system status")
        print("   /save      - Save current state")
        print("   /summary   - Get conversation summary")
        print("   /context   - Show current context")
        print("   /topics    - List conversation topics")
        print("   /entities  - Show tracked entities")
        print("   /relationships - Show entity relationships")
        print()
        print("Debug Commands:")
        print("   /correct   - Correct my last response")
        print("   /feedback  - Rate my last response")
        print("   /learning  - Show learning statistics")
        print("   exit       - End conversation")
        print("=" * 80)
        
        # Greet user
        greeting = self._get_greeting()
        print(f"\nA.L.I.C.E: {greeting}\n")
        
        if self.speech and self.voice_enabled:
            self.speech.speak(greeting, blocking=False)
        
        while self.running:
            try:
                # Get input
                user_input = input(f"\n{self.context.user_prefs.name}: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # Handle exit
                if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                    farewell = self._get_farewell()
                    print(f"\nA.L.I.C.E: {farewell}\n")
                    
                    if self.speech and self.voice_enabled:
                        self.speech.speak(farewell, blocking=True)
                    
                    self.shutdown()
                    break
                
                # Process input
                response = self.process_input(user_input, use_voice=self.voice_enabled)
                print(f"\nA.L.I.C.E: {response}")
                
            except KeyboardInterrupt:
                farewell = self._get_farewell()
                print(f"\n\nA.L.I.C.E: {farewell}")
                self.shutdown()
                break
            except Exception as e:
                logger.error(f"[ERROR] Error in interactive loop: {e}")
                print(f"\n[ERROR] Error: {e}")
    
    def run_voice_mode(self):
        """Run voice-activated mode"""
        if not self.speech:
            logger.error("[ERROR] Voice mode not available - speech engine not initialized")
            return
        
        self.running = True
        
        print("\n" + "=" * 80)
        print("Voice Mode Activated")
        print("=" * 80)
        print(f"\nListening for wake words: {', '.join(self.speech.config.wake_words)}")
        print("Say the wake word followed by your command.")
        print("Press Ctrl+C to exit voice mode.\n")
        
        def handle_voice_command(command: str):
            """Handle voice command"""
            if command.lower() in ['exit', 'quit', 'goodbye']:
                print("\nExiting voice mode...")
                self.speech.speak("Exiting voice mode. Goodbye!")
                self.speech.stop_listening()
                self.running = False
                return
            
            print(f"\nYou said: {command}")
            response = self.process_input(command, use_voice=True)
            print(f"A.L.I.C.E: {response}\n")
        
        # Start listening for wake word
        try:
            self.speech.listen_for_wake_word(handle_voice_command, background=False)
        except KeyboardInterrupt:
            print("\n\nExiting voice mode...")
            self.speech.stop_listening()
    
    def _handle_command(self, command: str):
        """Handle system commands"""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            print("\nAvailable Commands:")
            print("   /help              - Show this help message")
            print("   /voice             - Toggle voice mode on/off")
            print("   /clear             - Clear conversation history")
            print("   /memory            - Show memory system statistics")
            print("   /plugins           - List all available plugins")
            print("   /status            - Show system status and capabilities")
            print("   /location [City]   - Set or view your location")
            print("   /save              - Save current state manually")
            print("   /summary           - Get conversation summary")
            print("   /context           - Show current context")
            print("   /topics            - List conversation topics")
            print("   /entities          - Show tracked entities")
            print("   /relationships     - Show entity relationships")
            print()
            print("Debug Commands:")
            print("   /correct [type]    - Correct A.L.I.C.E's last response")
            print("   /feedback [rating] - Rate A.L.I.C.E's last response (1-5)")
            print("   /learning          - Show active learning statistics")
            print("   exit               - End conversation and exit")
        
        elif cmd == '/voice':
            if self.speech:
                self.voice_enabled = not self.voice_enabled
                status = "enabled" if self.voice_enabled else "disabled"
                print(f"\n[OK] Voice mode {status}")
            else:
                print("\n[ERROR] Voice engine not available")
        
        elif cmd == '/clear':
            self.llm.clear_history()
            self.context.clear_short_term_memory()
            print("\n[OK] Conversation history cleared")
        
        elif cmd == '/memory':
            stats = self.memory.get_statistics()
            print("\n📊 Memory Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        elif cmd == '/plugins':
            plugins = self.plugins.get_all_plugins()
            print("\nAvailable Plugins:")
            for plugin in plugins:
                status = "[ON]" if plugin['enabled'] else "[PAUSED]"
                print(f"   {status} {plugin['name']}: {plugin['description']}")
        
        elif cmd == '/status':
            print("\n📊 System Status:")
            status = self.context.get_system_status()
            print(f"   LLM Model: {status.get('llm_model', 'N/A')}")
            print(f"   Voice: {'Enabled' if status.get('voice_enabled') else 'Disabled'}")
            print(f"   Plugins: {len(self.plugins.plugins)}")
            print(f"   Capabilities: {len(status.get('capabilities', []))}")
            
            memory_stats = self.memory.get_statistics()
            print(f"   Total Memories: {memory_stats['total_memories']}")
        
        elif cmd.startswith('/location'):
            # Set location manually: /location City, Country
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                location_parts = parts[1].split(',')
                city = location_parts[0].strip()
                country = location_parts[1].strip() if len(location_parts) > 1 else None
                self.context.user_prefs.set_location(city, country)
                self.context.save_context()
                print(f"\n[OK] Location set to: {self.context.user_prefs.location}")
            else:
                current = self.context.user_prefs.location or "Not set"
                print(f"\nCurrent location: {current}")
                print("Usage: /location City, Country")
        
        elif cmd == '/save':
            self._save_conversation_state()
            self.context.save_context()
            self.memory._save_memories()
            print("\n[OK] Conversation state saved successfully")
        
        elif cmd == '/summary':
            if self.summarizer:
                try:
                    summary_text = self.summarizer.get_conversation_summary()
                    print("\n📝 Conversation Summary:")
                    print("=" * 50)
                    print(summary_text)
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get summary: {e}")
            else:
                print("\n[ERROR] Conversation summarizer not available")
        
        elif cmd == '/context':
            if self.summarizer:
                try:
                    context_summary = self.summarizer.get_context_summary()
                    print("\n🧠 Current Context:")
                    print("=" * 50)
                    print(context_summary)
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get context: {e}")
            else:
                print("\n[ERROR] Conversation summarizer not available")
        
        elif cmd == '/topics':
            if self.summarizer:
                try:
                    context = self.summarizer.get_detailed_context()
                    topics = context.get('frequent_topics', [])
                    print("\n🏷️ Conversation Topics:")
                    print("=" * 50)
                    if topics:
                        for i, topic in enumerate(topics, 1):
                            print(f"   {i}. {topic.title()}")
                    else:
                        print("   No topics identified yet.")
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get topics: {e}")
            else:
                print("\n[ERROR] Conversation summarizer not available")
        
        elif cmd == '/entities':
            if self.relationship_tracker:
                try:
                    stats = self.relationship_tracker.get_statistics()
                    print("\n👥 Tracked Entities:")
                    print("=" * 50)
                    print(f"Total entities: {stats['total_entities']}")
                    
                    if stats['most_connected_entities']:
                        print("\nMost connected entities:")
                        for entity, count in stats['most_connected_entities']:
                            print(f"   • {entity.title()}: {count} connections")
                    
                    if stats['entity_types']:
                        print(f"\nEntity types: {', '.join(stats['entity_types'].keys())}")
                    
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get entities: {e}")
            else:
                print("\n[ERROR] Entity relationship tracker not available")
        
        elif cmd == '/relationships':
            if self.relationship_tracker:
                try:
                    stats = self.relationship_tracker.get_statistics()
                    print("\n🔗 Entity Relationships:")
                    print("=" * 50)
                    print(f"Total relationships: {stats['total_relationships']}")
                    
                    if stats['relationship_types']:
                        print("\nRelationship types:")
                        for rel_type, count in stats['relationship_types'].items():
                            print(f"   • {rel_type.replace('_', ' ').title()}: {count}")
                    
                    if stats['recent_relationships']:
                        print(f"\nRecent relationships:")
                        for rel in stats['recent_relationships'][:5]:
                            source = rel['source_entity'].title()
                            target = rel['target_entity'].title()
                            rel_type = rel['relationship_type'].replace('_', ' ')
                            confidence = rel['confidence']
                            print(f"   • {source} {rel_type} {target} (confidence: {confidence:.2f})")
                    
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get relationships: {e}")
            else:
                print("\n[ERROR] Entity relationship tracker not available")
        
        elif cmd.startswith('/correct'):
            self._handle_correction_command(cmd)
        
        elif cmd.startswith('/feedback'):
            self._handle_feedback_command(cmd)
        
        elif cmd == '/learning':
            self._handle_learning_stats_command()
        
        else:
            print(f"\n[ERROR] Unknown command: {command}")
            print("   Type /help for available commands")
    
    def _get_greeting(self) -> str:
        """Get contextual greeting based on time of day"""
        hour = datetime.now().hour
        name = self.context.user_prefs.name
        
        if 5 <= hour < 12:
            greetings = [
                f"Morning, {name}. What are we working on today?",
                f"Good morning, {name}. How can I help?",
                f"Hey {name}, ready to get started?",
                f"Morning! What's on your mind, {name}?",
            ]
        elif 12 <= hour < 17:
            greetings = [
                f"Hey {name}, what can I help with?",
                f"Afternoon, {name}. What do you need?",
                f"Hi {name}. What are you working on?",
                f"Good afternoon, {name}. How's it going?",
            ]
        elif 17 <= hour < 22:
            greetings = [
                f"Evening, {name}. What can I do for you?",
                f"Hey {name}, need a hand with something?",
                f"Good evening, {name}. What's up?",
                f"Hi {name}. What are we tackling tonight?",
            ]
        else:
            greetings = [
                f"Late night, {name}. What are you working on?",
                f"Still up, {name}? How can I help?",
                f"Hey {name}, burning the midnight oil?",
                f"Evening, {name}. What do you need?",
            ]
        
        return random.choice(greetings)
    
    def _get_farewell(self) -> str:
        """Get varied farewell message"""
        name = self.context.user_prefs.name
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            farewells = [
                f"Have a great day, {name}!",
                f"Good luck with everything today, {name}.",
                f"See you later, {name}. Take care!",
                f"Catch you later, {name}!",
            ]
        elif 12 <= hour < 17:
            farewells = [
                f"Have a good one, {name}!",
                f"Take care, {name}.",
                f"See you around, {name}!",
                f"Later, {name}. Good luck with your projects!",
            ]
        elif 17 <= hour < 22:
            farewells = [
                f"Have a good evening, {name}!",
                f"Take it easy, {name}.",
                f"See you later, {name}!",
                f"Good night, {name}. Hope you get some rest!",
            ]
        else:
            farewells = [
                f"Get some rest, {name}!",
                f"Good night, {name}. Don't stay up too late!",
                f"See you tomorrow, {name}!",
                f"Take care, {name}. Sleep well!",
            ]
        
        return random.choice(farewells)
    
    def _handle_relationship_query(self, user_input: str, intent: str, entities: Dict[str, Any]) -> Optional[str]:
        """Handle relationship queries like 'who does John work for?' or 'tell me about Sarah'"""
        query_lower = user_input.lower()
        
        # Patterns for relationship queries
        relationship_query_patterns = [
            r'who does (\w+) work (?:for|with)',
            r'where does (\w+) live',
            r'who is (\w+)(?:\s+to\s+(\w+))?',
            r'tell me about (\w+)',
            r'what do you know about (\w+)',
            r"(\w+)(?:'s)?\s+(?:relationship|connection)\s+(?:with|to)\s+(\w+)",
            r'how are (\w+) and (\w+) (?:related|connected)',
            r'(\w+) and (\w+) relationship',
        ]
        
        for pattern in relationship_query_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entity_name = match.group(1)
                second_entity = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                
                # Get relationships for the entity
                relationships = self.relationship_tracker.get_entity_relationships(entity_name)
                
                if not relationships:
                    return f"I don't have any information about relationships involving {entity_name.title()}."
                
                # Format relationships
                response_parts = [f"Here's what I know about {entity_name.title()}:"]
                
                # Group relationships by type
                by_type = defaultdict(list)
                for rel in relationships:
                    if rel.source_entity == entity_name.lower():
                        by_type[rel.relationship_type].append(f"{rel.relationship_type.replace('_', ' ')} {rel.target_entity.title()}")
                    else:
                        by_type[rel.relationship_type].append(f"is {rel.relationship_type.replace('_', ' ')} by {rel.source_entity.title()}")
                
                for rel_type, connections in by_type.items():
                    if len(connections) == 1:
                        response_parts.append(f"• {connections[0]}")
                    else:
                        response_parts.append(f"• {rel_type.replace('_', ' ')}: {', '.join([c.replace(rel_type.replace('_', ' '), '').strip() for c in connections])}")
                
                # If asking about specific relationship between two entities
                if second_entity:
                    specific_rels = [
                        rel for rel in relationships
                        if (rel.source_entity == second_entity.lower() or rel.target_entity == second_entity.lower())
                    ]
                    if specific_rels:
                        response_parts.append(f"\nConnection with {second_entity.title()}:")
                        for rel in specific_rels:
                            if rel.source_entity == entity_name.lower():
                                response_parts.append(f"• {entity_name.title()} {rel.relationship_type.replace('_', ' ')} {rel.target_entity.title()}")
                            else:
                                response_parts.append(f"• {rel.source_entity.title()} {rel.relationship_type.replace('_', ' ')} {entity_name.title()}")
                
                return "\n".join(response_parts)
        
        # Check if user is asking for general relationship information
        if any(word in query_lower for word in ['relationships', 'connections', 'network']):
            stats = self.relationship_tracker.get_statistics()
            if stats['total_relationships'] == 0:
                return "I haven't tracked any entity relationships yet. Have a conversation mentioning people, places, or things and I'll start building a relationship map!"
            
            response_parts = [
                f"I've tracked {stats['total_relationships']} relationships between {stats['total_entities']} entities.",
            ]
            
            if stats['most_connected_entities']:
                response_parts.append("\nMost connected entities:")
                for entity, count in stats['most_connected_entities'][:3]:
                    response_parts.append(f"• {entity.title()}: {count} connections")
            
            return "\n".join(response_parts)
        
        return None
    
    def _handle_correction_command(self, command: str):
        """Handle correction commands"""
        if not self.last_user_input or not self.last_assistant_response:
            print("\n[ERROR] No previous interaction to correct")
            return
        
        parts = command.split(" ", 1)
        correction_type = parts[1] if len(parts) > 1 else ""
        
        print(f"\n📝 Correction Mode")
        print("=" * 50)
        print(f"Last input: {self.last_user_input}")
        print(f"Last response: {self.last_assistant_response[:200]}...")
        print()
        
        if not correction_type:
            print("Available correction types:")
            print("   intent     - Correct intent classification")
            print("   entity     - Correct entity extraction")
            print("   response   - Correct response quality")
            print("   sentiment  - Correct sentiment analysis")
            print("   factual    - Correct factual error")
            print()
            correction_type = input("Correction type: ").strip().lower()
        
        if correction_type == "intent":
            print(f"Current intent: {self.last_intent}")
            new_intent = input("Correct intent: ").strip()
            if new_intent:
                self.learning_manager.record_correction(
                    CorrectionType.INTENT_CLASSIFICATION,
                    self.last_user_input,
                    self.last_intent,
                    new_intent,
                    f"User corrected intent from '{self.last_intent}' to '{new_intent}'",
                    {"original_nlp_result": self.last_nlp_result}
                )
                print(f"✅ Recorded intent correction: {self.last_intent} → {new_intent}")
        
        elif correction_type == "entity":
            print(f"Current entities: {self.last_entities}")
            print("Enter correct entities (JSON format):")
            try:
                new_entities_input = input("Correct entities: ").strip()
                if new_entities_input:
                    new_entities = eval(new_entities_input)  # Simple eval for demo
                    self.learning_manager.record_correction(
                        CorrectionType.ENTITY_EXTRACTION,
                        self.last_user_input,
                        self.last_entities,
                        new_entities,
                        "User corrected entity extraction",
                        {"original_nlp_result": self.last_nlp_result}
                    )
                    print("✅ Recorded entity correction")
            except Exception as e:
                print(f"[ERROR] Invalid entity format: {e}")
        
        elif correction_type == "response":
            print("Enter the correct response:")
            correct_response = input("Correct response: ").strip()
            if correct_response:
                self.learning_manager.record_correction(
                    CorrectionType.RESPONSE_QUALITY,
                    self.last_user_input,
                    self.last_assistant_response,
                    correct_response,
                    "User provided better response",
                    {"intent": self.last_intent}
                )
                print("✅ Recorded response quality correction")
        
        elif correction_type == "sentiment":
            print(f"Current sentiment: {self.last_nlp_result.get('sentiment', {})}")
            new_sentiment = input("Correct sentiment (positive/negative/neutral): ").strip()
            if new_sentiment:
                self.learning_manager.record_correction(
                    CorrectionType.SENTIMENT_ANALYSIS,
                    self.last_user_input,
                    self.last_nlp_result.get('sentiment'),
                    new_sentiment,
                    f"User corrected sentiment to {new_sentiment}",
                    {"original_nlp_result": self.last_nlp_result}
                )
                print(f"✅ Recorded sentiment correction: {new_sentiment}")
        
        elif correction_type == "factual":
            print("Describe the factual error:")
            error_description = input("Error description: ").strip()
            correct_fact = input("Correct fact: ").strip()
            if error_description and correct_fact:
                self.learning_manager.record_correction(
                    CorrectionType.FACTUAL_ERROR,
                    self.last_user_input,
                    error_description,
                    correct_fact,
                    f"User corrected factual error: {error_description}",
                    {"response": self.last_assistant_response}
                )
                print("✅ Recorded factual correction")
        
        else:
            print(f"[ERROR] Unknown correction type: {correction_type}")
    
    def _handle_feedback_command(self, command: str):
        """Handle feedback commands"""
        if not self.last_user_input or not self.last_assistant_response:
            print("\n[ERROR] No previous interaction to rate")
            return
        
        parts = command.split(" ", 1)
        rating_str = parts[1] if len(parts) > 1 else ""
        
        try:
            rating = int(rating_str) if rating_str else None
        except ValueError:
            rating = None
        
        print(f"\n📊 Feedback Mode")
        print("=" * 50)
        print(f"Last input: {self.last_user_input}")
        print(f"Last response: {self.last_assistant_response[:200]}...")
        print()
        
        if rating is None:
            rating = int(input("Rate this response (1-5): ").strip())
        
        if 1 <= rating <= 5:
            feedback_type = FeedbackType.POSITIVE if rating >= 4 else FeedbackType.NEGATIVE
            comment = input("Additional comment (optional): ").strip()
            suggestion = ""
            
            if rating <= 3:
                suggestion = input("How could I improve? ").strip()
            
            self.learning_manager.record_feedback(
                feedback_type,
                self.last_user_input,
                self.last_assistant_response,
                rating,
                comment,
                suggestion,
                {"intent": self.last_intent, "entities": self.last_entities}
            )
            
            print(f"✅ Thank you for the {rating}/5 rating!")
            if suggestion:
                print(f"💡 I'll work on: {suggestion}")
        else:
            print("[ERROR] Rating must be between 1 and 5")
    
    def _handle_learning_stats_command(self):
        """Handle learning statistics command"""
        print(f"\n🧠 Active Learning Statistics")
        print("=" * 50)
        
        try:
            stats = self.learning_manager.get_learning_stats()
            
            print(f"Total corrections: {stats['total_corrections']}")
            print(f"Learning patterns: {stats['total_patterns']}")
            print(f"User feedback entries: {stats['total_feedback']}")
            print(f"Applied patterns: {stats['applied_patterns']}")
            print(f"Recent corrections (7 days): {stats['recent_corrections']}")
            print(f"Average user rating: {stats['average_user_rating']:.1f}/5.0")
            
            if stats['correction_types']:
                print("\nCorrection types:")
                for corr_type, count in stats['correction_types'].items():
                    print(f"   • {corr_type.replace('_', ' ').title()}: {count}")
            
            suggestions = self.learning_manager.suggest_improvements()
            if suggestions:
                print("\n💡 Improvement suggestions:")
                for suggestion in suggestions:
                    print(f"   • {suggestion}")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"[ERROR] Failed to get learning statistics: {e}")
    
    def _format_learning_guidance(self, guidance: Dict[str, Any]) -> str:
        """Format learning guidance for LLM context"""
        guidance_parts = ["Learning guidance:"]
        
        if guidance.get("preferred_words"):
            preferred = ", ".join(guidance["preferred_words"][:5])
            guidance_parts.append(f"Consider using these words: {preferred}")
        
        if guidance.get("avoid_words"):
            avoid = ", ".join(guidance["avoid_words"][:5])
            guidance_parts.append(f"Avoid using these words: {avoid}")
        
        if guidance.get("style_improvement"):
            guidance_parts.append(f"Style note: {guidance['style_improvement']}")
        
        return "\n".join(guidance_parts)
    
    def shutdown(self):
        """Gracefully shutdown ALICE"""
        logger.info(" Shutting down ALICE...")
        
        # Save conversation state
        self._save_conversation_state()
        
        # Save context and memory
        self.context.save_context()
        self.memory._save_memories()
        
        # Stop voice if active
        if self.speech:
            self.speech.stop_listening()
        
        self.running = False
        logger.info("[OK] ALICE shutdown complete")


# Main entry point
def main():
    """Main entry point for A.L.I.C.E"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A.L.I.C.E - Advanced AI Assistant")
    parser.add_argument("--voice", action="store_true", help="Enable voice interaction")
    parser.add_argument("--voice-only", action="store_true", help="Run in voice-only mode")
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model to use")
    parser.add_argument("--name", default="User", help="Your name")
    
    args = parser.parse_args()
    
    try:
        # Initialize ALICE
        alice = ALICE(
            voice_enabled=args.voice or args.voice_only,
            llm_model=args.model,
            user_name=args.name
        )
        
        # Run appropriate mode
        if args.voice_only:
            alice.run_voice_mode()
        else:
            alice.run_interactive()
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {e}")
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
