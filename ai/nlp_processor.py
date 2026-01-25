"""
Advanced NLP Processor for A.L.I.C.E
Features:
- Intent classification
- Entity extraction
- Sentiment analysis
- Semantic understanding
- Context-aware processing
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Intent:
    """Intent classification categories - Enhanced with comprehensive categories"""
    # Basic conversational
    GREETING = "greeting"
    FAREWELL = "farewell"
    CONVERSATION = "conversation"
    QUESTION = "question"
    
    # Information seeking
    INFORMATION = "information"
    SEARCH = "search"
    EXPLAIN = "explain"
    DEFINE = "define"
    COMPARE = "compare"
    RECOMMEND = "recommend"
    
    # Task management
    TASK = "task"
    REMINDER = "reminder"
    SCHEDULE = "schedule"
    CALENDAR = "calendar"
    TODO = "todo"
    DEADLINE = "deadline"
    
    # System & Commands
    COMMAND = "command"
    SYSTEM_CONTROL = "system_control"
    FILE_OPERATION = "file_operation"
    PROCESS_CONTROL = "process_control"
    SETTINGS = "settings"
    
    # Communication
    EMAIL = "email"
    MESSAGE = "message"
    CALL = "call"
    CONTACT = "contact"
    NOTIFICATION = "notification"
    
    # Media & Entertainment
    MUSIC = "music"
    VIDEO = "video"
    GAME = "game"
    NEWS = "news"
    PODCAST = "podcast"
    RADIO = "radio"
    
    # Productivity & Organization
    SCHEDULE = "schedule"
    CALENDAR = "calendar"
    MEETING = "meeting"
    APPOINTMENT = "appointment"
    REMINDER = "reminder"
    TODO = "todo"
    TASK = "task"
    NOTE = "note"
    DOCUMENT = "document"
    
    # Information & Data
    WEATHER = "weather"
    TIME = "time"
    DATE = "date"
    LOCATION = "location"
    TRANSLATION = "translation"
    CALCULATION = "calculation"
    UNIT_CONVERSION = "unit_conversion"
    
    # Web & Internet
    WEB_SEARCH = "web_search"
    BROWSE = "browse"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    SOCIAL_MEDIA = "social_media"
    
    # Learning & Education
    LEARN = "learn"
    STUDY = "study"
    TUTORIAL = "tutorial"
    COURSE = "course"
    QUIZ = "quiz"
    
    # Health & Fitness
    HEALTH = "health"
    FITNESS = "fitness"
    NUTRITION = "nutrition"
    MEDITATION = "meditation"
    SLEEP = "sleep"
    
    # Shopping & Finance
    SHOPPING = "shopping"
    FINANCE = "finance"
    BUDGET = "budget"
    INVESTMENT = "investment"
    BANKING = "banking"
    
    # Travel & Transportation
    TRAVEL = "travel"
    NAVIGATION = "navigation"
    TRANSPORTATION = "transportation"
    HOTEL = "hotel"
    FLIGHT = "flight"
    
    # Home & Lifestyle
    SMART_HOME = "smart_home"
    COOKING = "cooking"
    RECIPE = "recipe"
    SHOPPING_LIST = "shopping_list"
    HOME_AUTOMATION = "home_automation"
    
    # Work & Productivity
    WORK = "work"
    MEETING = "meeting"
    PROJECT = "project"
    DOCUMENT = "document"
    PRESENTATION = "presentation"
    EMAIL_WORK = "email_work"
    
    # Creative & Fun
    CREATIVE = "creative"
    JOKE = "joke"
    STORY = "story"
    POEM = "poem"
    ART = "art"
    WRITING = "writing"
    
    # Technical & Development
    PROGRAMMING = "programming"
    DEBUG = "debug"
    CODE_REVIEW = "code_review"
    TECH_SUPPORT = "tech_support"
    
    # Emergency & Safety
    EMERGENCY = "emergency"
    SAFETY = "safety"
    SECURITY = "security"
    BACKUP = "backup"
    
    # Personal & Emotional
    PERSONAL = "personal"
    EMOTION = "emotion"
    COMPLIMENT = "compliment"
    COMPLAINT = "complaint"
    GRATITUDE = "gratitude"
    APOLOGY = "apology"
    
    # Unknown
    UNKNOWN = "unknown"


class NLPProcessor:
    """
    Advanced NLP processing for A.L.I.C.E
    Handles intent detection, entity extraction, and semantic understanding
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Intent patterns (keyword-based for lightweight performance)
        self.intent_patterns = {
            # Basic conversational
            Intent.GREETING: [
                r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening|howdy|sup|what\'s up)\b',
            ],
            Intent.FAREWELL: [
                r'\b(bye|goodbye|see you|farewell|exit|quit|later|take care|catch you later)\b',
            ],
            Intent.QUESTION: [
                r'\b(what|when|where|who|why|how|which|can you|could you|would you)\b.*\?',
                r'^(what|when|where|who|why|how|which)',
            ],
            Intent.CONVERSATION: [
                r'\b(chat|talk|discuss|conversation|tell me)\b',
            ],
            
            # Information seeking
            Intent.INFORMATION: [
                r'\b(info|information|details|about|explain|tell me about)\b',
            ],
            Intent.EXPLAIN: [
                r'\b(explain|describe|elaborate|clarify|break down|walk me through)\b',
            ],
            Intent.DEFINE: [
                r'\b(define|definition|what is|what does.*mean|meaning of)\b',
            ],
            Intent.COMPARE: [
                r'\b(compare|comparison|difference between|vs|versus|better|worse)\b',
            ],
            
            # Calendar & Scheduling
            Intent.CALENDAR: [
                r'\b(calendar|schedule|agenda|events|appointments)\b',
            ],
            Intent.SCHEDULE: [
                r'\b(schedule|book|plan|set up|arrange|organize)\b',
            ],
            Intent.MEETING: [
                r'\b(meeting|conference|call|session|appointment)\b',
            ],
            Intent.APPOINTMENT: [
                r'\b(appointment|booking|reservation|visit)\b',
            ],
            Intent.REMINDER: [
                r'\b(remind|reminder|alert|notification|don\'t forget)\b',
            ],
            Intent.RECOMMEND: [
                r'\b(recommend|suggestion|suggest|advice|what should|which is better)\b',
            ],
            
            # Task management
            Intent.TASK: [
                r'\b(task|to.?do|assignment|project|work on)\b',
            ],
            Intent.REMINDER: [
                r'\b(remind|reminder|remember|don\'t forget|alert me)\b',
            ],
            Intent.SCHEDULE: [
                r'\b(schedule|plan|arrange|book|appointment|meeting)\b',
            ],
            Intent.CALENDAR: [
                r'\b(calendar|event|appointment|date|schedule|availability)\b',
            ],
            Intent.TODO: [
                r'\b(todo|to.?do|task list|checklist|add to list)\b',
            ],
            Intent.DEADLINE: [
                r'\b(deadline|due date|expires?|finish by|complete by)\b',
            ],
            
            # System & Commands
            Intent.COMMAND: [
                r'\b(start|stop|run|execute|launch|kill|terminate|command|execute)\b',
            ],
            Intent.SYSTEM_CONTROL: [
                r'\b(shutdown|restart|reboot|sleep|hibernate|volume|brightness|wifi|bluetooth)\b',
                r'\b(open|launch|start|run|execute)\b.*\b(app|application|program|game|software)\b',
                r'\b(open|launch|start|run)\s+\w+',
            ],
            Intent.FILE_OPERATION: [
                r'\b(file|folder|directory|document|create|delete|remove|move|copy|save|load)\b',
                r'\b(save|load|read|write|edit|modify)\b.*\b(file|document)\b',
            ],
            Intent.PROCESS_CONTROL: [
                r'\b(process|kill|terminate|stop|close|end task|task manager)\b',
            ],
            Intent.SETTINGS: [
                r'\b(settings|preferences|config|configure|setup|options)\b',
            ],
            
            # Communication
            Intent.EMAIL: [
                r'\b(emails?|gmail|mails?|inbox|messages?|compose|draft|send)\b',
                r'\b(check|read|show|list|send|reply|write|compose|draft|star|flag).*\b(email|mail|inbox)\b',
                r'\bunread\b.*\b(email|mail)\b',
                r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                r'\b(latest|last|recent|new).*\b(emails?|mails?)\b',
            ],
            Intent.MESSAGE: [
                r'\b(text|sms|message|chat|whatsapp|telegram|slack|discord)\b',
            ],
            Intent.CALL: [
                r'\b(call|phone|dial|ring|contact|video call|voice call)\b',
            ],
            Intent.CONTACT: [
                r'\b(contact|phone book|address book|contacts list)\b',
            ],
            Intent.NOTIFICATION: [
                r'\b(notification|alert|popup|notify|bell|badge)\b',
            ],
            
            # Media & Entertainment
            Intent.MUSIC: [
                r'\b(music|song|play|pause|skip|spotify|apple music|youtube music|sound|audio|playlist)\b',
            ],
            Intent.VIDEO: [
                r'\b(video|movie|film|youtube|netflix|streaming|watch|player)\b',
            ],
            Intent.GAME: [
                r'\b(game|gaming|play|steam|xbox|playstation|nintendo)\b',
            ],
            Intent.NEWS: [
                r'\b(news|headlines|breaking news|current events|newspaper|journalism)\b',
            ],
            Intent.PODCAST: [
                r'\b(podcast|listen|audio show|episode|podcasting)\b',
            ],
            Intent.RADIO: [
                r'\b(radio|fm|am|station|broadcast)\b',
            ],
            
            # Information & Data
            Intent.WEATHER: [
                r'\b(weather|temperature|forecast|rain|sunny|cloudy|storm|humidity|wind)\b',
            ],
            Intent.TIME: [
                r'\b(time|clock|hour|minute|second|now|current time)\b',
            ],
            Intent.DATE: [
                r'\b(date|today|tomorrow|yesterday|calendar|day|month|year)\b',
            ],
            Intent.LOCATION: [
                r'\b(location|address|where|place|coordinates|gps|map)\b',
            ],
            Intent.TRANSLATION: [
                r'\b(translate|translation|language|spanish|french|german|chinese|japanese)\b',
            ],
            Intent.CALCULATION: [
                r'\b(calculate|math|plus|minus|multiply|divide|equals|sum|total)\b',
                r'\d+\s*[\+\-\*\/\=]\s*\d+',
            ],
            Intent.UNIT_CONVERSION: [
                r'\b(convert|conversion|miles to km|feet to meters|celsius to fahrenheit)\b',
            ],
            
            # Web & Internet
            Intent.WEB_SEARCH: [
                r'\b(search|google|bing|yahoo|find|look for|look up)\b',
            ],
            Intent.BROWSE: [
                r'\b(browse|website|url|internet|web|browser|chrome|firefox|safari)\b',
            ],
            Intent.DOWNLOAD: [
                r'\b(download|get|fetch|pull|retrieve)\b',
            ],
            Intent.UPLOAD: [
                r'\b(upload|post|share|send|publish)\b',
            ],
            Intent.SOCIAL_MEDIA: [
                r'\b(facebook|twitter|instagram|linkedin|tiktok|snapchat|reddit|social)\b',
            ],
            
            # Learning & Education
            Intent.LEARN: [
                r'\b(learn|study|education|knowledge|understand|master)\b',
            ],
            Intent.STUDY: [
                r'\b(study|review|practice|homework|assignment|exam|test)\b',
            ],
            Intent.TUTORIAL: [
                r'\b(tutorial|guide|how to|walkthrough|instructions|demo)\b',
            ],
            Intent.COURSE: [
                r'\b(course|class|lesson|training|workshop|seminar)\b',
            ],
            Intent.QUIZ: [
                r'\b(quiz|test|exam|question|assessment|evaluation)\b',
            ],
            
            # Health & Fitness
            Intent.HEALTH: [
                r'\b(health|medical|doctor|symptom|illness|disease|medicine)\b',
            ],
            Intent.FITNESS: [
                r'\b(fitness|exercise|workout|gym|training|running|cycling|swimming)\b',
            ],
            Intent.NUTRITION: [
                r'\b(nutrition|diet|food|calories|vitamins|protein|carbs)\b',
            ],
            Intent.MEDITATION: [
                r'\b(meditation|mindfulness|relaxation|zen|breathing|calm)\b',
            ],
            Intent.SLEEP: [
                r'\b(sleep|tired|rest|nap|bedtime|insomnia|dream)\b',
            ],
            
            # Shopping & Finance
            Intent.SHOPPING: [
                r'\b(shop|shopping|buy|purchase|order|cart|checkout|amazon|ebay)\b',
            ],
            Intent.FINANCE: [
                r'\b(money|finance|financial|bank|account|balance|transaction)\b',
            ],
            Intent.BUDGET: [
                r'\b(budget|expense|spending|cost|price|afford|save)\b',
            ],
            Intent.INVESTMENT: [
                r'\b(invest|investment|stock|bond|portfolio|retirement|401k)\b',
            ],
            Intent.BANKING: [
                r'\b(bank|atm|deposit|withdraw|transfer|loan|credit|debit)\b',
            ],
            
            # Travel & Transportation
            Intent.TRAVEL: [
                r'\b(travel|trip|vacation|holiday|journey|destination)\b',
            ],
            Intent.NAVIGATION: [
                r'\b(navigate|direction|route|map|gps|turn|left|right)\b',
            ],
            Intent.TRANSPORTATION: [
                r'\b(transport|bus|train|subway|uber|taxi|car|bike|plane)\b',
            ],
            Intent.HOTEL: [
                r'\b(hotel|accommodation|room|booking|reservation|airbnb)\b',
            ],
            Intent.FLIGHT: [
                r'\b(flight|airplane|airport|boarding|ticket|airline)\b',
            ],
            
            # Home & Lifestyle
            Intent.SMART_HOME: [
                r'\b(smart home|lights|thermostat|security|camera|door|lock)\b',
            ],
            Intent.COOKING: [
                r'\b(cook|cooking|bake|baking|kitchen|chef|recipe)\b',
            ],
            Intent.RECIPE: [
                r'\b(recipe|ingredient|cook|bake|dish|meal|food)\b',
            ],
            Intent.SHOPPING_LIST: [
                r'\b(shopping list|grocery|groceries|buy|store|market)\b',
            ],
            Intent.HOME_AUTOMATION: [
                r'\b(automation|automatic|schedule|timer|routine|smart)\b',
            ],
            
            # Work & Productivity
            Intent.WORK: [
                r'\b(work|job|office|business|career|professional|colleague)\b',
            ],
            Intent.MEETING: [
                r'\b(meeting|conference|call|zoom|teams|presentation|agenda)\b',
            ],
            Intent.PROJECT: [
                r'\b(project|milestone|deadline|deliverable|team|collaboration)\b',
            ],
            Intent.DOCUMENT: [
                r'\b(document|doc|pdf|spreadsheet|presentation|report|file)\b',
            ],
            Intent.PRESENTATION: [
                r'\b(presentation|slide|powerpoint|keynote|present|demo)\b',
            ],
            
            # Creative & Fun
            Intent.CREATIVE: [
                r'\b(creative|create|design|art|draw|paint|sketch|imagine)\b',
            ],
            Intent.JOKE: [
                r'\b(joke|funny|humor|laugh|comic|pun|witty)\b',
            ],
            Intent.STORY: [
                r'\b(story|tale|narrative|plot|character|fiction)\b',
            ],
            Intent.POEM: [
                r'\b(poem|poetry|verse|rhyme|haiku|sonnet)\b',
            ],
            Intent.ART: [
                r'\b(art|artist|painting|sculpture|gallery|museum|creative)\b',
            ],
            Intent.WRITING: [
                r'\b(write|writing|author|novel|essay|blog|article)\b',
            ],
            
            # Technical & Development
            Intent.PROGRAMMING: [
                r'\b(code|coding|program|programming|developer|software|python|javascript|java)\b',
            ],
            Intent.DEBUG: [
                r'\b(debug|error|bug|fix|troubleshoot|problem|issue)\b',
            ],
            Intent.CODE_REVIEW: [
                r'\b(review|code review|feedback|optimization|refactor)\b',
            ],
            Intent.TECH_SUPPORT: [
                r'\b(tech support|technical|help|support|problem|issue|broken)\b',
            ],
            
            # Emergency & Safety
            Intent.EMERGENCY: [
                r'\b(emergency|urgent|help|911|crisis|danger|ambulance|fire|police)\b',
            ],
            Intent.SAFETY: [
                r'\b(safety|safe|secure|protection|guard|warning)\b',
            ],
            Intent.SECURITY: [
                r'\b(security|password|encryption|privacy|breach|hack)\b',
            ],
            Intent.BACKUP: [
                r'\b(backup|restore|recovery|save|archive|sync)\b',
            ],
            
            # Personal & Emotional
            Intent.PERSONAL: [
                r'\b(personal|private|family|friend|relationship|life)\b',
            ],
            Intent.EMOTION: [
                r'\b(feel|feeling|emotion|mood|happy|sad|angry|excited|nervous)\b',
            ],
            Intent.COMPLIMENT: [
                r'\b(great|awesome|excellent|amazing|wonderful|fantastic|good job)\b',
            ],
            Intent.COMPLAINT: [
                r'\b(complain|complaint|problem|issue|annoying|frustrated|unhappy)\b',
            ],
            Intent.GRATITUDE: [
                r'\b(thank|thanks|grateful|appreciate|gratitude)\b',
            ],
            Intent.APOLOGY: [
                r'\b(sorry|apologize|apology|regret|mistake|fault)\b',
            ],
        }
        
        # Entity patterns
        self.entity_patterns = {
            # Contact information
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            
            # Time and dates
            'time': r'\b([0-1]?[0-9]|2[0-3]):[0-5][0-9]\s*(am|pm)?\b',
            'time_12h': r'\b([0-1]?[0-9]):([0-5][0-9])\s*(am|pm)\b',
            'time_natural': r'\b(morning|afternoon|evening|night|noon|midnight)\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'relative_date': r'\b(today|tomorrow|yesterday|next week|last week|this week|next month|last month)\b',
            'day_of_week': r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            'month': r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            'duration': r'\b(\d+)\s*(minute|hour|day|week|month)s?\b',
            
            # Calendar-specific entities
            'event_type': r'\b(meeting|appointment|call|conference|session|interview|lunch|dinner|workshop|training)\b',
            'location': r'\b(?:at|in|@)\s+([^,\n]+?)(?=\s+(?:on|at|tomorrow|today|$))',
            'attendee': r'\b(?:with|invite|including)\s+([A-Za-z\s]+?)(?=\s+(?:on|at|tomorrow|today|$))',
            
            # Numbers and measurements
            'number': r'\b\d+\b',
            'percentage': r'\b\d+(\.\d+)?%\b',
            'currency': r'\$\d+(\.\d{2})?\b',
            'temperature': r'\b\d+°[CF]\b',
            'duration': r'\b\d+\s*(minute|hour|day|week|month|year)s?\b',
            
            # File and system
            'file_path': r'[A-Za-z]:\\[\\\S|*\S]?.*?\.[\w:]+',
            'file_extension': r'\.\w{2,4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'mac_address': r'\b[0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}[:-][0-9A-Fa-f]{2}\b',
            
            # Media and entertainment
            'music_service': r'\b(spotify|apple music|youtube music|pandora|amazon music)\b',
            'video_service': r'\b(netflix|youtube|hulu|disney plus|amazon prime|hbo)\b',
            'social_media': r'\b(facebook|twitter|instagram|linkedin|tiktok|snapchat|reddit)\b',
            
            # Programming and tech
            'programming_language': r'\b(python|javascript|java|c\+\+|c#|ruby|php|go|rust|swift)\b',
            'file_format': r'\b(pdf|doc|docx|txt|csv|json|xml|html|css|js|py|java)\b',
            'browser': r'\b(chrome|firefox|safari|edge|opera)\b',
            'operating_system': r'\b(windows|mac|linux|android|ios)\b',
            
            # Location and travel
            'country': r'\b(usa|canada|uk|france|germany|japan|china|australia|brazil)\b',
            'city': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Simple city name pattern
            'address': r'\b\d+\s+[A-Za-z\s]+(street|st|avenue|ave|road|rd|drive|dr|lane|ln)\b',
            
            # Health and fitness
            'exercise': r'\b(running|walking|swimming|cycling|yoga|weightlifting|cardio)\b',
            'body_part': r'\b(head|neck|shoulder|arm|hand|chest|back|leg|foot|knee|ankle)\b',
            'symptom': r'\b(headache|fever|cough|fatigue|pain|nausea|dizziness)\b',
            
            # Food and nutrition
            'food_item': r'\b(apple|banana|chicken|beef|rice|pasta|bread|milk|cheese|egg)\b',
            'meal_type': r'\b(breakfast|lunch|dinner|snack|appetizer|dessert)\b',
            'cuisine': r'\b(italian|chinese|mexican|indian|japanese|thai|french)\b',
            
            # Work and business
            'job_title': r'\b(manager|developer|engineer|designer|analyst|director|ceo|cto)\b',
            'company': r'\b(google|microsoft|apple|amazon|facebook|netflix|tesla)\b',
            'meeting_platform': r'\b(zoom|teams|skype|webex|meet|hangouts)\b',
        }
        
        logger.info("✅ NLP Processor initialized with advanced features")
    
    def process(self, text: str) -> Dict[str, any]:
        """
        Comprehensive text processing
        
        Returns:
            Dict containing:
            - tokens: List of tokens
            - intent: Detected intent
            - entities: Extracted entities
            - sentiment: Sentiment analysis
            - clean_text: Cleaned version of text
        """
        
        # Clean and normalize
        clean_text = self._clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(clean_text.lower())
        
        # Intent detection
        intent = self.detect_intent(text)
        
        # Entity extraction
        entities = self.extract_entities(text)
        
        # Sentiment analysis
        sentiment = self.analyze_sentiment(text)
        
        return {
            'tokens': tokens,
            'intent': intent,
            'entities': entities,
            'sentiment': sentiment,
            'clean_text': clean_text,
            'original_text': text
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\'-]', '', text)
        return text
    
    def detect_intent(self, text: str) -> str:
        """
        Detect user intent from text
        
        Args:
            text: Input text
            
        Returns:
            Intent category (from Intent class)
        """
        text_lower = text.lower()
        
        # Check specific intents first (before generic ones like QUESTION)
        priority_intents = [
            # Emergency & Critical
            Intent.EMERGENCY,
            Intent.SAFETY,
            Intent.SECURITY,
            
            # Basic conversational
            Intent.GREETING,
            Intent.FAREWELL,
            Intent.GRATITUDE,
            Intent.APOLOGY,
            
            # High-priority tasks
            Intent.REMINDER,
            Intent.DEADLINE,
            Intent.CALENDAR,
            Intent.SCHEDULE,
            
            # Communication
            Intent.EMAIL,
            Intent.MESSAGE,
            Intent.CALL,
            
            # System control
            Intent.COMMAND,
            Intent.SYSTEM_CONTROL,
            Intent.FILE_OPERATION,
            Intent.PROCESS_CONTROL,
            Intent.SETTINGS,
            
            # Information & time-sensitive
            Intent.WEATHER,
            Intent.TIME,
            Intent.DATE,
            Intent.LOCATION,
            
            # Media & entertainment
            Intent.MUSIC,
            Intent.VIDEO,
            Intent.NEWS,
            
            # Work & productivity
            Intent.MEETING,
            Intent.DOCUMENT,
            Intent.PROJECT,
            
            # Web & search
            Intent.WEB_SEARCH,
            Intent.BROWSE,
            
            # Task management
            Intent.TASK,
            Intent.TODO,
            
            # Creative & fun
            Intent.JOKE,
            Intent.STORY,
            Intent.CREATIVE,
            
            # Technical
            Intent.PROGRAMMING,
            Intent.DEBUG,
            Intent.TECH_SUPPORT,
            
            # Health & lifestyle
            Intent.HEALTH,
            Intent.FITNESS,
            Intent.COOKING,
            Intent.RECIPE,
            
            # Shopping & finance
            Intent.SHOPPING,
            Intent.FINANCE,
            Intent.BUDGET,
            
            # Travel
            Intent.TRAVEL,
            Intent.NAVIGATION,
            Intent.TRANSPORTATION,
            
            # Learning
            Intent.LEARN,
            Intent.TUTORIAL,
            Intent.COURSE,
            
            # Information seeking
            Intent.EXPLAIN,
            Intent.DEFINE,
            Intent.COMPARE,
            Intent.RECOMMEND,
            Intent.INFORMATION,
            Intent.CALCULATION,
            Intent.TRANSLATION,
        ]
        
        # Check priority intents first
        for intent in priority_intents:
            if intent in self.intent_patterns:
                for pattern in self.intent_patterns[intent]:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        logger.debug(f"Intent detected: {intent}")
                        return intent
        
        # Then check remaining intents (like QUESTION)
        for intent, patterns in self.intent_patterns.items():
            if intent not in priority_intents:
                for pattern in patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        logger.debug(f"Intent detected: {intent}")
                        return intent
        
        # Default to conversation or question based on punctuation
        if '?' in text:
            return Intent.QUESTION
        
        return Intent.CONVERSATION
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity_type] = matches
                logger.debug(f"Entities found - {entity_type}: {matches}")
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores (neg, neu, pos, compound)
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        # Categorize
        if scores['compound'] >= 0.05:
            category = 'positive'
        elif scores['compound'] <= -0.05:
            category = 'negative'
        else:
            category = 'neutral'
        
        scores['category'] = category
        return scores
    
    def vectorize(self, texts: List[str]):
        """
        Vectorize texts using TF-IDF
        
        Args:
            texts: List of text documents
            
        Returns:
            TF-IDF matrix
        """
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, text: str):
        """
        Transform single text using fitted vectorizer
        
        Args:
            text: Input text
            
        Returns:
            TF-IDF vector
        """
        return self.vectorizer.transform([text])
    
    def is_question(self, text: str) -> bool:
        """Check if text is a question"""
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'can', 'could', 'would', 'should']
        text_lower = text.lower()
        return '?' in text or any(text_lower.startswith(qw) for qw in question_words)
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract top keywords from text
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Tokenize and remove stopwords
        tokens = word_tokenize(text.lower())
        
        # Simple stopwords list
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        # Filter and get unique keywords
        keywords = [token for token in tokens if token.isalnum() and token not in stopwords]
        
        # Return top N by frequency
        from collections import Counter
        keyword_freq = Counter(keywords)
        return [word for word, _ in keyword_freq.most_common(top_n)]
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Transform both texts
            vec1 = self.transform(text1)
            vec2 = self.transform(text2)
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            return float(similarity)
        except:
            # Fallback to simple word overlap
            words1 = set(word_tokenize(text1.lower()))
            words2 = set(word_tokenize(text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            
            return overlap / total if total > 0 else 0.0
