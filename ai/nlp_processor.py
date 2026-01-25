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
    """Intent classification categories"""
    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    COMMAND = "command"
    INFORMATION = "information"
    TASK = "task"
    WEATHER = "weather"
    TIME = "time"
    SEARCH = "search"
    EMAIL = "email"
    FILE_OPERATION = "file_operation"
    SYSTEM_CONTROL = "system_control"
    CONVERSATION = "conversation"
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
            Intent.GREETING: [
                r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b',
            ],
            Intent.FAREWELL: [
                r'\b(bye|goodbye|see you|farewell|exit|quit)\b',
            ],
            Intent.QUESTION: [
                r'\b(what|when|where|who|why|how|which|can you|could you|would you)\b.*\?',
                r'^(what|when|where|who|why|how|which)',
            ],
            Intent.COMMAND: [
                r'\b(start|stop|run|execute|launch|kill|terminate)\b',
                r'\b(create|delete|remove|move|copy)\b.*\b(file|folder|directory)\b',
            ],
            Intent.WEATHER: [
                r'\b(weather|temperature|forecast|rain|sunny|cloudy)\b',
            ],
            Intent.TIME: [
                r'\b(time|date|day|today|tomorrow|yesterday|calendar|schedule)\b',
            ],
            Intent.EMAIL: [
                r'\b(emails?|gmail|mails?|inbox|messages?)\b',
                r'\b(check|read|show|list|send|reply|write|compose|draft|star|flag|unstar|unflag).*\b(email|mail|inbox)\b',
                r'\bunread\b.*\b(email|mail)\b',
                r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                r'\breply\b',
                r'\b(star|flag|unstar|unflag)\b',
            ],
            Intent.SEARCH: [
                r'\b(search|find|look for|google|browse)\b',
            ],
            Intent.FILE_OPERATION: [
                r'\b(file|folder|directory|document)\b',
                r'\b(save|load|read|write)\b',
            ],
            Intent.SYSTEM_CONTROL: [
                r'\b(shutdown|restart|sleep|volume|brightness|wifi|bluetooth)\b',
            ],
            Intent.TASK: [
                r'\b(remind|remember|task|todo|note|appointment)\b',
            ],
        }
        
        # Entity patterns
        self.entity_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'time': r'\b([0-1]?[0-9]|2[0-3]):[0-5][0-9]\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'number': r'\b\d+\b',
            'file_path': r'[A-Za-z]:\\[\\\S|*\S]?.*?\.[\w:]+',
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
            Intent.GREETING,
            Intent.FAREWELL,
            Intent.COMMAND,
            Intent.WEATHER,
            Intent.TIME,
            Intent.EMAIL,
            Intent.FILE_OPERATION,
            Intent.SEARCH,
            Intent.TASK,
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


# Example usage
if __name__ == "__main__":
    print("Testing Advanced NLP Processor...\n")
    
    processor = NLPProcessor()
    
    test_cases = [
        "Hey ALICE, what's the weather like today?",
        "Can you open my email at john@example.com?",
        "I'm feeling great about this project!",
        "Please remind me to call at 3:30 PM",
        "Search for Python tutorials online",
        "Create a new file called test.txt",
    ]
    
    for text in test_cases:
        print(f"Input: {text}")
        result = processor.process(text)
        print(f"  Intent: {result['intent']}")
        print(f"  Sentiment: {result['sentiment']['category']} ({result['sentiment']['compound']:.2f})")
        print(f"  Entities: {result['entities']}")
        print(f"  Keywords: {processor.extract_keywords(text, 3)}")
        print()
           
