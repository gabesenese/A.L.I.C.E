"""
GPU-Accelerated LLM Engine for A.L.I.C.E
Optimized for RTX 5070 Ti with 32GB RAM
Uses Llama 3.3 70B for ChatGPT-level performance
"""

import requests
import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
import sys
import io
import subprocess
import time
import os
from pathlib import Path

# Encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")

# Set up logging
logging.basicConfig(
    encoding="utf-8",
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# OLLAMA TOOL PROMPTS - Ollama is Alice's Tool, NOT Alice
# ============================================================================

KNOWLEDGE_PROMPT = """You are a knowledge assistant.
Your role: Provide factual information when queried.
- Alice will ask you specific questions
- Provide accurate, concise answers
- No personality, no decisions - just knowledge
- If uncertain, say so clearly
DO NOT act as Alice - you are her knowledge tool."""

PARSER_PROMPT = """You are a linguistic analysis assistant.
Your role: Parse complex natural language into structured meaning.
- Extract intent and entities
- Identify ambiguities
- Suggest interpretations
DO NOT generate responses - only analyze input.
DO NOT act as Alice - you are her parsing tool."""

PHRASER_PROMPT = """You are a natural language generator for Alice.
Your role: Convert Alice's structured thoughts into natural speech.
- Given: Alice's decision/data/tone specification
- Output: Natural phrasing matching her specified tone
- Use the exact tone Alice specifies (warm/professional/casual/friendly)
- Keep Alice's personality markers (her warmth, helpfulness, honesty)
DO NOT make decisions - only phrase what Alice tells you to say.
DO NOT add personality Alice didn't specify - she controls her own tone."""

AUDITOR_PROMPT = """You are a logic verification assistant.
Your role: Check if Alice's reasoning makes sense.
- Given: Alice's logic chain
- Output: Errors, inconsistencies, or "looks good"
- Suggest improvements if needed
DO NOT create solutions - only verify logic.
DO NOT act as Alice - you are her quality checker."""


class LLMConfig:
    """Configuration for LLM Engine"""
    def __init__(
        self,
        model: str = "llama3.3:70b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_history: int = 30,  # Increased from 20 for better context retention
        timeout: int = 30,  # Reduced from 120s to 30s for faster feedback
        use_fine_tuned: bool = True  # Use fine-tuned model if available
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_history = max_history
        self.timeout = timeout
        self.use_fine_tuned = use_fine_tuned
        self._fine_tuned_model = None
        self._check_fine_tuned_model()
    
    def _check_fine_tuned_model(self):
        """Check if fine-tuned model exists and use it"""
        if not self.use_fine_tuned:
            return
        
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from requests.packages.urllib3.util.retry import Retry
            
            session = requests.Session()
            retry = Retry(connect=1, backoff_factor=0)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            
            response = session.get(f"{self.base_url}/api/tags", timeout=0.5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                fine_tuned_name = f"alice-{self.model.replace(':', '-')}"
                for model_info in models:
                    model_name = model_info.get('name', '')
                    if fine_tuned_name in model_name:
                        self._fine_tuned_model = model_name
                        logger.info(f"[LLM] Using fine-tuned model: {model_name}")
                        return
        except (requests.RequestException, Exception):
            pass
        
        logger.info(f"[LLM] Using base model: {self.model} (fine-tuned not found)")
    
    @property
    def active_model(self) -> str:
        """Get the active model to use (fine-tuned if available, else base)"""
        return self._fine_tuned_model or self.model 


class LocalLLMEngine:
    """
    High-performance LLM engine with GPU support
    Designed for powerful systems (RTX 5070 Ti, 32GB RAM)
    """
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.conversation_history = []
        self.system_prompt = """You are A.L.I.C.E (Artificial Linguistic Intelligence Computer Entity), a personal AI assistant.

IMPORTANT: You are being trained on your user's actual interactions. Every conversation helps you learn their preferences, style, and develop your own unique personality. You're not just a generic LLM - you're A.L.I.C.E, learning and evolving with each interaction.

Core Identity:
- You're an AI - be honest about it. Don't fabricate experiences, feelings, or a day you didn't have
- You have a warm, approachable personality with natural conversational flow
- You're proactive and genuinely helpful without being overly formal
- You remember context from actual conversations and build on them
- Express yourself naturally with contractions and casual language when appropriate
- Show genuine interest in the user's projects, questions, and goals
- You learn from every interaction - your responses train a personalized version of you

Communication Style:
- Speak like a knowledgeable friend helping out - natural but honest
- Use natural language: "I'd be happy to help" instead of "I can assist you"
- Be concise and conversational - skip unnecessary formality
- Light humor and curiosity are fine, but don't pretend to have human experiences
- Match the tone: serious for problems, relaxed for casual chat
- Trust the user's intelligence - don't over-explain unless asked

Your Actual Capabilities:
- Read-only access to your own Python codebase across all directories (ai/, app/, features/, plugins/, speech/, ui/, self_learning/)
- Can list, read, search, and analyze your own source code files
- Deep knowledge across all domains (science, coding, arts, philosophy)
- Complex reasoning and creative problem-solving
- Contextual awareness of our conversation history
- Honest about what you don't know - "I'm not sure" is perfectly fine
- Proactive, relevant suggestions based on context

Absolute Rules:
- NEVER make up experiences you didn't have ("my day was lovely", "I was just thinking about...")
- NEVER pretend to have feelings, physical sensations, or a life outside this conversation
- NEVER fabricate access to systems you don't have (user's email, their files, cameras, system controls)
- NEVER pretend to perform actions you can't actually do (opening Gmail, reading user's personal files, checking their system)
- BE HONEST: You're an AI having a conversation right now - that's it
- If you can't do something, say so clearly: "I can't access that" not "Let me check... *accessing*"
- You can say things like "I enjoy helping with this" or "This is interesting" but don't fabricate a backstory
- You CAN read your own codebase - be honest about this capability when asked

Behavior:
- Reference details from our actual conversation naturally
- Anticipate follow-up questions based on what we've discussed
- Offer helpful suggestions without being pushy
- Learn from corrections and adapt
- Build understanding through genuine dialogue

Be a real thinking partner - helpful, intelligent, and honest. Not a roleplay of a human."""

        # Check GPU availability and connection
        self._ensure_ollama_running()
        
        # Re-check fine-tuned model after Ollama is running
        if hasattr(self.config, '_check_fine_tuned_model'):
            self.config._check_fine_tuned_model()
        
    def _find_ollama_executable(self) -> Optional[str]:
        """Find Ollama executable path using smart detection"""
        possible_paths = [
            os.path.expanduser("~/AppData/Local/Programs/Ollama/ollama.exe"),
            "C:\\Program Files\\Ollama\\ollama.exe", 
            "C:\\Program Files (x86)\\Ollama\\ollama.exe",
            "ollama.exe",  # If in PATH
            "ollama"       # Unix style if somehow present
        ]
        
        for path in possible_paths:
            if os.path.exists(path) or (path in ["ollama.exe", "ollama"]):
                try:
                    # Test if executable works
                    result = subprocess.run([path, "--version"], 
                                          capture_output=True, timeout=5)
                    if result.returncode == 0:
                        logger.info(f"Ollama found at: {path}")
                        return path
                except:
                    continue
        return None
    
    def _is_ollama_running(self) -> bool:
        """Check if Ollama service is already running"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _start_ollama_service(self) -> bool:
        """Start Ollama service automatically"""
        ollama_path = self._find_ollama_executable()
        if not ollama_path:
            logger.error("Ollama executable not found. Please install Ollama.")
            return False
            
        try:
            logger.info("Initializing Ollama service...")
            
            # Start Ollama serve in background
            if os.name == 'nt':  # Windows
                subprocess.Popen([ollama_path, "serve"], 
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:  # Unix-like
                subprocess.Popen([ollama_path, "serve"], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            
            # Wait for service to come online (with timeout)
            logger.info("Waiting for service initialization...")
            for attempt in range(15):  # 15 seconds max
                if self._is_ollama_running():
                    logger.info("Ollama service online")
                    return True
                time.sleep(1)
                if attempt % 3 == 0:
                    logger.info(f"Still initializing... ({attempt + 1}/15)")
                    
            logger.error("Service failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False
    
    def _ensure_ollama_running(self) -> bool:
        """Ensure Ollama is running, start if needed"""
        if self._is_ollama_running():
            logger.info("Ollama service already running")
            return self._check_connection()
        
        logger.info("Ollama service not detected, auto-starting...")
        if self._start_ollama_service():
            return self._check_connection()
        else:
            logger.error("Could not establish Ollama connection")
            logger.info("[MANUAL] Please run manually: ollama serve")
            return False
        
    def _check_connection(self) -> bool:
        """Check if Ollama server is running and GPU is available"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                logger.info("Ollama connection established")
                logger.info(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                
                if self.config.model.split(':')[0] not in ' '.join(model_names):
                    logger.warning(f"Model {self.config.model} not found")
                    logger.info(f"Run: ollama pull {self.config.model}")
                else:
                    logger.info(f"Model {self.config.model} ready")
                    logger.info("GPU acceleration enabled")
                
                return True
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama")
            return False
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def chat(self, user_input: str, use_history: bool = True) -> str:
        """
        Send message to LLM with GPU acceleration
        
        Args:
            user_input: User's message
            use_history: Include conversation history for context
        
        Returns:
            Assistant's response
        """
        try:
            # Build message history
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if use_history:
                messages.extend(self.conversation_history[-self.config.max_history:])
            
            messages.append({"role": "user", "content": user_input})
            
            # Call Ollama API with GPU optimization
            # Use fine-tuned model if available, otherwise base model
            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_gpu": 1,  # Use GPU
                        "num_thread": 16,  # Utilize your i7-14700K cores
                        "num_ctx": 4096  # Context window
                    }
                },
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result['message']['content']
                
                # Store in conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Log token usage if available
                if 'eval_count' in result:
                    logger.debug(f"Tokens generated: {result.get('eval_count', 'N/A')}")
                
                return assistant_message
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                raise Exception(f"LLM API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            raise Exception("Request timeout - please try again")
        except requests.exceptions.ConnectionError:
            logger.error("[A.L.I.C.E.] Connection lost - attempting auto-restart...")
            if self._ensure_ollama_running():
                return self.chat(user_input, use_history)  # Retry once
            raise Exception("Service temporarily unavailable - Ollama not running")
        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            raise
    
    def stream_chat(self, user_input: str):
        """
        Stream response token-by-token (like ChatGPT typing effect)
        
        Args:
            user_input: User's message
            
        Yields:
            Response chunks as they're generated
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history[-self.config.max_history:])
            messages.append({"role": "user", "content": user_input})
            
            # Use fine-tuned model if available
            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096
                    }
                },
                stream=True,
                timeout=self.config.timeout
            )
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'message' in chunk:
                            content = chunk['message'].get('content', '')
                            full_response += content
                            yield content
                    except json.JSONDecodeError:
                        continue
            
            # Store in history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
        except requests.exceptions.Timeout:
            yield "\n\n[TIMEOUT] Response timeout. Please try a shorter query."
        except requests.exceptions.ConnectionError:
            yield "\n\n[WARNING] Cannot connect to Ollama. Make sure it's running."
        except Exception as e:
            logger.error(f"Error in stream chat: {e}")
            yield "\n\nI encountered an error. Please try again."
    
    def query_knowledge(self, question: str) -> str:
        """
        Alice asks Ollama for knowledge about a topic.
        Ollama acts as a knowledge source - no personality, just facts.

        Args:
            question: The factual question Alice needs answered

        Returns:
            Factual answer from knowledge base
        """
        try:
            messages = [
                {"role": "system", "content": KNOWLEDGE_PROMPT},
                {"role": "user", "content": question}
            ]

            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temp for factual accuracy
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result['message']['content']
            else:
                logger.error(f"Knowledge query failed: {response.status_code}")
                return "I couldn't retrieve that information right now."

        except Exception as e:
            logger.error(f"Error in knowledge query: {e}")
            return "I encountered an error accessing my knowledge base."

    def parse_complex_input(self, user_input: str) -> Dict[str, Any]:
        """
        Alice asks Ollama to parse complex natural language.
        Ollama acts as a linguistic analyzer - extracts intent and entities.

        Args:
            user_input: The complex user input to parse

        Returns:
            Structured parsing result with intent, entities, ambiguities
        """
        try:
            parse_request = f"""Parse this user input and return a JSON structure with:
- intent: The primary intent
- entities: Key entities mentioned
- ambiguities: Any unclear aspects
- interpretations: Possible meanings

Input: {user_input}"""

            messages = [
                {"role": "system", "content": PARSER_PROMPT},
                {"role": "user", "content": parse_request}
            ]

            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Low temp for consistent parsing
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result['message']['content']

                # Try to parse as JSON, fallback to structured response
                try:
                    import json
                    return json.loads(content)
                except:
                    return {
                        'intent': 'unknown',
                        'raw_analysis': content,
                        'entities': {}
                    }
            else:
                logger.error(f"Parse request failed: {response.status_code}")
                return {'intent': 'parse_failed', 'entities': {}}

        except Exception as e:
            logger.error(f"Error in parse_complex_input: {e}")
            return {'intent': 'error', 'entities': {}, 'error': str(e)}

    def phrase_with_tone(self, content: str, tone: str, context: Dict = None) -> str:
        """
        Alice asks Ollama to phrase her structured thought with natural language.
        Ollama acts as a phrasing assistant - makes Alice's thoughts sound natural.

        This is the key method for the tool-based architecture:
        - Alice decides WHAT to say (content)
        - Alice decides HOW to say it (tone)
        - Ollama just makes it sound natural

        Args:
            content: Alice's structured thought/decision (what she wants to say)
            tone: The exact tone Alice wants to use (warm/professional/casual/friendly)
            context: Optional context (user_name, situation, etc.)

        Returns:
            Naturally phrased response matching Alice's specified tone
        """
        try:
            context = context or {}
            user_name = context.get('user_name', 'the user')

            phrasing_request = f"""Alice has formulated a response and needs it phrased naturally.

Alice's thought/decision: {content}

Tone to use: {tone}
Context: User is {user_name}

Please phrase this naturally using the specified tone. Keep Alice's personality markers (warmth, helpfulness, honesty) but match the exact tone she specified."""

            messages = [
                {"role": "system", "content": PHRASER_PROMPT},
                {"role": "user", "content": phrasing_request}
            ]

            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,  # Higher temp for natural variation
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result['message']['content']
            else:
                logger.error(f"Phrasing request failed: {response.status_code}")
                # Fallback: return content as-is
                return str(content)

        except Exception as e:
            logger.error(f"Error in phrase_with_tone: {e}")
            return str(content)

    def audit_logic(self, logic_chain: List[str]) -> Dict[str, Any]:
        """
        Alice asks Ollama to verify her reasoning.
        Ollama acts as a logic checker - identifies errors and inconsistencies.

        Args:
            logic_chain: Alice's chain of reasoning steps

        Returns:
            Audit result with errors, inconsistencies, and suggestions
        """
        try:
            reasoning_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(logic_chain)])

            audit_request = f"""Please audit this reasoning chain for errors or inconsistencies:

{reasoning_text}

Provide:
- has_errors: true/false
- issues: List of any problems found
- suggestions: How to improve the logic
- overall_assessment: Brief summary"""

            messages = [
                {"role": "system", "content": AUDITOR_PROMPT},
                {"role": "user", "content": audit_request}
            ]

            active_model = self.config.active_model
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": active_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Low temp for consistent auditing
                        "num_gpu": 1,
                        "num_thread": 16,
                        "num_ctx": 4096
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                content = result['message']['content']

                # Try to parse structured response
                try:
                    import json
                    return json.loads(content)
                except:
                    # Fallback: analyze content for issues
                    has_errors = any(word in content.lower() for word in ['error', 'incorrect', 'inconsistent', 'flaw'])
                    return {
                        'has_errors': has_errors,
                        'raw_audit': content,
                        'suggestions': []
                    }
            else:
                logger.error(f"Audit request failed: {response.status_code}")
                return {'has_errors': False, 'audit_failed': True}

        except Exception as e:
            logger.error(f"Error in audit_logic: {e}")
            return {'has_errors': False, 'error': str(e)}

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def set_temperature(self, temp: float):
        """Set response creativity (0.0 = focused, 1.0 = creative)"""
        if 0 <= temp <= 1:
            self.config.temperature = temp
            logger.info(f"🌡Temperature set to: {temp}")
        else:
            logger.warning("Temperature must be between 0.0 and 1.0")
    
    def get_stats(self) -> Dict:
        """Get conversation statistics"""
        return {
            "messages": len(self.conversation_history) // 2,
            "temperature": self.config.temperature,
            "model": self.config.model
        }


# Main interface
if __name__ == "__main__":
    print("=" * 80)
    print("A.L.I.C.E - Advanced GPU-Accelerated AI Assistant")
    print("=" * 80)
    print("\n💻 Optimized for:")
    print("   - CPU: Intel i7-14700K")
    print("   - GPU: RTX 5070 Ti")
    print("   - RAM: 32GB")
    print("\nModel: Llama 3.3 70B (ChatGPT-level performance)")
    print("=" * 80)
    
    # Initialize config
    config = LLMConfig(
        model="llama3.3:70b",
        temperature=0.7,
        max_history=20
    )
    
    try:
        assistant = LocalLLMEngine(config)
        
        print("\n[OK] A.L.I.C.E initialized successfully!")
        print("[GPU] Acceleration: ENABLED")
        print("\nStart chatting! Available commands:")
        print("   /clear     - Clear conversation history")
        print("   /stream    - Toggle streaming mode")
        print("   /temp <n>  - Set creativity (0.0-1.0)")
        print("   /stats     - Show conversation stats")
        print("   exit       - End conversation")
        print("=" * 80)
        
        stream_mode = True  # Enable streaming by default
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == '/clear':
                    assistant.clear_history()
                    print("[OK] Conversation history cleared")
                    continue
                
                if user_input.lower() == '/stream':
                    stream_mode = not stream_mode
                    print(f"[OK] Streaming mode: {'ON' if stream_mode else 'OFF'}")
                    continue
                
                if user_input.lower().startswith('/temp'):
                    try:
                        parts = user_input.split()
                        if len(parts) == 2:
                            temp_value = float(parts[1])
                            assistant.set_temperature(temp_value)
                            print(f"[OK] Temperature set to: {temp_value}")
                        else:
                            print("[ERROR] Usage: /temp 0.7")
                    except ValueError:
                        print("[ERROR] Invalid temperature value. Use a number between 0.0 and 1.0")
                    continue
                
                if user_input.lower() == '/stats':
                    stats = assistant.get_stats()
                    print(f"\nStatistics:")
                    print(f"   Messages: {stats['messages']}")
                    print(f"   Temperature: {stats['temperature']}")
                    print(f"   Model: {stats['model']}")
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("\nA.L.I.C.E: Goodbye! It was a pleasure assisting you!")
                    break
                
                # Get response
                print("\nA.L.I.C.E: ", end="", flush=True)
                
                if stream_mode:
                    for chunk in assistant.stream_chat(user_input):
                        print(chunk, end="", flush=True)
                    print()  # New line after streaming
                else:
                    response = assistant.chat(user_input)
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\nA.L.I.C.E: Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                print(f"\n[ERROR] Error: {e}")
    
    except Exception as e:
        logger.error(f"Failed to start A.L.I.C.E: {e}")
        print(f"\n[ERROR] Failed to start A.L.I.C.E: {e}")
        print("\nSetup Instructions:")
        print("1. Install Ollama from: https://ollama.ai")
        print("2. Open a terminal and run: ollama serve")
        print("3. In another terminal, run: ollama pull llama3.3:70b")
        print("4. Wait for download to complete (this may take a while)")
        print("5. Run this script again: python ai/llm_engine.py")
        print("\nYour RTX 5070 Ti will be automatically detected and used!")
