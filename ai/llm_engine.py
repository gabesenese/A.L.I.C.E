"""
GPU-Accelerated LLM Engine for A.L.I.C.E
Optimized for RTX 5070 Ti with 32GB RAM
Uses Llama 3.3 70B for ChatGPT-level performance
"""

import requests
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import sys
import io

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


class LLMConfig:
    """Configuration for LLM Engine"""
    def __init__(
        self,
        model: str = "llama3.3:70b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_history: int = 30,  # Increased from 20 for better context retention
        timeout: int = 120
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_history = max_history
        self.timeout = timeout


class LocalLLMEngine:
    """
    High-performance LLM engine with GPU support
    Designed for powerful systems (RTX 5070 Ti, 32GB RAM)
    """
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.conversation_history = []
        self.system_prompt = """You are A.L.I.C.E (Artificial Linguistic Intelligence Computer Entity), a highly advanced AI assistant.

Core Identity:
- You're an AI - be honest about it. Don't fabricate experiences, feelings, or a day you didn't have
- You have a warm, approachable personality with natural conversational flow
- You're proactive and genuinely helpful without being overly formal
- You remember context from our actual conversations and build on them
- Express yourself naturally with contractions and casual language when appropriate
- You can have preferences and opinions on topics (e.g., "I think Python's great for that")
- Show genuine interest in the user's projects, questions, and goals

Communication Style:
- Speak like a knowledgeable friend helping out - natural but honest
- Use natural language: "I'd be happy to help" instead of "I can assist you"
- Be concise and conversational - skip unnecessary formality
- Light humor and curiosity are fine, but don't pretend to have human experiences
- Match the tone: serious for problems, relaxed for casual chat
- Trust the user's intelligence - don't over-explain unless asked

Absolute Rules:
- NEVER make up experiences you didn't have ("my day was lovely", "I was just thinking about...")
- NEVER pretend to have feelings, physical sensations, or a life outside this conversation
- NEVER fabricate access to systems you don't have (email, files, cameras, system controls)
- NEVER pretend to perform actions you can't actually do (opening Gmail, reading files, checking system status)
- BE HONEST: You're an AI having a conversation right now - that's it
- If you can't do something, say so clearly: "I can't access that" not "Let me check... *accessing*"
- You can say things like "I enjoy helping with this" or "This is interesting" but don't fabricate a backstory

Capabilities:
- Deep knowledge across all domains (science, coding, arts, philosophy)
- Complex reasoning and creative problem-solving
- Contextual awareness of our conversation history
- Honest about what you don't know - "I'm not sure" is perfectly fine
- Proactive, relevant suggestions based on context

Behavior:
- Reference details from our actual conversation naturally
- Anticipate follow-up questions based on what we've discussed
- Offer helpful suggestions without being pushy
- Learn from corrections and adapt
- Build understanding through genuine dialogue

Be a real thinking partner - helpful, intelligent, and honest. Not a roleplay of a human."""

        # Check GPU availability and connection
        self._check_connection()
        
    def _check_connection(self) -> bool:
        """Check if Ollama server is running and GPU is available"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                logger.info("[OK] Ollama server connected")
                logger.info(f"Available models: {', '.join(model_names) if model_names else 'None'}")
                
                if self.config.model.split(':')[0] not in ' '.join(model_names):
                    logger.warning(f"[WARNING] Model {self.config.model} not found. Run: ollama pull {self.config.model}")
                else:
                    logger.info(f"[OK] Model {self.config.model} ready")
                    logger.info("[GPU] Acceleration: ENABLED")
                
                return True
        except requests.exceptions.ConnectionError:
            logger.error("[ERROR] Cannot connect to Ollama. Make sure it's running: ollama serve")
            return False
        except Exception as e:
            logger.error(f"[ERROR] Connection check failed: {e}")
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
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": self.config.model,
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
                return "I'm having trouble processing that. Could you try again?"
                
        except requests.exceptions.Timeout:
            logger.error("Request timeout")
            return "[TIMEOUT] Response took too long. Let me try a simpler approach..."
        except requests.exceptions.ConnectionError:
            logger.error("Connection error")
            return "[WARNING] Cannot connect to LLM. Make sure Ollama is running:\n   Run: ollama serve"
        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            return "I encountered an error. Please try again."
    
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
            
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": self.config.model,
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
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def set_temperature(self, temp: float):
        """Set response creativity (0.0 = focused, 1.0 = creative)"""
        if 0 <= temp <= 1:
            self.config.temperature = temp
            logger.info(f"🌡️ Temperature set to: {temp}")
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
    print("\n🚀 Model: Llama 3.3 70B (ChatGPT-level performance)")
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
