# A.L.I.C.E (Artificial Linguistic Intelligence Computer Entity)

A sophisticated AI personal assistant powered by advanced machine learning, natural language processing, and local LLM capabilities. Built to run on powerful systems with GPU acceleration for ChatGPT-level performance completely offline.

## Features

### Advanced AI Capabilities
- **Llama 3.1 8B Integration**: Local LLM via Ollama with efficient performance
- **GPU-Accelerated**: Optimized for RTX 5070 Ti and compatible systems
- **Intent Classification**: Automatically detects user intent (questions, commands, tasks, etc.)
- **Entity Extraction**: Identifies emails, dates, times, URLs, and more
- **Sentiment Analysis**: Understands emotional context of conversations
- **Semantic Understanding**: Deep comprehension of user queries
- **Enhanced Terminal UI**: Rich library-powered terminal with modern aesthetics

### Memory & Context Management
- **Episodic Memory**: Remembers conversations and events
- **Semantic Memory**: Stores facts and knowledge
- **Procedural Memory**: Retains how-to information
- **RAG (Retrieval Augmented Generation)**: Enhances responses with relevant memories
- **Vector Embeddings**: Semantic search across all stored memories
- **Persistent Context**: Maintains user preferences and conversation state across sessions

### Extensible Plugin System
- **Modular Architecture**: Easy to add new capabilities
- **Built-in Plugins**:
  - Weather Information
  - Time & Date
  - File Operations (create, read, move, delete)
  - System Control (volume, brightness, etc.)
  - Web Search
- **Custom Plugin Support**: Create your own plugins easily

### Voice Interaction
- **Speech-to-Text**: Multiple engine support (Google, Whisper, Vosk)
- **Text-to-Speech**: Natural voice responses with multiple voice options
- **Wake Word Detection**: Activate with "Hey Alice", "Alice", or "OK Alice"
- **Continuous Listening**: Always ready to respond
- **Voice Activity Detection**: Smart audio processing

### Task Execution & Automation
- **System Commands**: Execute shell commands safely
- **File Management**: Create, modify, and organize files
- **Application Control**: Open and manage applications
- **Workflow Automation**: Chain multiple tasks together
- **Scheduled Tasks**: Set reminders and timed actions

### Context-Aware Intelligence
- **Personalization**: Learns your preferences and habits
- **Topic Tracking**: Maintains conversation context
- **Proactive Suggestions**: Offers help based on context
- **Time-Aware**: Adjusts responses based on time of day
- **Location-Aware**: Provides location-specific information

## System Requirements

### Recommended Specifications
- **CPU**: Intel i7-14700K or equivalent
- **GPU**: RTX 5070 Ti (or any CUDA-compatible GPU)
- **RAM**: 16GB+ (for 8B model)
- **Storage**: 20GB+ free space (for models and data)
- **OS**: Windows 10/11, Linux, macOS

### Minimum Specifications
- **CPU**: Intel i5 or AMD Ryzen 5
- **GPU**: GTX 1060 6GB or better (optional, CPU works)
- **RAM**: 8GB (for 3B model)
- **Storage**: 10GB free space
- **OS**: Windows 10+, Ubuntu 20.04+, macOS 11+

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/A.L.I.C.E.git
cd A.L.I.C.E
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

### 4. Install Ollama (for LLM support)
- **Windows**: Download from [ollama.ai](https://ollama.ai)
- **Linux**: `curl https://ollama.ai/install.sh | sh`
- **macOS**: `brew install ollama`

### 5. Pull the LLM Model
```bash
ollama pull llama3.1:8b
```

### 6. Optional: Install Voice Dependencies
```bash
# Windows - PyAudio
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# macOS
brew install portaudio
pip install pyaudio
```

### 7. Optional: Install Advanced NLP (for better embeddings)
```bash
pip install sentence-transformers
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Usage

### 🌟 Production Mode (Enhanced Terminal)
**For regular use with Rich terminal UI and minimal logs:**
```bash
python -m app.alice
```

With options:
```bash
# Classic terminal mode (no Rich formatting)
python -m app.alice --ui classic

# Rich terminal mode (default)
python -m app.alice --ui rich

# Enable voice interaction
python -m app.alice --voice

# Specify LLM model
python -m app.alice --model llama3.1:8b
```

### 🔧 Debug Mode (Full Logs)
**For development and troubleshooting:**
```bash
python -m app.main
```

With voice support:
```bash
python -m app.main --voice
```

Voice-only mode (wake word activation):
```bash
python -m app.main --voice-only
```

Custom configuration:
```bash
python -m app.main --name "User" --model llama3.1:8b --voice --llm-policy minimal
```

### LLM Policy Options
```bash
# Minimal LLM usage (recommended for learning)
python -m app.main --llm-policy minimal

# Balanced (default production)
python -m app.main --llm-policy balanced

# Aggressive (highest quality, slowest)
python -m app.main --llm-policy aggressive
```

### Available Commands
- `/help` - Show available commands
- `/voice` - Toggle voice mode on/off
- `/clear` - Clear conversation history
- `/memory` - Show memory statistics
- `/summary` - Get conversation summary
- `/context` - Show current context
- `/topics` - List conversation topics
- `/entities` - Show tracked entities
- `/plugins` - List available plugins
- `/location` - Set or view your location
- `/status` - Show system status (includes routing stats, LLM usage)
- `/save` - Save current state
- `/correct` - Correct last response
- `/feedback` - Rate last response
- `/learning` - Show learning statistics and pattern suggestions
- `/policy` - Show current LLM policy settings
- `/patterns` - Show learned conversation patterns
- `exit` - End conversation

## Project Structure

```
A.L.I.C.E/
├── ai/
│   ├── core.py                         # UNIFIED ENGINE INTERFACE (single import point)
│   │
│   ├── UNIFIED ENGINES:
│   ├── context_engine.py               # Context + conversation state (replaces context_manager + advanced_context_handler)
│   ├── reasoning_engine.py             # World state + goals + verification (replaces world_state + reference_resolver + goal_resolver + verifier)
│   ├── learning_engine.py              # Pattern learning + training (replaces training_system + response_generator + ml_learner)
│   │
│   ├── ROUTING & POLICY:
│   ├── router.py                       # Strict priority-based routing (SELF_REFLECTION → CONVERSATIONAL → TOOL → RAG → LLM)
│   ├── llm_policy.py                   # LLM budget and approval controls
│   ├── simple_formatters.py            # Tool output formatters (no LLM)
│   ├── llm_context.py                  # Stable context schema for LLM
│   │
│   ├── INFRASTRUCTURE:
│   ├── errors.py                       # Structured error types (NLPError, ToolError, LLMError, etc.)
│   ├── service_degradation.py          # Graceful fallback when services down
│   │
│   ├── CORE SYSTEMS:
│   ├── llm_engine.py                   # Ollama LLM integration
│   ├── nlp_processor.py                # NLP with intent detection
│   ├── memory_system.py                # Long-term memory with RAG, consolidation, deduplication
│   ├── conversation_summarizer.py      # Conversation summarization
│   ├── conversational_engine.py        # Learned conversation patterns
│   │
│   ├── LEARNING & TRACKING:
│   ├── entity_relationship_tracker.py  # Entity & relationship tracking
│   ├── active_learning_manager.py      # Continuous learning with safety guardrails
│   │
│   ├── PLUGINS:
│   ├── plugin_system.py                # Plugin architecture
│   ├── calendar_plugin.py              # Google Calendar
│   ├── email_plugin.py                 # Gmail
│   ├── music_plugin.py                 # Music control
│   ├── notes_plugin.py                 # Notes management
│   ├── document_plugin.py              # Document RAG
│   ├── maps_plugin.py                  # Maps and location
│   │
│   ├── ANTICIPATORY AI (EXPERIMENTAL):
│   ├── event_bus.py                    # Event system
│   ├── observers.py                    # Observer pattern
│   ├── pattern_learner.py              # Behavioral patterns
│   ├── system_monitor.py               # System monitoring
│   ├── task_planner.py                 # Task planning
│   ├── plan_executor.py                # Plan execution
│   ├── proactive_assistant.py          # Proactive suggestions
│   ├── error_recovery.py               # Error handling
│   ├── smart_context_cache.py          # Context caching
│   ├── adaptive_context_selector.py    # Context optimization
│   ├── predictive_prefetcher.py        # Action prediction
│   ├── response_optimizer.py           # Response refinement
│   └── self_reflection.py              # Code introspection
│
├── speech/
│   ├── speech_engine.py       # Voice interaction system
│   ├── audio_segmentation.py  # Audio processing
│   └── phoneme_generation.py  # Speech synthesis
│
├── ui/
│   ├── rich_terminal.py       # Enhanced Rich terminal UI
│   └── __init__.py            # UI package exports
│
├── features/
│   └── welcome.py             # Welcome messages & greetings
│
├── data/
│   ├── context/               # User context and preferences
│   ├── memory/                # Long-term memory storage
│   ├── training/              # Training data and learned patterns
│   ├── notes/                 # User notes
│   ├── entities.json          # Tracked entities
│   └── relationships.json     # Entity relationships
│
├── config/
│   ├── config.json.template   # Configuration template
│   └── cred/                  # Local credentials (ignored)
│
├── memory/
│   ├── corrections.json       # User corrections
│   ├── learning_patterns.json # Learning patterns (versioned)
│   └── user_feedback.json     # User feedback data
│
├── app/
│   ├── alice.py               # Production interface (Rich UI)
│   └── main.py                # Debug interface (full logs)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE
```

### Key Architecture Principles

1. **Single Import Point**: Import from `ai.core` instead of individual modules
2. **Unified Engines**: Three core engines handle all intelligence
3. **Strict Routing**: Priority-based, LLM as last resort
4. **Safe Learning**: Versioned patterns, shadow mode, minimum examples
5. **Graceful Degradation**: System works even when services are down

## How It Works

### Strict Routing Order (LLM as Last Resort)

A.L.I.C.E uses a priority-based routing system to minimize LLM dependency and maximize speed:

```
User Input → NLP Processing (intent + entities)
    ↓
Priority 1: SELF_REFLECTION
    └─ Code introspection, training stats, system commands
    └─ Handled directly without external calls
    ↓
Priority 2: CONVERSATIONAL (Learned Patterns)
    └─ Greetings, farewells, chitchat, known patterns
    └─ No LLM - uses learned responses from past interactions
    ↓
Priority 3: TOOL_CALL (Plugins)
    └─ Email, calendar, weather, files, music, notes
    └─ Structured output + simple formatter (NO LLM)
    ↓
Priority 4: RAG_ONLY (Knowledge Retrieval)
    └─ Document queries, fact lookup
    └─ Retrieval without generation
    ↓
Priority 5: LLM_FALLBACK (Last Resort)
    └─ Complex questions, generation tasks
    └─ Requires user approval: "I don't know this yet. Want me to look it up?"
    └─ Logged for pattern learning
```

### Detailed Flow

**1. Input Processing**
```
User Input → NLP Processor → Intent Detection + Entity Extraction
```

**2. Routing Decision**
```
Intent + Confidence → Router → Routing Decision (Priority 1-5)
```

**3. Context Building** (only if needed)
```
User Prefs + Recent History + Relevant Memories → Structured Context
```

**4. Response Generation**

- **Priority 1-4**: Direct execution, no LLM
  - Self-reflection: System introspection
  - Conversational: Learned patterns
  - Tools: Plugin execution + simple formatter
  - RAG: Memory retrieval

- **Priority 5**: LLM Fallback
  - Check LLM policy (rate limit, approval)
  - User approves → Call LLM
  - Log to learning engine
  - After 3+ similar calls → Auto-create pattern

**5. Memory & Learning**
```
Every Interaction:
  → Store in episodic memory
  → Track entities and relationships
  → Periodic consolidation (every 100 turns)
  
LLM Calls:
  → Log for pattern learning
  → Suggest pattern creation after 3+ occurrences
  → Future calls use learned pattern (no LLM)
```

**6. Output**
```
Response → (Optional) Voice Synthesis → User
```

### LLM Policy Modes

Control how much A.L.I.C.E relies on the LLM:

```bash
# Minimal (default): LLM only for complex tasks, requires approval
python -m app.main --llm-policy minimal

# Balanced: LLM for tools + generation, no approval needed
python -m app.main --llm-policy balanced

# Aggressive: LLM for everything, highest quality but slower
python -m app.main --llm-policy aggressive
```

| Mode | Chitchat | Tool Formatting | Generation | User Approval | Rate Limit |
|------|----------|-----------------|------------|---------------|------------|
| minimal | Learned patterns only | Simple formatters | LLM allowed | Required | 5/min |
| balanced | Learned patterns only | LLM allowed | LLM allowed | Not required | 15/min |
| aggressive | LLM allowed | LLM allowed | LLM allowed | Not required | 30/min |

**Recommended**: Start with `minimal` to build learned patterns, then switch to `balanced` for production use.

## Customization

### Creating Custom Plugins

```python
from ai.plugin_system import PluginInterface

class MyCustomPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self.name = "MyPlugin"
        self.description = "Does something awesome"
        self.capabilities = ["custom_capability"]
    
    def initialize(self) -> bool:
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        return intent == "my_custom_intent"
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        return {
            "success": True,
            "response": "Custom plugin response!",
            "data": {}
        }
    
    def shutdown(self):
        pass
```

### Register Your Plugin

```python
from ai.plugin_system import PluginManager
from my_plugin import MyCustomPlugin

pm = PluginManager()
pm.register_plugin(MyCustomPlugin())
```

## Troubleshooting

### Out of Memory Error

**Error:** `model requires more system memory`

**Solution:** Switch to a smaller model:

```bash
# Download a smaller model
ollama pull llama3.2:3b

# Run ALICE with the smaller model
python -m app.alice --model llama3.2:3b
```

**Model Recommendations by RAM:**
- 8GB RAM: `llama3.2:3b` (~2GB model)
- 16GB RAM: `llama3.1:8b` (~5GB, recommended)
- 32GB RAM: `llama3.1:8b` or `mixtral:8x7b` (better quality)
- 64GB+ RAM: `llama3.3:70b` (best quality)

### Cannot Connect to Ollama

**Error:** `Cannot connect to Ollama`

**Solution:**
1. Make sure Ollama is running: `ollama serve`
2. Check if model is downloaded: `ollama list`
3. Pull the model if missing: `ollama pull llama3.1:8b`

### Voice Recognition Not Working

**Solution:**
1. Install PyAudio: `pip install pyaudio`
2. Test your microphone in Windows Sound settings
3. Try running with `--voice` flag

### Slow Responses

**Solutions:**
- Use a smaller model (`--model llama3.1:8b`)
- Close background applications
- Ensure Ollama is using your GPU (check Ollama logs)
- Clear conversation history with `/clear`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Llama 3.3 by Meta AI
- Ollama for local LLM inference
- OpenAI for inspiration
- The open-source community

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## Roadmap

- [x] Enhanced terminal UI with Rich library
- [x] Advanced NLP with slot filling and temporal parsing
- [x] Entity and relationship tracking
- [x] Conversation summarization
- [x] Active learning and self-improvement
- [x] Plugin system (Calendar, Email, Music, Notes, Documents)
- [ ] Web interface for remote access
- [ ] Mobile app integration
- [ ] Advanced computer vision capabilities
- [ ] Home automation integration
- [ ] Multi-language support
- [ ] Cloud sync for memories

---

Built for the future of personal AI assistants
