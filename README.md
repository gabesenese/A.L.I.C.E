# A.L.I.C.E (Artificial Linguistic Intelligence Computer Entity)

A sophisticated AI personal assistant powered by advanced machine learning, natural language processing, and local LLM capabilities. Built to run on powerful systems with GPU acceleration for ChatGPT-level performance completely offline.

## Features

### Advanced AI Capabilities
- **Llama 3.3 70B Integration**: ChatGPT-level performance running locally via Ollama
- **GPU-Accelerated**: Optimized for RTX 5070 Ti and high-end systems
- **Intent Classification**: Automatically detects user intent (questions, commands, tasks, etc.)
- **Entity Extraction**: Identifies emails, dates, times, URLs, and more
- **Sentiment Analysis**: Understands emotional context of conversations
- **Semantic Understanding**: Deep comprehension of user queries

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
- **RAM**: 32GB+ (for 70B model)
- **Storage**: 50GB+ free space (for models and data)
- **OS**: Windows 10/11, Linux, macOS

### Minimum Specifications
- **CPU**: Intel i5 or AMD Ryzen 5
- **GPU**: GTX 1060 6GB or better
- **RAM**: 16GB
- **Storage**: 20GB free space
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
ollama pull llama3.3:70b
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

### Interactive Console Mode (Default)
```bash
python main.py
```

### With Voice Support
```bash
python main.py --voice
```

### Voice-Only Mode (Wake Word Activation)
```bash
python main.py --voice-only
```

### Custom Configuration
```bash
python main.py --name "Tony Stark" --model llama3.3:70b --voice
```

### Available Commands
- `/help` - Show available commands
- `/voice` - Toggle voice mode on/off
- `/clear` - Clear conversation history
- `/memory` - Show memory statistics
- `/plugins` - List available plugins
- `/status` - Show system status
- `/save` - Save current state
- `exit` - End conversation

## Project Structure

```
A.L.I.C.E/
├── ai/
│   ├── ai_engine.py          # Legacy ML engine
│   ├── ai_model.py            # Neural network models
│   ├── llm_engine.py          # Ollama LLM integration
│   ├── nlp_processor.py       # Advanced NLP with intent detection
│   ├── context_manager.py     # Context and state management
│   ├── memory_system.py       # Long-term memory with RAG
│   ├── plugin_system.py       # Extensible plugin architecture
│   ├── task_executor.py       # Task automation framework
│   └── dataset.py             # Dataset management
├── speech/
│   ├── speech_engine.py       # Voice interaction system
│   ├── audio_segmentation.py  # Audio processing
│   └── phoneme_generation.py  # Speech synthesis
├── self_learning/
│   ├── self_learning.py       # Self-improvement system
│   ├── clean_database.py      # Data cleaning
│   └── reset_database.py      # Database management
├── features/
│   └── welcome.py             # Welcome messages
├── data/
│   ├── context/               # User context and preferences
│   └── memory/                # Long-term memory storage
├── models/
│   └── (AI models stored here)
├── main.py                    # Main orchestrator
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE
```

## How It Works

### 1. **Input Processing**
User input → NLP Processor → Intent Detection + Entity Extraction

### 2. **Context Enhancement**
Current Input + Conversation History + Relevant Memories → Enhanced Context

### 3. **Response Generation**
- **Plugin Check**: Can a plugin handle this?
  - Yes → Execute plugin
  - No → Use LLM with RAG context

### 4. **Memory Storage**
Interaction → Episodic Memory + Context Update → Save to Disk

### 5. **Output**
Text Response → (Optional) Voice Synthesis → User

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

**Error:** `model requires more system memory (39.8 GiB) than is available (33.4 GiB)`

**Solution:** The llama3.3:70b model requires about 40GB RAM. Switch to a smaller model:

```bash
# Download a smaller model (recommended for 32GB RAM systems)
ollama pull llama3.1:8b

# Run ALICE with the smaller model
python main.py --name "Your Name" --model llama3.1:8b
```

**Model Recommendations by RAM:**
- 8GB RAM: `llama3.2:3b` (2GB model)
- 16GB RAM: `mistral:7b` or `llama3.1:8b` (~5GB)
- 32GB RAM: `llama3.1:8b` or `mistral:7b` (70b won't fit)
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

- [ ] Web interface for remote access
- [ ] Mobile app integration
- [ ] Advanced computer vision capabilities
- [ ] Home automation integration
- [ ] Multi-language support
- [ ] Cloud sync for memories
- [ ] Advanced reasoning capabilities
- [ ] Tool use and function calling

---

Built for the future of personal AI assistants
