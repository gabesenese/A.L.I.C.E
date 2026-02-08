# A.L.I.C.E

**Advanced Linguistic Intelligence Computer Entity**

A sophisticated AI assistant with advanced memory systems, natural language understanding, and extensible plugin architecture.

## Features

- **Advanced Memory Systems**
  - Episodic memory for conversation history
  - Semantic memory for facts and knowledge
  - Entity tracking and relationship mapping

- **Natural Language Understanding**
  - Intent classification with NLP
  - Entity extraction
  - Context-aware responses
  - Smart routing between pattern matching and LLM

- **Plugin Architecture**
  - Weather forecasting
  - Web search
  - File operations
  - Calculator
  - Time/date queries
  - Extensible plugin system

- **Voice Interaction** (Optional)
  - Speech-to-text input
  - Text-to-speech output

- **Self-Learning**
  - Feedback system
  - Training data generation
  - Continuous improvement

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run A.L.I.C.E
python -m app.alice
```

## Usage

### Production Mode
```bash
python -m app.alice                    # Start with Rich UI
python -m app.alice --voice            # Enable voice interaction
python -m app.alice --model llama3.3:70b  # Use different LLM model
python -m app.alice --privacy-mode     # Disable memory storage
```

### Development Mode
```bash
python -m app.dev                      # Auto-reload on code changes
python -m app.dev --voice              # Dev mode with voice
python -m app.dev --no-watch           # Disable auto-reload
python -m app.dev --no-thinking        # Hide debug output
```

### LLM Policy Modes
- `default`: Balanced approach (pattern matching + LLM when needed)
- `minimal`: Patterns only, no LLM for chitchat/simple queries
- `strict`: No LLM at all, patterns and tools only

```bash
python -m app.alice --llm-policy minimal
```

## Commands

- `/help` - Show available commands
- `/memory` - Show memory statistics
- `/plugins` - List available plugins
- `/clear` - Clear conversation history
- `/voice` - Toggle voice mode
- `/location` - Set your location
- `/correct` - Correct last response
- `/feedback` - Rate last response

## Project Structure

```
A.L.I.C.E/
├── app/              # Main application entry points
├── ai/               # AI/NLP components
├── features/         # Core features and tools
├── plugins/          # Extensible plugin system
├── ui/               # User interface components
├── speech/           # Voice interaction
├── self_learning/    # Learning and improvement
└── data/             # Runtime data and memories
```

## Configuration

Location settings, user preferences, and API keys are stored in `data/context/`.

## License

Private project - All rights reserved

## Author

Gabriel

---

*Built with advanced AI technology for natural and intelligent interactions*
