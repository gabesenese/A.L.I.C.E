# A.L.I.C.E Command Directory

**Quick reference for all commands to run, test, and interact with A.L.I.C.E**

Last Updated: 2026-02-16 (Added Analytics & Memory Management commands)

---

## 🚀 Quick Start

```bash
# Run A.L.I.C.E (interactive mode)
python app/main.py

# Run with debug logging
python app/main.py --debug

# Run in test mode (non-interactive)
python app/main.py --test-mode
```

---

## 📋 Table of Contents

1. [Running A.L.I.C.E](#-running-alice)
2. [Testing & Validation](#-testing--validation)
3. [Training & Learning](#-training--learning)
4. [Automation & Scheduling](#-automation--scheduling)
5. [Monitoring & Debugging](#-monitoring--debugging)
6. [Analytics & Memory Management](#-analytics--memory-management)
7. [Development Tools](#%EF%B8%8F-development-tools)
8. [Coverage & Quality](#-coverage--quality)
9. [Git Operations](#-git-operations)

---

## 🤖 Running A.L.I.C.E

### Interactive Mode
```bash
# Standard interactive session
python app/main.py

# With voice enabled
python app/main.py --voice

# With debug logging
python app/main.py --debug

# Production demo mode
python scripts/production_demo.py
```

### Testing Specific Features
```bash
# Test autonomous capabilities
python test_autonomous.py

# Test confidence building
python test_confidence_building.py

# Test initialization
python test_init.py

# Test Alice knowledge
python -m pytest tests/test_alice_knowledge.py -v

# Test weather functionality
python -m pytest tests/test_weather_debug.py -v
```

---

## ✅ Testing & Validation

### Run All Tests
```bash
# Run complete test suite
python -m pytest tests/ -v

# Run with detailed output
python -m pytest tests/ -v --tb=long

# Run specific test categories
python -m pytest tests/integration/ -v      # Integration tests only
python -m pytest tests/unit/ -v             # Unit tests only (if exists)
```

### Integration Tests
```bash
# Response formulation pipeline
python -m pytest tests/integration/test_response_formulation_pipeline.py -v

# Learning cycle
python -m pytest tests/integration/test_learning_cycle.py -v

# Memory RAG system
python -m pytest tests/integration/test_memory_rag.py -v

# Plugin execution
python -m pytest tests/integration/test_plugin_execution.py -v

# NLP tokenizer layers
python -m pytest tests/integration/test_nlp_tokenizer_layers.py -v
```

### Individual Test Files
```bash
# Comprehensive test suite
python tests/comprehensive_test.py

# Enhanced tests
python tests/enhanced_tests.py

# Confidence building tests
python tests/test_confidence_building.py
```

### Component Testing
```bash
# Test audit components
python scripts/testing/test_audit_components.py

# Test audit pipeline
python scripts/testing/test_audit_pipeline.py

# Test automation systems
python scripts/testing/test_automation.py
```

---

## 🎓 Training & Learning

### Scenario-Based Training
```bash
# Generate training scenarios
python generate_scenarios.py

# Run scenarios and train
python scripts/training/run_scenarios_and_train.py

# Run scenarios and generate report
python scripts/training/run_and_report_scenarios.py

# Count available scenarios
python scripts/utilities/count_scenarios.py
```

### Learning Cycles
```bash
# Simple learning cycle
python scripts/training/simple_learning.py

# Full learning cycle
python scripts/training/run_learning_cycle.py

# Automated training
python scripts/automation/automated_training.py
```

### Response Templates
```bash
# Seed response templates for faster learning
python scripts/automation/seed_response_templates.py

# Create learning documentation
python scripts/automation/create_learning_docs.py
```

---

## ⏰ Automation & Scheduling

### Start Automation
```bash
# Start all automation services
python scripts/automation/start_automation.py

# Nightly training (supervised)
python scripts/automation/nightly_training.py

# Nightly training (autonomous)
python scripts/automation/nightly_training_autonomous.py

# Nightly audit scheduler
python scripts/automation/nightly_audit_scheduler.py
```

---

## 🔍 Monitoring & Debugging

### Live Monitoring
```bash
# Monitor Alice in real-time
python tools/monitoring/monitor_live.py

# Monitor training progress
python tools/monitoring/monitor_training.py

# Monitor audit progress
python tools/monitoring/monitor_audit_progress.py
```

### Debugging Tools
```bash
# Check training data
python tools/debugging/check_training_data.py

# Debug import issues
python tools/debugging/debug_import.py

# Debug learned patterns
python tools/debugging/debug_patterns.py

# Audit training data
python tools/auditing/training_data_auditor.py
```

### System Diagnostics
```bash
# Check Python environment
python --version
pip list | grep -E "ollama|sentence-transformers|torch"

# Check Alice installation
python -c "import ai; print(ai.__file__)"

# Check Ollama connection
curl http://localhost:11434/api/tags

# Test core imports
python -c "from ai.core.llm_engine import LocalLLMEngine; print('✓ Imports working')"

# Test facades
python -c "from ai.facades import *; print('✓ All facades working')"
```

### Log Analysis
```bash
# View recent errors
grep "ERROR" logs/alice.log | tail -20

# View all errors (excluding tests)
grep -i "error\|exception\|traceback" logs/alice.log | grep -v "test"

# Monitor logs in real-time
tail -f logs/alice.log
```

---

## 📊 Analytics & Memory Management

### Usage Analytics

```bash
# View usage analytics log
tail -50 data/analytics/usage_log.jsonl

# View formatted usage stats
cat data/analytics/usage_log.jsonl | jq '.'

# View recent interactions
tail -20 data/analytics/usage_log.jsonl | jq -r '[.timestamp, .intent, .plugin, .response_time_ms] | @csv'

# View daily statistics
cat data/analytics/daily_stats.json | jq '.'

# Count interactions by intent
cat data/analytics/usage_log.jsonl | jq -r '.intent' | sort | uniq -c | sort -rn

# Count plugin usage
cat data/analytics/usage_log.jsonl | jq -r '.plugin | select(. != null)' | sort | uniq -c | sort -rn

# Average response time
cat data/analytics/usage_log.jsonl | jq -r '.response_time_ms | select(. != null)' | awk '{sum+=$1; count++} END {print sum/count " ms"}'

# Monitor usage in real-time
tail -f data/analytics/usage_log.jsonl | jq '.'
```

### Memory Growth Monitor

```bash
# View memory growth snapshots
tail -10 data/analytics/memory_growth.jsonl

# View formatted memory growth
cat data/analytics/memory_growth.jsonl | jq '.'

# Check latest memory stats
cat data/analytics/memory_growth.jsonl | tail -1 | jq '.memory_stats'

# View memory size over time
cat data/analytics/memory_growth.jsonl | jq -r '[.timestamp, .memory_stats.total_memories] | @csv'

# View memory type breakdown
cat data/analytics/memory_growth.jsonl | tail -1 | jq '.type_breakdown'

# Monitor memory growth rate
cat data/analytics/memory_growth.jsonl | jq -r '[.timestamp, .file_stats.total_size_mb] | @csv'

# View memory growth config
cat data/analytics/memory_monitor_config.json | jq '.'
```

### Memory Pruner

```bash
# View memory pruning config
cat data/memory_pruning_config.json | jq '.'

# Check archived memories
ls -lh data/archives/

# View archived conversations
ls -lht data/archives/ | head -20

# Check pruning schedule
cat data/memory_pruning_config.json | jq '.retention_days'

# View importance thresholds
cat data/memory_pruning_config.json | jq '.importance_threshold'

# Check last pruning time
cat data/memory_pruning_config.json | jq '.last_prune_time'

# Manually trigger pruning (edit config to force)
# Set "last_prune_time": null in data/memory_pruning_config.json
```

### Background Embedding Generator

```bash
# Check if background embedding generator is running
# (Look for [BgEmbedding] in logs during startup)
grep "BgEmbedding" logs/alice.log | tail -10

# Monitor embedding generation in real-time
tail -f logs/alice.log | grep "BgEmbedding"

# Check embedding queue status
# (Visible in debug logs when running with --debug)
python app/main.py --debug | grep "embedding"
```

### RAG Document Indexer Plugin

```bash
# Using RAG indexer in A.L.I.C.E (interactive mode)
# Start A.L.I.C.E and type these commands:

# List indexed directories
# Input: "list rag directories"

# Add directory to index
# Input: "add directory to rag index: ~/Documents/notes"

# Remove directory from index
# Input: "remove directory from rag index: ~/Documents/notes"

# Trigger reindexing
# Input: "reindex rag documents"

# Check index status
# Input: "show rag index status"

# View RAG indexer configuration
cat config/rag_indexer_config.json | jq '.'

# Check indexed directories
cat config/rag_indexer_config.json | jq '.indexed_directories'

# View file extensions indexed
cat config/rag_indexer_config.json | jq '.file_extensions'

# Check auto-index status
cat config/rag_indexer_config.json | jq '.auto_index'
```

### Analytics Dashboard (Future Enhancement)

```bash
# Generate usage report (future feature)
python scripts/analytics/generate_usage_report.py

# Export analytics to CSV
cat data/analytics/usage_log.jsonl | jq -r '[.timestamp, .intent, .plugin, .response_time_ms, .success] | @csv' > analytics.csv

# Memory growth report
cat data/analytics/memory_growth.jsonl | jq -r '[.timestamp, .memory_stats.total_memories, .file_stats.total_size_mb] | @csv' > memory_growth.csv
```

---

## 🛠️ Development Tools

### Code Organization
```bash
# Reorganize project structure
python tools/reorganize_project.py

# Deprecate old modules
python scripts/utilities/deprecate_modules.py

# Promote learned patterns to production
python scripts/utilities/promote_patterns.py
```

### Component Verification
```bash
# Verify response formulator
python -c "from ai.core.response_formulator import ResponseFormulator; r = ResponseFormulator(); print('✓ ResponseFormulator working')"

# Verify phrasing learner
python -c "from ai.learning.phrasing_learner import PhrasingLearner; p = PhrasingLearner(); print('✓ PhrasingLearner working')"

# Verify LLM gateway
python -c "from ai.core.llm_gateway import get_llm_gateway; g = get_llm_gateway(); print('✓ LLM Gateway working')"

# Verify memory system
python -c "from ai.memory.memory_system import MemorySystem; m = MemorySystem(); print('✓ Memory system working')"

# Verify embedding manager
python -c "from ai.memory.embedding_manager import get_embedding_manager; e = get_embedding_manager(); print('✓ Embedding manager working')"

# Verify plugin manager
python -c "from ai.plugins.plugin_system import PluginManager; pm = PluginManager(); print('✓ Plugin manager working')"
```

### API Testing
```bash
# Test PhrasingLearner API
python -c "from ai.learning.phrasing_learner import PhrasingLearner; p = PhrasingLearner(); print('✓ API check passed')"

# Test specific API patterns
python -m pytest tests/ -k "response_formulation or phrasing" -v
```

---

## 📊 Coverage & Quality

### Test Coverage
```bash
# Install coverage tools (if not installed)
pip install pytest-cov

# Run tests with coverage report
python -m pytest tests/ --cov=ai --cov-report=html --cov-report=term

# View HTML coverage report
start htmlcov/index.html       # Windows
open htmlcov/index.html        # Mac
xdg-open htmlcov/index.html    # Linux

# Get detailed coverage report
python -m pytest --cov=ai --cov-report=term-missing

# Check coverage threshold (must be >70%)
python -m pytest tests/ --cov=ai --cov-fail-under=70

# Coverage by module
python -m pytest --cov=ai.core --cov-report=term
python -m pytest --cov=ai.memory --cov-report=term
python -m pytest --cov=ai.learning --cov-report=term
python -m pytest --cov=ai.plugins --cov-report=term
python -m pytest --cov=ai.facades --cov-report=term
```

### Code Quality
```bash
# Run linter
flake8 ai/ --max-line-length=120 --exclude=__pycache__

# Check code formatting
black ai/ --check

# Format code automatically
black ai/

# Security scan
bandit -r ai/ -ll

# Complexity analysis
radon cc ai/ -a -nb

# Type checking with mypy
mypy ai/core/ --ignore-missing-imports
mypy ai/ --html-report mypy_report --ignore-missing-imports
```

### Performance Benchmarking
```bash
# Response time benchmark
python scripts/benchmark_response_time.py --iterations 100

# Memory usage profiling
python -m memory_profiler app/main.py

# CPU profiling
python -m cProfile -o profile.stats app/main.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Stress testing
python scripts/stress_test.py --concurrent 10 --duration 60

# Memory leak detection
python scripts/check_memory_leaks.py
```

---

## 📝 Git Operations

### Status & Inspection
```bash
# Check current changes
git status

# View diff of changes
git diff

# View recent commit history
git log --oneline -10
git log --oneline --graph --decorate -20

# See files changed in last commit
git show --name-only

# View full diff of last commit
git log -p -1
```

### Branching & Committing
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Add all changes
git add .

# Commit with message
git commit -m "Description of changes"

# Commit checkpoint
git add . && git commit -m "Checkpoint: [description]"

# Push changes
git push origin feature/your-feature-name

# Switch back to main
git checkout main
```

### Rollback Commands
```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert to specific commit
git reset --hard <commit-hash>

# Restore specific file
git checkout HEAD -- <file-path>
git restore <file-path>

# Discard all local changes
git reset --hard HEAD
```

---

## 📚 Documentation

### View Documentation
```bash
# Plugin API documentation
cat docs/api/plugin_interface.md
cat docs/api/plugin_lifecycle.md
cat docs/api/plugin_best_practices.md
```

### Run Documentation Examples
```bash
# Simple plugin example
python docs/api/examples/simple_plugin.py

# Stateful plugin example
python docs/api/examples/stateful_plugin.py

# Async plugin example
python docs/api/examples/async_plugin.py
```

### Generate Documentation
```bash
# Generate pydoc for plugin system
python -m pydoc ai.plugins.plugin_system > docs/api/plugin_system_reference.txt

# Generate module documentation
pydoc -w ai.facades.core_facade
pydoc -w ai.memory.memory_system

# Generate full HTML documentation
pdoc --html --output-dir docs/generated ai/
```

---

## 🎯 Common Workflows

### Full System Check
```bash
# 1. Run all tests
python -m pytest tests/ -v

# 2. Check coverage
python -m pytest tests/ --cov=ai --cov-report=term

# 3. Verify all components
python -c "from ai.facades import *; print('✓ All facades working')"

# 4. Check for errors in logs
grep "ERROR" logs/alice.log | tail -20

# 5. Run Alice interactively
python app/main.py
```

### Before Committing
```bash
# 1. Run all tests
python -m pytest tests/ -v

# 2. Check code quality
flake8 ai/ --max-line-length=120 --exclude=__pycache__

# 3. Format code
black ai/ --check

# 4. Check git status
git status

# 5. Commit if all checks pass
git add . && git commit -m "Your commit message"
```

### Training A.L.I.C.E
```bash
# 1. Generate scenarios
python generate_scenarios.py

# 2. Run training
python scripts/training/run_scenarios_and_train.py

# 3. Monitor progress
python tools/monitoring/monitor_training.py

# 4. Verify learning
python -m pytest tests/integration/test_learning_cycle.py -v

# 5. Check training data quality
python tools/debugging/check_training_data.py
```

### Debugging Issues
```bash
# 1. Check logs
tail -100 logs/alice.log

# 2. Run diagnostic tests
python tests/test_weather_debug.py
python tests/test_alice_knowledge.py

# 3. Debug specific components
python tools/debugging/debug_import.py
python tools/debugging/debug_patterns.py

# 4. Monitor live
python tools/monitoring/monitor_live.py

# 5. Audit training data
python tools/auditing/training_data_auditor.py
```

---

## ⚡ Pro Tips

1. **Use `--test-mode` for non-interactive testing**
   ```bash
   python app/main.py --test-mode
   ```

2. **Monitor logs in real-time while testing**
   ```bash
   tail -f logs/alice.log
   ```

3. **Run tests with coverage to identify gaps**
   ```bash
   python -m pytest tests/ --cov=ai --cov-report=html
   ```

4. **Use pytest's `-k` flag to run specific tests**
   ```bash
   python -m pytest tests/ -k "weather or plugin" -v
   ```

5. **Check import health before major refactoring**
   ```bash
   python -c "from ai.facades import *; print('✓ All imports healthy')"
   ```

6. **Always check coverage before committing**
   ```bash
   python -m pytest tests/ --cov=ai --cov-fail-under=70
   ```

7. **Use black to auto-format code**
   ```bash
   black ai/ app/ tests/
   ```

8. **Profile before optimizing**
   ```bash
   python -m cProfile -o profile.stats app/main.py
   ```

---

## 🆘 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows CMD
$env:PYTHONPATH += ";$(Get-Location)"     # Windows PowerShell
```

**Ollama Connection Issues:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

**Test Failures:**
```bash
# Run with verbose output
python -m pytest tests/ -v --tb=long

# Run specific failing test
python -m pytest tests/test_file.py::test_name -v
```

**Memory Issues:**
```bash
# Check memory usage
python -m memory_profiler app/main.py

# Monitor system resources
htop  # Linux/Mac
taskmgr  # Windows
```

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| **Run A.L.I.C.E** | `python app/main.py` |
| **Run all tests** | `python -m pytest tests/ -v` |
| **Check coverage** | `python -m pytest tests/ --cov=ai --cov-report=term` |
| **Train Alice** | `python scripts/training/run_scenarios_and_train.py` |
| **Monitor live** | `python tools/monitoring/monitor_live.py` |
| **Debug logs** | `tail -f logs/alice.log` |
| **View usage analytics** | `tail -20 data/analytics/usage_log.jsonl \| jq '.'` |
| **View memory growth** | `tail -10 data/analytics/memory_growth.jsonl \| jq '.'` |
| **Check memory config** | `cat data/memory_pruning_config.json \| jq '.'` |
| **View RAG config** | `cat config/rag_indexer_config.json \| jq '.'` |
| **Check imports** | `python -c "from ai.facades import *"` |
| **Run benchmarks** | `python scripts/benchmark_response_time.py` |
| **Format code** | `black ai/ app/ tests/` |
| **Git status** | `git status` |

---

**Remember:**
- Always run tests before committing (`python -m pytest tests/ -v`)
- Maintain >70% test coverage
- Check logs after major changes
- Monitor memory usage for large refactors
- Use `--debug` flag when troubleshooting

**Last Updated**: 2026-02-16
