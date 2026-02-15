# A.L.I.C.E. Command Directory

Central reference for all test, verification, and validation commands organized by scenario.

---

## Phase 1A: Critical API Mismatch Fixes ✅

### Verify PhrasingLearner API Fixes
```bash
# Test that response formulation doesn't crash
python -c "from ai.core.response_formulator import ResponseFormulator; r = ResponseFormulator(); print('✓ API check passed')"

# Test learning pipeline integration
python -m pytest tests/ -k "response_formulation or phrasing" -v

# Manual end-to-end test
python app/main.py
# Then run: create a note called test
# Then run: show me my notes
```

**Status**: ✅ Complete (Commit: 074a013)

---

## Phase 1B: Bare Except Block Replacement

### Verify Exception Handling Improvements
```bash
# Run all tests to ensure no hidden exceptions
python -m pytest tests/ -v --tb=short

# Check error logs for proper exception messages
python -c "import logging; logging.basicConfig(level=logging.DEBUG); from ai.core import llm_engine"

# Grep logs for new detailed error messages
grep "ERROR" logs/alice.log | tail -20
```

**Status**: 🔄 In Progress

---

## Phase 1C: Type Hints Addition

### Verify Type Hints with mypy
```bash
# Check type hints on core modules
mypy ai/core/ --ignore-missing-imports

# Check all modified files
mypy ai/core/response_formulator.py ai/core/llm_engine.py ai/core/conversational_engine.py --strict

# Check learning modules
mypy ai/learning/ ai/training/ --ignore-missing-imports

# Full type coverage report
mypy ai/ --html-report mypy_report --ignore-missing-imports
open mypy_report/index.html
```

**Status**: 🔜 Pending

---

## Phase 2A: Facade Architecture

### Verify Facade Creation
```bash
# Test individual facades load correctly
python -c "from ai.facades.core_facade import CoreFacade; c = CoreFacade(); print('✓ CoreFacade initialized')"
python -c "from ai.facades.memory_facade import MemoryFacade; m = MemoryFacade(); print('✓ MemoryFacade initialized')"
python -c "from ai.facades.plugin_facade import PluginFacade; p = PluginFacade(); print('✓ PluginFacade initialized')"

# Test main.py with facades
python app/main.py --test-mode

# Check import count reduction
python -c "from app.main import ALICE; import sys; ai_modules = [m for m in sys.modules if m.startswith('ai.')]; print(f'AI modules loaded: {len(ai_modules)}')"

# Run integration tests
python -m pytest tests/integration/ -v
```

**Status**: 🔜 Pending

---

## Phase 2B: Memory System Decomposition

### Verify Memory System Refactoring
```bash
# Test EmbeddingManager standalone
python -c "from ai.memory.embedding_manager import EmbeddingManager; em = EmbeddingManager(); print(em.create_embedding('test'))"

# Test MemoryStore abstraction
python -c "from ai.memory.memory_store import InMemoryMemoryStore; store = InMemoryMemoryStore(); print('✓ MemoryStore working')"

# Test persistence
python -c "from ai.memory.persistence_manager import PersistenceManager; pm = PersistenceManager(); print('✓ Persistence working')"

# Full memory system test
python -m pytest tests/test_memory_system.py -v

# Memory consolidation test
python scripts/test_memory_consolidation.py

# Check memory usage improvement
python -c "from ai.memory.memory_system import MemorySystem; import sys; m = MemorySystem(); print(f'Memory size: {sys.getsizeof(m)} bytes')"
```

**Status**: 🔜 Pending

---

## Phase 2C: Integration Tests

### Run Integration Test Suite
```bash
# Response formulation pipeline
python -m pytest tests/integration/test_response_formulation_pipeline.py -v --cov=ai.core

# Learning cycle
python -m pytest tests/integration/test_learning_cycle.py -v --cov=ai.learning

# Memory RAG
python -m pytest tests/integration/test_memory_rag.py -v --cov=ai.memory

# Plugin execution
python -m pytest tests/integration/test_plugin_execution.py -v --cov=ai.plugins

# All integration tests
python -m pytest tests/integration/ -v --cov=ai --cov-report=html
open htmlcov/index.html
```

**Status**: 🔜 Pending

---

## Phase 3A: Unified Policy Manager

### Verify Policy Unification
```bash
# Test policy manager initialization
python -c "from ai.core.unified_policy import get_policy_manager; pm = get_policy_manager(); print(f'Thresholds: {pm.thresholds}')"

# Test threshold loading
python -c "from ai.core.unified_policy import get_policy_manager; pm = get_policy_manager(); pm.load_dynamic_thresholds('data/training/threshold.json'); print('✓ Thresholds loaded')"

# Test policy decisions
python -m pytest tests/test_unified_policy.py -v

# Compare old vs new policy behavior
python scripts/compare_policy_decisions.py
```

**Status**: 🔜 Pending

---

## Phase 3B: Comprehensive Testing

### Run Full Test Suite
```bash
# All unit tests
python -m pytest tests/unit/ -v

# All integration tests
python -m pytest tests/integration/ -v

# All end-to-end tests
python -m pytest tests/e2e/ -v

# Full suite with coverage
python -m pytest tests/ -v --cov=ai --cov-report=html --cov-report=term

# Coverage threshold check (must be >70%)
python -m pytest tests/ --cov=ai --cov-fail-under=70

# Coverage by module
python -m pytest tests/ --cov=ai.core --cov=ai.learning --cov=ai.memory --cov=ai.plugins --cov-report=term
```

**Status**: 🔜 Pending

---

## End-to-End Verification

### Complete System Validation
```bash
# Smoke tests - basic functionality
python app/main.py --test-mode

# Performance regression check
python scripts/benchmark_response_time.py

# Memory leak detection
python scripts/check_memory_leaks.py

# Run Alice with debug logging
python app/main.py --debug

# Check all logs for errors
grep -i "error\|exception\|traceback" logs/alice.log | grep -v "test"
```

---

## Development Utilities

### Code Quality Checks
```bash
# Run linter
flake8 ai/ --max-line-length=120 --exclude=__pycache__

# Check code formatting
black ai/ --check

# Security scan
bandit -r ai/ -ll

# Complexity analysis
radon cc ai/ -a -nb
```

### Git Operations
```bash
# Check current changes
git status

# View diff
git diff

# Create feature branch
git checkout -b refactor/comprehensive-improvements

# Commit checkpoint
git add . && git commit -m "Checkpoint: [description]"

# Push changes
git push origin refactor/comprehensive-improvements

# View commit history
git log --oneline -10
```

### Quick Diagnostics
```bash
# Check Python environment
python --version
pip list | grep -E "ollama|sentence-transformers|torch"

# Check Alice installation
python -c "import ai; print(ai.__file__)"

# Check Ollama connection
curl http://localhost:11434/api/tags

# Test import resolution
python -c "from ai.core.llm_engine import LocalLLMEngine; print('✓ Imports working')"
```

---

## Rollback Commands

### Revert to Previous State
```bash
# Revert last commit (keep changes)
git reset --soft HEAD~1

# Revert last commit (discard changes)
git reset --hard HEAD~1

# Revert to specific commit
git reset --hard <commit-hash>

# Restore specific file
git checkout HEAD -- <file-path>

# Switch back to main branch
git checkout main
```

---

## Documentation Generation

### Generate API Documentation
```bash
# Generate pydoc for plugin system
python -m pydoc ai.plugins.plugin_system > docs/api/plugin_system_reference.txt

# Generate module documentation
pydoc -w ai.facades.core_facade
pydoc -w ai.memory.memory_system

# Generate full documentation
pdoc --html --output-dir docs/generated ai/

# Validate documentation examples
python docs/api/examples/simple_plugin.py
python docs/api/examples/async_plugin.py
```

---

## Performance Monitoring

### Benchmark Commands
```bash
# Response time benchmark
python scripts/benchmark_response_time.py --iterations 100

# Memory usage profiling
python -m memory_profiler app/main.py

# CPU profiling
python -m cProfile -o profile.stats app/main.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Concurrent request testing
python scripts/stress_test.py --concurrent 10 --duration 60
```

---

## Notes

- Always run tests before committing
- Use `--test-mode` for non-interactive testing
- Check logs after major refactoring
- Monitor memory usage for large changes
- Keep coverage above 70%

**Last Updated**: Phase 1A Complete
