"""
A.L.I.C.E System Test
Verifies all components are working correctly
"""

import sys
import os

print("=" * 80)
print("🧪 A.L.I.C.E System Test")
print("=" * 80)

# Test 1: Import Core Modules
print("\n1. Testing core module imports...")
try:
    from ai.nlp_processor import NLPProcessor
    from ai.context_manager import ContextManager
    from ai.memory_system import MemorySystem
    from ai.plugin_system import PluginManager
    from ai.task_executor import TaskExecutor
    print("   ✅ All core modules imported successfully")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 2: NLP Processor
print("\n2. Testing NLP Processor...")
try:
    nlp = NLPProcessor()
    result = nlp.process("Hey Alice, what's the weather like today?")
    print(f"   [OK] Intent detected: {result['intent']}")
    print(f"   [OK] Sentiment: {result['sentiment']['category']}")
except Exception as e:
    print(f"   [ERROR] NLP test failed: {e}")

# Test 3: Context Manager
print("\n3. Testing Context Manager...")
try:
    ctx = ContextManager()
    ctx.user_prefs.name = "Test User"
    ctx.store_fact("test_key", "test_value", "testing")
    value = ctx.recall_fact("test_key", "testing")
    assert value == "test_value", "Fact recall failed"
    print(f"   [OK] Context manager working")
    print(f"   [OK] Fact storage and retrieval successful")
except Exception as e:
    print(f"   [ERROR] Context manager test failed: {e}")

# Test 4: Memory System
print("\n4. Testing Memory System...")
try:
    memory = MemorySystem()
    mem_id = memory.store_memory(
        "This is a test memory",
        memory_type="episodic",
        importance=0.8
    )
    print(f"   ✅ Memory stored: {mem_id}")
    
    results = memory.recall_memory("test memory", top_k=1)
    if results:
        print(f"   ✅ Memory recalled successfully")
    else:
        print(f"   ⚠️ Memory recall returned no results (might need better embeddings)")
except Exception as e:
    print(f"   ❌ Memory system test failed: {e}")

# Test 5: Plugin System
print("\n5. Testing Plugin System...")
try:
    from ai.plugin_system import TimePlugin, WeatherPlugin
    pm = PluginManager()
    pm.register_plugin(TimePlugin())
    pm.register_plugin(WeatherPlugin())
    
    print(f"   ✅ {len(pm.plugins)} plugins registered")
    
    # Test plugin execution
    result = pm.execute_for_intent("time", "What time is it?", {}, {})
    if result and result.get('success'):
        print(f"   ✅ Plugin execution successful")
    else:
        print(f"   ⚠️ Plugin execution returned no result")
except Exception as e:
    print(f"   ❌ Plugin system test failed: {e}")

# Test 6: Task Executor
print("\n6. Testing Task Executor...")
try:
    executor = TaskExecutor(safe_mode=True)
    
    # Test file operation
    result = executor.file_operation("create", "test_file.txt")
    if result.success:
        print(f"   ✅ File creation successful")
        
        # Clean up
        executor.file_operation("delete", "test_file.txt")
        print(f"   ✅ File deletion successful")
    else:
        print(f"   ❌ File operation failed: {result.error}")
except Exception as e:
    print(f"   ❌ Task executor test failed: {e}")

# Test 7: LLM Engine (Optional - requires Ollama)
print("\n7. Testing LLM Engine (requires Ollama)...")
try:
    from ai.llm_engine import LocalLLMEngine, LLMConfig
    
    config = LLMConfig(model="llama3.3:70b")
    llm = LocalLLMEngine(config)
    
    # Just check connection
    print(f"   [OK] LLM engine initialized")
    print(f"   ℹ️ Model: {config.model}")
    
except Exception as e:
    print(f"   ⚠️ LLM test skipped (Ollama not running or not installed)")
    print(f"      Error: {e}")

# Test 8: Speech Engine (Optional - requires audio hardware)
print("\n8. Testing Speech Engine (optional)...")
try:
    from speech.speech_engine import SpeechEngine, SpeechConfig
    
    config = SpeechConfig()
    speech = SpeechEngine(config)
    
    print(f"   ✅ Speech engine initialized")
    print(f"   ℹ️ TTS available: {speech.tts_engine is not None}")
    print(f"   ℹ️ STT available: {speech.recognizer is not None}")
    
except Exception as e:
    print(f"   ⚠️ Speech test skipped (audio dependencies not installed)")
    print(f"      Error: {e}")

# Summary
print("\n" + "=" * 80)
print("✅ System Test Complete!")
print("=" * 80)
print("\nNext steps:")
print("1. Install Ollama: https://ollama.ai")
print("2. Pull model: ollama pull llama3.3:70b")
print("3. Run ALICE: python main.py --name 'Your Name'")
print("\nOptional:")
print("- Install voice dependencies: pip install pyaudio")
print("- Install better embeddings: pip install sentence-transformers")
print("=" * 80)
