"""Quick test of Alice initialization"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing Alice initialization...")
    from app.main import ALICE

    print("Creating Alice instance...")
    alice = ALICE(
        llm_model="llama3.2:1b",
        privacy_mode=False,
        llm_policy="default"
    )

    print("SUCCESS: Alice initialized without errors")
    print(f"- Autonomous agent available: {hasattr(alice, 'autonomous_agent')}")
    print(f"- Error recovery available: {hasattr(alice, 'error_recovery')}")
    print(f"- Execution loop available: {hasattr(alice, 'execution_loop')}")

    # Shutdown cleanly
    alice.shutdown()
    print("\nAlice shutdown successfully")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
