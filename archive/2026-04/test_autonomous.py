"""
Test script for autonomous mode
"""
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_autonomous_imports():
    """Test that all autonomous mode modules can be imported"""
    print("Testing autonomous mode imports...")

    try:
        from ai.planning.autonomous_agent import AutonomousAgent, get_autonomous_agent
        print("- autonomous_agent imported successfully")
    except Exception as e:
        print(f"- ERROR importing autonomous_agent: {e}")
        return False

    try:
        from ai.planning.autonomous_execution_loop import AutonomousExecutionLoop, get_execution_loop
        print("- autonomous_execution_loop imported successfully")
    except Exception as e:
        print(f"- ERROR importing autonomous_execution_loop: {e}")
        return False

    try:
        from ai.planning.error_recovery import ErrorRecoverySystem, get_error_recovery, RecoveryStrategy
        print("- error_recovery imported successfully")
    except Exception as e:
        print(f"- ERROR importing error_recovery: {e}")
        return False

    return True

def test_autonomous_initialization():
    """Test that autonomous components can be initialized"""
    print("\nTesting autonomous mode initialization...")

    try:
        from ai.planning.autonomous_agent import get_autonomous_agent
        agent = get_autonomous_agent()
        print(f"- Autonomous agent created: {agent}")
    except Exception as e:
        print(f"- ERROR creating autonomous agent: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from ai.planning.error_recovery import get_error_recovery
        recovery = get_error_recovery()
        print(f"- Error recovery system created: {recovery}")
    except Exception as e:
        print(f"- ERROR creating error recovery: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        from ai.planning.autonomous_execution_loop import get_execution_loop
        loop = get_execution_loop()
        print(f"- Execution loop created: {loop}")
    except Exception as e:
        print(f"- ERROR creating execution loop: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_goal_decomposition():
    """Test goal decomposition"""
    print("\nTesting goal decomposition...")

    try:
        from ai.planning.autonomous_agent import get_autonomous_agent
        from ai.planning.persistent_goals import get_goal_system

        agent = get_autonomous_agent()
        goal_system = get_goal_system(storage_path="data/test_goals")

        # Test template decomposition (doesn't require LLM)
        steps = agent._template_decomposition("refactor authentication system")

        if steps:
            print(f"- Goal decomposed into {len(steps)} steps:")
            for i, step in enumerate(steps[:3], 1):
                print(f"  {i}. {step['description']}")
            return True
        else:
            print("- ERROR: No steps returned from decomposition")
            return False

    except Exception as e:
        print(f"- ERROR testing goal decomposition: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_classification():
    """Test error classification"""
    print("\nTesting error classification...")

    try:
        from ai.planning.error_recovery import get_error_recovery

        recovery = get_error_recovery()

        # Test various error types
        test_errors = [
            ("Connection timeout", "network_error"),
            ("No such file or directory: test.py", "file_not_found"),
            ("SyntaxError: invalid syntax", "syntax_error"),
            ("ImportError: No module named 'foo'", "import_error"),
            ("Permission denied", "permission_denied"),
        ]

        all_correct = True
        for error_msg, expected_type in test_errors:
            error_type = recovery._classify_error(Exception(error_msg))
            if error_type == expected_type:
                print(f"- '{error_msg}' -> {error_type}")
            else:
                print(f"- ERROR: '{error_msg}' -> {error_type} (expected {expected_type})")
                all_correct = False

        return all_correct

    except Exception as e:
        print(f"- ERROR testing error classification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("AUTONOMOUS MODE END-TO-END TEST")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Import Test", test_autonomous_imports()))
    results.append(("Initialization Test", test_autonomous_initialization()))
    results.append(("Goal Decomposition Test", test_goal_decomposition()))
    results.append(("Error Classification Test", test_error_classification()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed successfully")
        sys.exit(0)
    else:
        print(f"\n{total - passed} test(s) failed")
        sys.exit(1)
