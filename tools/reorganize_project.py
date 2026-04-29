"""
Project Structure Reorganization
=================================
Advanced engineering folder structure - clean, modular, scalable.

This script is intentionally quarantined because the map can drift as the
repository evolves. Always validate live usage and path existence before
executing file moves.
"""

import argparse
import shutil
from pathlib import Path

# Current root
ROOT = Path(__file__).resolve().parents[1]

# New structure
REORGANIZATION = {
    # Core AI - The brain
    "ai/core/": [
        "ai/llm_engine.py",
        "ai/llm_gateway.py",
        "ai/llm_policy.py",
        "ai/nlp_processor.py",
        "ai/intent_classifier.py",
        "ai/conversational_engine.py",
        "ai/knowledge_engine.py",
        "ai/reasoning_engine.py",
    ],
    # Learning Systems - How Alice improves
    "ai/learning/": [
        "ai/learning_engine.py",
        "ai/phrasing_learner.py",
        "ai/pattern_learner.py",
        "ai/pattern_miner.py",
        "ai/semantic_pattern_miner.py",
        "ai/active_learning_manager.py",
        "ai/self_reflection.py",
    ],
    # Memory & Context - Alice's recall
    "ai/memory/": [
        "ai/memory_system.py",
        "ai/context_engine.py",
        "ai/adaptive_context_selector.py",
        "ai/multimodal_context.py",
        "ai/smart_context_cache.py",
        "ai/predictive_prefetcher.py",
        "ai/conversation_summarizer.py",
    ],
    # Plugins - External capabilities
    "ai/plugins/": [
        "ai/plugin_system.py",
        "ai/calendar_plugin.py",
        "ai/document_plugin.py",
        "ai/email_plugin.py",
        "ai/file_operations_plugin.py",
        "ai/maps_plugin.py",
        "ai/memory_plugin.py",
        "ai/music_plugin.py",
        "ai/notes_plugin.py",
    ],
    # Training & Quality - Make Alice better
    "ai/training/": [
        "ai/ollama_teacher.py",
        "ai/ollama_auditor.py",
        "ai/ollama_scorer.py",
        "ai/ollama_feedback_injector.py",
        "ai/teacher_loop.py",
        "ai/teacher_comparison.py",
        "ai/run_offline_training.py",
        "ai/scenario_generator.py",
        "ai/scenario_runner.py",
        "ai/synthetic_corpus_generator.py",
    ],
    # Optimization & Monitoring
    "ai/optimization/": [
        "ai/autonomous_adjuster.py",
        "ai/audit_config_optimizer.py",
        "ai/metric_tracker.py",
        "ai/response_optimizer.py",
        "ai/runtime_thresholds.py",
        "ai/system_monitor.py",
    ],
    # Goals & Planning
    "ai/planning/": [
        "ai/goal_tracker.py",
        "ai/goal_from_llm.py",
        "ai/task_planner.py",
        "ai/task_executor.py",
        "ai/plan_executor.py",
        "ai/proactive_assistant.py",
    ],
    # Infrastructure
    "ai/infrastructure/": [
        "ai/event_bus.py",
        "ai/observers.py",
        "ai/router.py",
        "ai/errors.py",
        "ai/error_recovery.py",
        "ai/service_degradation.py",
        "ai/system_state.py",
        "ai/policy.py",
    ],
    # Data models & utilities
    "ai/models/": [
        "ai/ml_models.py",
        "ai/llm_context.py",
        "ai/simple_formatters.py",
        "ai/entity_relationship_tracker.py",
    ],
    # Tests - all testing
    "tests/": [
        "test_alice_knowledge.py",
        "test_confidence_building.py",
        "tests/comprehensive_test.py",
        "tests/enhanced_tests.py",
        "tests/test_clarification_gate.py",
        "tests/test_learning.py",
        "tests/test_os_architecture.py",
        "tests/test_weather_debug.py",
        "tests/test_weather_followup.py",
        "ai/test_audit_cycle.py",
    ],
    # Tools - development & maintenance
    "tools/auditing/": [
        "tools/training_data_auditor.py",
    ],
    "tools/debugging/": [
        "tools/debug_import.py",
        "tools/debug_patterns.py",
        "tools/check_training_data.py",
    ],
    "tools/monitoring/": [
        "tools/monitoring/monitor_training.py",
        "tools/monitoring/monitor_audit_progress.py",
        "tools/monitoring/monitor_live.py",
    ],
    # Scripts - automation
    "scripts/automation/": [
        "scripts/automation/automated_training.py",
        "scripts/automation/nightly_training.py",
        "scripts/automation/nightly_training_autonomous.py",
        "scripts/automation/start_automation.py",
        "scripts/automation/nightly_audit_scheduler.py",
    ],
    "scripts/training/": [
        "scripts/training/run_learning_cycle.py",
        "scripts/training/run_scenarios_and_train.py",
        "scripts/training/simple_learning.py",
        "scripts/training/run_and_report_scenarios.py",
    ],
    "scripts/testing/": [
        "scripts/testing/test_audit_components.py",
        "scripts/testing/test_audit_pipeline.py",
        "scripts/testing/test_automation.py",
    ],
    "scripts/utilities/": [
        "scripts/utilities/count_scenarios.py",
        "scripts/utilities/deprecate_modules.py",
        "ai/promote_patterns.py",
    ],
    # Keep in place (already organized or special)
    "_keep_in_place": [
        "app/main.py",
        "app/dev.py",
        "app/alice.py",
        "speech/speech_engine.py",
        "speech/speech_model.py",
        "speech/audio_segmentation.py",
        "speech/phoneme_generation.py",
        "ui/rich_terminal.py",
        "features/personal_events.py",
        "features/welcome.py",
    ],
    # Deprecated - move to archive or delete
    "_deprecated": [
        "ai/lab_simulator.py",  # Old testing
        "ai/red_team_tester.py",  # Moved to tests
        "ai/ollama_auditor_spec.py",  # Spec files (keep as docs?)
        "ai/ollama_teaching_spec.py",
        "scripts/production_demo.py",  # Demo file
        "app/example_events.py",  # Example file
        "generate_scenarios.py",  # Duplicate functionality
        "reports/assessment_report.py",  # Old report
    ],
}


def print_reorganization_plan():
    """Print the reorganization plan"""

    print("=" * 80)
    print("PROJECT REORGANIZATION PLAN")
    print("=" * 80)
    print()

    total_moves = 0

    for new_location, files in REORGANIZATION.items():
        if new_location.startswith("_"):
            continue

        print(f"\n{new_location}")
        print("-" * 80)
        for file in files:
            print(f"  {file} -> {new_location}")
            total_moves += 1

    print()
    print("=" * 80)
    print(f"Total files to reorganize: {total_moves}")
    print("=" * 80)
    print()

    # Show deprecated files
    print("\nDEPRECATED FILES (will move to archive/):")
    print("-" * 80)
    for file in REORGANIZATION.get("_deprecated", []):
        print(f"  {file}")

    print()


def _collect_mapped_sources():
    """Collect all mapped source paths from the reorganization table."""
    sources = []
    for new_location, files in REORGANIZATION.items():
        if new_location.startswith("_"):
            continue
        for file_path in files:
            sources.append(ROOT / file_path)
    return sources


def _map_health(min_existing_ratio: float = 0.60):
    """Return (is_healthy, stats) for mapped source path existence."""
    sources = _collect_mapped_sources()
    existing = [p for p in sources if p.exists()]
    missing = [p for p in sources if not p.exists()]
    total = len(sources)
    ratio = (len(existing) / total) if total else 0.0
    return ratio >= min_existing_ratio, {
        "total": total,
        "existing": len(existing),
        "missing": len(missing),
        "existing_ratio": ratio,
    }


def execute_reorganization(dry_run=True, min_existing_ratio: float = 0.60):
    """Execute the reorganization"""

    healthy, stats = _map_health(min_existing_ratio=min_existing_ratio)

    print("\n[MAP HEALTH]")
    print(
        f"  mapped={stats['total']} existing={stats['existing']} "
        f"missing={stats['missing']} ratio={stats['existing_ratio']:.1%}"
    )

    if dry_run:
        print("\n[DRY RUN MODE - No files will be moved]\n")
        print_reorganization_plan()
        return True

    if not healthy:
        print(
            "\n[BLOCKED] Reorganization map appears stale for current repository state."
        )
        print(
            f"[BLOCKED] Existing ratio {stats['existing_ratio']:.1%} is below "
            f"required {min_existing_ratio:.1%}."
        )
        print("[BLOCKED] Refresh the map first, then rerun intentionally.")
        return False

    print("EXECUTING REORGANIZATION...")
    print()

    # Create new directories
    for new_location in REORGANIZATION.keys():
        if new_location.startswith("_"):
            continue

        new_dir = ROOT / new_location
        new_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {new_location}")

    print()

    # Move files
    moved = 0
    for new_location, files in REORGANIZATION.items():
        if new_location.startswith("_"):
            continue

        for file_path in files:
            src = ROOT / file_path
            dst = ROOT / new_location / Path(file_path).name

            # When map and destination already align, treat as no-op.
            try:
                if src.resolve() == dst.resolve():
                    print(f"Already in place: {file_path}")
                    continue
            except FileNotFoundError:
                pass

            if src.exists():
                shutil.move(str(src), str(dst))
                print(f"Moved: {file_path} -> {new_location}")
                moved += 1
            else:
                print(f"Not found: {file_path}")

    print()
    print(f"Reorganization complete! Moved {moved} files.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quarantined project reorganization tool"
    )
    parser.add_argument("--execute", action="store_true", help="Actually move files")
    parser.add_argument(
        "--acknowledge-quarantine",
        action="store_true",
        help="Required safety acknowledgement for this stale-map tool",
    )
    parser.add_argument(
        "--min-existing-ratio",
        type=float,
        default=0.60,
        help="Minimum mapped-source existence ratio required for execution",
    )
    args = parser.parse_args()

    if not args.acknowledge_quarantine:
        print("\n[BLOCKED] This script is quarantined because path maps can drift.")
        print("[BLOCKED] Rerun with --acknowledge-quarantine to inspect or execute.")
        print("[BLOCKED] Example dry run:")
        print("  python tools/reorganize_project.py --acknowledge-quarantine")
        print("[BLOCKED] Example execute:")
        print("  python tools/reorganize_project.py --acknowledge-quarantine --execute")
        raise SystemExit(2)

    ok = execute_reorganization(
        dry_run=not args.execute,
        min_existing_ratio=args.min_existing_ratio,
    )

    if not args.execute:
        print()
        print("To execute reorganization intentionally, run:")
        print("  python tools/reorganize_project.py --acknowledge-quarantine --execute")

    raise SystemExit(0 if ok else 1)
