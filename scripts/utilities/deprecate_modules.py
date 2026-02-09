#!/usr/bin/env python3
"""
Module Deprecation & Cleanup Script
====================================

Moves unnecessary modules to 'deprecated/' folder while keeping them
for reference and historical reasons. This is the first step in
reducing A.L.I.C.E. from 69 → 15 core modules.

Usage:
  python scripts/deprecate_modules.py --dry-run  (see what would be moved)
  python scripts/deprecate_modules.py --confirm  (actually move them)
"""

import os
import shutil
import sys
from pathlib import Path

# Modules to keep in ai/ folder (production core only)
CORE_MODULES = {
    # NLP & LLM
    'nlp_processor.py',
    'llm_engine.py',
    'intent_classifier.py',
    'conversational_engine.py',
    
    # Memory & Context
    'memory_system.py',
    'context_engine.py',
    
    # Audit & Learning Loop
    'ollama_teacher.py',
    'ollama_auditor.py',
    'ollama_scorer.py',
    'ollama_feedback_injector.py',
    'ollama_teaching_spec.py',
    'ollama_auditor_spec.py',
    'metric_tracker.py',
    'nightly_audit_scheduler.py',
    
    # Learning
    'active_learning_manager.py',
    
    # Response Enhancement
    'response_optimizer.py',
    
    # Plugin System
    'plugin_system.py',
    
    # Error Handling (keep minimal)
    'errors.py',
}

MODULES_TO_DEPRECATE = {
    # Learning/Pattern modules (subsumed by active_learning_manager)
    'adaptive_context_selector.py',
    'autonomous_adjuster.py',
    'learning_engine.py',
    'pattern_learner.py',
    'pattern_miner.py',
    'promote_patterns.py',
    'self_reflection.py',
    'semantic_pattern_miner.py',
    'teacher_comparison.py',
    'teacher_loop.py',
    
    # Goal/Planning (not integrated)
    'goal_from_llm.py',
    'goal_tracker.py',
    'task_executor.py',
    'task_planner.py',
    'plan_executor.py',
    'reasoning_engine.py',
    
    # Policy/Routing (not integrated)
    'llm_gateway.py',
    'llm_policy.py',
    'policy.py',
    'router.py',
    
    # Context/LLM helpers (subsumed)
    'llm_context.py',
    'multimodal_context.py',
    'adaptive_context_selector.py',
    
    # Monitoring (move to standalone scripts)
    'system_monitor.py',
    'system_state.py',
    'service_degradation.py',
    'audit_config_optimizer.py',
    
    # Event/Observer pattern (over-engineered)
    'observers.py',
    'event_bus.py',
    
    # ML Models (outdated/not used)
    'ml_models.py',
    
    # Testing/Simulation only
    'red_team_tester.py',
    'scenario_generator.py',
    'scenario_runner.py',
    'lab_simulator.py',
    'test_audit_cycle.py',
    
    # Training/Generation (move to scripts)
    'run_offline_training.py',
    'synthetic_corpus_generator.py',
    
    # Conversation utilities (move to utils)
    'conversation_summarizer.py',
    'entity_relationship_tracker.py',
    
    # Caching (redundant to memory_system)
    'smart_context_cache.py',
    
    # Recovery/Error handling (keep simple)
    'error_recovery.py',
    
    # Optimization (v2.0 feature)
    'predictive_prefetcher.py',
    'proactive_assistant.py',
    'runtime_thresholds.py',
    
    # Plugin modules (v1.0 features - not critical)
    'calendar_plugin.py',
    'document_plugin.py',
    'email_plugin.py',
    'maps_plugin.py',
    'music_plugin.py',
    'notes_plugin.py',
    
    # Utils
    'simple_formatters.py',
}

def run_deprecation(dry_run=True):
    """Move modules to deprecated folder"""
    
    ai_dir = Path('ai')
    deprecated_dir = ai_dir / 'deprecated'
    
    # Create deprecated folder if it doesn't exist
    if not dry_run:
        deprecated_dir.mkdir(exist_ok=True)
        print(f"\n[OK] Created {deprecated_dir}/")
    
    # Find modules to move
    modules_to_move = []
    
    for py_file in ai_dir.glob('*.py'):
        filename = py_file.name
        
        # Skip __pycache__ and __init__
        if filename.startswith('__'):
            continue
        
        # Check if it should be deprecated
        if filename in MODULES_TO_DEPRECATE:
            modules_to_move.append((py_file, deprecated_dir / filename))
    
    if not modules_to_move:
        print("[WARN] No modules to deprecate found")
        return
    
    print(f"\n[INFO] Found {len(modules_to_move)} modules to deprecate:\n")
    
    for src, dst in sorted(modules_to_move):
        size = src.stat().st_size // 1024  # KB
        print(f"  {src.name:<40} ({size:>4} KB)")
    
    total_size = sum((src.stat().st_size for src, _ in modules_to_move)) // 1024
    print(f"\n  Total: {total_size} KB")
    
    if dry_run:
        print("\n[DRY-RUN] Add --confirm to actually move modules\n")
        return
    
    # Actually move them
    print("\n[MOVING] Deprecating modules...")
    failed = []
    
    for src, dst in modules_to_move:
        try:
            shutil.move(str(src), str(dst))
            print(f"  [OK] Moved {src.name}")
        except Exception as e:
            print(f"  [FAIL] {src.name}: {e}")
            failed.append(src.name)
    
    if failed:
        print(f"\n[ERROR] Failed to move {len(failed)} modules:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    
    # Create __init__.py in deprecated folder
    init_file = deprecated_dir / '__init__.py'
    if not init_file.exists():
        init_file.write_text("""\"\"\"
Deprecated A.L.I.C.E. Modules
=============================

These modules are no longer used in the production core.
Kept for reference and historical reasons only.

If you need functionality from these modules:
1. Check if it's been moved to a core module
2. If not, open an issue to re-integrate it
3. Or create a custom plugin using plugin_system.py

Active Period: February 2026 - Present
Deprecation Reason: Module consolidation for production v2.0
\"\"\"
""")
        print(f"  [OK] Created {init_file}")
    
    print("\n[DONE] Module deprecation complete!")
    print(f"\n[STATUS] Now using {len(CORE_MODULES)} core modules")
    print("         Consider also creating deprecated_imports.py")
    print("         to redirect old imports gracefully")

def verify_core_modules():
    """Verify all core modules exist"""
    
    print("\n[CHECK] Verifying core modules exist...")
    ai_dir = Path('ai')
    missing = []
    
    for module in sorted(CORE_MODULES):
        module_file = ai_dir / module
        if module_file.exists():
            size = module_file.stat().st_size // 1024
            print(f"  [OK] {module:<40} ({size:>4} KB)")
        else:
            print(f"  [MISS] {module}")
            missing.append(module)
    
    if missing:
        print(f"\n[ERROR] Missing {len(missing)} core modules!")
        for m in missing:
            print(f"  - {m}")
        return False
    
    print(f"\n[OK] All {len(CORE_MODULES)} core modules present")
    return True

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("A.L.I.C.E. MODULE DEPRECATION")
    print("=" * 70)
    
    dry_run = '--confirm' not in sys.argv
    
    # Verify core modules first
    if not verify_core_modules():
        sys.exit(1)
    
    # Run deprecation
    run_deprecation(dry_run=dry_run)
    
    if not dry_run:
        print("\n[NEXT] Update imports to remove deprecated references")
        print("[NEXT] Run tests to verify nothing broke:")
        print("       python -m pytest tests/ -v")
        print()
