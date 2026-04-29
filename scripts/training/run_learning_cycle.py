#!/usr/bin/env python3
"""
Run Learning Cycle Now
=====================

Immediately runs the teaching → auditing → learning cycle
without waiting for 2 AM. Alice learns in real-time.

This script:
1. Teacher generates test queries
2. Alice processes each query
3. Auditor grades her responses
4. Scorer extracts training signals
5. System improves

Run with: python scripts/training/run_learning_cycle.py
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Resolve to project root (scripts/training/ -> scripts/ -> project root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_learning_cycle():
    """Run one complete learning cycle right now"""

    print("\n" + "=" * 70)
    print("A.L.I.C.E. LEARNING CYCLE - RUNNING NOW")
    print("=" * 70)

    try:
        # Step 1: Initialize components
        print("\n[1/5] Initializing components...")
        from app.alice import ALICE
        from ai.core.llm_engine import LocalLLMEngine, LLMConfig
        from ai.ollama_teaching_spec import TEACHING_VECTORS

        alice = ALICE(debug=False)
        llm = LocalLLMEngine(config=LLMConfig(model="llama3.1:8b"))

        print("  Alice initialized")
        print("  LLM initialized")

        # Step 2: Generate test queries
        print("\n[2/5] Generating test queries...")
        from ai.training.ollama_teacher import OllamaTeacher

        teacher = OllamaTeacher(llm=llm)
        test_queries = []

        # TEACHING_VECTORS is a dict of domain -> list of TeachingVectors
        domains_to_test = list(TEACHING_VECTORS.keys())[:2]  # Start with 2 domains
        print(f"  Testing domains: {', '.join(domains_to_test)}")

        for domain in domains_to_test:
            try:
                # Get teaching vectors for this domain
                vectors = TEACHING_VECTORS.get(domain, [])
                if not vectors:
                    print(f"   {domain}: no teaching vectors found")
                    continue

                # Use first 1-2 skills from this domain
                for vector in vectors[:1]:
                    skill = vector.skill if hasattr(vector, "skill") else "general"
                    try:
                        queries = teacher.generate_test_queries(domain, skill, count=2)
                        test_queries.extend([(q, domain) for q in queries])
                        print(f"  {domain}/{skill}: generated {len(queries)} queries")
                    except Exception as e:
                        print(f"   {domain}/{skill}: {str(e)[:60]}")
            except Exception as e:
                print(f"   {domain}: {str(e)[:60]}")

        if not test_queries:
            print("  No test queries generated. Check LLM connection.")
            return False

        print(f"\n  Total: {len(test_queries)} test queries")

        # Step 3: Process queries and grade responses
        print("\n[3/5] Processing queries and grading responses...")
        from ai.training.ollama_auditor import OllamaAuditor

        auditor = OllamaAuditor(llm=llm)
        results = []

        for i, (query, domain) in enumerate(test_queries, 1):
            try:
                print(f"\n  [{i}/{len(test_queries)}] Query: {query[:50]}...")

                # Alice processes the query
                response = alice.process_input(query)
                print(f"       Response: {response[:60]}...")

                # Auditor grades it
                audit_result = auditor.audit_response(
                    domain=domain,
                    query=query,
                    response=response,
                )

                print(f"       Audit Score: {audit_result.overall_score:.1f}/5.0")

                results.append(
                    {
                        "query": query,
                        "domain": domain,
                        "response": response,
                        "audit_score": audit_result.overall_score,
                        "audit_obj": audit_result,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            except Exception as e:
                print(f"       Error: {e}")
                continue

        # Step 4: Extract training signals
        print("\n[4/5] Extracting training signals...")
        from ai.training.ollama_scorer import OllamaScorer

        scorer = OllamaScorer()
        signals = []

        for result in results:
            audit_obj = result.get("audit_obj")
            if not audit_obj:
                continue

            generated = scorer.score_audit(
                audit_obj,
                result["domain"],
                skill="general",
            )
            signals.extend(generated)

            for signal in generated:
                print(
                    f"  {result['domain']}: {signal.signal_type} "
                    f"(score: {result['audit_score']:.1f})"
                )

        # Step 5: Store feedback for training
        print("\n[5/5] Storing feedback for learning...")
        from ai.training.ollama_feedback_injector import create_injector

        injector = create_injector()
        injected_count = injector.inject_signals(signals)

        print(f"  Stored {injected_count} feedback signals")
        print("  File: data/training/audit_feedback.jsonl")

        # Summary
        print("\n" + "=" * 70)
        print("LEARNING CYCLE COMPLETE")
        print("=" * 70)

        positive = len([s for s in signals if s.signal_type == "positive"])
        improvement = len([s for s in signals if s.signal_type == "improvement"])
        negative = len([s for s in signals if s.signal_type == "negative"])

        print("\nResults:")
        print(f"  Positive signals:     {positive}")
        print(f"   Improvement needed:   {improvement}")
        print(f"  Negative signals:     {negative}")

        avg_score = (
            sum(r["audit_score"] for r in results) / len(results) if results else 0
        )
        print(f"\n  Average Score: {avg_score:.2f}/5.0")

        if positive + improvement > negative:
            print("\n  Status: IMPROVING - Alice is learning!")
        else:
            print("\n  Status: NEEDS WORK - More training required")

        print("\nNext steps:")
        print("  1. Review feedback: cat data/training/audit_feedback.jsonl")
        print("  2. Run again: python scripts/training/run_learning_cycle.py")
        print("  3. Monitor: python scripts/monitor_live.py")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_learning_cycle()
    sys.exit(0 if success else 1)
