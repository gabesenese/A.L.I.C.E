"""
Quick Audit Pipeline Test
Verifies full pipeline works before setting up automation
"""

import sys
import os
import json
import logging
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_test_pipeline(
    alice,
    llm,
    domains=None,
    skills_per_domain: int = 1,
    queries_per_skill: int = 2,
):
    """Run a compact end-to-end audit pipeline against selected domains."""
    from ai.ollama_teaching_spec import TEACHING_VECTORS
    from ai.training.ollama_teacher import create_teacher
    from ai.training.ollama_auditor import create_auditor
    from ai.training.ollama_scorer import create_scorer
    from ai.training.ollama_feedback_injector import create_injector
    from ai.optimization.metric_tracker import create_tracker

    teacher = create_teacher(llm)
    auditor = create_auditor(llm)
    scorer = create_scorer()
    injector = create_injector()
    tracker = create_tracker()

    if not domains:
        domains = list(TEACHING_VECTORS.keys())

    results = {
        "status": "complete",
        "domains_tested": 0,
        "total_queries": 0,
        "total_audits": 0,
        "total_signals": 0,
        "domain_results": {},
        "errors": [],
    }

    for domain in domains:
        vectors = TEACHING_VECTORS.get(domain, [])[:skills_per_domain]
        if not vectors:
            continue

        domain_results = {
            "skills_tested": 0,
            "queries": 0,
            "audits": 0,
            "signals": 0,
            "avg_score": 0.0,
        }
        domain_scores = []

        for vector in vectors:
            try:
                queries = teacher.generate_test_queries(
                    domain,
                    vector.skill,
                    count=queries_per_skill,
                )
                domain_results["skills_tested"] += 1
                domain_results["queries"] += len(queries)
                results["total_queries"] += len(queries)

                for query in queries:
                    response = alice.process_input(query)
                    audit_score = auditor.audit_response(domain, query, response)
                    signals = scorer.score_audit(audit_score, domain, vector.skill)
                    injector.inject_signals(signals)

                    domain_scores.append(audit_score.overall_score)
                    domain_results["audits"] += 1
                    domain_results["signals"] += len(signals)
                    results["total_audits"] += 1
                    results["total_signals"] += len(signals)
            except Exception as e:
                results["errors"].append(f"{domain}/{vector.skill}: {e}")

        if domain_scores:
            domain_results["avg_score"] = sum(domain_scores) / len(domain_scores)
            tracker.record_pre_training_score(domain, domain_results["avg_score"], {})

        results["domain_results"][domain] = domain_results
        results["domains_tested"] += 1

    results_file = PROJECT_ROOT / "data" / "training" / "test_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    """Run audit pipeline test"""

    print("\n" + "=" * 70)
    print("OLLAMA AUDIT PIPELINE TEST")
    print("=" * 70)

    # Step 1: Initialize Alice and LLM
    print("\n[1/3] Initializing ALICE and LLM...")
    try:
        import os

        # Disable semantic classifier download on network issues
        os.environ["ALICE_DISABLE_SEMANTIC_CLASSIFIER"] = "1"

        from app.alice import ALICE
        from ai.core.llm_engine import LocalLLMEngine, LLMConfig

        alice = ALICE(debug=False)
        llm = LocalLLMEngine(config=LLMConfig(model="llama3.1:8b"))
        print("Alice initialized")
        print("LLM engine ready")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return False

    # Step 2: Run test pipeline
    print("\n[2/3] Running end-to-end audit cycle...")
    print("     (Testing weather and email domains)")
    try:
        results = run_test_pipeline(
            alice,
            llm,
            domains=["weather", "email"],
            skills_per_domain=1,
            queries_per_skill=2,
        )

        if results["status"] != "complete":
            print(f"Test failed: {results.get('status')}")
            return False

        print("Pipeline test complete")
        print(f"  - Domains tested: {results['domains_tested']}")
        print(f"  - Total queries: {results['total_queries']}")
        print(f"  - Total audits: {results['total_audits']}")
        print(f"  - Signals generated: {results['total_signals']}")

    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Step 3: Show results
    print("\n[3/3] Analyzing results...")
    try:
        results_file = Path("data/training/test_results.json")
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)

            print("\n" + "=" * 70)
            print("TEST RESULTS SUMMARY")
            print("=" * 70)

            for domain, domain_data in data.get("domain_results", {}).items():
                print(f"\n{domain.upper()}")
                print(f"  Skills tested: {domain_data['skills_tested']}")
                print(f"  Queries: {domain_data['queries']}")
                print(f"  Audits: {domain_data['audits']}")
                print(f"  Signals: {domain_data['signals']}")
                print(f"  Avg score: {domain_data['avg_score']:.2f}/5.0")

            print("\n" + "=" * 70)
            print("OUTPUT ARTIFACTS")
            print("=" * 70)
            print("Test results: data/training/test_results.json")
            print("Feedback log: data/training/audit_feedback.jsonl")
            print("Domain datasets: data/training/{domain}_feedback.json")

    except Exception as e:
        print(f"Failed to show results: {e}")
        return False

    print("\n" + "=" * 70)
    print("PIPELINE TEST SUCCESSFUL")
    print("=" * 70)

    print("\nNEXT STEPS:")
    print("  1. Review results in: data/training/test_results.json")
    print("  2. Check domain feedback in: data/training/*_feedback.json")
    print("  3. If satisfied, run: python scripts/automation/start_automation.py")
    print("\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
