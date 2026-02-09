#!/usr/bin/env python3
"""
A.L.I.C.E Learning System
=========================

Integrated learning platform with 3 modes:
1. Single learning cycle: python scripts/simple_learning.py
2. Progress check: python scripts/simple_learning.py --progress
3. Continuous automation: python scripts/simple_learning.py --continuous [--interval SECONDS]

All modes:
- Generate test queries via Ollama
- Have Alice answer them
- Grade with Ollama auditor
- Extract signals and store
- Track performance metrics
"""

import sys
import os
import json
import logging
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import requests

# Fix Windows encoding issue
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format='%(message)s')

def call_ollama(prompt: str, model: str = "llama3.1:8b", timeout: int = 30) -> str:
    """Call Ollama directly via REST API"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=timeout
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"  [ERROR] Ollama error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"  [ERROR] Connection error: {e}")
        return ""

def generate_test_queries(domain: str, count: int = 2) -> list:
    """Generate test queries using Ollama"""
    
    domain_prompts = {
        "weather": "Generate {count} different weather-related questions someone might ask an assistant. Make them practical and varied. Just list the questions, one per line.",
        "code": "Generate {count} different programming questions someone might ask an AI. Cover different areas like bugs, optimization, explanation. Just list the questions, one per line.",
        "conversation": "Generate {count} different casual conversation questions for a personal assistant. Make them natural and varied. Just list the questions, one per line.",
    }
    
    template = domain_prompts.get(domain, "Generate {count} {domain}-related questions. Just list them, one per line.")
    prompt = template.format(count=count, domain=domain)
    
    print(f"  Generating {domain} queries...", end=" ", flush=True)
    response = call_ollama(prompt)
    
    if not response:
        return []
    
    # Parse responses (should be one per line)
    queries = [q.strip() for q in response.split('\n') if q.strip() and len(q.strip()) > 10]
    print(f"generated {len(queries)}")
    return queries[:count]

def ask_alice(query: str) -> str:
    """Ask Alice a question via simple prompt"""
    # For now, use Ollama directly as Alice
    print(f"    Alice answering: \"{query[:40]}...\"", end=" ", flush=True)
    
    prompt = f"""You are A.L.I.C.E, a helpful AI assistant. 
Answer this question naturally and helpfully:
{query}

Keep your answer concise (2-3 sentences max)."""
    
    response = call_ollama(prompt, timeout=60)
    print(f"done")
    return response

def grade_response(query: str, response: str, domain: str) -> dict:
    """Grade Alice's response using Ollama auditor"""
    print(f"    Grading response...", end=" ", flush=True)
    
    grading_prompt = f"""Rate this response on a scale of 1-5 for the following:
Domain: {domain}
Question: {query}
Response: {response}

Rate on: accuracy, clarity, helpfulness
Respond with just a number 1-5 and brief reason."""
    
    audit = call_ollama(grading_prompt, timeout=30)
    
    # Try to extract score
    try:
        score = float(audit[0]) if audit and audit[0].isdigit() else 3.0
    except:
        score = 3.0
    
    print(f"score: {score}/5")
    return {
        "score": score,
        "audit_text": audit,
        "grade": "positive" if score >= 4.0 else "improvement" if score >= 3.0 else "negative"
    }

def main():
    """Run the learning cycle"""
    
    print("\n" + "="*70)
    print("A.L.I.C.E. SIMPLE LEARNING CYCLE")
    print("="*70)
    
    # Check Ollama is running
    print("\n[1/4] Checking Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"  [OK] Ollama running")
            print(f"  [OK] Available models: {len(models)}")
        else:
            print("  [ERROR] Ollama not responding")
            return False
    except Exception as e:
        print(f"  [ERROR] Cannot connect to Ollama: {e}")
        return False
    
    # Generate test queries
    print("\n[2/4] Generating test queries...")
    domains = ["weather", "code", "conversation"]
    all_queries = []
    
    for domain in domains:
        queries = generate_test_queries(domain, count=2)
        all_queries.extend([(q, domain) for q in queries])
    
    if not all_queries:
        print("  [ERROR] No queries generated")
        return False
    
    print(f"  Total: {len(all_queries)} test queries")
    
    # Process queries and grade
    print("\n[3/4] Processing queries...")
    results = []
    
    for i, (query, domain) in enumerate(all_queries, 1):
        print(f"\n  [{i}/{len(all_queries)}] Domain: {domain}")
        
        # Get answer
        answer = ask_alice(query)
        if not answer:
            print(f"    [ERROR] No response")
            continue
        
        # Grade it
        grade = grade_response(query, answer, domain)
        
        results.append({
            "query": query,
            "domain": domain,
            "answer": answer,
            "score": grade["score"],
            "grade": grade["grade"],
            "timestamp": datetime.now().isoformat()
        })
    
    # Store results
    print("\n[4/4] Storing feedback...")
    feedback_file = Path("data/training/audit_feedback.jsonl")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(feedback_file, "a") as f:
        for result in results:
            f.write(json.dumps({
                "query": result["query"],
                "domain": result["domain"],
                "score": result["score"],
                "signal_type": result["grade"],
                "timestamp": result["timestamp"]
            }) + "\n")
    
    print(f"  [OK] Stored {len(results)} feedback signals")
    
    # Summary
    print("\n" + "="*70)
    print("LEARNING CYCLE SUMMARY")
    print("="*70)
    
    positive = len([r for r in results if r["grade"] == "positive"])
    improvement = len([r for r in results if r["grade"] == "improvement"])
    negative = len([r for r in results if r["grade"] == "negative"])
    
    print(f"\nResults:")
    print(f"  [+] Positive:     {positive}")
    print(f"  [*] Improvement:  {improvement}")
    print(f"  [-] Negative:     {negative}")
    
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        print(f"\n  Average Score: {avg_score:.2f}/5.0")
        
        if avg_score >= 3.5:
            print(f"\n  Status: [OK] Alice is responding well!")
        else:
            print(f"\n  Status: [WARN] Alice needs more training")
    
    print("\nNext steps:")
    print("  1. Check progress: python scripts/simple_learning.py --progress")
    print("  2. Run again: python scripts/simple_learning.py")
    print("  3. Continuous learning: python scripts/simple_learning.py --continuous")
    print()
    
    return True


def load_feedback():
    """Load all feedback signals"""
    feedback_file = Path("data/training/audit_feedback.jsonl")
    if not feedback_file.exists():
        return []
    
    signals = []
    with open(feedback_file) as f:
        for line in f:
            if line.strip():
                signals.append(json.loads(line))
    return signals


def show_progress():
    """Display learning progress dashboard"""
    signals = load_feedback()
    
    if not signals:
        print("No feedback yet. Run: python scripts/simple_learning.py")
        return True
    
    # Group by domain
    by_domain = defaultdict(list)
    for signal in signals:
        by_domain[signal["domain"]].append(signal["score"])
    
    print("\n" + "="*70)
    print("A.L.I.C.E. LEARNING PROGRESS")
    print("="*70)
    
    print(f"\nTotal signals collected: {len(signals)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-"*70)
    print("DOMAIN PERFORMANCE")
    print("-"*70)
    
    overall_scores = []
    for domain in sorted(by_domain.keys()):
        scores = by_domain[domain]
        avg = sum(scores) / len(scores)
        overall_scores.extend(scores)
        
        print(f"\n{domain.upper()}")
        print(f"  Samples:     {len(scores)}")
        print(f"  Average:     {avg:.2f}/5.0")
        print(f"  Best:        {max(scores):.1f}/5.0")
        print(f"  Worst:       {min(scores):.1f}/5.0")
        print(f"  Range:       {max(scores) - min(scores):.1f}")
    
    print("\n" + "-"*70)
    print("OVERALL STATS")
    print("-"*70)
    
    overall_avg = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print(f"\nAverage Score:  {overall_avg:.2f}/5.0")
    
    # Interpret
    if overall_avg >= 4.5:
        status = "[EXCELLENT] Alice is ready for production"
    elif overall_avg >= 4.0:
        status = "[GOOD] Alice is performing well"
    elif overall_avg >= 3.5:
        status = "[FAIR] Alice needs more training"
    elif overall_avg >= 3.0:
        status = "[WARN] Continue learning cycles"
    else:
        status = "[CRITICAL] Retrain Alice immediately"
    
    print(f"Status:         {status}")
    print("\n")
    return True


def run_continuous(interval: int = 3600, max_cycles: int = None):
    """Run learning cycles continuously"""
    print("\n" + "="*70)
    print("A.L.I.C.E. LEARNING AUTOMATION")
    print("="*70)
    
    print(f"\nInterval: {interval} seconds ({interval//60} minutes)")
    if max_cycles:
        print(f"Max cycles: {max_cycles}")
    else:
        print("Running continuously until interrupted")
    
    cycle_count = 0
    total_success = 0
    
    try:
        while True:
            cycle_count += 1
            
            print("\n" + "-"*70)
            print(f"LEARNING CYCLE {cycle_count}")
            print("-"*70 + "\n")
            
            # Run learning cycle
            if main_cycle():
                total_success += 1
            
            # Check progress
            show_progress()
            
            # Stop after N cycles if specified
            if max_cycles and cycle_count >= max_cycles:
                break
            
            # Wait before next cycle
            if not (max_cycles and cycle_count >= max_cycles):
                print(f"Waiting {interval} seconds before next cycle...")
                time.sleep(interval)
    
    except KeyboardInterrupt:
        pass
    
    finally:
        print("\n" + "="*70)
        print("AUTOMATION SUMMARY")
        print("="*70)
        print(f"Total cycles: {cycle_count}")
        print(f"Successful: {total_success}")
        print(f"Failed: {cycle_count - total_success}")
        if cycle_count > 0:
            success_rate = (total_success / cycle_count) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print()
        return True


def main_cycle():
    """Run a single learning cycle (extracted from main for reuse)"""
    return main(show_output=False)


def main(show_output=True):
    """Run the learning cycle"""
    
    if show_output:
        print("\n" + "="*70)
        print("A.L.I.C.E. SIMPLE LEARNING CYCLE")
        print("="*70)
    
    # Check Ollama is running
    if show_output:
        print("\n[1/4] Checking Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if show_output:
                print(f"  [OK] Ollama running")
                print(f"  [OK] Available models: {len(models)}")
        else:
            if show_output:
                print("  [ERROR] Ollama not responding")
            return False
    except Exception as e:
        if show_output:
            print(f"  [ERROR] Cannot connect to Ollama: {e}")
        return False
    
    # Generate test queries
    if show_output:
        print("\n[2/4] Generating test queries...")
    domains = ["weather", "code", "conversation"]
    all_queries = []
    
    for domain in domains:
        queries = generate_test_queries(domain, count=2)
        all_queries.extend([(q, domain) for q in queries])
    
    if not all_queries:
        if show_output:
            print("  [ERROR] No queries generated")
        return False
    
    if show_output:
        print(f"  Total: {len(all_queries)} test queries")
    
    # Process queries and grade
    if show_output:
        print("\n[3/4] Processing queries...")
    results = []
    
    for i, (query, domain) in enumerate(all_queries, 1):
        if show_output:
            print(f"\n  [{i}/{len(all_queries)}] Domain: {domain}")
        
        # Get answer
        answer = ask_alice(query)
        if not answer:
            if show_output:
                print(f"    [ERROR] No response")
            continue
        
        # Grade it
        grade = grade_response(query, answer, domain)
        
        results.append({
            "query": query,
            "domain": domain,
            "answer": answer,
            "score": grade["score"],
            "grade": grade["grade"],
            "timestamp": datetime.now().isoformat()
        })
    
    # Store results
    if show_output:
        print("\n[4/4] Storing feedback...")
    feedback_file = Path("data/training/audit_feedback.jsonl")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(feedback_file, "a") as f:
        for result in results:
            f.write(json.dumps({
                "query": result["query"],
                "domain": result["domain"],
                "score": result["score"],
                "signal_type": result["grade"],
                "timestamp": result["timestamp"]
            }) + "\n")
    
    if show_output:
        print(f"  [OK] Stored {len(results)} feedback signals")
    
    # Summary
    if show_output:
        print("\n" + "="*70)
        print("LEARNING CYCLE SUMMARY")
        print("="*70)
    
    positive = len([r for r in results if r["grade"] == "positive"])
    improvement = len([r for r in results if r["grade"] == "improvement"])
    negative = len([r for r in results if r["grade"] == "negative"])
    
    if show_output:
        print(f"\nResults:")
        print(f"  [+] Positive:     {positive}")
        print(f"  [*] Improvement:  {improvement}")
        print(f"  [-] Negative:     {negative}")
    
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        if show_output:
            print(f"\n  Average Score: {avg_score:.2f}/5.0")
            
            if avg_score >= 3.5:
                print(f"\n  Status: [OK] Alice is responding well!")
            else:
                print(f"\n  Status: [WARN] Alice needs more training")
        
        if show_output:
            print("\nNext steps:")
            print("  1. Check progress: python scripts/simple_learning.py --progress")
            print("  2. Run again: python scripts/simple_learning.py")
            print("  3. Continuous learning: python scripts/simple_learning.py --continuous")
    
    if show_output:
        print()
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A.L.I.C.E Learning System - Direct Ollama Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/simple_learning.py              # Run one learning cycle
  python scripts/simple_learning.py --progress   # Check learning progress
  python scripts/simple_learning.py --continuous # Run learning continuously (1 hour intervals)
  python scripts/simple_learning.py --continuous --interval 300  # Every 5 minutes
  python scripts/simple_learning.py --continuous --cycles 10     # Run 10 cycles then stop
        """
    )
    
    parser.add_argument("--progress", action="store_true",
                        help="Show learning progress and exit")
    parser.add_argument("--continuous", action="store_true",
                        help="Run learning cycles continuously")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between cycles (default: 3600 = 1 hour)")
    parser.add_argument("--cycles", type=int, default=None,
                        help="Stop after N cycles (only with --continuous)")
    
    args = parser.parse_args()
    
    try:
        if args.progress:
            success = show_progress()
        elif args.continuous:
            success = run_continuous(interval=args.interval, max_cycles=args.cycles)
        else:
            success = main()
        
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
