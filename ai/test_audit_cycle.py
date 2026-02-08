"""
Test Audit Cycle - End-to-end verification
Runs full pipeline before automating
"""

import logging
from typing import Dict, Any
from ai.ollama_teaching_spec import TEACHING_VECTORS
from ai.ollama_teacher import create_teacher
from ai.ollama_auditor import create_auditor
from ai.ollama_scorer import create_scorer
from ai.ollama_feedback_injector import create_injector
from ai.metric_tracker import create_tracker

logger = logging.getLogger(__name__)


def test_full_audit_pipeline(
    alice,
    llm_engine,
    domains: list = None,
    skills_per_domain: int = 1,
    queries_per_skill: int = 2
) -> Dict[str, Any]:
    """
    Run end-to-end audit pipeline to verify it works
    
    Args:
        alice: ALICE instance
        llm_engine: LLM engine for teacher/auditor
        domains: Which domains to test (default: all)
        skills_per_domain: Skills per domain to test
        queries_per_skill: Test queries per skill
    
    Returns:
        Test results dict
    """
    logger.info("="*60)
    logger.info("TESTING END-TO-END AUDIT PIPELINE")
    logger.info("="*60)
    
    # Initialize components
    teacher = create_teacher(llm_engine)
    auditor = create_auditor(llm_engine)
    scorer = create_scorer()
    injector = create_injector()
    tracker = create_tracker()
    
    # Determine domains to test
    if not domains:
        domains = list(TEACHING_VECTORS.keys())
    
    test_results = {
        'status': 'running',
        'domains_tested': 0,
        'total_queries': 0,
        'total_audits': 0,
        'total_signals': 0,
        'domain_results': {},
        'errors': []
    }
    
    for domain in domains[:3]:  # Start with first 3 domains
        logger.info(f"\nTesting domain: {domain}")
        print(f"\n{'='*40}")
        print(f"Domain: {domain}")
        print(f"{'='*40}")
        
        domain_results = {
            'skills_tested': 0,
            'queries': 0,
            'audits': 0,
            'signals': 0,
            'avg_score': 0.0,
            'dimension_scores': {}
        }
        
        # Get vectors for domain
        vectors = TEACHING_VECTORS.get(domain, [])[:skills_per_domain]
        
        all_audit_scores = []
        
        for vector in vectors:
            skill = vector.skill
            logger.info(f"  Skill: {skill}")
            
            try:
                # Step 1: Generate test queries
                logger.info(f"    [1/5] Generating queries...")
                queries = teacher.generate_test_queries(domain, skill, count=queries_per_skill)
                domain_results['queries'] += len(queries)
                test_results['total_queries'] += len(queries)
                print(f"    Generated {len(queries)} queries")
                
                for q in queries[:1]:  # Show first query
                    print(f"      Example: {q[:60]}...")
                
                # Step 2-4: Get response, audit, score
                for query in queries:
                    try:
                        logger.info(f"    [2/5] Getting Alice response...")
                        response = alice.process_input(query)
                        
                        logger.info(f"    [3/5] Auditing response...")
                        audit_score = auditor.audit_response(domain, query, response)
                        all_audit_scores.append(audit_score)
                        domain_results['audits'] += 1
                        test_results['total_audits'] += 1
                        
                        logger.info(f"    [4/5] Scoring audit...")
                        signals = scorer.score_audit(audit_score, domain, skill)
                        domain_results['signals'] += len(signals)
                        test_results['total_signals'] += len(signals)
                        
                        # Log result
                        print(f"\n    Query: {query[:50]}...")
                        print(f"    Alice: {response[:100]}...")
                        print(f"    Score: {audit_score.overall_score:.1f}/5.0")
                        print(f"    Signals: {len(signals)}")
                    
                    except Exception as e:
                        logger.error(f"Error processing query: {e}")
                        test_results['errors'].append(str(e))
                
                domain_results['skills_tested'] += 1
            
            except Exception as e:
                logger.error(f"Error with skill {skill}: {e}")
                test_results['errors'].append(str(e))
        
        # Step 5: Inject signals
        logger.info(f"    [5/5] Injecting signals into training...")
        injected = injector.inject_signals(
            [s for signals in [scorer.score_audit(a, domain, 'test') for a in all_audit_scores] for s in signals]
        )
        print(f"    Injected {injected} signals")
        
        # Track metrics
        if all_audit_scores:
            avg_score = sum(a.overall_score for a in all_audit_scores) / len(all_audit_scores)
            domain_results['avg_score'] = avg_score
            tracker.record_pre_training_score(domain, avg_score, {})
        
        test_results['domains_tested'] += 1
        test_results['domain_results'][domain] = domain_results
    
    # Summary
    test_results['status'] = 'complete'
    
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Domains tested: {test_results['domains_tested']}")
    logger.info(f"Total queries: {test_results['total_queries']}")
    logger.info(f"Total audits: {test_results['total_audits']}")
    logger.info(f"Total signals: {test_results['total_signals']}")
    
    if test_results['errors']:
        logger.warning(f"Errors encountered: {len(test_results['errors'])}")
        for error in test_results['errors'][:3]:
            logger.warning(f"  - {error}")
    
    logger.info("="*60)
    logger.info("Pipeline test complete")
    logger.info("="*60 + "\n")
    
    # Save test results
    import json
    from pathlib import Path
    
    results_path = Path("data/training/test_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    logger.info(f"Test results saved to {results_path}")
    
    return test_results


if __name__ == "__main__":
    # For standalone testing
    print("Import this module and call test_full_audit_pipeline(alice, llm_engine)")
