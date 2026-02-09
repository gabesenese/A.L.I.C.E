"""
Nightly Audit Scheduler - Automation engine
Runs audit → grade → improve cycle without manual intervention
"""

import logging
import schedule
import threading
from typing import Callable, Dict, List, Any
from datetime import datetime, time
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditScheduler:
    """Schedules and runs nightly audit cycles"""
    
    def __init__(self):
        self.scheduler = schedule.Scheduler()
        self.running = False
        self.scheduler_thread = None
        self.audit_function: Callable = None
        self.schedule_time = "02:00"  # 2 AM by default
        self.last_run = None
        self.run_count = 0
    
    def set_audit_function(self, func: Callable):
        """Set the function to call for each audit cycle"""
        self.audit_function = func
    
    def schedule_daily(self, hour: int = 2, minute: int = 0) -> 'AuditScheduler':
        """
        Schedule audit to run daily at specific time
        
        Args:
            hour: Hour (0-23)
            minute: Minute (0-59)
        
        Returns:
            Self for chaining
        """
        self.schedule_time = f"{hour:02d}:{minute:02d}"
        
        # Schedule the job
        def job_wrapper():
            return self._run_audit_cycle()
        
        self.scheduler.every().day.at(self.schedule_time).do(job_wrapper)
        
        logger.info(f"Scheduled daily audit at {self.schedule_time}")
        return self
    
    def schedule_every_n_hours(self, hours: int = 6) -> 'AuditScheduler':
        """
        Schedule audit to run every N hours
        
        Args:
            hours: Hours between runs
        
        Returns:
            Self for chaining
        """
        def job_wrapper():
            return self._run_audit_cycle()
        
        self.scheduler.every(hours).hours.do(job_wrapper)
        
        logger.info(f"Scheduled audit every {hours} hours")
        return self
    
    def _run_audit_cycle(self) -> Dict[str, Any]:
        """
        Run single audit cycle
        
        Returns:
            Results of audit
        """
        logger.info("="*60)
        logger.info("STARTING AUTOMATED AUDIT CYCLE")
        logger.info("="*60)
        
        self.last_run = datetime.now()
        self.run_count += 1
        
        if not self.audit_function:
            logger.error("No audit function set!")
            return {'status': 'error', 'reason': 'no_audit_function'}
        
        try:
            # Run audit
            result = self.audit_function()
            
            logger.info("="*60)
            logger.info(f"AUDIT COMPLETE (Run #{self.run_count})")
            logger.info("="*60)
            
            return {
                'status': 'success',
                'run_number': self.run_count,
                'timestamp': self.last_run.isoformat(),
                'result': result
            }
        
        except Exception as e:
            logger.error(f"Audit cycle failed: {e}", exc_info=True)
            
            return {
                'status': 'error',
                'run_number': self.run_count,
                'error': str(e)
            }
    
    def start(self) -> threading.Thread:
        """
        Start scheduler in background thread
        
        Returns:
            Scheduler thread
        """
        if self.running:
            logger.warning("Scheduler already running")
            return self.scheduler_thread
        
        self.running = True
        
        def run_scheduler():
            logger.info("Scheduler thread started")
            while self.running:
                self.scheduler.run_pending()
                # Check every 60 seconds
                import time as time_module
                time_module.sleep(60)
            logger.info("Scheduler thread stopped")
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Audit scheduler started in background")
        return self.scheduler_thread
    
    def stop(self):
        """Stop scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Audit scheduler stopped")
    
    def get_next_run_time(self) -> datetime:
        """Get time of next scheduled run"""
        for job in self.scheduler.jobs:
            if job.next_run:
                return job.next_run
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            'running': self.running,
            'scheduled_jobs': len(self.scheduler.jobs),
            'run_count': self.run_count,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.get_next_run_time().isoformat() if self.get_next_run_time() else None,
            'schedule_time': self.schedule_time
        }


class AutomatedAuditRunner:
    """Runs complete audit → score → inject → train cycle automatically"""
    
    def __init__(
        self,
        alice,
        teacher,
        auditor,
        scorer,
        injector,
        tracker
    ):
        """
        Initialize automated runner
        
        Args:
            alice: ALICE instance
            teacher: OllamaTeacher instance
            auditor: OllamaAuditor instance
            scorer: OllamaScorer instance
            injector: FeedbackInjector instance
            tracker: MetricTracker instance
        """
        self.alice = alice
        self.teacher = teacher
        self.auditor = auditor
        self.scorer = scorer
        self.injector = injector
        self.tracker = tracker
        self.scheduler = AuditScheduler()
        self.scheduler.set_audit_function(self.run_full_cycle)
    
    def run_full_cycle(self) -> Dict[str, Any]:
        """
        Run complete audit cycle:
        1. Generate test queries (teacher)
        2. Get Alice responses
        3. Grade responses (auditor)
        4. Convert to training signals (scorer)
        5. Inject into training data (injector)
        6. Track metrics (tracker)
        
        Returns:
            Cycle results
        """
        logger.info("Running full automated audit cycle...")
        
        cycle_results = {
            'domains': {},
            'total_tests': 0,
            'signals_generated': 0,
            'signals_injected': 0
        }
        
        # Record pre-training scores
        domains = list(self.teacher.teacher.TEACHING_VECTORS.keys())
        
        for domain in domains:
            logger.info(f"\nProcessing domain: {domain}")
            
            # Generate test queries
            vectors = [v for v in self.teacher.teacher.TEACHING_VECTORS[domain]]
            
            domain_scores = {'tests': 0, 'avg_score': 0}
            domain_audit_scores = []
            
            for vector in vectors:
                logger.info(f"  Skill: {vector.skill}")
                
                # Generate queries
                queries = self.teacher.generate_test_queries(domain, vector.skill, count=2)
                
                domain_scores['tests'] += len(queries)
                cycle_results['total_tests'] += len(queries)
                
                # Get Alice responses and audit them
                for query in queries:
                    try:
                        # Get response from Alice
                        response = self.alice.process_input(query)
                        
                        # Audit response
                        audit_score = self.auditor.audit_response(domain, query, response)
                        domain_audit_scores.append(audit_score)
                        
                        # Generate training signals
                        signals = self.scorer.score_audit(audit_score, domain, vector.skill)
                        
                        # Inject signals
                        count = self.injector.inject_signals(signals)
                        cycle_results['signals_injected'] += count
                        cycle_results['signals_generated'] += len(signals)
                    
                    except Exception as e:
                        logger.error(f"Error processing query: {e}")
                        continue
            
            # Record domain metrics
            if domain_audit_scores:
                avg_score = sum(s.overall_score for s in domain_audit_scores) / len(domain_audit_scores)
                domain_scores['avg_score'] = avg_score
                
                dimension_scores = self._aggregate_dimension_scores(domain_audit_scores)
                self.tracker.record_pre_training_score(domain, avg_score, dimension_scores)
            
            cycle_results['domains'][domain] = domain_scores
        
        # Save datasets
        for domain in domains:
            self.injector.save_domain_dataset(domain)
        
        logger.info(f"\nCycle summary:")
        logger.info(f"  Total tests: {cycle_results['total_tests']}")
        logger.info(f"  Signals generated: {cycle_results['signals_generated']}")
        logger.info(f"  Signals injected: {cycle_results['signals_injected']}")
        
        return cycle_results
    
    def _aggregate_dimension_scores(self, audit_scores):
        """Aggregate dimension scores from multiple audits"""
        from ai.ollama_auditor_spec import ScoringDimension
        
        dimension_scores = {}
        
        for dimension in ScoringDimension:
            scores = [
                a.scores.get(dimension, 0)
                for a in audit_scores
                if dimension in a.scores
            ]
            if scores:
                dimension_scores[dimension.value] = sum(scores) / len(scores)
        
        return dimension_scores
    
    def start_scheduler(self, hour: int = 2, minute: int = 0):
        """Start automated scheduler"""
        self.scheduler.schedule_daily(hour, minute)
        self.scheduler.start()
        return self.scheduler
    
    def stop_scheduler(self):
        """Stop scheduler"""
        self.scheduler.stop()


def create_scheduler(alice, teacher, auditor, scorer, injector, tracker) -> AutomatedAuditRunner:
    """Factory to create automated runner with scheduler"""
    return AutomatedAuditRunner(alice, teacher, auditor, scorer, injector, tracker)
