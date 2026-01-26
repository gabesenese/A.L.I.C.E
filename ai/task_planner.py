"""
Task Planner for A.L.I.C.E
Strictly separates understanding → planning → execution
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Status of a plan"""
    DRAFT = "draft"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(Enum):
    """Status of an individual step"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan"""
    step_id: int
    action: str  # e.g., "read_file", "call_plugin", "query_llm"
    params: Dict[str, Any]  # Action parameters
    depends_on: List[int] = field(default_factory=list)  # Step IDs this depends on
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def can_execute(self, completed_steps: set) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep_id in completed_steps for dep_id in self.depends_on)


@dataclass
class ExecutionPlan:
    """Complete plan for executing a task"""
    plan_id: str
    goal: str  # User's original request
    intent: str  # Detected intent
    steps: List[PlanStep]
    status: PlanStatus = PlanStatus.DRAFT
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_next_step(self, completed_steps: set) -> Optional[PlanStep]:
        """Get the next executable step"""
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.can_execute(completed_steps):
                return step
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if all steps are completed"""
        return all(
            step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
            for step in self.steps
        )
    
    @property
    def has_failed(self) -> bool:
        """Check if any critical step failed"""
        return any(step.status == StepStatus.FAILED for step in self.steps)


class TaskPlanner:
    """
    Creates execution plans from understood intents
    Strictly separates planning from execution
    """
    
    def __init__(self):
        self._plan_counter = 0
    
    def create_plan(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> ExecutionPlan:
        """
        Create an execution plan from understood intent
        
        Args:
            intent: Detected intent (e.g., "summarize_notes")
            entities: Extracted entities (dates, topics, etc.)
            context: Current context
        
        Returns:
            ExecutionPlan ready for execution
        """
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter}"
        
        goal = self._reconstruct_goal(intent, entities)
        steps = self._generate_steps(intent, entities, context)
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            goal=goal,
            intent=intent,
            steps=steps,
            status=PlanStatus.READY,
            metadata={
                'entities': entities,
                'context': context
            }
        )
        
        logger.info(f"Created plan {plan_id} with {len(steps)} steps for intent '{intent}'")
        
        return plan
    
    def _reconstruct_goal(self, intent: str, entities: Dict[str, Any]) -> str:
        """Reconstruct user's goal from intent and entities"""
        if intent == "summarize_notes":
            topic = entities.get('topic', 'all')
            return f"Summarize {topic} notes"
        
        elif intent == "check_calendar":
            timeframe = entities.get('timeframe', 'today')
            return f"Check calendar for {timeframe}"
        
        elif intent == "send_email":
            recipient = entities.get('recipient', 'someone')
            return f"Send email to {recipient}"
        
        elif intent == "play_music":
            query = entities.get('query', 'music')
            return f"Play {query}"
        
        elif intent == "create_note":
            title = entities.get('title', 'note')
            return f"Create note: {title}"
        
        else:
            return f"Execute: {intent}"
    
    def _generate_steps(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> List[PlanStep]:
        """
        Generate execution steps for an intent
        This is the core planning logic
        """
        steps = []
        
        # SUMMARIZE NOTES
        if intent == "summarize_notes":
            topic = entities.get('topic', 'all')
            timeframe = entities.get('timeframe')
            
            # Step 1: Load notes
            steps.append(PlanStep(
                step_id=1,
                action="plugin.notes.list",
                params={'topic': topic, 'timeframe': timeframe}
            ))
            
            # Step 2: Read note contents
            steps.append(PlanStep(
                step_id=2,
                action="plugin.notes.read_multiple",
                params={'note_ids': '{{step_1_result}}'},
                depends_on=[1]
            ))
            
            # Step 3: Generate summary with LLM
            steps.append(PlanStep(
                step_id=3,
                action="llm.summarize",
                params={
                    'content': '{{step_2_result}}',
                    'focus': topic
                },
                depends_on=[2]
            ))
        
        # CHECK CALENDAR
        elif intent == "check_calendar":
            timeframe = entities.get('timeframe', 'today')
            
            # Step 1: Query calendar
            steps.append(PlanStep(
                step_id=1,
                action="plugin.calendar.get_events",
                params={'timeframe': timeframe}
            ))
            
            # Step 2: Format results
            steps.append(PlanStep(
                step_id=2,
                action="format.calendar_events",
                params={'events': '{{step_1_result}}'},
                depends_on=[1]
            ))
        
        # SEND EMAIL
        elif intent == "send_email":
            recipient = entities.get('recipient')
            subject = entities.get('subject', '')
            body = entities.get('body', '')
            
            # Step 1: Validate recipient
            steps.append(PlanStep(
                step_id=1,
                action="plugin.email.validate_recipient",
                params={'recipient': recipient}
            ))
            
            # Step 2: Draft email (if body is incomplete)
            if not body:
                steps.append(PlanStep(
                    step_id=2,
                    action="llm.draft_email",
                    params={'subject': subject, 'context': context},
                    depends_on=[1]
                ))
                body_ref = '{{step_2_result}}'
                send_depends = [1, 2]
            else:
                body_ref = body
                send_depends = [1]
            
            # Step 3: Send email
            steps.append(PlanStep(
                step_id=3,
                action="plugin.email.send",
                params={
                    'to': recipient,
                    'subject': subject,
                    'body': body_ref
                },
                depends_on=send_depends
            ))
        
        # PLAY MUSIC
        elif intent == "play_music":
            query = entities.get('query', '')
            
            # Step 1: Search for music
            steps.append(PlanStep(
                step_id=1,
                action="plugin.music.search",
                params={'query': query}
            ))
            
            # Step 2: Play top result
            steps.append(PlanStep(
                step_id=2,
                action="plugin.music.play",
                params={'track_id': '{{step_1_result.top_match}}'},
                depends_on=[1]
            ))
        
        # CREATE NOTE
        elif intent == "create_note":
            title = entities.get('title', 'New Note')
            content = entities.get('content', '')
            
            # Step 1: Create note
            steps.append(PlanStep(
                step_id=1,
                action="plugin.notes.create",
                params={'title': title, 'content': content}
            ))
        
        # GENERIC QUESTION
        elif intent == "question":
            query = entities.get('query', '')
            
            # Step 1: Check memory for relevant context
            steps.append(PlanStep(
                step_id=1,
                action="memory.search",
                params={'query': query, 'top_k': 3}
            ))
            
            # Step 2: Query LLM with context
            steps.append(PlanStep(
                step_id=2,
                action="llm.answer",
                params={
                    'query': query,
                    'context': '{{step_1_result}}'
                },
                depends_on=[1]
            ))
        
        # DEFAULT: Single LLM call
        else:
            steps.append(PlanStep(
                step_id=1,
                action="llm.process",
                params={'intent': intent, 'entities': entities}
            ))
        
        return steps
    
    def validate_plan(self, plan: ExecutionPlan) -> bool:
        """
        Validate a plan before execution
        Checks: circular dependencies, invalid actions, etc.
        """
        # Check for circular dependencies
        visited = set()
        
        def has_cycle(step_id: int, path: set) -> bool:
            if step_id in path:
                return True
            
            path.add(step_id)
            
            step = next((s for s in plan.steps if s.step_id == step_id), None)
            if step:
                for dep in step.depends_on:
                    if has_cycle(dep, path.copy()):
                        return True
            
            return False
        
        for step in plan.steps:
            if has_cycle(step.step_id, set()):
                logger.error(f"Circular dependency detected in plan {plan.plan_id}")
                return False
        
        # Check that all dependencies exist
        step_ids = {s.step_id for s in plan.steps}
        
        for step in plan.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    logger.error(f"Step {step.step_id} depends on non-existent step {dep}")
                    return False
        
        return True
    
    def explain_plan(self, plan: ExecutionPlan) -> str:
        """Generate human-readable explanation of the plan"""
        explanation = [f"Plan: {plan.goal}\n"]
        
        for step in plan.steps:
            deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
            explanation.append(f"{step.step_id}. {step.action}{deps}")
        
        return "\n".join(explanation)


# Global instance
_planner = None


def get_planner() -> TaskPlanner:
    """Get the global task planner instance"""
    global _planner
    if _planner is None:
        _planner = TaskPlanner()
    return _planner
