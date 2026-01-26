"""
Plan Executor for A.L.I.C.E
Executes plans created by TaskPlanner
Strictly separated from planning logic
"""

from typing import Dict, Any, Callable, Optional
import logging
import re

from .task_planner import ExecutionPlan, PlanStep, StepStatus, PlanStatus
from .event_bus import get_event_bus, EventType, EventPriority
from .system_state import get_state_tracker

logger = logging.getLogger(__name__)


class PlanExecutor:
    """
    Executes plans step-by-step
    Handles: dependencies, errors, rollbacks
    """
    
    def __init__(self, plugin_manager, llm_engine, memory_system):
        self.plugin_manager = plugin_manager
        self.llm_engine = llm_engine
        self.memory_system = memory_system
        
        self.event_bus = get_event_bus()
        self.state_tracker = get_state_tracker()
        
        # Action handlers
        self._action_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default action handlers"""
        # Plugin actions
        self._action_handlers['plugin.*'] = self._execute_plugin_action
        
        # LLM actions
        self._action_handlers['llm.summarize'] = self._execute_llm_summarize
        self._action_handlers['llm.answer'] = self._execute_llm_answer
        self._action_handlers['llm.draft_email'] = self._execute_llm_draft
        self._action_handlers['llm.process'] = self._execute_llm_process
        
        # Memory actions
        self._action_handlers['memory.search'] = self._execute_memory_search
        
        # Formatting actions
        self._action_handlers['format.*'] = self._execute_format
    
    def execute(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Execute a plan step-by-step
        
        Args:
            plan: Execution plan to run
        
        Returns:
            Final result
        """
        logger.info(f"Executing plan {plan.plan_id}: {plan.goal}")
        
        # Create system task
        task_id = self.state_tracker.create_task(
            name=plan.goal,
            metadata={'plan_id': plan.plan_id, 'intent': plan.intent}
        )
        
        self.state_tracker.start_task(task_id)
        plan.status = PlanStatus.EXECUTING
        
        completed_steps = set()
        step_results = {}
        
        try:
            while not plan.is_complete and not plan.has_failed:
                # Get next executable step
                next_step = plan.get_next_step(completed_steps)
                
                if next_step is None:
                    # No executable steps left but not complete = deadlock
                    if not plan.is_complete:
                        raise RuntimeError("Plan deadlock: no executable steps remaining")
                    break
                
                # Execute step
                logger.info(f"Executing step {next_step.step_id}: {next_step.action}")
                next_step.status = StepStatus.RUNNING
                
                # Update progress
                progress = len(completed_steps) / len(plan.steps)
                self.state_tracker.update_task_progress(task_id, progress)
                
                try:
                    # Resolve parameter references
                    params = self._resolve_params(next_step.params, step_results)
                    
                    # Execute action
                    result = self._execute_action(next_step.action, params)
                    
                    # Store result
                    next_step.result = result
                    next_step.status = StepStatus.COMPLETED
                    step_results[next_step.step_id] = result
                    completed_steps.add(next_step.step_id)
                    
                    logger.info(f"Step {next_step.step_id} completed successfully")
                
                except Exception as e:
                    logger.error(f"Step {next_step.step_id} failed: {e}")
                    next_step.status = StepStatus.FAILED
                    next_step.error = str(e)
                    
                    # Emit failure event
                    self.event_bus.emit(
                        EventType.TASK_FAILED,
                        data={
                            'plan_id': plan.plan_id,
                            'step_id': next_step.step_id,
                            'action': next_step.action,
                            'error': str(e)
                        },
                        priority=EventPriority.HIGH
                    )
                    
                    # Fail the plan
                    plan.status = PlanStatus.FAILED
                    self.state_tracker.fail_task(task_id, str(e))
                    
                    return {
                        'success': False,
                        'error': str(e),
                        'failed_step': next_step.step_id
                    }
            
            # Plan completed successfully
            plan.status = PlanStatus.COMPLETED
            self.state_tracker.complete_task(task_id, result=step_results)
            
            # Return final result (last step's result)
            final_result = plan.steps[-1].result if plan.steps else None
            
            logger.info(f"Plan {plan.plan_id} completed successfully")
            
            return {
                'success': True,
                'result': final_result,
                'all_results': step_results
            }
        
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            plan.status = PlanStatus.FAILED
            self.state_tracker.fail_task(task_id, str(e))
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def _resolve_params(self, params: Dict[str, Any], step_results: Dict[int, Any]) -> Dict[str, Any]:
        """
        Resolve parameter references like {{step_1_result}}
        
        Args:
            params: Parameters with potential references
            step_results: Results from previous steps
        
        Returns:
            Resolved parameters
        """
        resolved = {}
        
        for key, value in params.items():
            if isinstance(value, str) and '{{' in value:
                # Extract reference
                match = re.search(r'\{\{step_(\d+)_result(?:\.(.+))?\}\}', value)
                
                if match:
                    step_id = int(match.group(1))
                    field = match.group(2)
                    
                    if step_id in step_results:
                        result = step_results[step_id]
                        
                        # Navigate to field if specified
                        if field and isinstance(result, dict):
                            resolved[key] = result.get(field)
                        else:
                            resolved[key] = result
                    else:
                        raise ValueError(f"Step {step_id} result not available")
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        
        return resolved
    
    def _execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """
        Execute a single action
        
        Args:
            action: Action name (e.g., "plugin.notes.list")
            params: Action parameters
        
        Returns:
            Action result
        """
        # Find matching handler
        handler = None
        
        # Try exact match first
        if action in self._action_handlers:
            handler = self._action_handlers[action]
        else:
            # Try wildcard match
            for pattern, h in self._action_handlers.items():
                if '*' in pattern:
                    prefix = pattern.replace('.*', '')
                    if action.startswith(prefix):
                        handler = h
                        break
        
        if handler is None:
            raise ValueError(f"No handler for action: {action}")
        
        return handler(action, params)
    
    def _execute_plugin_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute a plugin action"""
        # Parse action: plugin.{name}.{method}
        parts = action.split('.')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid plugin action format: {action}")
        
        plugin_name = parts[1]
        method_name = parts[2]
        
        # Get plugin
        plugin = self.plugin_manager.get_plugin(plugin_name)
        
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        # Call method
        if not hasattr(plugin, method_name):
            raise ValueError(f"Plugin {plugin_name} has no method {method_name}")
        
        method = getattr(plugin, method_name)
        return method(**params)
    
    def _execute_llm_summarize(self, action: str, params: Dict[str, Any]) -> str:
        """Execute LLM summarization"""
        content = params.get('content', '')
        focus = params.get('focus', '')
        
        prompt = f"Summarize the following content"
        if focus:
            prompt += f" focusing on {focus}"
        prompt += f":\n\n{content}"
        
        return self.llm_engine.generate(prompt, max_tokens=500)
    
    def _execute_llm_answer(self, action: str, params: Dict[str, Any]) -> str:
        """Execute LLM question answering"""
        query = params.get('query', '')
        context = params.get('context', '')
        
        prompt = f"Answer this question"
        if context:
            prompt += f" using the following context:\n\n{context}\n\nQuestion: {query}"
        else:
            prompt += f": {query}"
        
        return self.llm_engine.generate(prompt)
    
    def _execute_llm_draft(self, action: str, params: Dict[str, Any]) -> str:
        """Execute LLM email drafting"""
        subject = params.get('subject', '')
        context = params.get('context', {})
        
        prompt = f"Draft a professional email with subject: {subject}"
        
        return self.llm_engine.generate(prompt, max_tokens=300)
    
    def _execute_llm_process(self, action: str, params: Dict[str, Any]) -> str:
        """Execute generic LLM processing"""
        intent = params.get('intent', '')
        entities = params.get('entities', {})
        
        prompt = f"Process this intent: {intent}\nEntities: {entities}"
        
        return self.llm_engine.generate(prompt)
    
    def _execute_memory_search(self, action: str, params: Dict[str, Any]) -> List[Dict]:
        """Execute memory search"""
        query = params.get('query', '')
        top_k = params.get('top_k', 3)
        
        results = self.memory_system.search(query, top_k=top_k)
        
        return results
    
    def _execute_format(self, action: str, params: Dict[str, Any]) -> str:
        """Execute formatting action"""
        # Simple formatting handlers
        if action == "format.calendar_events":
            events = params.get('events', [])
            
            if not events:
                return "No events found."
            
            formatted = []
            for event in events:
                formatted.append(f"- {event.get('title')} at {event.get('time')}")
            
            return "\n".join(formatted)
        
        return str(params)


# Global instance - will be initialized by ALICE
_executor = None


def get_executor() -> Optional[PlanExecutor]:
    """Get the global executor instance"""
    return _executor


def initialize_executor(plugin_manager, llm_engine, memory_system):
    """Initialize the global executor"""
    global _executor
    _executor = PlanExecutor(plugin_manager, llm_engine, memory_system)
    return _executor
