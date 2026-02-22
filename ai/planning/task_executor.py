"""
Task Execution Framework for A.L.I.C.E
Enables intelligent automation and system control
Features:
- System command execution
- File operations
- Application control
- Automation workflows
- Scheduled tasks
"""

import os
import subprocess
import logging
import platform
import shutil
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskResult:
    """Result of task execution"""
    def __init__(
        self, 
        success: bool,
        message: str,
        data: Any = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.message = message
        self.data = data
        self.error = error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp
        }


class TaskExecutor:
    """
    Advanced task execution system
    Handles system automation and intelligent control
    """
    
    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode  # Prevent dangerous commands
        self.system = platform.system()
        self.task_history: List[Dict] = []
        self.scheduled_tasks: List[Dict] = []
        
        # Dangerous commands to block in safe mode
        self.dangerous_commands = [
            'rm -rf /',
            'format',
            'del /f /s /q',
            'shutdown /s /f /t 0',
            'dd if=/dev/zero',
        ]
        
        logger.info(f"[OK] Task Executor initialized (System: {self.system}, Safe Mode: {safe_mode})")
    
    def execute_command(
        self, 
        command: str,
        shell: bool = True,
        timeout: Optional[int] = 30,
        capture_output: bool = True
    ) -> TaskResult:
        """
        Execute a system command
        
        Args:
            command: Command to execute
            shell: Use shell
            timeout: Command timeout in seconds
            capture_output: Capture stdout/stderr
            
        Returns:
            TaskResult with command output
        """
        # Safety check
        if self.safe_mode and any(dangerous in command.lower() for dangerous in self.dangerous_commands):
            logger.warning(f" Blocked dangerous command: {command}")
            return TaskResult(
                success=False,
                message="Command blocked for safety",
                error="Dangerous command detected"
            )
        
        try:
            logger.info(f"Executing command: {command}")
            
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            
            # Log to history
            self.task_history.append({
                "type": "command",
                "command": command,
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            
            return TaskResult(
                success=success,
                message="Command executed successfully" if success else "Command failed",
                data={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            )
            
        except subprocess.TimeoutExpired:
            logger.error(f"[ERROR] Command timeout: {command}")
            return TaskResult(
                success=False,
                message="Command timed out",
                error="Timeout"
            )
        except Exception as e:
            logger.error(f"[ERROR] Command error: {e}")
            return TaskResult(
                success=False,
                message="Command execution failed",
                error=str(e)
            )
    
    def open_application(self, app_name: str) -> TaskResult:
        """
        Open an application
        
        Args:
            app_name: Name of application to open
        """
        try:
            if self.system == "Windows":
                # Handle protocol URLs (like steam://, com.epicgames.launcher://)
                if "://" in app_name:
                    result = self.execute_command(f'start "" "{app_name}"', timeout=10)
                # Handle direct executable names  
                elif app_name.lower() in ['notepad', 'calc', 'mspaint', 'explorer']:
                    result = self.execute_command(f"start {app_name}", timeout=5)
                # Handle application names with spaces or complex names
                else:
                    result = self.execute_command(f'start "" "{app_name}"', timeout=10)
            elif self.system == "Darwin":
                # macOS
                result = self.execute_command(f"open -a '{app_name}'")
            else:
                # Linux
                result = self.execute_command(f"nohup {app_name} &")
            
            if result.success:
                logger.info(f"[OK] Opened application: {app_name}")
                return TaskResult(
                    success=True,
                    message=f"Opened {app_name}",
                    data={"app": app_name}
                )
            else:
                return result
                
        except Exception as e:
            logger.error(f"[ERROR] Error opening {app_name}: {e}")
            return TaskResult(
                success=False,
                message=f"Failed to open {app_name}",
                error=str(e)
            )
    
    def file_operation(
        self,
        operation: str,
        source: str,
        destination: Optional[str] = None
    ) -> TaskResult:
        """
        Perform file operations
        
        Args:
            operation: Operation type (create, delete, copy, move, read)
            source: Source file/folder path
            destination: Destination path (for copy/move)
        """
        try:
            if operation == "create":
                # Create file or directory
                if source.endswith('/') or source.endswith('\\'):
                    os.makedirs(source, exist_ok=True)
                    message = f"Created directory: {source}"
                else:
                    # Create file
                    os.makedirs(os.path.dirname(source), exist_ok=True)
                    with open(source, 'w') as f:
                        f.write("")
                    message = f"Created file: {source}"
                
                return TaskResult(success=True, message=message)
            
            elif operation == "delete":
                if os.path.isdir(source):
                    shutil.rmtree(source)
                    message = f"Deleted directory: {source}"
                else:
                    os.remove(source)
                    message = f"Deleted file: {source}"
                
                return TaskResult(success=True, message=message)
            
            elif operation == "copy":
                if not destination:
                    return TaskResult(success=False, message="Destination required for copy", error="Missing destination")
                
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                    message = f"Copied directory: {source} → {destination}"
                else:
                    shutil.copy2(source, destination)
                    message = f"Copied file: {source} → {destination}"
                
                return TaskResult(success=True, message=message)
            
            elif operation == "move":
                if not destination:
                    return TaskResult(success=False, message="Destination required for move", error="Missing destination")
                
                shutil.move(source, destination)
                return TaskResult(success=True, message=f"Moved: {source} → {destination}")
            
            elif operation == "read":
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return TaskResult(
                    success=True,
                    message=f"Read file: {source}",
                    data={"content": content, "path": source}
                )
            
            else:
                return TaskResult(success=False, message=f"Unknown operation: {operation}", error="Invalid operation")
                
        except FileNotFoundError:
            return TaskResult(success=False, message=f"File not found: {source}", error="FileNotFoundError")
        except PermissionError:
            return TaskResult(success=False, message=f"Permission denied: {source}", error="PermissionError")
        except Exception as e:
            logger.error(f"[ERROR] File operation error: {e}")
            return TaskResult(success=False, message="File operation failed", error=str(e))
    
    def get_system_info(self) -> TaskResult:
        """Get system information"""
        try:
            info = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            }
            
            # Get additional info based on OS
            if self.system == "Windows":
                # Get CPU and memory info
                result = self.execute_command("wmic cpu get name", capture_output=True)
                if result.success:
                    cpu_info = result.data.get('stdout', '').strip()
                    info['cpu_details'] = cpu_info
            
            return TaskResult(
                success=True,
                message="System information retrieved",
                data=info
            )
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting system info: {e}")
            return TaskResult(success=False, message="Failed to get system info", error=str(e))
    
    def control_volume(self, action: str, level: Optional[int] = None) -> TaskResult:
        """
        Control system volume
        
        Args:
            action: Action (up, down, mute, set)
            level: Volume level (0-100) for 'set' action
        """
        try:
            if self.system == "Windows":
                # Windows volume control using nircmd (requires installation) or PowerShell
                if action == "up":
                    command = "(Get-AudioDevice -Playback).Volume += 10"
                elif action == "down":
                    command = "(Get-AudioDevice -Playback).Volume -= 10"
                elif action == "mute":
                    command = "(Get-AudioDevice -Playback).Mute = $true"
                elif action == "set" and level is not None:
                    command = f"(Get-AudioDevice -Playback).Volume = {level}"
                else:
                    return TaskResult(success=False, message="Invalid volume action", error="Invalid action")
                
                # Note: This requires AudioDeviceCmdlets module
                # Install with: Install-Module -Name AudioDeviceCmdlets
                result = self.execute_command(f"powershell -Command \"{command}\"")
                
                return TaskResult(
                    success=True,
                    message=f"Volume {action}",
                    data={"action": action, "level": level}
                )
            else:
                return TaskResult(
                    success=False,
                    message="Volume control not implemented for this OS",
                    error="Not implemented"
                )
                
        except Exception as e:
            logger.error(f"[ERROR] Volume control error: {e}")
            return TaskResult(success=False, message="Volume control failed", error=str(e))
    
    def schedule_task(
        self,
        task_name: str,
        task_function: Callable,
        delay_seconds: int,
        repeat: bool = False,
        interval: Optional[int] = None
    ) -> TaskResult:
        """
        Schedule a task to run later
        
        Args:
            task_name: Name of task
            task_function: Function to execute
            delay_seconds: Delay before first execution
            repeat: Repeat task
            interval: Interval for repeated tasks (seconds)
        """
        def run_scheduled_task():
            time.sleep(delay_seconds)
            
            while True:
                try:
                    logger.info(f" Executing scheduled task: {task_name}")
                    task_function()
                    
                    if not repeat:
                        break
                    
                    if interval:
                        time.sleep(interval)
                    else:
                        break
                        
                except Exception as e:
                    logger.error(f"[ERROR] Scheduled task error ({task_name}): {e}")
                    break
        
        # Start task in background thread
        thread = threading.Thread(target=run_scheduled_task, daemon=True)
        thread.start()
        
        # Track scheduled task
        self.scheduled_tasks.append({
            "name": task_name,
            "scheduled_at": datetime.now().isoformat(),
            "delay": delay_seconds,
            "repeat": repeat,
            "interval": interval
        })
        
        return TaskResult(
            success=True,
            message=f"Task '{task_name}' scheduled",
            data={"delay": delay_seconds, "repeat": repeat}
        )
    
    def create_workflow(self, workflow_name: str, tasks: List[Dict]) -> TaskResult:
        """
        Execute a workflow (series of tasks)
        
        Args:
            workflow_name: Name of workflow
            tasks: List of task dictionaries with 'type' and parameters
            
        Example:
            tasks = [
                {"type": "command", "command": "echo Hello"},
                {"type": "file", "operation": "create", "source": "test.txt"},
                {"type": "app", "name": "notepad"}
            ]
        """
        results = []
        
        logger.info(f"Starting workflow: {workflow_name}")
        
        for i, task in enumerate(tasks, 1):
            task_type = task.get("type")
            
            try:
                if task_type == "command":
                    result = self.execute_command(task.get("command", ""))
                elif task_type == "file":
                    result = self.file_operation(
                        task.get("operation"),
                        task.get("source"),
                        task.get("destination")
                    )
                elif task_type == "app":
                    result = self.open_application(task.get("name"))
                else:
                    result = TaskResult(success=False, message=f"Unknown task type: {task_type}", error="Invalid task type")
                
                results.append({
                    "step": i,
                    "task": task,
                    "result": result.to_dict()
                })
                
                # Stop on failure if critical
                if not result.success and task.get("critical", False):
                    logger.error(f" Workflow failed at step {i}")
                    break
                    
            except Exception as e:
                logger.error(f" Workflow error at step {i}: {e}")
                results.append({
                    "step": i,
                    "task": task,
                    "error": str(e)
                })
                break
        
        success = all(r.get("result", {}).get("success", False) for r in results)
        
        return TaskResult(
            success=success,
            message=f"Workflow '{workflow_name}' {'completed' if success else 'failed'}",
            data={"workflow": workflow_name, "results": results}
        )
    
    def get_task_history(self, limit: int = 10) -> List[Dict]:
        """Get recent task history"""
        return self.task_history[-limit:]
    
    def clear_history(self):
        """Clear task history"""
        self.task_history = []
        logger.info(" Task history cleared")


# Example usage
if __name__ == "__main__":
    print("Testing Task Execution Framework...\n")
    
    executor = TaskExecutor(safe_mode=True)
    
    # Test 1: System info
    print("1. Getting system information...")
    result = executor.get_system_info()
    if result.success:
        print(f"   [OK] System: {result.data.get('system')} {result.data.get('release')}")
    
    # Test 2: File operations
    print("\n2. Testing file operations...")
    
    # Create a test file
    result = executor.file_operation("create", "test_alice.txt")
    print(f"   Create: {result.message}")
    
    # Read file
    result = executor.file_operation("read", "test_alice.txt")
    print(f"   Read: {result.message}")
    
    # Delete file
    result = executor.file_operation("delete", "test_alice.txt")
    print(f"   Delete: {result.message}")
    
    # Test 3: Command execution
    print("\n3. Testing command execution...")
    if platform.system() == "Windows":
        result = executor.execute_command("echo Hello from ALICE!")
        print(f"   Command output: {result.data.get('stdout', '').strip()}")
    
    # Test 4: Workflow
    print("\n4. Testing workflow...")
    workflow_tasks = [
        {"type": "command", "command": "echo Step 1: Starting workflow"},
        {"type": "file", "operation": "create", "source": "workflow_test.txt"},
        {"type": "command", "command": "echo Step 3: Workflow complete"},
    ]
    
    result = executor.create_workflow("test_workflow", workflow_tasks)
    print(f"   Workflow: {result.message}")
    print(f"   Steps completed: {len(result.data.get('results', []))}")
    
    # Cleanup
    if os.path.exists("workflow_test.txt"):
        os.remove("workflow_test.txt")
    
    # Test 5: Scheduled task
    print("\n5. Testing scheduled task...")
    
    def test_task():
        print("    Scheduled task executed!")
    
    result = executor.schedule_task("test_reminder", test_task, delay_seconds=2)
    print(f"   {result.message}")
    print("   Waiting for task...")
    time.sleep(3)
    
    print("\n[OK] All tests complete!")
