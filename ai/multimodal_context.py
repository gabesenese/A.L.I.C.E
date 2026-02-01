"""
Multimodal Context: Captures system state, active windows, processes, telemetry.
Fuses these signals with ALICE's reasoning for contextual pattern learning.
"""

import json
import os
import platform
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime


class MultimodalContext:
    def __init__(self):
        """Initialize multimodal context capture."""
        self.platform = platform.system()
        self.history_path = "data/context/multimodal_history.jsonl"

    def get_active_windows(self) -> List[Dict[str, Any]]:
        """Get list of active windows/processes."""
        try:
            if self.platform == "Windows":
                return self._get_windows_windows()
            elif self.platform == "Darwin":
                return self._get_macos_windows()
            else:
                return self._get_linux_windows()
        except Exception as e:
            return [{"error": str(e)}]

    def _get_windows_windows(self) -> List[Dict[str, Any]]:
        """Get active windows on Windows."""
        try:
            import ctypes
            import win32gui
            import win32process

            windows = []
            def enum_callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    try:
                        title = win32gui.GetWindowText(hwnd)
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if title.strip():
                            windows.append({
                                "title": title,
                                "pid": pid,
                                "visible": True
                            })
                    except:
                        pass
                return True

            win32gui.EnumWindows(enum_callback, None)
            return windows[:10]  # Top 10 windows
        except:
            return [{"platform": "windows", "fallback": True}]

    def _get_macos_windows(self) -> List[Dict[str, Any]]:
        """Get active windows on macOS."""
        try:
            # macOS window enumeration would require additional libraries
            return [{"platform": "macos", "info": "Requires specialized libraries"}]
        except:
            return []

    def _get_linux_windows(self) -> List[Dict[str, Any]]:
        """Get active windows on Linux."""
        try:
            # Linux window enumeration would require X11 libraries
            return [{"platform": "linux", "info": "Requires X11 libraries"}]
        except:
            return []

    def get_system_telemetry(self) -> Dict[str, Any]:
        """Get system telemetry: CPU, memory, disk usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "memory": {
                    "total_mb": memory.total / (1024 * 1024),
                    "available_mb": memory.available / (1024 * 1024),
                    "percent": memory.percent,
                    "used_mb": memory.used / (1024 * 1024)
                },
                "disk": {
                    "total_gb": disk.total / (1024 ** 3),
                    "used_gb": disk.used / (1024 ** 3),
                    "free_gb": disk.free / (1024 ** 3),
                    "percent": disk.percent
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def get_running_processes(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top N running processes by CPU/memory."""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cpu_percent": proc.info['cpu_percent'] or 0,
                        "memory_percent": proc.info['memory_percent'] or 0
                    })
                except psutil.NoSuchProcess:
                    pass

            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            return processes[:top_n]
        except Exception as e:
            return [{"error": str(e)}]

    def get_editor_state(self) -> Dict[str, Any]:
        """Get state of editor (VS Code, IDE, etc.)."""
        try:
            editor_state = {
                "vs_code_running": False,
                "ide_running": False,
                "terminal_open": False,
                "files_open": [],
                "git_status": None
            }

            # Check for VS Code
            processes = [p.name() for p in psutil.process_iter(['name'])]
            editor_state["vs_code_running"] = "code.exe" in processes or "code" in processes
            editor_state["terminal_open"] = "conhost.exe" in processes or "bash" in processes

            return editor_state
        except Exception as e:
            return {"error": str(e)}

    def get_application_context(self) -> Dict[str, Any]:
        """Get application-level context signals."""
        return {
            "active_windows": self.get_active_windows(),
            "telemetry": self.get_system_telemetry(),
            "top_processes": self.get_running_processes(5),
            "editor_state": self.get_editor_state(),
            "timestamp": datetime.now().isoformat()
        }

    def correlate_with_interaction(self, user_input: str, response: str) -> Dict[str, Any]:
        """
        Correlate user interaction with system context.
        Example: "VS Code + failing test output" → different help pattern needed.
        """
        context = self.get_application_context()

        correlation = {
            "user_input": user_input,
            "response": response,
            "system_context": context,
            "patterns": self._infer_patterns(user_input, context),
            "timestamp": datetime.now().isoformat()
        }

        return correlation

    def _infer_patterns(self, user_input: str, context: Dict[str, Any]) -> List[str]:
        """Infer contextual patterns from input + system state."""
        patterns = []

        # VS Code + code-related query
        if context.get("editor_state", {}).get("vs_code_running"):
            if any(keyword in user_input.lower() for keyword in ["error", "debug", "test", "run"]):
                patterns.append("code_debugging_context")

        # High system load
        telemetry = context.get("telemetry", {})
        if telemetry.get("cpu_percent", 0) > 80:
            patterns.append("high_system_load")

        if context.get("memory", {}).get("percent", 0) > 85:
            patterns.append("memory_pressure")

        # Terminal activity
        if context.get("terminal_open"):
            patterns.append("terminal_active")

        # Time-based pattern
        hour = datetime.now().hour
        if hour < 9:
            patterns.append("early_morning")
        elif hour >= 17:
            patterns.append("evening")

        return patterns

    def save_interaction_context(self, correlation: Dict[str, Any]):
        """Save interaction context to history."""
        os.makedirs(os.path.dirname(self.history_path) or '.', exist_ok=True)
        with open(self.history_path, 'a') as f:
            f.write(json.dumps(correlation) + '\n')

    def get_context_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent interaction contexts."""
        history = []
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))

        return history[-limit:]

    def get_context_statistics(self) -> Dict[str, Any]:
        """Get statistics about captured contexts."""
        history = self.get_context_history(1000)

        patterns = {}
        for item in history:
            for pattern in item.get("patterns", []):
                patterns[pattern] = patterns.get(pattern, 0) + 1

        avg_cpu = sum(item.get("system_context", {}).get("telemetry", {}).get("cpu_percent", 0)
                     for item in history) / max(len(history), 1)

        return {
            "total_interactions": len(history),
            "pattern_frequencies": patterns,
            "avg_cpu_load": avg_cpu,
            "timestamps": [item.get("timestamp") for item in history[-10:]]
        }
