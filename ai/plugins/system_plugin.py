"""
System Monitoring Plugin for A.L.I.C.E — real-time system health and status.

Handles intents: system:status, system:processes, system:resources
Triggered by:
  "system status", "how's the system", "what's running",
  "cpu usage", "memory usage", "disk space", "open ports"
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("[SystemPlugin] psutil not installed — install with: pip install psutil")

from ai.plugins.plugin_system import PluginInterface


_INTENTS = {
    "system:status",
    "system:processes",
    "system:resources",
    "system:disk",
    "system:network",
    "system:ports",
}

_TRIGGER_PHRASES = [
    "system status", "system health", "how's the system", "how is the system",
    "what's running", "what is running", "cpu usage", "cpu load",
    "memory usage", "ram usage", "disk space", "disk usage",
    "open ports", "network status", "running processes", "process list",
    "resource usage", "system info", "system report",
]


class SystemPlugin(PluginInterface):
    """Real-time system monitoring — CPU, memory, disk, processes, ports."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "SystemPlugin"
        self.version = "1.0.0"
        self.description = "Monitor system resources, processes, disk, and network"
        self.enabled = True
        self.capabilities = list(_INTENTS)

    def initialize(self) -> bool:
        if not PSUTIL_AVAILABLE:
            logger.warning("[SystemPlugin] psutil unavailable — limited functionality")
        return True

    def shutdown(self) -> None:
        pass

    def can_handle(self, intent: str, entities: Dict, query: str = "") -> bool:
        if intent in _INTENTS:
            return True
        q = (query or "").lower()
        return any(phrase in q for phrase in _TRIGGER_PHRASES)

    def execute(
        self, intent: str, query: str, entities: Dict, context: Dict
    ) -> Dict[str, Any]:
        q = (query or "").lower()

        if "process" in q or "running" in q or intent == "system:processes":
            return self._top_processes()
        if "port" in q or intent == "system:ports":
            return self._open_ports()
        if "disk" in q or intent == "system:disk":
            return self._disk_info()
        if "network" in q or intent == "system:network":
            return self._network_info()
        return self._full_status()

    # ------------------------------------------------------------------ reports

    def _full_status(self) -> Dict[str, Any]:
        parts: List[str] = []

        # CPU
        if PSUTIL_AVAILABLE:
            cpu_pct = psutil.cpu_percent(interval=0.5)
            cpu_count = psutil.cpu_count()
            parts.append(f"CPU: {cpu_pct:.1f}% ({cpu_count} cores)")

            # Memory
            mem = psutil.virtual_memory()
            used_gb = mem.used / 1e9
            total_gb = mem.total / 1e9
            parts.append(f"RAM: {used_gb:.1f} / {total_gb:.1f} GB ({mem.percent:.0f}%)")

            # Disk (primary drive)
            disk = psutil.disk_usage("/")
            free_gb = disk.free / 1e9
            total_d = disk.total / 1e9
            parts.append(f"Disk: {free_gb:.1f} GB free of {total_d:.1f} GB ({disk.percent:.0f}% used)")

            # Top 5 CPU processes
            top = self._get_top_processes(n=5)
            if top:
                proc_lines = "  " + "\n  ".join(
                    f"{p['name']} ({p['cpu']:.1f}% CPU, {p['mem']:.0f} MB)" for p in top
                )
                parts.append(f"Top processes:\n{proc_lines}")

            # Battery if available
            try:
                batt = psutil.sensors_battery()
                if batt:
                    status = "charging" if batt.power_plugged else "on battery"
                    parts.append(f"Battery: {batt.percent:.0f}% ({status})")
            except Exception:
                pass
        else:
            parts.append("System info limited — psutil not installed.")
            parts.append(f"Platform: {platform.system()} {platform.release()}")
            parts.append(f"Hostname: {socket.gethostname()}")

        # Git status of CWD project
        git_info = self._git_status()
        if git_info:
            parts.append(f"Git: {git_info}")

        report = "\n".join(parts)
        return {
            "success": True,
            "response": f"System status:\n{report}",
            "data": {"report": report},
        }

    def _top_processes(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {"success": False, "response": "psutil not available."}
        top = self._get_top_processes(n=10)
        lines = "\n".join(
            f"  {i+1}. {p['name']} — {p['cpu']:.1f}% CPU, {p['mem']:.0f} MB RAM"
            for i, p in enumerate(top)
        )
        return {
            "success": True,
            "response": f"Top running processes:\n{lines}",
            "data": {"processes": top},
        }

    def _disk_info(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {"success": False, "response": "psutil not available."}
        lines: List[str] = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                free = usage.free / 1e9
                total = usage.total / 1e9
                lines.append(
                    f"  {part.mountpoint} ({part.fstype}): "
                    f"{free:.1f} GB free / {total:.1f} GB ({usage.percent:.0f}% used)"
                )
            except PermissionError:
                pass
        text = "\n".join(lines) if lines else "No disk info available."
        return {"success": True, "response": f"Disk usage:\n{text}", "data": {"disks": lines}}

    def _open_ports(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {"success": False, "response": "psutil not available."}
        listening: List[Dict] = []
        try:
            for conn in psutil.net_connections(kind="inet"):
                if conn.status == "LISTEN" and conn.laddr:
                    port = conn.laddr.port
                    pid = conn.pid
                    name = "unknown"
                    if pid:
                        try:
                            name = psutil.Process(pid).name()
                        except Exception:
                            pass
                    listening.append({"port": port, "process": name, "pid": pid})
        except Exception as e:
            return {"success": False, "response": f"Could not list ports: {e}"}

        listening.sort(key=lambda x: x["port"])
        lines = "\n".join(
            f"  :{p['port']} — {p['process']} (pid {p['pid']})" for p in listening[:20]
        )
        text = lines if lines else "No listening ports found."
        return {"success": True, "response": f"Open listening ports:\n{text}", "data": {"ports": listening}}

    def _network_info(self) -> Dict[str, Any]:
        if not PSUTIL_AVAILABLE:
            return {"success": False, "response": "psutil not available."}
        try:
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            lines: List[str] = []
            for iface, addr_list in addrs.items():
                st = stats.get(iface)
                if not st or not st.isup:
                    continue
                ips = [a.address for a in addr_list if a.family == socket.AF_INET]
                if ips:
                    lines.append(f"  {iface}: {', '.join(ips)}")
            text = "\n".join(lines) if lines else "No active network interfaces."
        except Exception as e:
            text = f"Network info unavailable: {e}"
        return {"success": True, "response": f"Network interfaces:\n{text}"}

    # ------------------------------------------------------------------ helpers

    def _get_top_processes(self, n: int = 10) -> List[Dict]:
        procs: List[Dict] = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
            try:
                info = p.info
                cpu = info.get("cpu_percent") or 0.0
                mem_bytes = (info.get("memory_info") or None)
                mem_mb = mem_bytes.rss / 1e6 if mem_bytes else 0.0
                procs.append({"pid": info["pid"], "name": info["name"] or "?", "cpu": cpu, "mem": mem_mb})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sorted(procs, key=lambda x: x["cpu"], reverse=True)[:n]

    def _git_status(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "status", "--short", "--branch"],
                capture_output=True, text=True, timeout=3,
                cwd=os.getcwd(),
            )
            if result.returncode != 0:
                return None
            lines = result.stdout.strip().splitlines()
            branch_line = lines[0] if lines else ""
            changes = len([ln for ln in lines[1:] if ln.strip()])
            branch = branch_line.replace("## ", "").split("...")[0]
            if changes:
                return f"{branch} ({changes} uncommitted change{'s' if changes != 1 else ''})"
            return f"{branch} (clean)"
        except Exception:
            return None

    # ------------------------------------------------------------------ snapshot for world model

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight snapshot suitable for the world model."""
        if not PSUTIL_AVAILABLE:
            return {}
        try:
            cpu = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return {
                "cpu_percent": cpu,
                "ram_percent": mem.percent,
                "ram_used_gb": round(mem.used / 1e9, 2),
                "ram_total_gb": round(mem.total / 1e9, 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / 1e9, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception:
            return {}
