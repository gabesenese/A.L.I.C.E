"""
System Monitor for A.L.I.C.E
Continuous OS-level awareness: apps, processes, file changes
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import time
import logging
import psutil

from .event_bus import get_event_bus, EventType, EventPriority

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Information about a running process"""
    pid: int
    name: str
    status: str
    cpu_percent: float
    memory_mb: float
    started_at: datetime


class SystemMonitor:
    """
    Monitors system state continuously
    Tracks: running apps, active window, file changes, network activity
    """
    
    def __init__(self):
        self.event_bus = get_event_bus()
        
        # Process tracking
        self._known_processes: Dict[int, ProcessInfo] = {}
        self._known_app_names: Set[str] = set()
        
        # File watching (simplified - full implementation would use watchdog)
        self._watched_paths: Set[str] = set()
        
        # Active window tracking (Windows only for now)
        self._active_window: Optional[str] = None
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        
        # Configuration
        self._scan_interval = 10  # seconds
        self._ignore_system_processes = True
    
    def start_monitoring(self):
        """Start continuous system monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="system_monitor"
        )
        self._monitor_thread.start()
        logger.info("System monitoring started")
        
        # Emit event
        self.event_bus.emit(
            EventType.SYSTEM_BUSY,
            data={'component': 'system_monitor', 'status': 'started'},
            priority=EventPriority.LOW
        )
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self._scan_processes()
                self._check_watched_files()
                # self._check_active_window()  # Would need platform-specific impl
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self._scan_interval)
    
    def _scan_processes(self):
        """Scan for new/closed processes"""
        try:
            current_pids = set()
            
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_info']):
                try:
                    pinfo = proc.info
                    pid = pinfo['pid']
                    name = pinfo['name']
                    
                    # Skip system processes if configured
                    if self._ignore_system_processes and self._is_system_process(name):
                        continue
                    
                    current_pids.add(pid)
                    
                    # New process detected
                    if pid not in self._known_processes:
                        memory_mb = pinfo['memory_info'].rss / (1024 * 1024) if pinfo.get('memory_info') else 0
                        
                        proc_info = ProcessInfo(
                            pid=pid,
                            name=name,
                            status=pinfo.get('status', 'unknown'),
                            cpu_percent=pinfo.get('cpu_percent', 0.0) or 0.0,
                            memory_mb=memory_mb,
                            started_at=datetime.now()
                        )
                        
                        self._known_processes[pid] = proc_info
                        
                        # New app launched
                        if name not in self._known_app_names:
                            self._known_app_names.add(name)
                            
                            # Emit event for notable apps
                            if self._is_notable_app(name):
                                self.event_bus.emit(
                                    EventType.PLUGIN_NOTIFICATION,
                                    data={
                                        'type': 'app_launched',
                                        'app': name,
                                        'pid': pid
                                    },
                                    priority=EventPriority.LOW,
                                    source='system_monitor'
                                )
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Detect closed processes
            closed_pids = set(self._known_processes.keys()) - current_pids
            
            for pid in closed_pids:
                proc_info = self._known_processes.pop(pid)
                
                # Emit event for notable app closures
                if self._is_notable_app(proc_info.name):
                    self.event_bus.emit(
                        EventType.PLUGIN_NOTIFICATION,
                        data={
                            'type': 'app_closed',
                            'app': proc_info.name,
                            'pid': pid,
                            'uptime': (datetime.now() - proc_info.started_at).total_seconds()
                        },
                        priority=EventPriority.LOW,
                        source='system_monitor'
                    )
        
        except Exception as e:
            logger.error(f"Process scan error: {e}")
    
    def _is_system_process(self, name: str) -> bool:
        """Check if process is a system process"""
        system_processes = {
            'System', 'svchost.exe', 'csrss.exe', 'lsass.exe', 'smss.exe',
            'winlogon.exe', 'services.exe', 'dwm.exe', 'explorer.exe',
            'RuntimeBroker.exe', 'SearchIndexer.exe', 'conhost.exe'
        }
        return name in system_processes
    
    def _is_notable_app(self, name: str) -> bool:
        """Check if app is notable enough to track"""
        notable_keywords = [
            'chrome', 'firefox', 'edge', 'brave',  # Browsers
            'code', 'pycharm', 'visual', 'sublime',  # IDEs
            'word', 'excel', 'powerpoint', 'outlook',  # Office
            'spotify', 'discord', 'slack', 'teams',  # Social/Media
            'notepad', 'photoshop', 'premiere',  # Tools
            'python', 'node', 'java', 'git'  # Dev tools
        ]
        
        name_lower = name.lower()
        return any(keyword in name_lower for keyword in notable_keywords)
    
    def _check_watched_files(self):
        """Check watched files for changes (simplified)"""
        # Full implementation would use watchdog library
        # For now, just a placeholder
        import os
        
        for path in list(self._watched_paths):
            try:
                if os.path.exists(path):
                    # Could check modification time here
                    pass
            except Exception as e:
                logger.error(f"Error checking {path}: {e}")
    
    def watch_file(self, filepath: str):
        """Add a file to watch list"""
        self._watched_paths.add(filepath)
        logger.info(f"Watching file: {filepath}")
    
    def unwatch_file(self, filepath: str):
        """Remove a file from watch list"""
        self._watched_paths.discard(filepath)
    
    def get_running_apps(self) -> List[str]:
        """Get list of currently running notable apps"""
        apps = set()
        
        for proc_info in self._known_processes.values():
            if self._is_notable_app(proc_info.name):
                apps.add(proc_info.name)
        
        return sorted(apps)
    
    def is_app_running(self, app_name: str) -> bool:
        """Check if a specific app is running"""
        app_name_lower = app_name.lower()
        
        for proc_info in self._known_processes.values():
            if app_name_lower in proc_info.name.lower():
                return True
        
        return False
    
    def get_system_snapshot(self) -> Dict[str, Any]:
        """Get current system state snapshot"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'running_apps': self.get_running_apps(),
                'total_processes': len(self._known_processes)
            }
        
        except Exception as e:
            logger.error(f"Failed to get system snapshot: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            'monitoring': self._monitoring,
            'tracked_processes': len(self._known_processes),
            'known_apps': len(self._known_app_names),
            'watched_files': len(self._watched_paths),
            'scan_interval': self._scan_interval
        }


# Global instance
_system_monitor = None


def get_system_monitor() -> SystemMonitor:
    """Get the global system monitor instance"""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor
