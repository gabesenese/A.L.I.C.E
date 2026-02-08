"""
Development Mode for A.L.I.C.E
Auto-reloads when code changes are detected
"""

import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from threading import Thread, Event

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog for file monitoring...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ALICEReloader(FileSystemEventHandler):
    """Watches for file changes and triggers ALICE reload"""
    
    def __init__(self, restart_callback, debounce_seconds=2):
        super().__init__()
        self.restart_callback = restart_callback
        self.debounce_seconds = debounce_seconds
        self.last_reload_time = 0
        self.ignored_patterns = {
            '__pycache__',
            '.pyc',
            '.git',
            'data/',
            'memory/',
            'config/cred/',
            '.md',
            '.txt',
            '.json'
        }
    
    def should_ignore(self, path):
        """Check if file should be ignored"""
        path_str = str(path)
        
        # Ignore non-Python files
        if not path_str.endswith('.py'):
            return True
        
        # Ignore patterns
        for pattern in self.ignored_patterns:
            if pattern in path_str:
                return True
        
        return False
    
    def on_modified(self, event):
        """Handle file modification"""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        # Debounce - avoid multiple reloads for same change
        current_time = time.time()
        if current_time - self.last_reload_time < self.debounce_seconds:
            return
        
        self.last_reload_time = current_time
        
        # event.src_path can be relative or absolute depending on OS/watchdog
        src_path = Path(event.src_path)
        if not src_path.is_absolute():
            src_path = (Path.cwd() / src_path).resolve()
        try:
            file_path = src_path.relative_to(Path.cwd().resolve())
        except ValueError:
            file_path = Path(event.src_path)
        logger.info(f"Code changed: {file_path}")
        
        # Trigger reload
        self.restart_callback()


class ALICERunner:
    """Manages ALICE process with auto-reload"""

    def __init__(self, voice_enabled=False, model='llama3.1:8b', show_thinking=True, llm_policy='default'):
        self.voice_enabled = voice_enabled
        self.model = model
        self.show_thinking = show_thinking  # In dev mode, show A.L.I.C.E thinking steps
        self.llm_policy = llm_policy

        self.process = None
        self.should_run = Event()
        self.should_run.set()

        self.restart_requested = Event()
        self.restart_count = 0

    def build_command(self):
        """Build ALICE command"""
        cmd = [sys.executable, 'alice.py', '--model', self.model]

        if self.voice_enabled:
            cmd.append('--voice')

        # Add LLM policy flag
        if self.llm_policy != 'default':
            cmd.extend(['--llm-policy', self.llm_policy])

        # Dev mode: show A.L.I.C.E thinking (intent, plugins, verifier)
        if getattr(self, 'show_thinking', True):
            cmd.append('--debug')

        return cmd

    def start_alice(self):
        """Start ALICE process"""
        if self.process and self.process.poll() is None:
            logger.info("Stopping existing ALICE instance...")
            self.stop_alice()
            time.sleep(1)

        logger.info(f" Starting ALICE (restart #{self.restart_count})...")
        logger.info(f"   Model: {self.model}, Voice: {self.voice_enabled}")

        cmd = self.build_command()
        
        try:
            # Start in same terminal
            self.process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                stdin=sys.stdin,
                bufsize=0
            )
            
            self.restart_count += 1
            logger.info(" ALICE running")
            
        except Exception as e:
            logger.error(f"Failed to start ALICE: {e}")
    
    def stop_alice(self):
        """Stop ALICE process"""
        if self.process and self.process.poll() is None:
            try:
                # Try graceful shutdown first
                self.process.terminate()
                
                # Wait up to 3 seconds
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.process.kill()
                    self.process.wait()
                
                logger.info("✓ ALICE stopped")
                
            except Exception as e:
                logger.error(f"Error stopping ALICE: {e}")
        
        self.process = None
    
    def request_restart(self):
        """Request ALICE restart"""
        self.restart_requested.set()
    
    def run(self):
        """Main run loop"""
        self.start_alice()
        
        try:
            while self.should_run.is_set():
                # Check if process died
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.returncode
                    
                    if exit_code != 0:
                        logger.warning(f"ALICE crashed (exit code {exit_code})")
                        logger.info("Restarting in 3 seconds...")
                        time.sleep(3)
                        self.start_alice()
                
                # Check for restart request
                if self.restart_requested.is_set():
                    self.restart_requested.clear()
                    self.stop_alice()
                    time.sleep(1)
                    self.start_alice()
                
                time.sleep(0.5)
        
        finally:
            self.stop_alice()
    
    def shutdown(self):
        """Shutdown runner"""
        logger.info("Shutting down...")
        self.should_run.clear()
        self.stop_alice()


def run_dev_mode(voice_enabled=False, model='llama3.1:8b', watch=True, show_thinking=True, llm_policy='default'):
    """
    Run ALICE in development mode with auto-reload

    Args:
        voice_enabled: Enable voice
        model: LLM model
        watch: Enable file watching
        show_thinking: Show A.L.I.C.E thinking steps (intent, plugins, verifier)
        llm_policy: LLM policy mode (default/minimal/strict)
    """
    print("=" * 70)
    print("A.L.I.C.E - Development Mode")
    print("=" * 70)
    print(f"Voice: {'Enabled' if voice_enabled else 'Disabled'}")
    print(f"Model: {model}")
    print(f"LLM Policy: {llm_policy}")
    print(f"Auto-reload: {'Enabled' if watch else 'Disabled'}")
    print(f"Thinking steps: {'On (--debug)' if show_thinking else 'Off'}")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Create runner
    runner = ALICERunner(voice_enabled=voice_enabled, model=model, show_thinking=show_thinking, llm_policy=llm_policy)
    
    # Set up file watcher
    observer = None
    if watch:
        event_handler = ALICEReloader(restart_callback=runner.request_restart)
        observer = Observer()
        
        # Watch main directories
        watch_paths = ['ai/', 'app/', 'speech/', 'ui/', 'features/', 'self_learning/']
        watch_paths.append('.')  # Watch root for alice.py and other root files
        
        for path in watch_paths:
            if os.path.exists(path):
                observer.schedule(event_handler, path, recursive=True)
                logger.info(f" Watching: {path}")
        
        observer.start()
        logger.info("✓ File watcher started")
        print()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        logger.info("\n Shutdown signal received")
        runner.shutdown()
        if observer:
            observer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run in separate thread
    runner_thread = Thread(target=runner.run, daemon=True)
    runner_thread.start()
    
    # Keep main thread alive
    try:
        while runner_thread.is_alive():
            runner_thread.join(timeout=1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Keyboard interrupt")
        runner.shutdown()
    finally:
        if observer:
            observer.stop()
            observer.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run A.L.I.C.E in development mode with auto-reload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dev.py                    # Run with defaults (auto-reload)
  python dev.py --voice            # Enable voice mode
  python dev.py --no-watch         # Disable auto-reload
  python dev.py --model llama3.2:3b  # Use different model
        """
    )

    parser.add_argument(
        '--voice',
        action='store_true',
        help='Enable voice interaction'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1:8b',
        help='LLM model to use (default: llama3.1:8b)'
    )

    parser.add_argument(
        '--no-watch',
        action='store_true',
        help='Disable file watching (no auto-reload)'
    )

    parser.add_argument(
        '--no-thinking',
        action='store_true',
        help='Disable A.L.I.C.E thinking steps in dev mode'
    )

    parser.add_argument(
        '--llm-policy',
        type=str,
        choices=['default', 'minimal', 'strict'],
        default='default',
        help='LLM policy mode: minimal (patterns only), strict (no LLM), default (balanced)'
    )

    args = parser.parse_args()

    run_dev_mode(
        voice_enabled=args.voice,
        model=args.model,
        watch=not args.no_watch,
        show_thinking=not args.no_thinking,
        llm_policy=args.llm_policy
    )
