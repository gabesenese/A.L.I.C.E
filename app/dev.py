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

    def __init__(self, restart_callback, debounce_seconds=1.0):
        super().__init__()
        self.restart_callback = restart_callback
        self.debounce_seconds = debounce_seconds
        self.last_reload_time = 0
        self.startup_time = time.time()
        self.startup_grace_period = 1.5  # Reduced from 3.0 to 1.5 seconds

        self.ignored_patterns = {
            '__pycache__',
            '.pyc',
            '.git',
            '.pytest_cache',
            '.venv',
            'venv',
            'env',
            '.swp',
            '.tmp',
            '~'
        }
        self.ignored_paths = {
            'data/',
            'memory/',
            'config/cred/',
            'scripts/data/'
        }

        # Track file modification times to detect real changes
        self.file_mtimes = {}
        print("[DEV] File watcher initialized - watching for .py changes")
        logger.debug("File watcher initialized with startup grace period")

    def should_ignore(self, path):
        """Check if file should be ignored"""
        path_str = str(path).replace('\\', '/')

        # Ignore non-Python files
        if not path_str.endswith('.py'):
            logger.debug(f"Ignoring non-Python file: {path_str}")
            return True

        # Ignore pattern matches
        for pattern in self.ignored_patterns:
            if pattern in path_str:
                logger.debug(f"Ignoring file matching pattern '{pattern}': {path_str}")
                return True

        # Ignore specific paths
        for ignored_path in self.ignored_paths:
            if ignored_path in path_str:
                logger.debug(f"Ignoring file in path '{ignored_path}': {path_str}")
                return True

        return False

    def file_changed(self, path, is_move=False):
        """Check if file actually changed based on modification time"""
        try:
            current_mtime = os.path.getmtime(path)
            path_str = str(path)

            # First time seeing this file
            if path_str not in self.file_mtimes:
                self.file_mtimes[path_str] = current_mtime
                logger.debug(f"First time tracking file: {path_str}")
                # For moved files (atomic saves), treat as changed even if first time
                if is_move:
                    logger.debug(f"File moved (atomic save) - treating as changed")
                    return True
                return False  # Don't trigger on first observation for regular modifications

            # Check if modification time changed
            if current_mtime != self.file_mtimes[path_str]:
                logger.debug(f"File modification detected: {path_str} (mtime changed)")
                self.file_mtimes[path_str] = current_mtime
                return True

            logger.debug(f"File unchanged: {path_str} (mtime same)")
            return False

        except OSError as e:
            logger.debug(f"Error checking file modification: {e}")
            return False

    def on_any_event(self, event):
        """Debug: log all events"""
        logger.debug(f"File event: {event.event_type} - {event.src_path}")

    def on_modified(self, event):
        """Handle file modification"""
        if event.is_directory:
            return

        if self.should_ignore(event.src_path):
            return

        # Grace period - ignore events during startup
        time_since_startup = time.time() - self.startup_time
        if time_since_startup < self.startup_grace_period:
            logger.debug(f"Ignoring event during startup grace period ({time_since_startup:.2f}s < {self.startup_grace_period}s)")
            return

        # Check if file actually changed
        if not self.file_changed(event.src_path):
            logger.debug(f"Ignoring event - file didn't actually change")
            return

        # Debounce - avoid multiple reloads for same change
        current_time = time.time()
        time_since_last = current_time - self.last_reload_time
        if time_since_last < self.debounce_seconds:
            logger.debug(f"Debouncing reload (last reload {time_since_last:.2f}s ago)")
            return

        self.last_reload_time = current_time

        # Get relative path for display
        src_path = Path(event.src_path)
        try:
            if not src_path.is_absolute():
                src_path = (Path.cwd() / src_path).resolve()
            file_path = src_path.relative_to(Path.cwd().resolve())
        except (ValueError, OSError):
            file_path = src_path.name

        print(f"\n[DEV] Code changed: {file_path}")
        print("[DEV] Reloading A.L.I.C.E...\n")
        logger.info(f"Code changed: {file_path}")
        logger.info("Reloading A.L.I.C.E...")

        # Trigger reload
        self.restart_callback()

    def on_created(self, event):
        """Handle file creation"""
        self.on_modified(event)

    def on_moved(self, event):
        """Handle file moves (atomic saves on Windows)"""
        if event.is_directory:
            return

        # On Windows, editors often save by creating temp file then renaming to original
        # The dest_path is the final Python file location
        if hasattr(event, 'dest_path'):
            dest_path = event.dest_path

            # Only process .py files
            if not dest_path.endswith('.py'):
                logger.debug(f"Ignoring moved non-Python file: {dest_path}")
                return

            if self.should_ignore(dest_path):
                return

            # Use same grace period and change detection logic
            time_since_startup = time.time() - self.startup_time
            if time_since_startup < self.startup_grace_period:
                logger.debug(f"Ignoring move event during startup grace period")
                return

            # Check if file actually changed
            if not self.file_changed(dest_path, is_move=True):
                logger.debug(f"Ignoring move - file didn't actually change")
                return

            # Debounce
            current_time = time.time()
            time_since_last = current_time - self.last_reload_time
            if time_since_last < self.debounce_seconds:
                logger.debug(f"Debouncing reload (last reload {time_since_last:.2f}s ago)")
                return

            self.last_reload_time = current_time

            # Get relative path for display
            src_path = Path(dest_path)
            try:
                if not src_path.is_absolute():
                    src_path = (Path.cwd() / src_path).resolve()
                file_path = src_path.relative_to(Path.cwd().resolve())
            except (ValueError, OSError):
                file_path = src_path.name

            print(f"\n[DEV] Code changed: {file_path}")
            print("[DEV] Reloading A.L.I.C.E...\n")
            logger.info(f"Code changed (moved): {file_path}")
            logger.info("Reloading A.L.I.C.E...")

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
        # Get the path to alice.py (in the app directory)
        alice_script = Path(__file__).parent / 'alice.py'
        cmd = [sys.executable, str(alice_script), '--model', self.model]

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

        logger.info(f"Starting ALICE (restart #{self.restart_count})...")
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
            logger.info("ALICE running")

        except Exception as e:
            logger.error(f"Failed to start ALICE: {e}")
    
    def stop_alice(self):
        """Stop ALICE process"""
        if self.process and self.process.poll() is None:
            try:
                logger.debug("Attempting to stop A.L.I.C.E process...")

                # Try graceful shutdown first
                self.process.terminate()
                logger.debug("Sent terminate signal")

                # Wait up to 3 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=3)
                    logger.debug("Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    logger.debug("Timeout expired, force killing...")
                    self.process.kill()
                    try:
                        self.process.wait(timeout=1)
                        logger.debug("Process killed")
                    except subprocess.TimeoutExpired:
                        logger.error("Failed to kill process even after kill signal")

                logger.debug("A.L.I.C.E stopped")

            except Exception as e:
                logger.error(f"Error stopping ALICE: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        self.process = None
    
    def request_restart(self):
        """Request ALICE restart"""
        logger.debug("Restart requested via file watcher")
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
                    logger.debug("Processing restart request...")
                    self.restart_requested.clear()
                    logger.info("Stopping for reload...")
                    self.stop_alice()
                    time.sleep(0.5)
                    self.start_alice()

                time.sleep(0.1)  # Faster polling for quicker response

        finally:
            self.stop_alice()
    
    def shutdown(self):
        """Shutdown runner"""
        logger.info("Shutting down...")
        self.should_run.clear()
        self.stop_alice()


def run_dev_mode(voice_enabled=False, model='llama3.1:8b', watch=True, show_thinking=True, llm_policy='default', debug=False):
    """
    Run ALICE in development mode with auto-reload

    Args:
        voice_enabled: Enable voice
        model: LLM model
        watch: Enable file watching
        show_thinking: Show A.L.I.C.E thinking steps (intent, plugins, verifier)
        llm_policy: LLM policy mode (default/minimal/strict)
        debug: Enable debug logging for file watcher
    """
    # Set logging level based on debug flag
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    print("=" * 70)
    print("A.L.I.C.E - Development Mode")
    print("=" * 70)
    print(f"Voice: {'Enabled' if voice_enabled else 'Disabled'}")
    print(f"Model: {model}")
    print(f"LLM Policy: {llm_policy}")
    print(f"Auto-reload: {'Enabled' if watch else 'Disabled'}")
    print(f"Thinking steps: {'On (--debug)' if show_thinking else 'Off'}")
    print(f"Debug logging: {'On' if debug else 'Off'}")
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

        # Get project root (one level up from app/)
        project_root = Path(__file__).parent.parent

        # Watch main directories (relative to project root)
        watch_dirs = ['ai', 'app', 'speech', 'ui', 'features', 'self_learning', 'plugins']

        logger.info("Setting up file watcher...")
        logger.info(f"Project root: {project_root}")
        print(f"\n[DEV]  Watching directories for .py file changes:")
        watched_count = 0
        for dir_name in watch_dirs:
            watch_path = project_root / dir_name
            if watch_path.exists():
                observer.schedule(event_handler, str(watch_path), recursive=True)
                print(f"[DEV]     {dir_name}/")
                logger.info(f"   Watching: {dir_name}/")
                watched_count += 1
            else:
                logger.debug(f"   Skipping (not found): {dir_name}/")

        if watched_count > 0:
            observer.start()
            print(f"[DEV]  File watcher active ({watched_count} directories)")
            print(f"[DEV]     Auto-reload enabled - edit any .py file to trigger reload")
            print(f"[DEV]      Grace period: {event_handler.startup_grace_period}s after startup\n")
            logger.info(f"File watcher active ({watched_count} directories)")
            logger.info(f"   Auto-reload enabled for .py files")
            logger.info(f"   Startup grace period: {event_handler.startup_grace_period}s")
        else:
            logger.warning("No directories found to watch!")
            observer = None
        print()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        logger.info("\nShutdown signal received")
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
        logger.info("\nKeyboard interrupt")
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

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging for file watcher and reload diagnostics'
    )

    args = parser.parse_args()

    run_dev_mode(
        voice_enabled=args.voice,
        model=args.model,
        watch=not args.no_watch,
        show_thinking=not args.no_thinking,
        llm_policy=args.llm_policy,
        debug=args.debug
    )
