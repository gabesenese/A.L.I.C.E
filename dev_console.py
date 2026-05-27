#!/usr/bin/env python3
"""ALICE Dev Console — Rich terminal interface for the development workflow."""

import json
import msvcrt
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import requests as _requests

    _requests_ok = True
except ImportError:
    _requests_ok = False

console = Console()

PROJECT_ROOT = Path(__file__).parent
MODEL = os.environ.get("ALICE_OLLAMA_MODEL", "alice_ollama")
OLLAMA_HOST = os.environ.get("ALICE_OLLAMA_HOST", "http://localhost:11434")


# ── Status ────────────────────────────────────────────────────────────────────


class AliceStatus:
    """Live snapshot of project health for the status sidebar."""

    def __init__(self):
        self.ollama_online = False
        self.ollama_models: list[str] = []
        self.ruff_errors: int = -1
        self.errors_24h: int = 0
        self.memory_db_size: str = "—"
        self.refresh()

    def refresh(self):
        self.ollama_online, self.ollama_models = self._check_ollama()
        self.ruff_errors = self._ruff_errors()
        self.errors_24h = self._error_count_24h()
        self.memory_db_size = self._memory_db_size()

    def _check_ollama(self) -> tuple[bool, list[str]]:
        if not _requests_ok:
            return False, []
        try:
            resp = _requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return True, models
        except Exception:
            pass
        return False, []

    def _ruff_errors(self) -> int:
        try:
            result = subprocess.run(
                ["ruff", "check", ".", "--exit-zero", "--statistics"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            total = 0
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    total += int(parts[0])
            return total
        except Exception:
            return -1

    def _error_count_24h(self) -> int:
        log = PROJECT_ROOT / "logs" / "errors.json"
        if not log.exists():
            return 0
        try:
            cutoff = datetime.now() - timedelta(hours=24)
            count = 0
            with open(log, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp") or entry.get("ts") or entry.get("time", "")
                        if ts and datetime.fromisoformat(ts[:19]) >= cutoff:
                            count += 1
                    except Exception:
                        pass
            return count
        except Exception:
            return 0

    def _memory_db_size(self) -> str:
        db = PROJECT_ROOT / "data" / "memory" / "alice.db"
        if not db.exists():
            return "—"
        size = db.stat().st_size
        if size >= 1_048_576:
            return f"{size / 1_048_576:.1f} MB"
        if size >= 1024:
            return f"{size / 1024:.0f} KB"
        return f"{size} B"

    def panel(self) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", min_width=10)
        t.add_column(min_width=18)

        t.add_row(
            "Ollama",
            Text("● ONLINE", style="bold green") if self.ollama_online else Text("● OFFLINE", style="bold red"),
        )
        t.add_row("Model", Text(MODEL, style="cyan"))
        t.add_row("Memory DB", Text(self.memory_db_size))

        if self.ruff_errors == 0:
            ruff_val = Text("✓ clean", style="bold green")
        elif self.ruff_errors == -1:
            ruff_val = Text("—", style="dim")
        else:
            ruff_val = Text(f"⚠  {self.ruff_errors} errors", style="bold yellow")
        t.add_row("Ruff", ruff_val)

        err_val = (
            Text("0 in last 24h", style="green")
            if self.errors_24h == 0
            else Text(f"{self.errors_24h} in last 24h", style="yellow")
        )
        t.add_row("Errors", err_val)

        return Panel(t, title="[dim]System Status[/dim]", border_style="dim blue", padding=(1, 2))


# ── Input ─────────────────────────────────────────────────────────────────────


def getch() -> str:
    """Single-keypress via msvcrt — no Enter needed."""
    ch = msvcrt.getch()
    if ch == b"\x03":
        raise KeyboardInterrupt
    if ch in (b"\xe0", b"\x00"):
        msvcrt.getch()  # discard second byte of special keys
        return ""
    return ch.decode("utf-8", errors="ignore").lower()


def wait_for_key(msg: str = "  Press any key to return..."):
    console.print(f"\n[dim]{msg}[/dim]")
    try:
        msvcrt.getch()
    except KeyboardInterrupt:
        pass


# ── Command runner ─────────────────────────────────────────────────────────────


def run_cmd(cmd: str, title: str = "", pause: bool = True):
    """Run a shell command with Rich-formatted output, streaming live."""
    console.print()
    if title:
        console.rule(f"[bold]{title}[/bold]", style="dim blue")
    console.print(f"  [dim]$ {cmd}[/dim]\n")
    start = time.time()
    try:
        proc = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
        elapsed = time.time() - start
        console.print()
        if proc.returncode == 0:
            console.print(f"  [bold green]✓[/bold green]  Completed in {elapsed:.1f}s")
        else:
            console.print(f"  [bold red]✗[/bold red]  Exited {proc.returncode} in {elapsed:.1f}s")
    except KeyboardInterrupt:
        console.print("\n  [yellow]Interrupted[/yellow]")
    if pause:
        wait_for_key()


def run_sequence(steps: list[tuple[str, str]], pause: bool = True):
    """Run several commands in sequence, each with its own section header."""
    for title, cmd in steps:
        console.print()
        console.rule(f"[bold]{title}[/bold]", style="dim blue")
        console.print(f"  [dim]$ {cmd}[/dim]\n")
        start = time.time()
        try:
            proc = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)
            elapsed = time.time() - start
            badge = (
                "[bold green]✓[/bold green]"
                if proc.returncode == 0
                else f"[bold red]✗[/bold red]  exit {proc.returncode}"
            )
            console.print(f"\n  {badge}  {title} — {elapsed:.1f}s")
        except KeyboardInterrupt:
            console.print("\n  [yellow]Interrupted[/yellow]")
            break
    if pause:
        wait_for_key()


# ── Layout helpers ─────────────────────────────────────────────────────────────


def print_header(subtitle: str = ""):
    console.clear()
    body = "[bold white]A · L · I · C · E[/bold white]"
    if subtitle:
        body += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(body, style="blue", padding=(0, 4)))
    console.print()


def menu(
    options: list[tuple[str, str, str]],
    status: AliceStatus,
    title: str = "Menu",
    valid: str = "",
    back: str = "b",
) -> str:
    """Render two-column menu + status panel, return pressed key."""
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan", min_width=4)
    t.add_column(style="bold white", min_width=16)
    t.add_column(style="dim")
    for key, label, desc in options:
        t.add_row(key, label, desc)
    if back:
        t.add_row("", "", "")
        t.add_row(back.upper(), "Back", "")

    left = Panel(t, title=f"[dim]{title}[/dim]", border_style="dim blue", padding=(1, 2))
    console.print(Columns([left, status.panel()], equal=False, expand=True))
    console.print()
    console.print("  Select: ", end="")

    accepted = valid + back
    while True:
        ch = getch()
        if ch in accepted:
            console.print(ch.upper())
            return ch


# ── Pre-flight ─────────────────────────────────────────────────────────────────


def preflight_ollama(status: AliceStatus) -> bool:
    """Warn if Ollama is offline. Returns True = proceed, False = abort."""
    if status.ollama_online:
        return True
    console.print(
        Panel(
            "[bold yellow]⚠  Ollama appears to be offline.[/bold yellow]\n\n"
            "    ALICE can't respond without it. Start Ollama first,\n"
            "    or press [bold]P[/bold] to proceed anyway.",
            border_style="yellow",
            padding=(1, 4),
        )
    )
    console.print("  [bold](P)[/bold]roceed  [bold](A)[/bold]bort: ", end="")
    while True:
        ch = getch()
        if ch == "p":
            console.print("P")
            return True
        if ch in ("a", "b", "\x1b"):
            console.print("A")
            return False


# ── Model picker ───────────────────────────────────────────────────────────────


def pick_model(status: AliceStatus) -> str:
    """List Ollama models and return user selection."""
    models = status.ollama_models
    if not models:
        console.print("  [dim]No models found in Ollama. Type model name:[/dim]  ", end="")
        return input().strip() or MODEL

    console.print()
    console.rule("[dim]Available models[/dim]", style="dim blue")
    console.print()
    for i, m in enumerate(models, 1):
        tag = "  [dim](fine-tuned)[/dim]" if "alice" in m.lower() else ""
        console.print(f"  [bold cyan]{i}[/bold cyan]  {m}{tag}")
    console.print()
    console.print("  [dim]Enter number or name (blank = current):[/dim]  ", end="")
    choice = input().strip()
    if not choice:
        return MODEL
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
    return choice


# ── Submenus ───────────────────────────────────────────────────────────────────


def submenu_debug(status: AliceStatus):
    while True:
        print_header("Debug")
        opts = [
            ("1", "Standard", f"--model {MODEL} --debug"),
            ("2", "Full logs", "app.main (raw stdout, no UI)"),
            ("3", "Dev + debug", "auto-reload + --debug"),
            ("4", "Custom model", "select from Ollama list"),
            ("5", "Minimal", "--llm-policy minimal"),
            ("6", "Strict", "--llm-policy strict"),
            ("7", "Privacy", "--privacy-mode"),
            ("8", "Voice", "--debug --voice"),
        ]
        ch = menu(opts, status, title="Debug", valid="12345678")
        if ch == "b":
            return

        if not preflight_ollama(status):
            continue
        print_header("Debug")

        if ch == "1":
            run_cmd(f"python -m app.alice --model {MODEL} --debug", title="Debug — standard")
        elif ch == "2":
            run_cmd("python -m app.main", title="Debug — full logs")
        elif ch == "3":
            run_cmd("python app\\dev.py --debug", title="Debug — dev + debug")
        elif ch == "4":
            m = pick_model(status)
            run_cmd(f"python -m app.alice --model {m} --debug", title=f"Debug — model: {m}")
        elif ch == "5":
            run_cmd(
                f"python -m app.alice --model {MODEL} --debug --llm-policy minimal",
                title="Debug — minimal policy",
            )
        elif ch == "6":
            run_cmd(
                f"python -m app.alice --model {MODEL} --debug --llm-policy strict",
                title="Debug — strict policy",
            )
        elif ch == "7":
            run_cmd(
                f"python -m app.alice --model {MODEL} --debug --privacy-mode",
                title="Debug — privacy mode",
            )
        elif ch == "8":
            run_cmd(
                f"python -m app.alice --model {MODEL} --debug --voice",
                title="Debug — voice",
            )


def submenu_test(status: AliceStatus):
    while True:
        print_header("Test")
        opts = [
            ("1", "All suites", "unit + integration + e2e + golden"),
            ("2", "Unit", "tests/unit"),
            ("3", "Integration", "tests/integration"),
            ("4", "E2E", "tests/e2e"),
            ("5", "Golden", "tests/golden"),
            ("6", "Collect", "list all tests, no run"),
        ]
        ch = menu(opts, status, title="Test", valid="123456")
        if ch == "b":
            return
        print_header("Test")

        cmds = {
            "1": "python -m pytest tests/unit tests/integration tests/e2e tests/golden -v",
            "2": "python -m pytest tests/unit -v",
            "3": "python -m pytest tests/integration -v",
            "4": "python -m pytest tests/e2e -v",
            "5": "python -m pytest tests/golden -v",
            "6": "python -m pytest --collect-only -q",
        }
        run_cmd(cmds[ch], title=f"Test — option {ch}")


def submenu_quality(status: AliceStatus):
    while True:
        print_header("Quality")
        opts = [
            ("1", "Lint", "ruff check ."),
            ("2", "Format", "ruff format ."),
            ("3", "Fix", "auto-fix → format → re-lint"),
            ("4", "Full check", "lint then all tests"),
        ]
        ch = menu(opts, status, title="Quality", valid="1234")
        if ch == "b":
            return
        print_header("Quality")

        if ch == "1":
            run_cmd("ruff check .", title="Lint")
        elif ch == "2":
            run_cmd("ruff format .", title="Format")
        elif ch == "3":
            run_sequence(
                [
                    ("Auto-fix", "ruff check . --fix --unsafe-fixes --exit-zero"),
                    ("Format", "ruff format ."),
                    ("Re-lint", "ruff check ."),
                ]
            )
        elif ch == "4":
            run_sequence(
                [
                    ("Lint", "ruff check ."),
                    ("Tests", "python -m pytest tests/unit tests/integration tests/e2e tests/golden -v"),
                ]
            )

        status.ruff_errors = status._ruff_errors()


def submenu_diagnostics(status: AliceStatus):
    while True:
        print_header("Diagnostics")
        opts = [
            ("1", "Startup doctor", "tools/auditing/startup_doctor.py"),
            ("2", "Memory health", "scripts/check_memory_health.py"),
            ("3", "Benchmark", "core_benchmark_gate.py run"),
            ("4", "Recent logs", "tail logs/alice.log (last 30 lines)"),
            ("5", "Open logs dir", "explorer logs\\"),
        ]
        ch = menu(opts, status, title="Diagnostics", valid="12345")
        if ch == "b":
            return
        print_header("Diagnostics")

        if ch == "1":
            run_cmd("python tools/auditing/startup_doctor.py", title="Startup Doctor")
        elif ch == "2":
            run_cmd("python scripts/check_memory_health.py", title="Memory Health")
        elif ch == "3":
            run_cmd("python tools/auditing/core_benchmark_gate.py run", title="Benchmark")
        elif ch == "4":
            _show_log_tail()
        elif ch == "5":
            logs_dir = PROJECT_ROOT / "logs"
            if logs_dir.exists():
                subprocess.Popen(["explorer", str(logs_dir)])
                console.print("  [dim]Opened logs\\ in Explorer.[/dim]")
            else:
                console.print("  [red]logs\\ folder not found.[/red]")
            wait_for_key()


def _show_log_tail():
    log_file = PROJECT_ROOT / "logs" / "alice.log"
    if not log_file.exists():
        console.print(Panel("[dim]logs/alice.log not found.[/dim]", border_style="dim"))
        wait_for_key()
        return
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    body = Text("\n".join(lines[-30:]), style="dim white")
    console.print(Panel(body, title="[dim]logs/alice.log — last 30 lines[/dim]", border_style="dim blue"))
    wait_for_key()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    status = AliceStatus()

    while True:
        print_header()

        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold cyan", min_width=4)
        t.add_column(style="bold white", min_width=16)
        t.add_column(style="dim")
        main_opts = [
            ("1", "Run", "dev mode with auto-reload"),
            ("2", "Debug", "launch options"),
            ("3", "Test", "run test suites"),
            ("4", "Quality", "lint / format / fix"),
            ("5", "Diagnostics", "health checks & benchmarks"),
            ("6", "Shell", "interactive venv shell"),
            ("7", "Exit", ""),
        ]
        for key, label, desc in main_opts:
            t.add_row(key, label, desc)

        left = Panel(t, title="[dim]Menu[/dim]", border_style="dim blue", padding=(1, 2))
        console.print(Columns([left, status.panel()], equal=False, expand=True))
        console.print()
        console.print("  Select: ", end="")

        while True:
            ch = getch()
            if ch in "1234567":
                console.print(ch)
                break

        if ch == "1":
            if preflight_ollama(status):
                print_header("Run")
                run_cmd("python app\\dev.py", title="Run — auto-reload")
        elif ch == "2":
            submenu_debug(status)
        elif ch == "3":
            submenu_test(status)
        elif ch == "4":
            submenu_quality(status)
        elif ch == "5":
            submenu_diagnostics(status)
        elif ch == "6":
            print_header("Shell")
            console.print("  [dim]Interactive shell (venv active). Type [bold]exit[/bold] to return.[/dim]\n")
            subprocess.run("cmd /k", shell=True, cwd=PROJECT_ROOT)
        elif ch == "7":
            console.print("\n  [dim]Goodbye.[/dim]\n")
            break

        status.refresh()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n  [dim]Interrupted.[/dim]\n")
        sys.exit(0)
