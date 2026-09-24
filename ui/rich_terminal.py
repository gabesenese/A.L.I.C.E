"""
Rich Terminal UI for A.L.I.C.E
Beautiful terminal interface with modern color scheme
"""

from contextlib import contextmanager
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from rich.live import Live
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich import box
from datetime import datetime
import re
import time


_MARKDOWN_LINE = re.compile(r"^\s*(#{1,6}\s|[-*>]\s|\d+\.\s|\|)", re.MULTILINE)
_CODE_SPAN = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)


def _looks_like_markdown(text: str) -> bool:
    """Markdown only when a line starts with a marker or there is a code fence.

    A mid-sentence "- " or "**" is prose, not structure.
    """
    return "```" in text or bool(_MARKDOWN_LINE.search(text))


def _protect_underscores(text: str) -> str:
    """Escape underscores outside code, so __init__.py is not rendered as bold "init".

    The model writes emphasis with asterisks; an underscore in her replies is
    nearly always part of a name.
    """
    parts = _CODE_SPAN.split(text)
    return "".join(part if i % 2 else part.replace("_", "\\_") for i, part in enumerate(parts))


class ThinkingStatus:
    """What the spinner shows: a phase, and the elapsed seconds once it is slow."""

    def __init__(self, style: str = "", phase: str = "thinking…", clock=time.monotonic):
        self._clock = clock
        self._started = clock()
        self._style = style
        self.phase = phase
        self._spinner = Spinner("dots2")

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def label(self) -> str:
        elapsed = self._clock() - self._started
        return self.phase if elapsed < 2 else f"{self.phase} {elapsed:.0f}s"

    def __rich__(self):
        self._spinner.update(text=Text(self.label(), style=self._style))
        return self._spinner


class RichTerminalUI:
    """Enhanced terminal UI using Rich library"""

    def __init__(self, user_name="User"):
        self.console = Console()
        self.user_name = user_name
        self.conversation_history = []

        # Futuristic tech color scheme - sleek and modern
        self.colors = {
            "user": "bright_blue",  # Electric blue for user messages
            "assistant": "bright_white",  # Bright white for Alice
            "accent": "grey70",  # Grey for accents/links (terminal feel)
            "success": "bright_green",  # Success messages
            "error": "bright_red",  # Errors
            "warning": "yellow1",  # Warnings
            "info": "grey50",  # Info/secondary text
            "border": "grey50",  # Grey panel borders
            "dim_border": "grey35",  # Subtle dark borders
        }

    def clear(self):
        """Clear the console"""
        self.console.clear()

    def _get_goal_line(self) -> str:
        """Return a formatted goal line from the top active goal, or empty string."""
        try:
            from ai.goals.goal_engine import get_goal_engine

            goal = get_goal_engine().top_goal()
            if not goal:
                return ""
            desc = goal.description[:60].rstrip()
            (goal.next_action or "").strip()[:50]
            accent = self.colors["accent"]
            self.colors["info"]
            return f"[{accent}][ {desc} ][/{accent}]"
        except Exception:
            return ""

    def show_welcome(self):
        """Display welcome banner"""
        self.clear()

        # ASCII Art - align as one block so "justify center" doesn't shift each line
        ascii_lines = [
            "    ___    __    ____  _______  ______",
            "   /   |  / /   /  _/ / ____/  / ____/",
            "  / /| | / /    / /  / /      / __/   ",
            " / ___ |/ /____/ /_ / /____  / /___   ",
            "/_/  |_/_____/___/ /_____/  /_____/   ",
        ]
        # Same width for every line so the block stays rectangular
        art_width = max(len(line) for line in ascii_lines)
        lines_padded = [line.ljust(art_width) for line in ascii_lines]
        # Center the whole block (one unit), not line-by-line
        try:
            width = getattr(self.console.size, "width", None) or getattr(self.console, "width", 80) or 80
        except Exception:
            width = 80
        margin = max(0, (width - art_width) // 2)
        centered_block = "\n".join(" " * margin + line for line in lines_padded)

        # Welcome panel with modern cyan borders
        welcome_panel = Panel(
            Text(centered_block, style="bright_white", justify="left"),
            border_style=self.colors["border"],
            box=box.ROUNDED,
        )
        self.console.print(welcome_panel)
        self.console.print()

        # Goal line — top active goal + next action
        goal_line = self._get_goal_line()
        if goal_line:
            self.console.print(goal_line, justify="center")
            self.console.print()

        # Info panel - sleek futuristic design
        current_time = datetime.now()
        # Only the date. The "System ready." line and the backronym made the
        # session open like a boot loader; she opens it herself now.
        info_text = f"[{self.colors['info']}]{current_time.strftime('%A, %B %d, %Y')}[/{self.colors['info']}]"

        info_panel = Panel(info_text, border_style=self.colors["dim_border"], box=box.MINIMAL)
        self.console.print(info_panel, justify="center")
        self.console.print()

    @contextmanager
    def thinking_spinner(self):
        """Spinner while a turn runs, with the seconds it has taken so far.

        A fixed "thinking…" looked the same at second 2 of a normal answer and
        at second 80 of a model that will time out; the counter tells them apart.
        """
        status = ThinkingStatus(style=self.colors["info"])
        with Live(status, console=self.console, transient=True, refresh_per_second=12):
            yield status

    def print_user_input(self, text):
        """Record the user's line. It is already on screen after the ❯ prompt,
        so echoing it again with a name and timestamp only made it a log."""
        self.conversation_history.append(("user", text))

    def print_assistant_response(self, text):
        """Print her reply the way she wrote it: no frame, no timestamp, no reflow.

        Replies used to go into a bordered panel titled "A.L.I.C.E" with a
        timestamp, after a reflow that split two or three sentences into
        one-sentence paragraphs. That read as a log record, not speech.
        """
        if not text:
            return
        text = str(text).strip()
        self.console.print()
        if _looks_like_markdown(text):
            try:
                self.console.print(Markdown(_protect_underscores(text)))
            except Exception:
                self.console.print(Text(text, overflow="fold"))
        else:
            self.console.print(Text(text, style=self.colors["assistant"], overflow="fold"))
        self.conversation_history.append(("assistant", text))
        self.console.print()

    def print_error(self, text):
        """Display error message"""
        self.console.print(f"[{self.colors['error']}]{text}[/{self.colors['error']}]")
        self.console.print()

    def print_info(self, text):
        """Display info message"""
        self.console.print(f"[{self.colors['info']}]{text}[/{self.colors['info']}]")

    def get_input(self):
        """Get user input with nice prompt"""
        try:
            user_input = Prompt.ask(
                f"\n[{self.colors['user']}]❯[/{self.colors['user']}]",
                default="",
                show_default=False,
            )
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return None

    def show_help(self):
        """Display help information"""
        help_text = f"""[bold {self.colors["accent"]}]Available Commands:[/bold {self.colors["accent"]}]

[{self.colors["warning"]}]General Commands:[/{self.colors["warning"]}]
  /help         - Show this help message
    /exit, /quit  - Exit A.L.I.C.E
  /clear        - Clear conversation history
  /save         - Save current state
  /status       - Show system status

[{self.colors["warning"]}]Memory & Context:[/{self.colors["warning"]}]
  /memory       - Show memory statistics
  /summary      - Get conversation summary
  /context      - Show current context
  /topics       - List conversation topics
  /entities     - Show tracked entities
    /relationships- Show entity relationships
    /mem-list     - List memories by type
    /mem-search   - Search memories by similarity
    /mem-delete   - Delete memory by ID
    /patterns     - Review proposed patterns

[{self.colors["warning"]}]Plugins:[/{self.colors["warning"]}]
  /plugins      - List available plugins
  /location     - Set or view your location

[{self.colors["warning"]}]Voice & Settings:[/{self.colors["warning"]}]
  /voice        - Toggle voice mode

[{self.colors["warning"]}]Learning & Feedback:[/{self.colors["warning"]}]
  /correct      - Correct my last response
  /feedback     - Rate my last response
  /learning     - Show learning statistics
    /realtime-status - Show live learning metrics
    /formulation  - Show formulation learning status
    /autolearn    - Show learning audit report

[{self.colors["warning"]}]Autonomous Mode:[/{self.colors["warning"]}]
    /autonomous <start|stop|pause|resume|status>
    /goals        - Show active and completed goals
"""
        panel = Panel(
            help_text,
            title=f"[bold {self.colors['accent']}]A.L.I.C.E Help[/bold {self.colors['accent']}]",
            border_style=self.colors["accent"],
            box=box.DOUBLE,
        )
        self.console.print(panel)
        self.console.print()
