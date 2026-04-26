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
from rich.layout import Layout
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import box
from datetime import datetime
import time
import random
import re


class RichTerminalUI:
    """Enhanced terminal UI using Rich library"""

    def __init__(self, user_name="User"):
        self.console = Console()
        self.user_name = user_name
        self.conversation_history = []
        self.used_greetings = set()  # Track used greeting signatures to avoid repeats

        # Futuristic tech color scheme - sleek and modern
        self.colors = {
            'user': 'bright_blue',           # Electric blue for user messages
            'assistant': 'bright_white',      # Bright white for Alice
            'accent': 'grey70',               # Grey for accents/links (terminal feel)
            'success': 'bright_green',        # Success messages
            'error': 'bright_red',            # Errors
            'warning': 'yellow1',             # Warnings
            'info': 'grey50',                 # Info/secondary text
            'border': 'grey50',               # Grey panel borders
            'dim_border': 'grey35',           # Subtle dark borders
        }

        # Time-based greeting components. We combine these dynamically to avoid
        # limited repeated lines while keeping greeting tone time-aware.
        self.greeting_bank = {
            'early_morning': {
                "openers": [
                    "Good early morning, {name}.",
                    "Morning, {name}.",
                    "You are up early, {name}.",
                    "Hello, {name}.",
                ],
                "context": [
                    "This is a clean window to plan the day before noise kicks in.",
                    "Great time to lock one priority and execute it fully.",
                    "Quiet start like this is ideal for focused setup work.",
                    "If we align now, the rest of the day gets easier.",
                ],
            },
            'morning': {
                "openers": [
                    "Good morning, {name}.",
                    "Morning, {name}.",
                    "Hello, {name}.",
                    "Hi, {name}.",
                ],
                "context": [
                    "Let's set the top outcomes for today.",
                    "Good time to pick one high-impact task and move it forward.",
                    "We can map the day into clear steps before execution starts.",
                    "If you share your top priority, I can structure the first sprint.",
                ],
            },
            'afternoon': {
                "openers": [
                    "Good afternoon, {name}.",
                    "Afternoon, {name}.",
                    "Hey, {name}.",
                    "Hi, {name}.",
                ],
                "context": [
                    "Perfect checkpoint to re-prioritize and close the critical path.",
                    "We can recover momentum fast with one concrete next action.",
                    "This is a good slot to clear blockers and finish strong.",
                    "If context has shifted, we can replan in one pass.",
                ],
            },
            'evening': {
                "openers": [
                    "Good evening, {name}.",
                    "Evening, {name}.",
                    "Hello, {name}.",
                    "Hi, {name}.",
                ],
                "context": [
                    "Ideal time to wrap open loops and prepare tomorrow's handoff.",
                    "We can turn today's progress into a clean next-step plan.",
                    "If you're winding down, I can summarize and stage tomorrow's priorities.",
                    "A short review now can save time tomorrow morning.",
                ],
            },
            'night': {
                "openers": [
                    "Good evening, {name}.",
                    "Late session, {name}.",
                    "Still in motion, {name}.",
                    "Hello, {name}.",
                ],
                "context": [
                    "Let's keep this focused and move one thing to done.",
                    "Night sessions work best with tight scope and clear output.",
                    "I can keep this lean: one target, one plan, one execution pass.",
                    "If energy is low, we can prioritize only what matters now.",
                ],
            },
            'late_night': {
                "openers": [
                    "Late night, {name}.",
                    "Still online, {name}.",
                    "Hello, {name}.",
                    "Quiet hours, {name}.",
                ],
                "context": [
                    "Let's keep it minimal and high-value.",
                    "I can help you finish one important task and park the rest.",
                    "Best move now is a narrow objective with no distraction.",
                    "If you want, we can prepare a precise restart plan for tomorrow.",
                ],
            },
        }
        self.agentic_prompts = [
            "What outcome should we drive first?",
            "Share one priority and I'll turn it into the next actions.",
            "Want a quick status sweep and a concrete plan?",
            "Give me the target and I'll map the shortest path.",
            "Point me at the blocker and I'll propose the next move.",
            "If you name the goal, I'll stage execution steps now.",
        ]
        
    def clear(self):
        """Clear the console"""
        self.console.clear()

    def _get_time_period(self):
        """Get current time period for greeting"""
        hour = datetime.now().hour
        if 5 <= hour < 7:
            return 'early_morning'
        elif 7 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        elif 21 <= hour < 24:
            return 'night'
        else:  # 0-5am
            return 'late_night'

    def _get_greeting(self):
        """Get a non-repeating, time-aware greeting with agentic intent."""
        period = self._get_time_period()
        bank = self.greeting_bank.get(period, self.greeting_bank["afternoon"])
        combos = [
            (opener, context, prompt)
            for opener in bank["openers"]
            for context in bank["context"]
            for prompt in self.agentic_prompts
        ]

        unused = []
        for opener, context, prompt in combos:
            signature = (period, opener, context, prompt)
            if signature not in self.used_greetings:
                unused.append((opener, context, prompt, signature))

        if not unused:
            self.used_greetings = {
                sig for sig in self.used_greetings if sig[0] != period
            }
            for opener, context, prompt in combos:
                signature = (period, opener, context, prompt)
                unused.append((opener, context, prompt, signature))

        opener, context, prompt, signature = random.choice(unused)
        self.used_greetings.add(signature)

        opener_text = opener.format(name=self.user_name)
        return f"{opener_text} {context} {prompt}"

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
            border_style=self.colors['border'],
            box=box.ROUNDED
        )
        self.console.print(welcome_panel)
        self.console.print()

        # Get dynamic greeting
        greeting_text = self._get_greeting()
        self.console.print(greeting_text, style=self.colors['assistant'], justify="center")
        self.console.print()

        # Info panel - sleek futuristic design
        current_time = datetime.now()
        info_text = f"""[{self.colors['accent']}]A.L.I.C.E[/{self.colors['accent']}] [{self.colors['info']}]>>[/{self.colors['info']}] Advanced Linguistic Intelligence Computer Entity
[{self.colors['info']}]{current_time.strftime('%A, %B %d, %Y')} | {current_time.strftime('%H:%M:%S')}[/{self.colors['info']}]

System ready. Type [{self.colors['accent']}]/help[/{self.colors['accent']}] for available commands
"""

        info_panel = Panel(
            info_text,
            border_style=self.colors['dim_border'],
            box=box.MINIMAL
        )
        self.console.print(info_panel, justify="center")
        self.console.print()

    def show_loading(self, message="Initializing A.L.I.C.E systems"):
        """Show loading progress bar with percentage"""
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40, style=self.colors['accent'], complete_style=self.colors['success']),
            TextColumn(f"[{self.colors['success']}]{{task.percentage:>3.0f}}%"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"[{self.colors['accent']}]{message}", total=100)

            # Simulate loading progress (0-100%)
            for i in range(100):
                progress.update(task, advance=1)
                time.sleep(0.02)  # 2 seconds total (100 steps * 0.02s)

    @contextmanager
    def thinking_spinner(self):
        """Context manager that shows an animated spinner while ALICE is processing."""
        spinner = Spinner("dots2", text=f"[{self.colors['info']}]thinking…[/{self.colors['info']}]")
        with Live(spinner, console=self.console, transient=True, refresh_per_second=12):
            yield

    def print_user_input(self, text):
        """Display user input"""
        ts = datetime.now().strftime("%H:%M")
        self.console.print(
            f"[{self.colors['user']}]❯ {self.user_name}[/{self.colors['user']}]  "
            f"[{self.colors['dim_border']}]{ts}[/{self.colors['dim_border']}]\n  {text}"
        )
        self.conversation_history.append(("user", text))

    def _format_assistant_terminal_text(self, text: str) -> str:
        """Apply display-only spacing for long plain-text replies in terminal UI."""
        value = str(text or "").strip()
        if not value:
            return value

        if "\n" in value:
            return value

        md_markers = ("**", "##", "```", "| ", "- ", "* ")
        if any(marker in value for marker in md_markers):
            return value

        if re.search(r"^\s*\d+\.\s+", value):
            return value

        normalized = re.sub(r"\s+", " ", value).strip()

        # Structured one-line outputs often contain repeated section labels.
        # Reflow those into readable paragraph blocks before fallback sentence spacing.
        section_labels = (
            "Project Concept",
            "Objective",
            "Project Direction",
            "Direction",
            "Domain",
            "Key Features",
            "Features",
            "Next Steps",
            "Deliverables",
            "Summary",
            "Goals",
            "Goal",
            "Scope",
            "Timeline",
            "Risks",
            "Approach",
        )
        heading_pattern = re.compile(
            r"\b(" + "|".join(re.escape(label) for label in section_labels) + r"):"
        )
        heading_matches = list(heading_pattern.finditer(normalized))
        if len(heading_matches) >= 2:
            structured = normalized
            structured = heading_pattern.sub(r"\n\n\1:", structured).strip()
            structured = re.sub(r"\n{3,}", "\n\n", structured)

            # If headings are followed by inline numbered lists, put each item on its own line.
            structured = re.sub(r"\s+(?=\d+\.\s+)", "\n", structured)
            structured = re.sub(r"\n{3,}", "\n\n", structured).strip()

            if "\n" in structured:
                return structured

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
        if len(sentences) < 2:
            return value

        # Keep short lead-ins attached to the next sentence for smoother reading.
        if len(sentences) >= 2:
            lead_tokens = re.findall(r"\b\w+\b", sentences[0])
            if len(lead_tokens) <= 2:
                sentences = [f"{sentences[0]} {sentences[1]}".strip(), *sentences[2:]]

        if len(normalized) < 120:
            return value

        return "\n\n".join(sentences)

    def print_assistant_response(self, text):
        """Display assistant response with consistent, clean formatting."""
        if not text:
            return
        text = text.strip()
        display_text = self._format_assistant_terminal_text(text)

        # Detect content that benefits from Markdown rendering
        _MD_MARKERS = ('**', '##', '```', '| ', '- ', '* ')
        is_multiline = '\n' in display_text
        has_markdown = any(m in display_text for m in _MD_MARKERS)
        # Numbered list: lines starting with digit+dot (e.g. "1. Item")
        has_numbered = any(
            line.lstrip().startswith(tuple(f'{i}.' for i in range(1, 20)))
            for line in display_text.splitlines()
        )

        if has_markdown or has_numbered:
            try:
                content = Markdown(display_text)
            except Exception:
                content = Text(display_text)
        elif is_multiline:
            content = Text(display_text)

        if is_multiline or has_markdown or has_numbered:
            ts = datetime.now().strftime("%H:%M")
            try:
                panel_width = max(40, self.console.width - 2)
            except Exception:
                panel_width = 98
            panel = Panel(
                content,
                title=f"[{self.colors['assistant']}]A.L.I.C.E[/{self.colors['assistant']}]",
                subtitle=f"[{self.colors['dim_border']}]{ts}[/{self.colors['dim_border']}]",
                border_style=self.colors['dim_border'],
                box=box.ROUNDED,
                padding=(0, 2),
                width=panel_width,
            )
            self.console.print()
            self.console.print(panel)
        else:
            self.console.print()
            self.console.print(f"[{self.colors['assistant']}]A.L.I.C.E:[/{self.colors['assistant']}]", end=" ")
            self.console.print(Text(display_text, overflow="fold"))

        self.conversation_history.append(("assistant", text))
        self.console.print()

    def print_error(self, text):
        """Display error message"""
        self.console.print(f"[{self.colors['error']}]ERROR:[/{self.colors['error']}] {text}")
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
        help_text = f"""[bold {self.colors['accent']}]Available Commands:[/bold {self.colors['accent']}]

[{self.colors['warning']}]General Commands:[/{self.colors['warning']}]
  /help         - Show this help message
    /exit, /quit  - Exit A.L.I.C.E
  /clear        - Clear conversation history
  /save         - Save current state
  /status       - Show system status

[{self.colors['warning']}]Memory & Context:[/{self.colors['warning']}]
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

[{self.colors['warning']}]Plugins:[/{self.colors['warning']}]
  /plugins      - List available plugins
  /location     - Set or view your location

[{self.colors['warning']}]Voice & Settings:[/{self.colors['warning']}]
  /voice        - Toggle voice mode

[{self.colors['warning']}]Learning & Feedback:[/{self.colors['warning']}]
  /correct      - Correct my last response
  /feedback     - Rate my last response
  /learning     - Show learning statistics
    /realtime-status - Show live learning metrics
    /formulation  - Show formulation learning status
    /autolearn    - Show learning audit report

[{self.colors['warning']}]Autonomous Mode:[/{self.colors['warning']}]
    /autonomous <start|stop|pause|resume|status>
    /goals        - Show active and completed goals
"""
        panel = Panel(
            help_text,
            title=f"[bold {self.colors['accent']}]A.L.I.C.E Help[/bold {self.colors['accent']}]",
            border_style=self.colors['accent'],
            box=box.DOUBLE
        )
        self.console.print(panel)
        self.console.print()

    def show_goodbye(self):
        """Display goodbye message"""
        goodbye_panel = Panel(
            f"[{self.colors['accent']}]Goodbye! A.L.I.C.E shutting down...[/{self.colors['accent']}]",
            border_style=self.colors['border'],
            box=box.ROUNDED
        )
        self.console.print()
        self.console.print(goodbye_panel, justify="center")
        self.console.print()
