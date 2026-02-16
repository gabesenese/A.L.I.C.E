"""
Rich Terminal UI for A.L.I.C.E
Beautiful terminal interface with modern color scheme
"""

from rich.console import Console
from rich.panel import Panel
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


class RichTerminalUI:
    """Enhanced terminal UI using Rich library"""

    def __init__(self, user_name="User"):
        self.console = Console()
        self.user_name = user_name
        self.conversation_history = []
        self.used_greetings = set()  # Track used greetings to avoid repeats

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

        # Time-based greeting variations (won't repeat in same session)
        self.greetings = {
            'early_morning': [  # 5-7am
                ("Rise and shine", "Early bird catches the worm!"),
                ("Good morning", "You're up bright and early!"),
                ("Morning", "Starting the day fresh?"),
                ("Hey there", "Ready to tackle the day?"),
            ],
            'morning': [  # 7-12pm
                ("Good morning", "Hope you're having a great start!"),
                ("Morning", "Ready for a productive day?"),
                ("Hi", "Beautiful morning, isn't it?"),
                ("Hey", "Coffee kicked in yet?"),
            ],
            'afternoon': [  # 12-5pm
                ("Good afternoon", "How's your day going?"),
                ("Hey", "Getting through the afternoon?"),
                ("Hi there", "Halfway through the day!"),
                ("Afternoon", "Hope you had a good lunch!"),
            ],
            'evening': [  # 5-9pm
                ("Good evening", "Winding down for the day?"),
                ("Evening", "How was your day?"),
                ("Hey", "Time to relax?"),
                ("Hi", "Almost time to unwind!"),
            ],
            'night': [  # 9pm-12am
                ("Good evening", "Working late tonight?"),
                ("Hey there", "Burning the midnight oil?"),
                ("Evening", "Night owl, are we?"),
                ("Hi", "Still going strong?"),
            ],
            'late_night': [  # 12am-5am
                ("Well hello", "Quite the late night session!"),
                ("Hey", "Can't sleep either?"),
                ("Hi there", "Pulling an all-nighter?"),
                ("Hello", "The quiet hours are the best, aren't they?"),
            ]
        }
        
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
        """Get a non-repeating greeting based on time of day"""
        period = self._get_time_period()
        available_greetings = self.greetings[period]

        # Filter out used greetings
        unused = [g for g in available_greetings if g not in self.used_greetings]

        # If all used, reset and start over
        if not unused:
            self.used_greetings.clear()
            unused = available_greetings

        # Pick a random unused greeting
        greeting = random.choice(unused)
        self.used_greetings.add(greeting)

        return greeting

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
        greeting, follow_up = self._get_greeting()
        greeting_text = f"{greeting}, {self.user_name}! {follow_up}"
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

    def print_user_input(self, text):
        """Display user input"""
        self.console.print(f"[{self.colors['user']}]{self.user_name}:[/{self.colors['user']}] {text}")
        self.conversation_history.append(("user", text))

    def print_assistant_response(self, text):
        """Display assistant response with nice formatting"""
        # Check if it's markdown-like content
        if any(marker in text for marker in ['**', '##', '- ', '1.']):
            md = Markdown(text)
            panel = Panel(
                md,
                title=f"[{self.colors['assistant']}]A.L.I.C.E[/{self.colors['assistant']}]",
                border_style=self.colors['dim_border'],
                box=box.MINIMAL,
                padding=(0, 2)
            )
            self.console.print(panel)
        else:
            self.console.print(f"[{self.colors['assistant']}]A.L.I.C.E:[/{self.colors['assistant']}] {text}")

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
            user_input = Prompt.ask(f"\n[{self.colors['user']}]{self.user_name}[/{self.colors['user']}]")
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return None

    def show_help(self):
        """Display help information"""
        help_text = f"""[bold {self.colors['accent']}]Available Commands:[/bold {self.colors['accent']}]

[{self.colors['warning']}]General Commands:[/{self.colors['warning']}]
  /help         - Show this help message
  /clear        - Clear conversation history
  /save         - Save current state
  /status       - Show system status
  /exit, exit   - Exit A.L.I.C.E

[{self.colors['warning']}]Memory & Context:[/{self.colors['warning']}]
  /memory       - Show memory statistics
  /summary      - Get conversation summary
  /context      - Show current context
  /topics       - List conversation topics
  /entities     - Show tracked entities

[{self.colors['warning']}]Plugins:[/{self.colors['warning']}]
  /plugins      - List available plugins
  /location     - Set or view your location

[{self.colors['warning']}]Voice & Settings:[/{self.colors['warning']}]
  /voice        - Toggle voice mode

[{self.colors['warning']}]Learning & Feedback:[/{self.colors['warning']}]
  /correct      - Correct my last response
  /feedback     - Rate my last response
  /learning     - Show learning statistics
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
