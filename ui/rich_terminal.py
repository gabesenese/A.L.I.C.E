"""
Rich Terminal UI for A.L.I.C.E
Beautiful terminal interface with proper rendering
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich import box
from datetime import datetime
import time


class RichTerminalUI:
    """Enhanced terminal UI using Rich library"""
    
    def __init__(self, user_name="User"):
        self.console = Console()
        self.user_name = user_name
        self.conversation_history = []
        
    def clear(self):
        """Clear the console"""
        self.console.clear()
    
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
        
        # Welcome panel - modern minimal design
        welcome_panel = Panel(
            Text(centered_block, style="bright_white", justify="left"),
            #title=f"[white]Welcome, {self.user_name}[/white]",
            border_style="dim white",
            box=box.ROUNDED
        )
        self.console.print(welcome_panel)
        self.console.print()
        
        # Greeting
        hour = datetime.now().hour
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Good evening"
        
        greeting_text = f"{time_greeting}, {self.user_name}!"
        self.console.print(greeting_text, style="white", justify="center")
        self.console.print()
        
        # Info panel - sleek modern design
        info_text = f"""[dim white]A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity[/dim white]
[dim]{datetime.now().strftime('%A, %B %d, %Y')} • {datetime.now().strftime('%I:%M %p')}[/dim]

Type [white]/help[/white] for available commands
"""
        
        info_panel = Panel(
            info_text,
            border_style="dim white",
            box=box.MINIMAL
        )
        self.console.print(info_panel, justify="center")
        self.console.print()
    
    def show_loading(self, message="Initializing A.L.I.C.E systems"):
        """Show loading animation"""
        with self.console.status(f"[bold cyan]{message}...", spinner="dots") as status:
            time.sleep(2)  # Simulated loading delay
    
    def print_user_input(self, text):
        """Display user input"""
        self.console.print(f"[blue]{self.user_name}:[/blue] {text}")
        self.conversation_history.append(("user", text))
    
    def print_assistant_response(self, text):
        """Display assistant response with nice formatting"""
        # Check if it's markdown-like content
        if any(marker in text for marker in ['**', '##', '- ', '1.']):
            md = Markdown(text)
            panel = Panel(
                md,
                title="[white]A.L.I.C.E[/white]",
                border_style="dim white",
                box=box.MINIMAL,
                padding=(0, 2)
            )
            self.console.print(panel)
        else:
            self.console.print(f"[white]A.L.I.C.E:[/white] {text}")
        
        self.conversation_history.append(("assistant", text))
        self.console.print()
    
    def print_error(self, text):
        """Display error message"""
        self.console.print(f"[red]ERROR:[/red] {text}")
        self.console.print()
    
    def print_info(self, text):
        """Display info message"""
        self.console.print(f"[dim]{text}[/dim]")
    
    def get_input(self):
        """Get user input with nice prompt"""
        try:
            user_input = Prompt.ask(f"\n[bold green]{self.user_name}[/bold green]")
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return None
    
    def show_help(self):
        """Display help information"""
        help_text = """[bold cyan]Available Commands:[/bold cyan]

[yellow]General Commands:[/yellow]
  /help         - Show this help message
  /clear        - Clear conversation history
  /save         - Save current state
  /status       - Show system status
  /exit, exit   - Exit A.L.I.C.E

[yellow]Memory & Context:[/yellow]
  /memory       - Show memory statistics
  /summary      - Get conversation summary
  /context      - Show current context
  /topics       - List conversation topics
  /entities     - Show tracked entities

[yellow]Plugins:[/yellow]
  /plugins      - List available plugins
  /location     - Set or view your location

[yellow]Voice & Settings:[/yellow]
  /voice        - Toggle voice mode

[yellow]Learning & Feedback:[/yellow]
  /correct      - Correct my last response
  /feedback     - Rate my last response
  /learning     - Show learning statistics
"""
        panel = Panel(
            help_text,
            title="[bold]A.L.I.C.E Help[/bold]",
            border_style="yellow",
            box=box.DOUBLE
        )
        self.console.print(panel)
        self.console.print()
    
    def show_goodbye(self):
        """Display goodbye message"""
        goodbye_panel = Panel(
            "[bold cyan]Goodbye! A.L.I.C.E shutting down...[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )
        self.console.print()
        self.console.print(goodbye_panel, justify="center")
        self.console.print()
