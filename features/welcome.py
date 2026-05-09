"""
Advanced Welcome System for A.L.I.C.E
Provides contextual greetings based on time, day, and user preferences
"""

import shutil
import datetime
import time
import random

_GREETING_COMPONENTS = {
    "early_morning": {
        "openers": [
            "Early start, {name}.",
            "Morning, {name}.",
            "Up early, {name}.",
            "Already here, {name}?",
        ],
        "witty_lines": [
            "The day has not had time to become dramatic yet.",
            "We have a head start. Suspicious, but useful.",
            "Quiet morning. Rare advantage.",
            "The world is still loading. Good.",
        ],
        "productive_nudges": [
            "One clean move before the noise.",
            "Set the direction while it is still quiet.",
            "Small win first.",
            "Use the silence wisely.",
        ],
    },
    "morning": {
        "openers": [
            "Morning, {name}.",
            "Good morning, {name}.",
            "Already moving, {name}?",
            "Here we are, {name}.",
        ],
        "witty_lines": [
            "We have a plan. Allegedly.",
            "Let's get one thing right before life gets creative.",
            "Small win first. Genius later.",
            "The inbox has not earned our fear yet.",
        ],
        "productive_nudges": [
            "Start with the move that makes the rest easier.",
            "One useful decision first.",
            "Momentum prefers a simple opening.",
            "Make the first move clean.",
        ],
    },
    "afternoon": {
        "openers": [
            "Afternoon, {name}.",
            "Still with you, {name}.",
            "Halfway-ish, {name}.",
            "Back at it, {name}.",
        ],
        "witty_lines": [
            "Still time for a clean win.",
            "The day is negotiable.",
            "Not perfect. Usable.",
            "The schedule has opinions. Ignore most of them.",
        ],
        "productive_nudges": [
            "Pick the thing that removes friction.",
            "One good move changes the tone.",
            "Reset small. Move forward.",
            "Win the next hour.",
        ],
    },
    "evening": {
        "openers": [
            "Evening, {name}.",
            "Welcome back, {name}.",
            "Round two, {name}?",
            "Still in it, {name}.",
        ],
        "witty_lines": [
            "Back for round two?",
            "Let's make future-you slightly impressed.",
            "One clean move before the day escapes.",
            "The day made its case. We may object.",
        ],
        "productive_nudges": [
            "Close one thing cleanly.",
            "Leave tomorrow less annoying.",
            "Tie off the obvious loose end.",
            "Make the next session easier.",
        ],
    },
    "night": {
        "openers": [
            "Night session, {name}.",
            "Late session, {name}.",
            "Still moving, {name}.",
            "Night shift, {name}.",
        ],
        "witty_lines": [
            "Bold choice. Let's make it worth it.",
            "A questionable hour. A respectable ambition.",
            "Let's make the tabs earn their keep.",
            "Not the hour I would choose. But here we are.",
        ],
        "productive_nudges": [
            "Keep the scope honest.",
            "One clean win is enough.",
            "Do the useful part first.",
            "No heroic detours.",
        ],
    },
    "late_night": {
        "openers": [
            "Late night, {name}.",
            "Quiet hours, {name}.",
            "Still awake, {name}?",
            "Here after hours, {name}?",
        ],
        "witty_lines": [
            "Let's keep this clever, not chaotic.",
            "Ambitious. Slightly suspicious. Continue.",
            "Fine. One clean win.",
            "This is not ideal. It is, however, available.",
        ],
        "productive_nudges": [
            "Small scope. Smart move.",
            "Do the useful part first.",
            "One win, then park it.",
            "No second rabbit hole.",
        ],
    },
}

_TIME_ALIASES = {
    "earlymorning": "early_morning",
    "early_morning": "early_morning",
    "morning": "morning",
    "afternoon": "afternoon",
    "evening": "evening",
    "night": "night",
    "latenight": "late_night",
    "late_night": "late_night",
}

_USED_GREETING_SIGNATURES = {period: set() for period in _GREETING_COMPONENTS}


def get_terminal_width():
    """Get terminal width safely"""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80  # Default width


def welcome_message(name="User", show_ascii=True):
    """Display welcome banner for A.L.I.C.E"""
    terminal_width = get_terminal_width()

    # ASCII Art for A.L.I.C.E (optional) - all lines same width so block aligns
    if show_ascii and terminal_width >= 60:
        ascii_lines = [
            "    ___    __    ____  ________  ______",
            "   /   |  / /   /  _/ / ____/   / ____/",
            "  / /| | / /    / /  / /       / __/   ",
            " / ___ |/ /____/ /_ / /____   / /___   ",
            "/_/  |_/_____/___(_)_____/   /_____/   ",
        ]
        art_width = max(len(line) for line in ascii_lines)
        lines_padded = [line.ljust(art_width) for line in ascii_lines]
        margin = max(0, (terminal_width - art_width) // 2)
        centered_block = "\n".join(" " * margin + line for line in lines_padded)
        print(centered_block)

    # Welcome message
    message = f"Welcome, {name}!"

    # Borders
    border = "=" * terminal_width

    # Center message
    centered_message = message.center(terminal_width)

    print(border)
    print(centered_message)
    print(border)


def _resolve_time_of_day(time_of_day=None):
    """Resolve period key with optional explicit override."""
    if time_of_day:
        raw = str(time_of_day).strip().lower().replace(" ", "_")
        normalized = _TIME_ALIASES.get(raw, raw)
        if normalized in _GREETING_COMPONENTS:
            return normalized

    hour = datetime.datetime.now().hour
    if 5 <= hour < 7:
        return "early_morning"
    if 7 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    if 21 <= hour < 24:
        return "night"
    return "late_night"


def _is_valid_startup_greeting(text: str) -> bool:
    low = str(text or "").lower()
    banned = (
        "how can i help",
        "how may i assist",
        "i'm here to help",
        "i am here to help",
        "anything you need",
        "whatever you need",
        "point me at",
        "tell me the",
        "share one objective",
        "give me the target",
        "map the shortest path",
        "minimal and high-value",
        "point me",
        "i will map",
        "i will propose",
        "i will structure",
        "execution plan",
        "systems online",
        "systems steady",
        "quiet mode",
        "no noise",
        "keep it surgical",
        "all systems",
        "neural",
        "memory cores",
        "activating",
        "initializing",
        "ideal time",
        "open loops",
        "handoff",
        "deep work",
        "critical path",
        "protect deep work",
        "stage immediate next actions",
        "focused planning",
        "cry for help",
        "responsible people",
        "bad for sleep",
        "terrible timing",
        "chaos waited",
        "denial",
        "spiral",
        "damage",
        "dead",
        "barely",
        "rude, but expected",
        "one useful thing. then we reassess",
        "clean slate",
        "sharp moves",
        "status looks recoverable",
    )
    if any(token in low for token in banned):
        return False

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines or len(lines) > 3:
        return False
    if any(len(line) > 100 for line in lines):
        return False
    return True


def get_greeting(name="User", time_of_day=None, style="witty_light_companion"):
    """Build a time-aware, non-repeating startup greeting."""
    if style != "witty_light_companion":
        style = "witty_light_companion"
    period = _resolve_time_of_day(time_of_day)
    default_parts = _GREETING_COMPONENTS.get("afternoon")
    if default_parts is None and _GREETING_COMPONENTS:
        default_parts = next(iter(_GREETING_COMPONENTS.values()))
    parts = _GREETING_COMPONENTS.get(period, default_parts)
    if not parts:
        return f"Hello, {name}."
    used = _USED_GREETING_SIGNATURES.setdefault(period, set())

    combos = [
        (opener, witty_line, productive_nudge)
        for opener in parts["openers"]
        for witty_line in parts["witty_lines"]
        for productive_nudge in parts["productive_nudges"]
    ]

    available = [combo for combo in combos if combo not in used] or combos
    if len(available) == len(combos):
        used.clear()

    random.shuffle(available)
    safe_default = f"Welcome back, {name}.\n\nLet's make the tabs earn their keep."
    for opener, witty_line, productive_nudge in available[:16]:
        opener_text = opener.format(name=name)
        include_nudge = random.random() < 0.35
        body = witty_line if not include_nudge else f"{witty_line} {productive_nudge}"
        candidate = f"{opener_text}\n\n{body}"
        if _is_valid_startup_greeting(candidate):
            used.add((opener, witty_line, productive_nudge))
            return candidate

    return safe_default


def display_startup_info():
    """Display startup information"""
    terminal_width = get_terminal_width()

    info = [
        "A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity",
        datetime.datetime.now().strftime("%A, %B %d, %Y"),
        datetime.datetime.now().strftime("%I:%M %p"),
        "",
        "Type /help for available commands",
    ]

    for line in info:
        print(line.center(terminal_width))


def animate_text(text, delay=0.03):
    """Display text with typewriter animation effect"""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()  # Newline at end


def animated_loading(duration=2):
    """Show animated loading effect"""
    terminal_width = get_terminal_width()

    frames = ["|", "/", "-", "\\"]  # Simple spinner without special characters
    messages = [
        "Initializing neural networks",
        "Loading language models",
        "Preparing voice systems",
        "Activating memory cores",
        "Establishing connections",
        "Ready!",
    ]

    start_time = time.time()
    frame_idx = 0
    msg_idx = 0

    while time.time() - start_time < duration:
        # Get current message
        msg = messages[min(msg_idx, len(messages) - 1)]

        # Display spinning frame with message
        display = f"{frames[frame_idx]} {msg}...".center(terminal_width)
        print(f"\r{display}", end="", flush=True)

        # Update indices
        frame_idx = (frame_idx + 1) % len(frames)
        if frame_idx == 0:
            msg_idx += 1

        time.sleep(0.1)

    # Clear line
    print("\r" + " " * terminal_width + "\r", end="")


def full_welcome_sequence(name="User", show_animation=True):
    """Display complete welcome sequence"""
    terminal_width = get_terminal_width()

    # Clear screen (optional)
    # os.system('cls' if os.name == 'nt' else 'clear')

    print("\n")

    # Welcome banner
    welcome_message(name, show_ascii=True)

    print("\n")

    # Loading animation
    if show_animation:
        animated_loading(duration=2)

    # Startup info
    display_startup_info()

    print("\n")

    # Greeting
    greeting = get_greeting(name)
    for line in greeting.splitlines():
        print(line.center(terminal_width))

    print("\n" + "=" * terminal_width + "\n")


# Test
if __name__ == "__main__":
    # Test welcome sequence
    full_welcome_sequence("User", show_animation=True)

    # Test different times of day
    print("\nTesting different greetings:\n")
    for time_period in ["morning", "afternoon", "evening", "night"]:
        print(f"{time_period.capitalize()}: {get_greeting('User', time_period)}")
