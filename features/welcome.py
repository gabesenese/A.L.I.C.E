"""
Advanced Welcome System for A.L.I.C.E
Provides contextual greetings based on time, day, and user preferences
"""

import datetime
import json
import random
import shutil
import time
from pathlib import Path

from features.welcome_validation import (
    is_valid_startup_greeting,
)

_GREETING_MESSAGES = {
    "early_morning": [
        "Early start, {name}.\n\nThe day has not had time to become dramatic yet.",
        "Morning, {name}.\n\nWe have a head start. Suspicious, but useful.",
        "Up early, {name}.\n\nSmall win first.",
        "Morning, {name}.\n\nUse the silence wisely.",
    ],
    "morning": [
        "Morning, {name}.\n\nWe have a plan. Allegedly.",
        "Good morning, {name}.\n\nLet's get one thing right before life gets creative.",
        "Morning, {name}.\n\nSmall win first. Genius later.",
        "Morning, {name}.\n\nMake the first move clean.",
    ],
    "afternoon": [
        "Afternoon, {name}.\n\nStill time for a clean win.",
        "Afternoon, {name}.\n\nThe day is negotiable.",
        "Back at it, {name}.\n\nNot perfect. Usable.",
        "Afternoon, {name}.\n\nWin the next hour.",
    ],
    "evening": [
        "Evening, {name}.\n\nBack for round two.",
        "Welcome back, {name}.\n\nLet's make future-you slightly impressed.",
        "Evening, {name}.\n\nOne clean move before the day escapes.",
        "Evening, {name}.\n\nLeave tomorrow less annoying.",
    ],
    "night": [
        "Night session, {name}.\n\nBold choice. Let's make it worth it.",
        "Late session, {name}.\n\nA questionable hour. A respectable ambition.",
        "Night shift, {name}.\n\nLet's make the tabs earn their keep.",
        "Night session, {name}.\n\nOne clean win is enough.",
    ],
    "late_night": [
        "Late night, {name}.\n\nLet's keep this clever, not chaotic.",
        "Quiet hours, {name}.\n\nSmall scope. Smart move.",
        "Still awake, {name}.\n\nFine. One clean win.",
        "Late night, {name}.\n\nAmbitious. Slightly suspicious. Continue.",
    ],
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

_USED_GREETING_SIGNATURES = {period: set() for period in _GREETING_MESSAGES}
_APPROVED_GREETINGS_PATH = Path("data/welcome_greetings.approved.json")
_FALLBACK_GREETING = "Welcome back, {name}.\n\nLet's make the tabs earn their keep."


def get_terminal_width():
    """Get terminal width safely"""
    try:
        return shutil.get_terminal_size().columns
    except Exception:
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
        if normalized in _GREETING_MESSAGES:
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
    return is_valid_startup_greeting(text, require_name_placeholder=False)


def _load_approved_greetings() -> dict[str, list[str]]:
    if not _APPROVED_GREETINGS_PATH.exists():
        return {}
    try:
        data = json.loads(_APPROVED_GREETINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    output: dict[str, list[str]] = {}
    for period in _GREETING_MESSAGES:
        items = data.get(period)
        if not isinstance(items, list):
            continue
        valid_items = [
            str(item)
            for item in items
            if isinstance(item, str)
            and is_valid_startup_greeting(item, require_name_placeholder=True)
        ]
        if valid_items:
            output[period] = valid_items
    return output


def get_greeting(name="User", time_of_day=None, style="curated"):
    """Build a time-aware startup greeting using approved/curated full messages."""
    if style != "curated":
        style = "curated"
    period = _resolve_time_of_day(time_of_day)
    approved = _load_approved_greetings()
    source = approved if approved else _GREETING_MESSAGES
    options = list(source.get(period) or source.get("afternoon") or [])
    if not options:
        return _FALLBACK_GREETING.format(name=name)

    used = _USED_GREETING_SIGNATURES.setdefault(period, set())
    available = [text for text in options if text not in used] or list(options)
    if len(available) == len(options):
        used.clear()
    random.shuffle(available)

    for template in available:
        candidate = str(template).format(name=name)
        if is_valid_startup_greeting(candidate):
            used.add(template)
            return candidate

    return _FALLBACK_GREETING.format(name=name)


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
