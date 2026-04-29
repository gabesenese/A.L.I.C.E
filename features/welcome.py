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
            "Good early morning, {name}.",
            "Morning, {name}.",
            "You are up early, {name}.",
            "Hello, {name}.",
        ],
        "context": [
            "Quiet hours are ideal for focused planning.",
            "This is a strong window to align priorities before the day gets noisy.",
            "Early momentum now usually compounds through the day.",
            "If we set direction now, execution gets simpler later.",
        ],
        "agentic_prompt": [
            "Share one objective and I will map the first two actions.",
            "Tell me the top priority and I will stage an execution plan.",
            "Give me the target and I will outline the shortest path.",
            "Point me to the blocker and I will propose the next move.",
        ],
    },
    "morning": {
        "openers": [
            "Good morning, {name}.",
            "Morning, {name}.",
            "Hello, {name}.",
            "Hi, {name}.",
        ],
        "context": [
            "Good time to lock in outcomes for today.",
            "We can start with one high-impact task and build momentum.",
            "A quick plan now can prevent context switching later.",
            "Morning planning is the easiest way to protect deep work blocks.",
        ],
        "agentic_prompt": [
            "Name the first goal and I will break it into concrete steps.",
            "Share the critical task and I will prep the execution sequence.",
            "Tell me what matters most and I will frame the next actions.",
            "If you give me the objective, I will structure a practical sprint.",
        ],
    },
    "afternoon": {
        "openers": [
            "Good afternoon, {name}.",
            "Afternoon, {name}.",
            "Hello, {name}.",
            "Hi, {name}.",
        ],
        "context": [
            "This is a good checkpoint to re-prioritize.",
            "We can recover momentum quickly with one clear decision.",
            "A focused reset now can still close the day strong.",
            "This slot is ideal for clearing blockers on the critical path.",
        ],
        "agentic_prompt": [
            "Share the current blocker and I will suggest the next move.",
            "Tell me the top outcome and I will map the fastest route.",
            "Give me your status and I will produce a practical plan.",
            "Name one target and I will stage immediate next actions.",
        ],
    },
    "evening": {
        "openers": [
            "Good evening, {name}.",
            "Evening, {name}.",
            "Hello, {name}.",
            "Hi, {name}.",
        ],
        "context": [
            "This is a good time to close open loops.",
            "A short review now can make tomorrow cleaner.",
            "We can convert today's progress into a clear handoff.",
            "Evening work is strongest when the scope is explicit.",
        ],
        "agentic_prompt": [
            "Share what remains and I will prioritize finishing order.",
            "Tell me what is pending and I will build a closeout plan.",
            "Give me your target and I will map a clean wrap-up path.",
            "Name tomorrow's priority and I will prepare the first steps now.",
        ],
    },
    "night": {
        "openers": [
            "Late session, {name}.",
            "Good evening, {name}.",
            "Still in motion, {name}.",
            "Hello, {name}.",
        ],
        "context": [
            "Night sessions work best with tight scope.",
            "Let's keep this lean and outcome-focused.",
            "A single clear objective is the best move at this hour.",
            "We can reduce noise and execute one important task.",
        ],
        "agentic_prompt": [
            "Point to one objective and I will define the exact next step.",
            "Share the target and I will keep the plan concise.",
            "Tell me what needs to be done tonight and I will structure it.",
            "Give me the priority and I will run a focused action sequence.",
        ],
    },
    "late_night": {
        "openers": [
            "Late night, {name}.",
            "Quiet hours, {name}.",
            "Still online, {name}.",
            "Hello, {name}.",
        ],
        "context": [
            "Best approach now is minimal scope and high value.",
            "We should optimize for one decisive outcome.",
            "Late-hour progress is strongest when complexity stays low.",
            "Let's execute only what matters now and park the rest.",
        ],
        "agentic_prompt": [
            "Name one must-do item and I will drive a focused plan.",
            "Share the immediate priority and I will outline the quickest route.",
            "Tell me what cannot wait and I will structure the next actions.",
            "Give me the critical task and I will keep execution tight.",
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


def get_greeting(name="User", time_of_day=None):
    """Build a time-aware, non-repeating greeting with agentic intent."""
    period = _resolve_time_of_day(time_of_day)
    default_parts = _GREETING_COMPONENTS.get("afternoon")
    if default_parts is None and _GREETING_COMPONENTS:
        default_parts = next(iter(_GREETING_COMPONENTS.values()))
    parts = _GREETING_COMPONENTS.get(period, default_parts)
    if not parts:
        return f"Hello, {name}."
    used = _USED_GREETING_SIGNATURES.setdefault(period, set())

    combos = [
        (opener, context, prompt)
        for opener in parts["openers"]
        for context in parts["context"]
        for prompt in parts["agentic_prompt"]
    ]

    available = [combo for combo in combos if combo not in used]
    if not available:
        used.clear()
        available = combos

    opener, context, prompt = random.choice(available)
    used.add((opener, context, prompt))

    opener_text = opener.format(name=name)
    return f"{opener_text} {context} {prompt}"


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
    print(greeting.center(terminal_width))

    print("\n" + "=" * terminal_width + "\n")


# Test
if __name__ == "__main__":
    # Test welcome sequence
    full_welcome_sequence("User", show_animation=True)

    # Test different times of day
    print("\nTesting different greetings:\n")
    for time_period in ["morning", "afternoon", "evening", "night"]:
        print(f"{time_period.capitalize()}: {get_greeting('User', time_period)}")
