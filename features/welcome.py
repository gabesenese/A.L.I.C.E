"""
Advanced Welcome System for A.L.I.C.E
Provides contextual greetings based on time, day, and user preferences
"""

import shutil
import datetime
import time
import random


def get_terminal_width():
    """Get terminal width safely"""
    try:
        return shutil.get_terminal_size().columns
    except:
        return 80  # Default width


def welcome_message(name="User", show_ascii=True):
    """Display welcome banner for A.L.I.C.E"""
    terminal_width = get_terminal_width()

    # ASCII Art for ALICE (optional)
    if show_ascii and terminal_width >= 60:
        ascii_art = """
    ___    __    ____  ________  ______
   /   |  / /   /  _/ / ____/   / ____/
  / /| | / /    / /  / /       / __/   
 / ___ |/ /____/ /_ / /____   / /___   
/_/  |_/_____/___(_)_____/   /_____/   
"""
        print(ascii_art.center(terminal_width))
    
    # Welcome message
    message = f"Welcome, {name}!"
    
    # Borders
    border = "=" * terminal_width
    
    # Center message
    centered_message = message.center(terminal_width)
    
    print(border)
    print(centered_message)
    print(border)


def get_greeting(name="User", time_of_day=None):
    """Get contextual greeting based on time of day"""
    if time_of_day is None:
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
    
    # Greeting templates
    greetings = {
        "morning": [
            f"Good morning, {name}! Ready to start the day?",
            f"Good morning, {name}! I hope you slept well.",
            f"Rise and shine, {name}! What can I help you with today?",
        ],
        "afternoon": [
            f"Good afternoon, {name}! How's your day going?",
            f"Good afternoon, {name}! What can I assist you with?",
            f"Hello, {name}! Productive afternoon so far?",
        ],
        "evening": [
            f"Good evening, {name}! How was your day?",
            f"Good evening, {name}! Ready to wind down?",
            f"Evening, {name}! What can I help you with tonight?",
        ],
        "night": [
            f"Good evening, {name}! Burning the midnight oil?",
            f"Hello, {name}! Working late tonight?",
            f"Good evening, {name}! Hope you're doing well.",
        ]
    }
    
    return random.choice(greetings.get(time_of_day, greetings["afternoon"]))


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
        print(char, end='', flush=True)
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
        "Ready!"
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
