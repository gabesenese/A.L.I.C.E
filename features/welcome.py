import shutil
import datetime
import time

def welcome_message(name):
    terminal_width = shutil.get_terminal_size().columns

    # Message
    message = f"Welcome {name}"

    # Top and Bottom Borders
    border = "#" * terminal_width

    # Center Message 
    message = message.center(terminal_width)

    print(border)
    print(message)
    print(border)

def greetings():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour<12:
        # Speak -- > Good Morning
    elif hour>= 12 and hour<18:
        # Speak -- > Good Afternoon
    else:
        # Speak -- > Good Evening

# Test
welcome_message("Welcome Mr. Senese")