"""
A.L.I.C.E - Production Interface
Clean, user-facing version with multiple UI options

For debugging with full logs, use: python main.py
For clean user experience with UI: python alice.py
"""

import os
import sys
import logging
import argparse
import threading

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set minimal logging
logging.basicConfig(level=logging.ERROR, format='%(message)s')
for logger_name in ['tensorflow', 'torch', 'sentence_transformers', 'transformers']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Import after setting up environment
from features.welcome import welcome_message, get_greeting, display_startup_info
from main import ALICE


def start_alice_rich(voice_enabled=False, llm_model="llama3.1:8b", user_name="Gabriel", debug=False, privacy_mode=False, llm_policy="default"):
    """Start A.L.I.C.E with Rich terminal UI"""
    try:
        from ui.rich_terminal import RichTerminalUI
    except ImportError:
        print("Rich library not found. Installing...")
        os.system(f"{sys.executable} -m pip install rich")
        from ui.rich_terminal import RichTerminalUI
    
    ui = RichTerminalUI(user_name)
    ui.show_welcome()
    ui.show_loading("Initializing A.L.I.C.E systems")
    
    # Initialize ALICE with stdout suppressed (unless debug, so thinking is visible)
    import io
    old_stdout = sys.stdout
    if not debug:
        sys.stdout = io.StringIO()
    
    try:
        alice = ALICE(
            voice_enabled=voice_enabled,
            llm_model=llm_model,
            user_name=user_name,
            debug=debug,
            privacy_mode=privacy_mode
        )
        if not debug:
            sys.stdout = old_stdout
        
        ui.clear()
        ui.show_welcome()
        if debug:
            ui.print_info("Debug mode: A.L.I.C.E thinking steps will appear above each response.")
        if privacy_mode:
            ui.print_info("🔒 Privacy mode: Episodic memories will not be saved.")
        ui.print_info("")
        
        # Main interaction loop
        while True:
            user_input = ui.get_input()
            
            if user_input is None or user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                ui.show_goodbye()
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.startswith('/'):
                if user_input == '/help':
                    ui.show_help()
                    continue
                else:
                    # Handle other commands through ALICE's command handler
                    alice._handle_command(user_input)
                    continue
            
            # Process input
            try:
                response = alice.process_input(user_input, use_voice=voice_enabled)
                ui.print_assistant_response(response)
            except Exception as e:
                ui.print_error(str(e))
        
        alice.shutdown()
        
    except Exception as e:
        sys.stdout = old_stdout
        ui.print_error(f"Error starting A.L.I.C.E: {e}")
        ui.print_info("\nFor detailed error logs, run: python main.py")


def start_alice(voice_enabled=False, llm_model="llama3.1:8b", user_name="Gabriel", debug=False, privacy_mode=False, llm_policy="default"):
    """
    Start A.L.I.C.E with welcome screen (classic terminal mode)
    
    Args:
        voice_enabled: Enable voice interaction
        llm_model: LLM model to use
        user_name: User's name for personalization
        debug: Show thinking steps (used by dev.py)
        privacy_mode: Disable episodic memory storage for privacy
        llm_policy: LLM policy mode (default/minimal/strict)
    """
    # Clear screen for clean start
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display welcome banner
    welcome_message(user_name, show_ascii=True)
    print()
    
    # Display greeting (simple, no animation to avoid glitches)
    greeting = get_greeting(user_name)
    print(greeting)
    print()
    
    # Display startup info
    display_startup_info()
    print()
    print("Initializing A.L.I.C.E systems...")
    if privacy_mode:
        print("🔒 Privacy mode: Episodic memories will not be saved.")
    print()
    
    # Initialize ALICE with stdout suppressed to avoid debug messages (unless debug)
    import io
    
    try:
        # Redirect stdout to suppress initialization messages (unless debug)
        old_stdout = sys.stdout
        if not debug:
            sys.stdout = io.StringIO()
        
        alice = ALICE(
            voice_enabled=voice_enabled,
            llm_model=llm_model,
            user_name=user_name,
            debug=debug,
            privacy_mode=privacy_mode,
            llm_policy=llm_policy
        )
        
        # Restore stdout
        if not debug:
            sys.stdout = old_stdout
        if debug:
            print("Debug mode: A.L.I.C.E thinking steps will appear above each response.\n")
        
        # Clear the console completely for a fresh start
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Show welcome banner again with clean screen
        welcome_message(user_name, show_ascii=True)
        print()
        print(get_greeting(user_name))
        print()
        print("Ready! Type /help for available commands")
        print()
        
        # Start interactive mode (skip welcome since we already showed it)
        alice.run_interactive(skip_welcome=True)
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! A.L.I.C.E shutting down...")
    except Exception as e:
        print(f"\n[ERROR] Error starting A.L.I.C.E: {e}")
        print("\nFor detailed error logs, run: python main.py")
        sys.exit(1)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alice.py                           # Start with classic terminal (default: Gabriel)
  python alice.py --ui rich                 # Start with Rich terminal UI
  python alice.py --ui gui                  # Start with Tkinter GUI
  python alice.py --voice                   # Enable voice interaction
  python alice.py --model llama3.3:70b      # Specify LLM model
  python alice.py --privacy-mode            # Run without saving episodic memories
  
For debugging with full logs:
  python main.py
        """
    )
    
    parser.add_argument(
        '--ui',
        type=str,
        choices=['classic', 'rich'],
        default='rich',
        help='UI mode: classic (standard terminal), rich (enhanced terminal with panels)'
    )
    
    # Commented out --name option, using Gabriel as default
    # parser.add_argument(
    #     '--name',
    #     type=str,
    #     default='Gabriel',
    #     help='Your name for personalization (default: Gabriel)'
    # )
    
    parser.add_argument(
        '--voice',
        action='store_true',
        help='Enable voice interaction (speech-to-text and text-to-speech)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='llama3.1:8b',
        help='LLM model to use (default: llama3.1:8b)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show A.L.I.C.E thinking steps (intent, plugins, verifier). Used by dev.py'
    )
    
    parser.add_argument(
        '--privacy-mode',
        action='store_true',
        help='Disable episodic memory storage for privacy (no conversation history saved)'
    )
    
    parser.add_argument(
        '--llm-policy',
        type=str,
        choices=['default', 'minimal', 'strict'],
        default='default',
        help='LLM policy: minimal (patterns only, no LLM for chitchat/tools), strict (no LLM at all), default (balanced)'
    )
    
    args = parser.parse_args()
    
    # Default user name
    user_name = "Gabriel"
    
    # Start A.L.I.C.E with selected UI
    if args.ui == 'rich':
        start_alice_rich(
            voice_enabled=args.voice,
            llm_model=args.model,
            user_name=user_name,
            debug=args.debug,
            privacy_mode=args.privacy_mode,
            llm_policy=args.llm_policy
        )
    else:  # classic
        start_alice(
            voice_enabled=args.voice,
            llm_model=args.model,
            user_name=user_name,
            debug=args.debug,
            privacy_mode=args.privacy_mode,
            llm_policy=args.llm_policy
        )


if __name__ == "__main__":
    main()
