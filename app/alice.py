"""
A.L.I.C.E - Production Interface
Clean, user-facing version with multiple UI options

For debugging with full logs, use: python -m app.main
For clean user experience with UI: python -m app.alice
"""

import os
import sys
from pathlib import Path
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

# Ensure project root is on sys.path when running as a script
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import after setting up environment
from app.main import ALICE


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


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alice.py                           # Start A.L.I.C.E (default: Gabriel)
  python alice.py --voice                   # Enable voice interaction
  python alice.py --model llama3.3:70b      # Specify LLM model
  python alice.py --privacy-mode            # Run without saving episodic memories

For debugging with full logs:
  python main.py
        """
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

    # Start A.L.I.C.E with Rich UI
    start_alice_rich(
        voice_enabled=args.voice,
        llm_model=args.model,
        user_name=user_name,
        debug=args.debug,
        privacy_mode=args.privacy_mode,
        llm_policy=args.llm_policy
    )


if __name__ == "__main__":
    main()
