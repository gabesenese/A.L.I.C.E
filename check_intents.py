"""Check what intents the NLP assigns in a clean context."""
import sys, logging
logging.disable(logging.CRITICAL)

sys.path.insert(0, '.')
from ai.core.nlp_processor import NLPProcessor

nlp = NLPProcessor()
tests = [
    "what time is it?",
    "what's today's date?",
    "what's the weather in London?",
    "what's the weather like?",
    "search my notes for project ideas",
    "delete note titled meeting agenda",
    "show all my notes",
    "read my work note",
]
for t in tests:
    r = nlp.process(t)
    print(f"{r.intent} ({r.intent_confidence:.2f}) | {t!r}")
