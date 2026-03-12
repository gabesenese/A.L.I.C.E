import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
import sys; sys.path.insert(0, '.')
from ai.core.nlp_processor import NLPProcessor
nlp = NLPProcessor()
tests = [
    ('do i have any notes?', 'notes:query_exist'),
    ('how many notes do i have?', 'notes:query_exist'),
    ('what is in the grocery list?', 'notes:read_content'),
    ("what's inside the grocery list?", 'notes:read_content'),
    ('add milk to my grocery list', 'notes:append'),
    ('add milk, eggs and butter to my grocery list note', 'notes:append'),
    ('what time is it?', 'time:current'),
    ("what's today's date?", 'time:current'),
]
for text, expected in tests:
    result = nlp.process(text)
    got = result.intent
    conf = result.intent_confidence
    ok = 'PASS' if got == expected else 'FAIL'
    print(f'{ok}: {text!r} => {got!r} (expected {expected!r}, conf={conf:.2f})')
