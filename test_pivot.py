"""Quick test for domain-pivot guard in _retrieval_first_parse."""
from app.main import ALICE

a = ALICE(voice_enabled=False, user_name='TestUser', debug=False)

# Prime with notes context (like test_scenarios does)
a.process_input('do i have any notes?')
a.process_input('find my work notes')
a.process_input('read my work note')

# Now test domain-pivot queries
tests = [
    ("what time is it?", "status_inquiry"),
    ("what's today's date?", "status_inquiry"),
    ("what's the weather in London?", "weather:current"),
]
print()
all_pass = True
for t, expected in tests:
    a.process_input(t)
    intent = a.nlp.context.last_intent
    ok = 'OK' if intent == expected else 'FAIL'
    if ok == 'FAIL':
        all_pass = False
    print(f'[{ok}] {t!r} -> intent={intent} (expected={expected})')

print()
print("All pass!" if all_pass else "SOME FAILED")
