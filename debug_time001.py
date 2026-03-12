import sys, logging; logging.disable(logging.CRITICAL); sys.path.insert(0, '.')
from app.main import ALICE
a = ALICE(voice_enabled=False, user_name='TestUser', debug=False)
# Run the test sequence that precedes time_001 in a fresh ALICE instance
# (ScenarioRunner reuses one ALICE instance, so state carries over)
a.process_input('show my notes')
a.process_input("what's in it?")
print('context after notes_014:', a.nlp.context.last_intent, '|', a.nlp.context.last_plugin)
r = a.process_input('what time is it?')
print('time_001 intent:', a.nlp.context.last_intent)
print('time_001 response:', r[:100])
