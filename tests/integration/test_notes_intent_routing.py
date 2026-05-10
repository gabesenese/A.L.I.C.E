from ai.core.nlp_processor import NLPProcessor


def test_open_notes_existence_maps_to_notes_query_exist():
    nlp = NLPProcessor()
    result = nlp.process("do i have any open notes?")
    assert result.intent in {"notes:query_exist", "notes:list"}


def test_open_notes_variant_not_hardcoded_sentence():
    nlp = NLPProcessor()
    result = nlp.process("great weather today, are there any open notes?")
    assert result.intent in {"notes:query_exist", "notes:list"}
    assert not result.intent.startswith("weather:")
