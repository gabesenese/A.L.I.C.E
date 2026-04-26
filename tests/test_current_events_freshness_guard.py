from pathlib import Path


def test_transcript_smoke_fixture_exists():
    path = Path(__file__).parent / "e2e" / "data" / "transcript_smoke.jsonl"
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() != ""
