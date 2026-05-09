from __future__ import annotations

import json
from urllib.error import URLError

from tools import generate_welcome_greetings as gen


def test_generator_handles_ollama_unavailable(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gen, "_ollama_generate", lambda **kwargs: (_ for _ in ()).throw(URLError("down")))
    out = tmp_path / "candidates.json"
    approved = tmp_path / "approved.json"
    code = gen.main(
        [
            "--output",
            str(out),
            "--approved-output",
            str(approved),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "Ollama unavailable" in captured.err
    assert not out.exists()
    assert not approved.exists()


def test_candidate_output_shape(monkeypatch, tmp_path):
    payload = {
        "period": "morning",
        "greetings": [
            "Morning, {name}.\n\nWe have a plan. Allegedly.",
            "Morning, {name}?\n\nHow can I help?",
        ],
    }
    monkeypatch.setattr(gen, "_ollama_generate", lambda **kwargs: json.dumps(payload))
    out = tmp_path / "candidates.json"
    approved = tmp_path / "approved.json"
    code = gen.main(
        [
            "--period",
            "morning",
            "--per-period",
            "2",
            "--output",
            str(out),
            "--approved-output",
            str(approved),
        ]
    )
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    rows = data["periods"]["morning"]
    assert len(rows) == 2
    assert rows[0]["valid"] is True
    assert rows[1]["valid"] is False
    assert isinstance(rows[1]["validation_reasons"], list)
    assert approved.exists()


def test_extract_json_block_handles_wrapped_content():
    raw = 'text before {"period":"night","greetings":["Night session, {name}.\\n\\nBold choice."]} text after'
    parsed = gen._extract_json_block(raw)
    assert parsed.get("period") == "night"
