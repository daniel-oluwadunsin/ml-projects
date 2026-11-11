from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_dashboard_starts_and_shows_research_scope():
    app = AppTest.from_file("app/main.py", default_timeout=30).run()
    assert not app.exception
    assert any("Multilingual Misinformation Detection" in item.value for item in app.title)
    assert any("English-only" in item.value for item in app.warning)
