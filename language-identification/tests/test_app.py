from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_predicts_and_shows_calibrated_confidence() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=90).run()
    assert not app.exception
    app.text_area[0].set_value(
        "La langue française utilise des accents et des combinaisons de lettres qui lui donnent une apparence particulière. Un paragraphe plus long fournit généralement de meilleurs indices qu'une phrase très courte."
    )
    next(button for button in app.button if button.label == "Identify language").click().run()

    assert not app.exception
    assert any("Predicted language: **French**" in item.value for item in app.success)
    assert any(metric.label == "Calibrated confidence" for metric in app.metric)
