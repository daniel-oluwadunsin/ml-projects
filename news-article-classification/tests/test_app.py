from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_classifies_article_and_shows_probabilities() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=90).run()
    assert not app.exception
    app.text_input[0].set_value("Late goal sends the home side into the final")
    app.text_area[0].set_value(
        "The winning team scored in the closing minutes after a fast counterattack. Its goalkeeper made several key saves, while the coach praised the players for maintaining their shape under pressure throughout the second half."
    )
    next(button for button in app.button if button.label == "Classify article").click().run()

    assert not app.exception
    assert any("Predicted category: **Sports**" in item.value for item in app.success)
    assert any(metric.label == "Calibrated confidence" for metric in app.metric)
