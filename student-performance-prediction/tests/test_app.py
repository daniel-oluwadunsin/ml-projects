from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_form_returns_real_prediction() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=90).run()
    assert not app.exception
    assert len(app.button) == 1

    app.button[0].click().run()
    assert not app.exception
    assert any("Estimated final Mathematics grade" in item.value for item in app.success)
    assert [metric.label for metric in app.metric] == ["MAE", "RMSE", "R²"]
