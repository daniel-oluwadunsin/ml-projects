from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.ui import MODEL_PATH

APP_ROOT = Path(__file__).resolve().parents[1] / "app"


def test_dashboard_starts_and_shows_research_scope():
    app = AppTest.from_file(APP_ROOT / "main.py", default_timeout=30).run()
    assert not app.exception
    assert any("Multilingual Misinformation Detection" in item.value for item in app.title)
    assert any("English-only" in item.value for item in app.warning)


def test_trained_dashboard_makes_a_claim_only_prediction():
    if not MODEL_PATH.is_file():
        pytest.skip("Run the FEVER baseline before testing a trained-model prediction")
    app = AppTest.from_file(APP_ROOT / "main.py", default_timeout=30).run()
    app.switch_page("pages/claim_analysis.py").run()
    app.text_area[0].set_value("A test claim for the English FEVER model.").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Predicted FEVER label" in item.value for item in app.subheader)


@pytest.mark.parametrize(
    "page_path",
    [
        "pages/overview.py",
        "pages/claim_analysis.py",
        "pages/model_comparison.py",
        "pages/language_performance.py",
        "pages/explainability.py",
        "pages/error_analysis.py",
        "pages/dataset_explorer.py",
        "pages/methodology.py",
        "pages/about.py",
    ],
)
def test_every_dashboard_page_loads(page_path):
    app = AppTest.from_file(APP_ROOT / "main.py", default_timeout=30).run()
    app.switch_page(page_path).run()
    assert not app.exception
