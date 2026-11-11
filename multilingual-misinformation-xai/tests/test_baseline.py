from __future__ import annotations

import joblib

from ml.explainability.linear_explanations import (
    global_feature_importance,
    local_feature_contributions,
)
from ml.models.baseline import build_baseline, predict_with_probabilities


def trained_model():
    texts = [
        "the sky is blue",
        "water is wet",
        "the sky is not blue",
        "water is not wet",
        "a person lives in London",
        "London is a city",
    ]
    labels = ["SUPPORTED", "SUPPORTED", "REFUTED", "REFUTED", "NOT_ENOUGH_INFORMATION", "SUPPORTED"]
    return build_baseline(max_features=500, min_df=1, max_iter=500).fit(texts, labels)


def test_pipeline_predicts_probabilities_and_roundtrips(tmp_path):
    model = trained_model()
    predictions, probabilities = predict_with_probabilities(model, ["the sky is blue"])
    assert predictions.shape == (1,)
    assert probabilities.shape == (1, 3)
    assert abs(float(probabilities.sum()) - 1.0) < 1e-8
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    assert restored.predict(["the sky is blue"]).tolist() == predictions.tolist()


def test_local_contributions_sum_to_selected_class_score():
    model = trained_model()
    result = local_feature_contributions(model, "the sky is blue", "SUPPORTED", top_n=100)
    classifier = model.named_steps["classifier"]
    predicted_score = model.decision_function(["the sky is blue"])[0]
    class_index = list(classifier.classes_).index("SUPPORTED")
    assert abs(result["linear_score"] - predicted_score[class_index]) < 1e-5
    assert result["feature_contributions"]


def test_global_features_use_fitted_coefficients():
    model = trained_model()
    result = global_feature_importance(model, "SUPPORTED", top_n=5)
    assert len(result["positive"]) == 5
    assert len(result["negative"]) == 5
    assert result["positive"][0]["coefficient"] > result["negative"][0]["coefficient"]
