"""Tests for the Student D (Trading Opportunity Rating) page.

The pure helpers are tested directly; the page itself is exercised with Streamlit's ``AppTest`` so a
broken widget or a crash on a degraded API response is caught here rather than in the browser.
"""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import main as core
from students import student_d as page
from students.student_d import metadata_value
from tests.conftest import TOR_METADATA, make_tor_features, make_tor_prediction


# ------------------------------------------------------------ pure helpers
def test_rating_order_matches_the_contract():
    """CONTRACT.md section 5 fixes both the strings and their order (worst -> best)."""
    assert page.TOR_LABELS == ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
    assert [page.rating_index(label) for label in page.TOR_LABELS] == [0, 1, 2, 3, 4]
    assert page.rating_index("Not a rating") is None
    assert page.rating_index(None) is None


@pytest.mark.parametrize(
    ("confidence", "band"),
    [(1.0, "High"), (0.7, "High"), (0.69, "Moderate"), (0.45, "Moderate"), (0.44, "Low"), (0.0, "Low"),
     (None, "Unknown")],
)
def test_confidence_bands(confidence, band):
    assert page.confidence_band(confidence) == band


def test_format_input_uses_the_unit_a_reader_expects():
    assert page.format_input("Return20", 0.0412) == "+4.1%"
    assert page.format_input("Volatility20", 0.0168) == "1.68% per day"
    assert page.format_input("RelativeVolume", 1.34) == "1.34x average"
    assert page.format_input("MA20", 88120.3) == "$88,120.30"
    assert page.format_input("MA20", 0.123456) == "$0.1235"
    assert page.format_input("VolumeSMA20", 820.0) == "820"
    assert page.format_input("Return20", None) == "unavailable"


def test_probabilities_frame_keeps_all_five_ratings_in_scale_order():
    """A rating the model gives no mass to must still appear, as an explicit zero."""
    prediction = make_tor_prediction(label="Buy", confidence=0.6)
    prediction["probabilities"] = {"Buy": 0.6, "Hold": 0.4}  # API omitted the other three
    frame = page.probabilities_frame(prediction)

    assert list(frame["rating"]) == page.TOR_LABELS
    assert frame.loc[frame["rating"] == "Strong Sell", "probability"].iloc[0] == 0.0
    assert frame["predicted"].sum() == 1
    assert frame.loc[frame["predicted"], "rating"].iloc[0] == "Buy"


def test_probabilities_sum_to_one_for_a_full_response():
    frame = page.probabilities_frame(make_tor_prediction())
    assert frame["probability"].sum() == pytest.approx(1.0)


def test_comparison_frame_ranks_best_opportunity_first_and_keeps_failures():
    results = [make_tor_prediction("BTC", "Hold", 0.5), None, make_tor_prediction("SOL", "Strong Buy", 0.8)]
    frame = page.comparison_frame(results, ["BTC", "ETH", "SOL"])

    assert list(frame["Token"]) == ["SOL", "BTC", "ETH"]
    assert frame.loc[0, "Rating"] == "Strong Buy"
    assert frame.loc[2, "Rating"] == "Unavailable"  # the failure is listed, not dropped


def test_comparison_frame_breaks_ties_on_confidence():
    results = [make_tor_prediction("BTC", "Buy", 0.40), make_tor_prediction("ETH", "Buy", 0.75)]
    frame = page.comparison_frame(results, ["BTC", "ETH"])
    assert list(frame["Token"]) == ["ETH", "BTC"]


def test_baseline_frame_sorts_by_macro_f1_and_includes_the_model():
    frame = page.baseline_frame(TOR_METADATA)
    assert frame.loc[0, "Approach"] == "Carry today's rating forward"  # persistence wins on this data
    assert frame["is_model"].sum() == 1
    assert frame.loc[frame["is_model"], "Macro-F1"].iloc[0] == pytest.approx(0.3424)


def test_beats_persistence_is_honest_about_this_model():
    assert page.beats_persistence(TOR_METADATA) is False
    better = {**TOR_METADATA, "metrics": {**TOR_METADATA["metrics"], "test_macro_f1": 0.9}}
    assert page.beats_persistence(better) is True
    assert page.beats_persistence({}) is None


def test_confusion_frame_shares_are_per_actual_class():
    frame = page.confusion_frame(TOR_METADATA)
    assert len(frame) == 25
    for _actual, group in frame.groupby("actual"):
        total = group["count"].sum()
        if total:
            assert group["share"].sum() == pytest.approx(1.0)


def test_importance_frame_attributes_every_input_to_an_owner():
    frame = page.importance_frame(TOR_METADATA)
    assert frame.loc[0, "Feature"] == "MA5_MA20_Ratio"
    assert "-" not in set(frame["Owner"])
    assert frame.loc[frame["Feature"] == "RelativeVolume", "Owner"].iloc[0] == "Student D"


def test_degradation_frame_labels_every_scenario():
    frame = page.degradation_frame(TOR_METADATA)
    assert len(frame) == 5
    assert "All services available" in set(frame["Scenario"])
    assert not any(name.endswith("_down") for name in frame["Scenario"])


def test_explanation_uses_only_api_values():
    text = page.explain_in_words(make_tor_prediction(label="Buy", confidence=0.52, current="Hold"))
    assert "**Buy**" in text and "2026-10-31" in text and "2026-10-24" in text
    assert "52%" in text and "moderate" in text
    assert "Today's rating is **Hold**, so this forecast is an improvement." in text


def test_explanation_flags_no_change_and_missing_inputs():
    unchanged = page.explain_in_words(make_tor_prediction(label="Hold", current="Hold"))
    assert "expects no change" in unchanged

    degraded = page.explain_in_words(make_tor_prediction(missing=["Return20", "OS"]))
    assert "unavailable" in degraded and "20-day return" in degraded


def test_explanation_survives_an_empty_response():
    assert "did not return a rating" in page.explain_in_words({})


def test_charts_build():
    prediction = make_tor_prediction()
    history = page.history_frame(make_tor_features(30)["history"])
    charts = [
        page.probability_chart(page.probabilities_frame(prediction)),
        page.confusion_chart(page.confusion_frame(TOR_METADATA)),
        page.importance_chart(page.importance_frame(TOR_METADATA)),
        page.volume_chart(history, "BTC"),
        page.price_chart(history, "BTC"),
    ]
    for chart in charts:
        assert chart.to_dict()  # raises if the spec is invalid


# ------------------------------------------------------------ the page itself
def _run(fake) -> AppTest:
    # AppTest re-executes the function's own source in a fresh module, so the import has to happen
    # inside the script rather than at module level (same pattern as tests/test_student_b.py).
    def script():
        from students import student_d

        student_d.render()

    app = AppTest.from_function(script, default_timeout=30).run()
    assert not app.exception, app.exception
    return app


def test_page_renders_the_rating(fake_tor_api):
    app = _run(fake_tor_api)
    text = " ".join(m.value for m in app.markdown)
    assert "Trading Opportunity Rating" in text
    assert "Vinnie Bui" in text and "26160438" in text
    assert any(m.label == "Confidence" for m in app.metric)
    assert ("/predict", {"token": "BTC", "forecast_horizon": 7}) in fake_tor_api.calls


def test_page_offers_every_supported_token(fake_tor_api):
    """The model is pooled over all eight tokens, so the picker must not be BTC-only."""
    app = _run(fake_tor_api)
    # The widget exposes the formatted labels ("BTC - Bitcoin"), so compare the symbols.
    assert [o.split(" - ")[0] for o in app.selectbox[0].options] == list(core.TOKENS)


def test_page_warns_when_the_model_loses_to_persistence(fake_tor_api):
    app = _run(fake_tor_api)
    warnings = " ".join(w.value for w in app.warning)
    assert "does **not** beat" in warnings or "does not beat" in warnings


def test_page_degrades_without_crashing_when_the_api_is_down(fake_tor_api):
    fake_tor_api.down = True
    app = _run(fake_tor_api)
    assert app.error, "an outage must be reported to the user"
    assert any("could not be reached" in e.value for e in app.error)


# ------------------------------------------------------------ the real service's shape
# Captured from a running Student D API (GET /model-metadata). The service groups its
# non-contract extras under `performance` and `training`, so a page that only looked at
# the top level rendered an empty comparison table and never showed the persistence
# warning - which is the one thing this page exists to be honest about.
LIVE_METADATA = json.loads(
    (Path(__file__).parent / "fixtures" / "student_d_model_metadata.json").read_text()
)


def test_metadata_value_finds_nested_blocks():
    assert metadata_value({"baselines": {"a": 1}}, "baselines") == {"a": 1}
    assert metadata_value({"performance": {"baselines": {"a": 1}}}, "baselines") == {"a": 1}
    assert metadata_value({"training": {"random_seed": 42}}, "random_seed") == 42
    # An empty block must not shadow a populated one further down.
    assert metadata_value({"baselines": {}, "performance": {"baselines": {"a": 1}}}, "baselines") == {"a": 1}
    # Documented rename: the service calls the tuned parameters `hyperparameters`.
    assert metadata_value({"hyperparameters": {"max_depth": 5}}, "best_params") == {"max_depth": 5}
    assert metadata_value({}, "baselines", {}) == {}


@pytest.mark.parametrize(
    "key",
    ["metrics", "baselines", "confusion_matrix", "feature_importance", "degradation",
     "per_class_recall", "per_token", "labels_in_test", "metrics_note", "training_period",
     "best_params", "search_space", "best_cv_macro_f1", "cv_strategy", "n_trials",
     "supported_tokens", "data_source", "random_seed"],
)
def test_every_block_the_page_needs_is_reachable_in_the_live_response(key):
    assert metadata_value(LIVE_METADATA, key) is not None, f"{key} not found in the live response"


def test_page_helpers_work_against_the_live_response():
    """The regression this file exists for: the real nesting must still populate the page."""
    baselines = page.baseline_frame(LIVE_METADATA)
    assert len(baselines) == 5                      # the model plus four baselines
    assert baselines["is_model"].sum() == 1
    assert page.beats_persistence(LIVE_METADATA) is False

    assert len(page.confusion_frame(LIVE_METADATA)) == 25
    assert len(page.degradation_frame(LIVE_METADATA)) == 5
    assert not page.importance_frame(LIVE_METADATA).empty
