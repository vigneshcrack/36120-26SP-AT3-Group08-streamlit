import pytest
from streamlit.testing.v1 import AppTest

from students import student_b as page
from tests.conftest import make_history, make_prediction


# ------------------------------------------------------------ pure helpers
@pytest.mark.parametrize("value,label", [(0, "Low"), (19.9, "Low"), (20, "Moderate"), (45, "High"),
                                         (60, "Very high"), (100, "Very high"), (None, "Unknown")])
def test_risk_level_matches_api_bands(value, label):
    assert page.risk_level(value) == label


def test_format_input():
    assert page.format_input("Return20", -0.0663) == "-6.6%"
    assert page.format_input("Volatility20", 0.0143) == "1.43% per day"
    assert page.format_input("MaxPrice20", 86973.6) == "$86,973.60"
    assert page.format_input("MaxPrice20", 0.123456) == "$0.1235"
    assert page.format_input("GapRisk", None) == "unavailable"


def test_contributions_sorted_by_size_with_direction():
    frame = page.contributions_frame(make_prediction())
    assert list(frame["feature"]) == ["Volatility20", "Return20", "DR", "MRI", "GapRisk"]
    assert frame.loc[0, "direction"] == page.LOWERS and frame.loc[1, "direction"] == page.RAISES


def test_explanation_uses_api_numbers():
    text = page.explain_in_words(make_prediction())
    assert "15.5/100 (Low)" in text and "fall to **14.7**" in text and "2026-10-31" in text
    assert "20-day return** (-6.6%, +0.6 points)" in text
    assert "20-day volatility** (1.43% per day, -0.7 points)" in text


def test_explanation_without_current_level():
    text = page.explain_in_words(make_prediction(current=None, model_used="fallback"))
    assert "could not be calculated" in text


def test_comparison_frame_sorts_and_keeps_failures():
    frame = page.comparison_frame([make_prediction("BTC", 10, 12), None, make_prediction("SOL", 30, 25)],
                                  ["BTC", "ETH", "SOL"])
    assert list(frame["Token"]) == ["SOL", "BTC", "ETH"]
    assert frame.loc[0, "Change"] == pytest.approx(-5)
    assert frame.loc[2, "Risk level"] == "Unavailable"


def test_charts_build():
    prediction = make_prediction()
    for chart in [page.risk_scale_chart(15.49, 14.74), page.risk_scale_chart(None, 14.74),
                  page.contributions_chart(page.contributions_frame(prediction)),
                  page.history_chart(make_history(30), prediction), page.history_chart(make_history(5), None),
                  page.comparison_chart(page.comparison_frame([prediction], ["BTC"]))]:
        assert chart.to_dict()


def test_history_components_add_up_to_mri():
    long = page.history_frame(make_history(10))
    totals = long.groupby("date")["points"].sum()
    mri = long.drop_duplicates("date").set_index("date")["MRI"]
    assert (totals - mri).abs().max() < 1e-9


# ------------------------------------------------------------ the page itself
def _app():
    def script():
        from students import student_b

        student_b.render()

    return AppTest.from_function(script, default_timeout=30)


def test_page_renders_forecast(fake_api):
    at = _app().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Risk today"] == "15.5"
    assert metrics["Forecast in 7 days"] == "14.7"
    assert metrics["Average error of this model"] == "3.70 points"
    assert not at.error
    paths = [p for p, _ in fake_api.calls]
    assert {"/predict", "/risk-history", "/model-metadata"} <= set(paths)


def test_page_handles_api_outage(fake_api):
    fake_api.down = True
    at = _app().run()
    assert not at.exception
    assert any("could not be reached" in e.value for e in at.error)


def test_page_shows_fallback_warning(fake_api, monkeypatch):
    original = fake_api.__class__.__call__

    def fallback(self, base_url, path, payload=None):
        result = original(self, base_url, path, payload)
        if path == "/predict":
            result.data = make_prediction(current=None, model_used="fallback")
        return result

    monkeypatch.setattr(fake_api.__class__, "__call__", fallback)
    at = _app().run()
    assert not at.exception
    assert any("backup model" in w.value for w in at.warning)


def test_changing_token_requests_new_forecast(fake_api):
    at = _app().run()
    at.selectbox(key="b_token").select("SOL").run()
    assert not at.exception
    assert ("/predict", {"token": "SOL", "forecast_horizon": 7}) in fake_api.calls


def test_new_helpers():
    history = make_history(30)
    comp = page.composition_frame(history)
    assert comp["points"].sum() == pytest.approx(history[-1]["MRI"])
    assert comp["share"].sum() == pytest.approx(1.0)
    assert page.composition_chart(comp).to_dict()
    assert page.price_chart(history, "BTC").to_dict()
    assert "0.678" in page.pipeline_dot(0.678)


def test_all_tabs_render_with_api_tab(fake_api):
    at = _app().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert {"Model used", "R² (test)", "Riskiest", "Tokens with rising risk"} <= set(labels)
    assert at.json  # live response shown on the API tab


def test_main_app_and_overview_load(fake_api):
    at = AppTest.from_file("app/main.py", default_timeout=30).run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"Services online", "Riskiest next week", "Calmest next week"} <= labels


def test_overview_survives_outage(fake_api):
    fake_api.down = True
    at = AppTest.from_file("app/main.py", default_timeout=30).run()
    assert not at.exception
    assert any("waking up" in i.value for i in at.info)
