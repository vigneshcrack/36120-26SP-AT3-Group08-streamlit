#imports
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import main as core
from students import student_a as page


def make_api_history(days: int = 30) -> list[dict]:
    dates = pd.date_range(end="2026-10-24", periods=days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        parts = {
            "return_strength_points": 20.0 + (i % 5),
            "trend_strength_points": 15.0 + (i % 3),
            "relative_volume_points": 10.0,
            "price_stability_points": 9.7,
        }
        rows.append(
            {
                "date": d.date().isoformat(),
                "close": 70000.0 + 150.0 * i,
                "API": sum(parts.values()),
                **parts,
            }
        )
    return rows


def make_api_prediction(token: str = "BTC", current: float | None = 54.7, forecast: float = 58.2) -> dict:
    return {
        "prediction": forecast,
        "predicted_api_t7": forecast,
        "forecast_horizon": 7,
        "prediction_date": "2026-10-31",
        "target_date": "2026-10-31",
        "token": token,
        "symbol": token,
        "asset_name": "Bitcoin",
        "as_of_date": "2026-10-24",
        "reference_date": "2026-10-24",
        "current_api": current,
        "current_api_score": current,
        "expected_change": 0.0 if current is None else forecast - current,
        "performance_level": page.performance_level(forecast),
        "performance_category": page.performance_level(forecast),
        "model_used": "primary",
        "model_version": "1.0.0",
        "missing_features": [],
        "peer_status": {"student_b": "ok", "student_c": "local_fallback", "student_d": "local_fallback"},
        "feature_sources": {
            "Return20": "owned by Student A",
            "TrendStrength": "student_b_api",
            "RelativeVolume": "local_fallback (shared package)",
            "Volatility20": "local_fallback (shared package)",
        },
        "inputs": {
            "Return20": 0.045,
            "TrendStrength": 0.018,
            "RelativeVolume": 1.12,
            "Volatility20": 0.019,
            "Return5": 0.015,
            "DailyReturn": 0.008,
            "Momentum": 3100.0,
            "API": current,
        },
        "components_current": {
            "RS": 0.6125,
            "TS": 0.59,
            "RV": 0.56,
            "PS": 0.981,
            "return_strength_points": 24.5,
            "trend_strength_points": 17.7,
            "relative_volume_points": 11.2,
            "price_stability_points": 9.81,
        },
        "explanation": {
            "baseline": 53.3,
            "contributions": {
                "Return20": 2.4,
                "TrendStrength": 1.8,
                "RelativeVolume": -0.6,
                "Volatility20": 0.3,
                "Return5": 0.5,
                "API": 0.5,
            },
            "method": "XGBoost exact tree SHAP values (pred_contribs)",
        },
        "history": make_api_history(30),
    }


API_METADATA = {
    "project": "Asset Performance Index API",
    "algorithm": "XGBoost Regressor",
    "version": "1.0.0",
    "forecast_horizon": 7,
    "feature_list": ["Asset_ID", "Return20", "TrendStrength"],
    "feature_importance": {"API": 0.21, "Return13": 0.14, "TS": 0.11},
    "best_hyperparameters": {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 350},
    "test_metrics": {
        "rmse": 13.0336,
        "mae": 9.6631,
        "r2": 0.5230,
        "naive_baseline_rmse": 16.4969,
        "naive_baseline_mae": 11.8267,
    },
    "per_coin_test_metrics": [
        {"Asset": "BTC", "Test_Rows": 33, "Model_RMSE": 7.4098, "Model_MAE": 5.7520, "Naive_RMSE": 9.2773},
    ],
}


@pytest.fixture
def fake_student_a_api(monkeypatch):
    class Fake:
        down = False
        calls: list = []

        def __call__(self, base_url, path, payload=None):
            self.calls.append((path, payload))
            if self.down:
                return core.ApiResult(ok=False, error="The service could not be reached.")
            if path == "/predict":
                return core.ApiResult(ok=True, data=make_api_prediction(payload["token"]))
            if path == "/model-metadata":
                return core.ApiResult(ok=True, data=API_METADATA)
            return core.ApiResult(ok=False, error="Not Found (HTTP 404)", status=404)

    fake = Fake()
    fake.calls = []
    monkeypatch.setattr(core, "cached_call", fake)
    monkeypatch.setattr(core, "cached_call_many", lambda base, path, payloads: [fake(base, path, p) for p in payloads])
    monkeypatch.setattr(
        core,
        "health",
        lambda url, **k: core.ApiResult(
            ok=not fake.down, data={"status": "ok"}, error=None if not fake.down else "down", elapsed_seconds=0.2
        ),
    )
    monkeypatch.setattr(core, "api_url", lambda key: "https://group08-studenta-api-api.onrender.com")
    return fake


def test_performance_level_bands():
    assert page.performance_level(20.0) == "Weak"
    assert page.performance_level(40.0) == "Moderate"
    assert page.performance_level(60.0) == "Good"
    assert page.performance_level(80.0) == "Strong"
    assert page.performance_level(None) == "Unknown"


def test_page_renders_forecast_and_tabs(fake_student_a_api):
    def script():
        from students import student_a

        student_a.render()

    at = AppTest.from_function(script, default_timeout=30).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Performance today"] == "54.7"
    assert metrics["Forecast in 7 days"] == "58.2"
    assert not at.error


def test_page_handles_outage(fake_student_a_api):
    fake_student_a_api.down = True

    def script():
        from students import student_a

        student_a.render()

    at = AppTest.from_function(script, default_timeout=30).run()
    assert not at.exception
    assert any("could not be reached" in e.value for e in at.error)