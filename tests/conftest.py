"""Shared fixtures: realistic API responses (shapes copied from the live Student B API)."""

from __future__ import annotations

import pandas as pd
import pytest

from app import main as core


def make_prediction(token: str = "BTC", current: float | None = 15.49, forecast: float = 14.74,
                    model_used: str = "primary") -> dict:
    return {
        "prediction": forecast, "forecast_horizon": 7, "prediction_date": "2026-10-31", "token": token,
        "as_of_date": "2026-10-24", "current_mri": current, "risk_level": "Low", "model_used": model_used,
        "model_version": "1.0.0", "missing_features": [] if model_used == "primary" else ["Return20", "MRI"],
        "feature_sources": {"Return20": "local_fallback (shared package)", "Volatility20": "student_c_api",
                            "MaxPrice20": "student_c_api", "GapRisk": "student_c_api",
                            "DR": "derived: MaxPrice20 + Kraken close", "MRI": "derived: brief formula"},
        "inputs": {"Return20": -0.0663, "Volatility20": 0.0143, "MaxPrice20": 86973.6, "GapRisk": 0.00002,
                   "DR": 0.081, "MRI": current},
        "explanation": {"baseline": 15.2, "contributions": {"Return20": 0.62, "Volatility20": -0.71,
                                                            "DR": -0.25, "GapRisk": 0.01, "MRI": -0.13},
                        "method": "LightGBM SHAP values"},
    }


def make_history(days: int = 30) -> list[dict]:
    dates = pd.date_range(end="2026-10-24", periods=days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        parts = {"volatility_points": 8 + i % 5, "drawdown_points": 4 + i % 3,
                 "downside_return_points": 2.0, "gap_risk_points": 0.01}
        rows.append({"date": d.date().isoformat(), "close": 90000 + 100 * i, "MRI": sum(parts.values()), **parts})
    return rows


METADATA = {
    "target": "Market Risk Index", "algorithm": "LightGBM Regressor", "forecast_horizon": 7,
    "model_version": "1.0.0", "training_period": {"start": "2024-11-22", "end": "2026-10-15"},
    "metrics": {"test_mae": 3.7037, "test_r2": 0.6297, "persistence_test_mae": 3.9105,
                "mean_baseline_test_mae": 11.2923,
                "test_skill_vs_persistence": 0.0529},
    "permutation_importance_test_mae_increase": {"MRI": 5.55, "Return20": 0.37, "DR": 0.05},
    "degradation_test_mae": {"All features available": 3.70, "Student A down": 4.40},
    "shrink": 0.678,
    "hyperparameters": {"primary": {"learning_rate": 0.0061, "n_estimators": 317, "objective": "l1"},
                        "fallback": {"learning_rate": 0.0095, "n_estimators": 552, "objective": "l2"}},
    "optuna": {"cv": "TimeSeriesSplit(5), gap = 7", "level": {"n_trials": 40, "best_cv_mae": 8.12},
               "residual": {"n_trials": 60, "best_cv_mae": 7.53}, "persistence_cv_mae": 7.96},
    "search_space": {"learning_rate": "log-uniform [0.005, 0.2]"},
}


@pytest.fixture
def fake_api(monkeypatch):
    """Replace cached_call with canned responses; set ``fake_api.down = True`` to simulate an outage."""

    class Fake:
        down = False
        calls: list[tuple[str, dict | None]] = []

        def __call__(self, base_url, path, payload=None):
            self.calls.append((path, payload))
            if self.down:
                return core.ApiResult(ok=False, error="The service could not be reached.")
            if path == "/predict":
                return core.ApiResult(ok=True, data=make_prediction(payload["token"]))
            if path == "/risk-history":
                return core.ApiResult(ok=True, data={"token": payload["token"],
                                                           "history": make_history(payload["days"])})
            if path == "/model-metadata":
                return core.ApiResult(ok=True, data=METADATA)
            return core.ApiResult(ok=False, error="Not Found (HTTP 404)", status=404)

    fake = Fake()
    fake.calls = []
    monkeypatch.setattr(core, "cached_call", fake)
    monkeypatch.setattr(core, "health", lambda url, **k: core.ApiResult(
        ok=not fake.down, data={"status": "ok"}, error=None if not fake.down else "down", elapsed_seconds=0.2))
    return fake
