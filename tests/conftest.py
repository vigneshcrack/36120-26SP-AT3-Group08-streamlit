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


# ---------------------------------------------------------------- Student D (TOR)
# Shapes copied from CONTRACT.md section 6.4/6.5 and from models/student_d/metrics.json in the
# experimentation repository, so the page is tested against what the service really returns.
TOR_LABELS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]


def make_tor_prediction(token: str = "BTC", label: str = "Buy", confidence: float = 0.52,
                        current: str | None = "Hold", missing: list[str] | None = None) -> dict:
    others = [r for r in TOR_LABELS if r != label]
    spread = (1.0 - confidence) / len(others)
    return {
        "prediction": label,
        "confidence": confidence,
        "probabilities": {label: confidence, **{r: spread for r in others}},
        "forecast_horizon": 7,
        "prediction_date": "2026-10-31",
        "token": token,
        "as_of_date": "2026-10-24",
        "current_rating": current,
        "model_version": "1.0.0",
        "missing_features": missing or [],
        "feature_sources": {
            "Return20": "student_a_api",
            "MA5": "student_b_api",
            "MA20": "student_b_api",
            "MA5_MA20_Ratio": "derived: Student B MA5 / MA20",
            "Volatility20": "local_fallback (shared package)",
            "RelativeVolume": "owned by Student D",
            "OS": "derived: brief formula",
        },
        "inputs": {"Return20": 0.0412, "MA5": 91234.5, "MA20": 88120.3, "MA5_MA20_Ratio": 1.0353,
                   "Volatility20": 0.0168, "RelativeVolume": 1.34, "OS": 0.6131},
    }


def make_tor_features(days: int = 30, token: str = "BTC") -> dict:
    dates = pd.date_range(end="2026-10-24", periods=days, freq="D")
    history = [
        {"date": d.date().isoformat(), "close": 88000 + 120 * i,
         "VolumeSMA20": 820.0 + i, "RelativeVolume": 0.9 + (i % 7) * 0.1}
        for i, d in enumerate(dates)
    ]
    return {"token": token, "as_of_date": "2026-10-24", "source": "kraken",
            "features": {"VolumeSMA20": history[-1]["VolumeSMA20"],
                         "RelativeVolume": history[-1]["RelativeVolume"]},
            "history": history}


TOR_METADATA = {
    "target": "Trading Opportunity Rating", "algorithm": "LightGBM Classifier", "forecast_horizon": 7,
    "model_version": "1.0.0", "training_period": {"start": "2024-11-22", "end": "2026-07-04"},
    "supported_tokens": ["BTC", "ETH", "XRP", "SOL", "DOGE", "XLM", "TRX", "XMR"],
    "classification_labels": TOR_LABELS,
    "features": ["Return20", "RelativeVolume", "Volatility20", "MA5", "MA20", "MA5_MA20_Ratio", "OS"],
    "metrics": {"test_macro_f1": 0.3424, "test_balanced_accuracy": 0.4552, "test_accuracy": 0.4648,
                "test_log_loss": 1.1186, "test_ece": 0.095},
    "metrics_note": "macro-F1 averages over all five TOR labels, including any absent from the test window.",
    "baselines": {
        "majority_class": {"accuracy": 0.4167, "balanced_accuracy": 0.25, "macro_f1": 0.1176},
        "stratified_random": {"accuracy": 0.3281, "balanced_accuracy": 0.246, "macro_f1": 0.1963},
        "persistence": {"accuracy": 0.5365, "balanced_accuracy": 0.4469, "macro_f1": 0.3646},
        "lightgbm_defaults": {"accuracy": 0.4688, "balanced_accuracy": 0.3618, "macro_f1": 0.2933},
    },
    "per_class_recall": {"Strong Sell": 0.0, "Sell": 0.3611, "Hold": 0.3844, "Buy": 0.5345,
                         "Strong Buy": 0.541},
    "labels_in_test": ["Sell", "Hold", "Buy", "Strong Buy"],
    "confusion_matrix": {"labels": TOR_LABELS,
                         "matrix": [[0, 0, 0, 0, 0], [0, 13, 18, 5, 0], [0, 21, 123, 142, 34],
                                    [0, 4, 92, 155, 39], [0, 0, 19, 37, 66]]},
    "feature_importance": {"MA5_MA20_Ratio": 0.4151, "Return20": 0.1751, "Volatility20": 0.1272,
                           "MA5": 0.0842, "MA20": 0.0803, "OS": 0.0674, "RelativeVolume": 0.0507},
    "degradation": {
        "all_peers_available": {"macro_f1": 0.3424, "accuracy": 0.4648},
        "student_a_down": {"macro_f1": 0.3728, "accuracy": 0.4792},
        "student_b_down": {"macro_f1": 0.2667, "accuracy": 0.4036},
        "student_c_down": {"macro_f1": 0.3337, "accuracy": 0.4583},
        "all_peers_down": {"macro_f1": 0.281, "accuracy": 0.4193},
    },
    "per_token": {"SOL": {"macro_f1": 0.3926, "accuracy": 0.5, "rows": 96},
                  "TRX": {"macro_f1": 0.153, "accuracy": 0.3646, "rows": 96}},
    "best_cv_macro_f1": 0.3546, "n_trials": 40, "random_seed": 42,
    "cv_strategy": "TimeSeriesSplit(n_splits=5, gap=7) over the DISTINCT DATES of the training period",
    "data_source": "Kraken public REST API /0/public/OHLC (interval=1440)",
    "best_params": {"max_depth": 5, "num_leaves": 20, "n_estimators": 158, "class_weight": "balanced"},
    "search_space": {"max_depth": "int [3, 12]", "num_leaves": "int [8, min(255, 2**max_depth)]"},
}


@pytest.fixture
def fake_tor_api(monkeypatch):
    """Canned Student D responses; set ``fake_tor_api.down = True`` to simulate an outage."""

    class Fake:
        down = False
        calls: list[tuple[str, dict | None]] = []

        def __call__(self, base_url, path, payload=None):
            self.calls.append((path, payload))
            if self.down:
                return core.ApiResult(ok=False, error="The service could not be reached.")
            if path == "/predict":
                return core.ApiResult(ok=True, data=make_tor_prediction(payload["token"]))
            if path == "/features":
                return core.ApiResult(ok=True, data=make_tor_features(payload.get("history_days", 30),
                                                                     payload["token"]))
            if path == "/model-metadata":
                return core.ApiResult(ok=True, data=TOR_METADATA)
            return core.ApiResult(ok=False, error="Not Found (HTTP 404)", status=404)

    fake = Fake()
    fake.calls = []
    monkeypatch.setattr(core, "cached_call", fake)
    monkeypatch.setattr(core, "cached_call_many",
                        lambda base, path, payloads: [fake(base, path, p) for p in payloads])
    monkeypatch.setattr(core, "health", lambda url, **k: core.ApiResult(
        ok=not fake.down, data={"status": "ok"}, error=None if not fake.down else "down", elapsed_seconds=0.2))
    monkeypatch.setattr(core, "api_url", lambda key: "https://example.invalid")
    return fake
