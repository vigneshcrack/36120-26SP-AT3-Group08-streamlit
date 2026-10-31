"""Student C page - Trend Category (TC), a five-class trend regime seven days ahead.

The page answers: *will this coin be trending up, down or sideways a week from now - and how sure
is the model?* Required outputs (AT3 brief): Current Trend, Prediction Confidence Score and Feature
Importance. Everything shown comes from the Student C FastAPI service over HTTP; the page never runs
the model. The only calculations here are display helpers (moving averages and the TrendScore band
for the price chart, from the closes the API returns).

Layout mirrors the other student pages: hero, token picker, then four tabs
(Trend forecast · All 8 tokens · Model & methodology · API). Pure helpers at the top are unit-tested
in ``tests/test_student_c.py``; everything that touches Streamlit lives below them.
"""

from __future__ import annotations

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from app import main as core
from app.main import DEFAULT_TOKEN, FORECAST_HORIZON, MEMBERS, TOKENS, api_url

STUDENT_KEY = "student_c"

# CONTRACT.md section 5: exact strings returned by the API, ordered most bearish -> most bullish.
TC_LABELS: list[str] = ["Strong Downtrend", "Downtrend", "Sideways", "Uptrend", "Strong Uptrend"]

TREND_COLOURS: dict[str, str] = {
    "Strong Downtrend": "#b02525",
    "Downtrend": "#e8706f",
    "Sideways": "#9a9994",
    "Uptrend": "#4fae6a",
    "Strong Uptrend": "#1e7a3c",
}
TREND_ICONS: dict[str, str] = {
    "Strong Downtrend": "⏬", "Downtrend": "🔻", "Sideways": "➡️", "Uptrend": "🔺", "Strong Uptrend": "⏫",
}
TREND_MEANING: dict[str, str] = {
    "Strong Downtrend": "a sharp, sustained decline - short-term average well below the 20-day average",
    "Downtrend": "prices drifting lower - the short-term average is below the 20-day average",
    "Sideways": "no clear direction - the averages are within 2% of each other",
    "Uptrend": "prices drifting higher - the short-term average is above the 20-day average",
    "Strong Uptrend": "a sharp, sustained rally - short-term average well above the 20-day average",
}
FEATURE_LABELS: dict[str, str] = {
    "Return20": "20-day return",
    "Return5": "5-day return",
    "MA5_MA20_Ratio": "Trend ratio (MA5 / MA20)",
    "TrendScore": "TrendScore today",
    "TrendScoreProj7": "Projected TrendScore (flat price)",
    "MA5": "5-day moving average",
    "MA20": "20-day moving average",
    "Volatility20": "20-day volatility",
    "MaxDrawdown": "Drawdown from 20-day high",
    "GapRisk": "Gap risk",
}
PEER_NAMES = {"student_a": "Student A (Return20, Return5)", "student_b": "Student B (MA5, MA20)"}
HISTORY_OPTIONS = [30, 60, 90, 180, 365]
CONFIDENCE_BANDS: list[tuple[float, str]] = [(0.65, "High"), (0.45, "Moderate"), (0.0, "Low")]


# ----------------------------------------------------------------- pure helpers (tested)
def trend_index(label: str | None) -> int | None:
    return TC_LABELS.index(label) if label in TC_LABELS else None


def classify_score(score: float | None) -> str | None:
    """Brief's TC rules (display only, for the history chart): Sideways is closed on both ends."""
    if score is None or pd.isna(score):
        return None
    if score < -0.10:
        return "Strong Downtrend"
    if score < -0.02:
        return "Downtrend"
    if score <= 0.02:
        return "Sideways"
    if score <= 0.10:
        return "Uptrend"
    return "Strong Uptrend"


def confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "Unknown"
    return next(name for threshold, name in CONFIDENCE_BANDS if confidence >= threshold)


def direction_change(current: str | None, forecast: str | None) -> str:
    """Plain-English comparison of today's trend with the 7-day forecast."""
    a, b = trend_index(current), trend_index(forecast)
    if a is None or b is None:
        return "Today's trend is unavailable."
    if a == b:
        return f"The model expects the **{forecast}** to continue."
    if (a - 2) * (b - 2) < 0:  # opposite sides of Sideways
        return f"The model expects a **reversal**: from {current} to {forecast}."
    if a == 2 or b == 2:
        return f"The model expects the trend to move from {current} to **{forecast}**."
    verb = "strengthen" if abs(b - 2) > abs(a - 2) else "ease"
    return f"The model expects the trend to {verb}: from {current} to **{forecast}**."


def format_value(name: str, value: float | None) -> str:
    if value is None or pd.isna(value):
        return "missing"
    if name in {"Return20", "Return5", "MaxDrawdown"}:
        return f"{value:+.1%}"
    if name == "Volatility20":
        return f"{value:.2%} per day"
    if name in {"TrendScore", "TrendScoreProj7"}:
        return f"{value:+.3f}"
    if name == "MA5_MA20_Ratio":
        return f"{value:.3f}"
    if name == "GapRisk":
        return f"{value:.4%}"
    return f"{value:,.4f}" if abs(value) < 10 else f"{value:,.2f}"


def format_metric(value, spec: str = ".3f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "-"


def probabilities_frame(prediction: dict) -> pd.DataFrame:
    probs = prediction.get("probabilities") or {}
    return pd.DataFrame([
        {"Trend": label, "Probability": float(probs.get(label, 0.0) or 0.0), "order": i,
         "Predicted": label == prediction.get("prediction")}
        for i, label in enumerate(TC_LABELS)
    ])


def importance_frame(metadata: dict) -> pd.DataFrame:
    importance = metadata.get("feature_importance") or {}
    rows = [{"Feature": FEATURE_LABELS.get(k, k), "key": k, "Importance": float(v)} for k, v in importance.items()]
    return pd.DataFrame(rows).sort_values("Importance", ascending=False) if rows else pd.DataFrame()


def explanation_frame(prediction: dict) -> pd.DataFrame:
    rows = [{"Feature": FEATURE_LABELS.get(e["feature"], e["feature"]), "Contribution": e.get("contribution") or 0.0,
             "Value": format_value(e["feature"], e.get("value"))}
            for e in prediction.get("explanation") or [] if e.get("feature")]
    return pd.DataFrame(rows)


def history_frame(history: list[dict], keep_days: int | None = None) -> pd.DataFrame:
    """Closes from /features history + display-only MA5, MA20, TrendScore and its band."""
    frame = pd.DataFrame(history)
    if frame.empty or "close" not in frame:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["MA5"] = frame["close"].rolling(5).mean()
    frame["MA20"] = frame["close"].rolling(20).mean()
    frame["TrendScore"] = 0.7 * (frame["MA5"] / frame["MA20"] - 1) + 0.3 * frame["close"].pct_change(20)
    frame["Trend"] = frame["TrendScore"].map(classify_score)
    return frame.tail(keep_days).reset_index(drop=True) if keep_days else frame


def comparison_frame(results: list[dict | None], tokens: list[str]) -> pd.DataFrame:
    rows = []
    for token, r in zip(tokens, results, strict=True):
        if r is None:
            rows.append({"Token": token, "Name": TOKENS.get(token, token), "Today": "-",
                         "In 7 days": "Unavailable", "Confidence": None, "order": -1})
            continue
        current = (r.get("current") or {}).get("trend")
        rows.append({"Token": token, "Name": TOKENS.get(token, token),
                     "Today": f"{TREND_ICONS.get(current, '')} {current or '-'}",
                     "In 7 days": f"{TREND_ICONS.get(r.get('prediction'), '')} {r.get('prediction')}",
                     "Confidence": r.get("confidence"), "order": trend_index(r.get("prediction")) or 0})
    frame = pd.DataFrame(rows).sort_values(["order", "Confidence"], ascending=[False, False])
    return frame.drop(columns="order").reset_index(drop=True)


def baseline_frame(metadata: dict) -> pd.DataFrame:
    names = {"persistence": "Persistence (today's trend carried forward)",
             "flat_price_projection": "Flat-price projection rule",
             "xgboost_defaults": "XGBoost, default settings",
             "stratified_random": "Random guess (class frequencies)", "majority_class": "Always the most common trend"}
    rows = [{"Model": "Tuned XGBoost (this service)",
             "Macro-F1": (metadata.get("metrics") or {}).get("test_macro_f1"), "is_model": True}]
    rows += [{"Model": names.get(k, k), "Macro-F1": v, "is_model": False}
             for k, v in (metadata.get("baselines") or {}).items()]
    return pd.DataFrame(rows).sort_values("Macro-F1", ascending=False).reset_index(drop=True)


def degradation_frame(metadata: dict) -> pd.DataFrame:
    names = {"all_peers_available": "All services available", "student_a_down": "Student A down",
             "student_b_down": "Student B down", "all_peers_down": "Students A and B both down"}
    deg = metadata.get("degradation_macro_f1") or {}
    return pd.DataFrame([{"Scenario": names.get(k, k), "Macro-F1": v} for k, v in deg.items()])


def confusion_frame(metadata: dict) -> pd.DataFrame:
    cm = metadata.get("confusion_matrix") or {}
    labels, matrix = cm.get("labels") or TC_LABELS, cm.get("matrix") or []
    return pd.DataFrame([{"Actual": a, "Predicted": p, "Count": matrix[i][j]}
                         for i, a in enumerate(labels) for j, p in enumerate(labels)]) if matrix else pd.DataFrame()


# ----------------------------------------------------------------- charts
def probability_chart(frame: pd.DataFrame) -> alt.LayerChart:
    base = alt.Chart(frame).encode(
        y=alt.Y("Trend:N", sort=TC_LABELS[::-1], title=None),
        x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title="Probability"),
    )
    bars = base.mark_bar(cornerRadius=4, height=22).encode(
        color=alt.Color("Trend:N", scale=alt.Scale(domain=TC_LABELS, range=[TREND_COLOURS[t] for t in TC_LABELS]),
                        legend=None),
        opacity=alt.condition("datum.Predicted", alt.value(1.0), alt.value(0.45)),
        tooltip=["Trend", alt.Tooltip("Probability:Q", format=".1%")],
    )
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text("Probability:Q", format=".0%"))
    return (bars + text).properties(height=200)


def trend_price_chart(frame: pd.DataFrame, token: str) -> alt.LayerChart:
    long = frame.melt("date", ["close", "MA5", "MA20"], var_name="Series", value_name="Price").dropna()
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None), y=alt.Y("Price:Q", scale=alt.Scale(zero=False), title=f"{token} (USD)"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["close", "MA5", "MA20"],
                                                    range=["#0b0b0b", "#2a78d6", "#eb6834"]),
                        legend=alt.Legend(orient="top", title=None)),
        strokeWidth=alt.condition("datum.Series == 'close'", alt.value(1.2), alt.value(2)),
        tooltip=["date:T", "Series:N", alt.Tooltip("Price:Q", format=",.4f")],
    )
    return lines.properties(height=260)


def trend_band_chart(frame: pd.DataFrame) -> alt.LayerChart:
    data = frame.dropna(subset=["TrendScore"])
    bars = alt.Chart(data).mark_bar().encode(
        x=alt.X("date:T", title=None), y=alt.Y("TrendScore:Q", title="TrendScore"),
        color=alt.Color("Trend:N", scale=alt.Scale(domain=TC_LABELS, range=[TREND_COLOURS[t] for t in TC_LABELS]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["date:T", alt.Tooltip("TrendScore:Q", format="+.3f"), "Trend:N"],
    )
    rules = alt.Chart(pd.DataFrame({"y": [-0.10, -0.02, 0.02, 0.10]})).mark_rule(
        strokeDash=[4, 4], color="#8a8984").encode(y="y:Q")
    return (bars + rules).properties(height=200)


def importance_chart(frame: pd.DataFrame) -> alt.Chart:
    return alt.Chart(frame).mark_bar(cornerRadius=3, color="#2a78d6").encode(
        x=alt.X("Importance:Q", axis=alt.Axis(format="%"), title="Share of total gain"),
        y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")],
    ).properties(height=40 * len(frame) + 30)


def explanation_chart(frame: pd.DataFrame) -> alt.Chart:
    return alt.Chart(frame).mark_bar(cornerRadius=3).encode(
        x=alt.X("Contribution:Q", title="Push towards the predicted trend (log-odds)"),
        y=alt.Y("Feature:N", sort=alt.EncodingSortField("Contribution", op="sum", order="descending"), title=None,
                axis=alt.Axis(labelLimit=260)),
        color=alt.condition("datum.Contribution > 0", alt.value("#4fae6a"), alt.value("#e8706f")),
        tooltip=["Feature", "Value", alt.Tooltip("Contribution:Q", format="+.3f")],
    ).properties(height=40 * len(frame) + 30)


def confusion_chart(frame: pd.DataFrame) -> alt.LayerChart:
    base = alt.Chart(frame).encode(x=alt.X("Predicted:N", sort=TC_LABELS), y=alt.Y("Actual:N", sort=TC_LABELS))
    return (base.mark_rect().encode(color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None))
            + base.mark_text().encode(text="Count:Q")).properties(height=260)


# ----------------------------------------------------------------- page sections
def _base_url() -> str | None:
    return api_url(STUDENT_KEY)


def _payload(token: str, **extra) -> dict:
    return {"token": token, "forecast_horizon": FORECAST_HORIZON, **extra}


def _show_failure(what: str, result: core.ApiResult) -> None:
    st.error(f"Could not load the {what}: {result.error}")
    st.caption("The Trend Category API runs on a free Render instance that sleeps when idle; the first request "
               "can take about a minute. Press **Refresh data** to try again. The other pages keep working.")


def _trend_badge(label: str | None, prefix: str) -> str:
    colour = TREND_COLOURS.get(label, "#9a9994")
    return (f'<div style="border-left:6px solid {colour};padding:10px 14px;background:rgba(0,0,0,0.03);'
            f'border-radius:6px;margin-bottom:6px"><span class="g8-muted">{escape(prefix)}</span><br>'
            f'<strong style="font-size:1.15rem">{TREND_ICONS.get(label, "")} {escape(str(label or "-"))}</strong>'
            f'<br><span class="g8-muted">{escape(TREND_MEANING.get(label, ""))}</span></div>')


def _forecast_tab(token: str, history_days: int) -> None:
    base = _base_url()
    with st.spinner("Getting the trend forecast (the first request can take up to a minute)..."):
        result = core.cached_call(base, "/predict", _payload(token))
        features = core.cached_call(base, "/features", _payload(token, history_days=min(365, history_days + 25)))

    if not result.ok:
        _show_failure("trend forecast", result)
    else:
        p = result.data
        current = p.get("current") or {}
        label, confidence = p.get("prediction"), p.get("confidence")

        left, right = st.columns(2)
        left.markdown(_trend_badge(current.get("trend"), f"Current trend · {current.get('date', p.get('as_of_date'))}"),
                      unsafe_allow_html=True)
        right.markdown(_trend_badge(label, f"Forecast trend · {p.get('prediction_date', '')} (in "
                                           f"{FORECAST_HORIZON} days)"), unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confidence", "-" if confidence is None else f"{confidence:.0%}", border=True,
                  help="Probability the model gives to the forecast trend. The model is well calibrated on the "
                       "test period (ECE about 0.05), so 60% means right about 6 times in 10.")
        c2.metric("Confidence level", confidence_band(confidence), border=True)
        c3.metric("TrendScore today", format_value("TrendScore", current.get("trend_score")), border=True,
                  help="0.7 x (MA5/MA20 - 1) + 0.3 x Return20, from today's prices.")
        c4.metric("If the price stays flat", format_value("TrendScoreProj7", current.get("projected_trend_score_t7")),
                  current.get("projected_trend_if_flat") or "", delta_color="off", delta_arrow="off", border=True,
                  help="TrendScore in 7 days if the price did not move - Student C's projection feature.")
        st.markdown(direction_change(current.get("trend"), label))

        core.section("How sure is the model?")
        a, b = st.columns([3, 2])
        with a:
            st.altair_chart(probability_chart(probabilities_frame(p)), width="stretch")
        with b:
            expl = explanation_frame(p)
            if not expl.empty:
                st.markdown("**Why this forecast** (SHAP, this prediction)")
                st.altair_chart(explanation_chart(expl), width="stretch")
        st.caption("All five probabilities sum to 100%. One tall bar is a confident call; similar bars mean the "
                   "model is unsure and the forecast should carry less weight.")

        peer_status = p.get("peer_status") or {}
        down = [PEER_NAMES.get(k, k) for k, v in peer_status.items() if not str(v).startswith("ok")]
        if down:
            st.warning("Forecast produced in **degraded mode**: " + ", ".join(down) + " did not answer, so "
                       f"{', '.join(FEATURE_LABELS.get(f, f) for f in p.get('missing_features') or [])} were "
                       "missing. The model is trained for this and Student C's own projection still works - see "
                       "the resilience table under **Model & methodology**.", icon="⚠️")
        with st.expander("Inputs used for this forecast and where they came from"):
            used, sources = p.get("features_used") or {}, p.get("feature_sources") or {}
            st.dataframe(pd.DataFrame([{"Input": FEATURE_LABELS.get(k, k), "Value": format_value(k, v),
                                        "Source": sources.get(k, "derived from the inputs above")}
                                       for k, v in used.items()]), hide_index=True, width="stretch")

    if features.ok and (features.data or {}).get("history"):
        frame = history_frame(features.data["history"], keep_days=history_days)
        if not frame.empty:
            core.section(f"Price and trend - last {len(frame)} days")
            st.altair_chart(trend_price_chart(frame, token), width="stretch")
            st.altair_chart(trend_band_chart(frame), width="stretch")
            st.caption("Top: close with the 5-day (blue) and 20-day (orange) moving averages - when blue is above "
                       "orange the short-term trend is up. Bottom: the TrendScore each day, coloured by its trend "
                       "category; dashed lines are the class boundaries (+/-0.02 and +/-0.10).")
    elif not features.ok:
        _show_failure("price history", features)


def _compare_tab() -> None:
    st.markdown("The 7-day trend forecast for all eight tokens, most bullish first. One pooled model serves every "
                "token, so the forecasts are directly comparable.")
    tokens = list(TOKENS)
    with st.spinner("Forecasting all eight tokens..."):
        results = core.cached_call_many(_base_url(), "/predict", [_payload(t) for t in tokens])
    data = [r.data if r.ok else None for r in results]
    if all(d is None for d in data):
        _show_failure("forecasts", results[0])
        return
    st.dataframe(comparison_frame(data, tokens), hide_index=True, width="stretch",
                 column_config={"Confidence": st.column_config.ProgressColumn(
                     "Confidence", format="percent", min_value=0, max_value=1)})


def _model_tab() -> None:
    result = core.cached_call(_base_url(), "/model-metadata")
    if not result.ok:
        _show_failure("model information", result)
        return
    meta = result.data
    metrics, period = meta.get("metrics") or {}, meta.get("training_period") or {}
    st.markdown(f"**{meta.get('algorithm', 'XGBoost Classifier')}** · version {meta.get('model_version', '?')} · "
                f"target: {meta.get('target', 'Trend Category')} {FORECAST_HORIZON} days ahead · trained on "
                f"{period.get('start', '?')} to {period.get('end', '?')} · "
                f"{len(meta.get('supported_tokens') or []) or 8} tokens (Kraken daily candles).")

    core.section("Accuracy on the unseen test period")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro-F1", format_metric(metrics.get("test_macro_f1")), border=True,
              help="Primary metric: average F1 over the five trends, so the rare strong trends count fully.")
    c2.metric("Balanced accuracy", format_metric(metrics.get("test_balanced_accuracy")), border=True)
    c3.metric("Within one class", format_metric(metrics.get("test_within_one_class"), ".0%"), border=True,
              help="Forecasts that are the right trend or a neighbouring one.")
    c4.metric("Calibration error (ECE)", format_metric(metrics.get("test_ece")), border=True,
              help="Gap between stated confidence and how often the model is right. Lower is better.")

    core.section("Feature importance")
    left, right = st.columns([3, 2])
    imp = importance_frame(meta)
    with left:
        if not imp.empty:
            st.altair_chart(importance_chart(imp), width="stretch")
    with right:
        st.markdown("The **projected TrendScore** is Student C's own engineered feature: the TrendScore the coin "
                    "would have in 7 days if its price stayed flat. 13 of the 20 days inside next week's 20-day "
                    "average are already known today, so this projection carries most of the signal - and it is "
                    "computed by this service itself, so it is available even if every peer is down.")

    core.section("Compared with the alternatives")
    st.dataframe(baseline_frame(meta).drop(columns="is_model"), hide_index=True, width="stretch",
                 column_config={"Macro-F1": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Carrying today's trend forward is a strong rule because trends are sticky. The model beats it "
               "on macro-F1 by a small margin; its main advantage is a calibrated confidence for every call.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Where the model gets it wrong** (test period)")
        cm = confusion_frame(meta)
        if not cm.empty:
            st.altair_chart(confusion_chart(cm), width="stretch")
    with b:
        st.markdown("**When a teammate's service is down**")
        deg = degradation_frame(meta)
        if not deg.empty:
            st.dataframe(deg, hide_index=True, width="stretch",
                         column_config={"Macro-F1": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Trained with peer features randomly blanked, so the service keeps answering - and stays near "
                   "baseline quality - when Student A or B is unreachable.")
        per_token = meta.get("per_token_macro_f1") or {}
        if per_token:
            st.markdown("**Macro-F1 by token**")
            st.dataframe(pd.DataFrame([{"Token": k, "Macro-F1": v} for k, v in per_token.items()])
                         .sort_values("Macro-F1", ascending=False), hide_index=True, width="stretch",
                         column_config={"Macro-F1": st.column_config.NumberColumn(format="%.3f")})

    with st.expander("How the model was tuned"):
        st.markdown(f"- **Search:** Optuna TPE, {meta.get('n_trials', '?')} trials per study (two studies: with and "
                    f"without class balancing, the winner chosen on the validation period)\n"
                    f"- **Cross-validation:** {meta.get('cv_strategy', 'not reported')}")
        best = meta.get("best_params") or {}
        st.dataframe(pd.DataFrame([{"Hyperparameter": k, "Selected": f"{v:.4g}" if isinstance(v, float) else str(v)}
                                   for k, v in best.items()]), hide_index=True, width="stretch")


def _api_tab(token: str) -> None:
    base = _base_url()
    if base:
        st.markdown(f"Base URL: `{base}` · interactive documentation at [{base}/docs]({base}/docs)")
        health = core.health(base)
        st.markdown("Status: " + (core.pill(f"Online · {health.elapsed_seconds:.2f} s", "ok") if health.ok
                                  else core.pill("Waking up / unavailable", "warn")), unsafe_allow_html=True)
    else:
        st.markdown("Status: " + core.pill("Not deployed yet", "off"), unsafe_allow_html=True)
        st.caption("Set `STUDENT_C_API_URL` as an environment variable or in `.streamlit/secrets.toml`.")
    member = MEMBERS[STUDENT_KEY]
    st.markdown(
        "| Method | Endpoint | Purpose |\n|---|---|---|\n"
        "| GET | `/` | Service metadata and endpoint list |\n"
        "| GET | `/health` | Model loaded, peers configured |\n"
        f"| POST | `/features` | Student C's features ({', '.join(member.owns)}) for peers, with history |\n"
        "| POST | `/predict` | 7-day trend with confidence, probabilities and SHAP explanation |\n"
        "| GET | `/model-metadata` | Target, features, importance, labels, metrics |"
    )
    st.code(f"curl -X POST {base or 'https://<service>'}/predict -H 'Content-Type: application/json' \\\n"
            f"     -d '{{\"token\": \"{token}\", \"forecast_horizon\": {FORECAST_HORIZON}}}'", language="bash")
    if base:
        result = core.cached_call(base, "/predict", _payload(token))
        if result.ok:
            st.markdown(f"**Live response for {token}**")
            st.json(result.data, expanded=1)
    st.code('{"error": "Unsupported cryptocurrency", "supported_tokens": ["BTC", "ETH", ...]}   # HTTP 400\n'
            '{"error": "Unsupported forecast_horizon", "supported_horizons": [7]}             # HTTP 422\n'
            '{"error": "Market data source unavailable", "retry_after_seconds": 60}           # HTTP 503',
            language="text")


# ----------------------------------------------------------------- home-page card
def render_overview_card(token: str) -> None:
    """Compact Student C visualisation for the home page (current vs 7-day trend + probabilities)."""
    result = core.cached_call(_base_url(), "/predict", _payload(token))
    with st.container(border=True):
        st.markdown(f"**Trend Category (C)** · {token}")
        if not result.ok:
            st.caption("Trend service unavailable right now - the rest of the app is unaffected.")
            return
        p = result.data
        current = (p.get("current") or {}).get("trend")
        st.markdown(f"Today **{TREND_ICONS.get(current, '')} {current or '-'}** → in {FORECAST_HORIZON} days "
                    f"**{TREND_ICONS.get(p.get('prediction'), '')} {p.get('prediction')}** "
                    f"({p.get('confidence', 0):.0%} confident)")
        st.altair_chart(probability_chart(probabilities_frame(p)).properties(height=150), width="stretch")


# ----------------------------------------------------------------- entry point
def render() -> None:
    core.inject_css()
    member = MEMBERS[STUDENT_KEY]
    core.hero(
        f"Student {member.letter} · {member.name} ({member.student_id})",
        "Trend Category - 7-day outlook",
        "Up, down or sideways? This forecasts the trend regime of a cryptocurrency one week ahead - from Strong "
        "Downtrend to Strong Uptrend - with a calibrated confidence and an explanation of each call.",
        ["XGBoost Classifier", "5 trend regimes", "All 8 tokens · pooled model", "Calibrated confidence",
         "Peer-failure resilient"],
    )
    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    token = c1.selectbox("Cryptocurrency", list(TOKENS), index=list(TOKENS).index(DEFAULT_TOKEN),
                         format_func=lambda t: f"{t} - {TOKENS[t]}", key="c_token")
    history_days = c2.select_slider("History shown (days)", options=HISTORY_OPTIONS, value=90, key="c_days")
    if c3.button("Refresh data", key="c_refresh", width="stretch"):
        core.clear_cache()
        st.rerun()

    tabs = st.tabs(["Trend forecast", "All 8 tokens", "Model & methodology", "API"])
    with tabs[0]:
        _forecast_tab(token, history_days)
    with tabs[1]:
        _compare_tab()
    with tabs[2]:
        _model_tab()
    with tabs[3]:
        _api_tab(token)

    with st.expander("How the Trend Category is calculated"):
        st.latex(r"\mathrm{TrendScore} = 0.7\left(\frac{\mathrm{MA}_5}{\mathrm{MA}_{20}} - 1\right) + 0.3\,R_{20}")
        st.markdown(
            "- $\\mathrm{MA}_5, \\mathrm{MA}_{20}$ - 5- and 20-day moving averages of the close (Student B)\n"
            "- $R_{20}$ - 20-day return (Student A)\n\n"
            "Categories: **Strong Downtrend** below -0.10 · **Downtrend** -0.10 to -0.02 · **Sideways** -0.02 to "
            "+0.02 · **Uptrend** +0.02 to +0.10 · **Strong Uptrend** above +0.10. The model predicts the category "
            "seven days from now using only information available today."
        )
    st.caption("This tool gives an estimate for educational purposes. It is not financial advice; crypto-assets "
               "are volatile and you can lose money.")


if __name__ == "__main__":
    render()
