"""Student B page - Market Risk Index (MRI) 7-day forecast.

Author: Vignesh Selvam (25740287), Group 08.

The page calls Student B's FastAPI service (``/predict``, ``/risk-history`` and
``/model-metadata``) and turns the responses into a forecast, a plain-language
explanation, a risk-history chart, a cross-token comparison and an honest summary of
model accuracy and limitations. All charts are built by the pure functions below, which
are unit-tested without a running API.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import main as core
from app.main import DEFAULT_TOKEN, FORECAST_HORIZON, MEMBERS, TOKENS, api_url

STUDENT_KEY = "student_b"

# risk bands used by the API (display aid only) and their reserved status colours
RISK_BANDS: list[tuple[float, float, str, str]] = [
    (0, 20, "Low", "#0ca30c"),
    (20, 40, "Moderate", "#fab219"),
    (40, 60, "High", "#ec835a"),
    (60, 100, "Very high", "#d03b3b"),
]
RAISES, LOWERS = "Raises risk", "Lowers risk"
DIRECTION_COLOURS = {RAISES: "#e34948", LOWERS: "#2a78d6"}  # diverging red / blue
COMPONENTS = {  # column -> (label, colour) in fixed categorical order
    "volatility_points": ("Volatility (40 %)", "#2a78d6"),
    "drawdown_points": ("Drawdown (30 %)", "#eb6834"),
    "downside_return_points": ("Negative 20-day return (20 %)", "#1baf7a"),
    "gap_risk_points": ("Gap risk (10 %)", "#eda100"),
}
FEATURE_LABELS = {
    "Return20": "20-day return",
    "Volatility20": "20-day volatility",
    "MaxPrice20": "20-day high price",
    "GapRisk": "Gap risk",
    "DR": "Drawdown from 20-day high",
    "MRI": "Today's risk level",
}
HISTORY_OPTIONS = [30, 60, 90, 180, 365]


# ----------------------------------------------------------------- pure helpers (tested)
def risk_level(value: float | None) -> str:
    if value is None:
        return "Unknown"
    for low, high, label, _ in RISK_BANDS:
        if low <= value < high:
            return label
    return "Very high"


def format_input(name: str, value: float | None) -> str:
    """Human-readable value of a model input."""
    if value is None:
        return "unavailable"
    if name in {"Return20", "DR", "GapRisk"}:
        return f"{value:+.1%}" if name == "Return20" else f"{value:.2%}"
    if name == "Volatility20":
        return f"{value:.2%} per day"
    if name == "MaxPrice20":
        return f"${value:,.4f}" if value < 1 else f"${value:,.2f}"
    return f"{value:.1f}"


def contributions_frame(prediction: dict) -> pd.DataFrame:
    """One row per model input with its contribution (MRI points), largest effect first."""
    contributions = prediction["explanation"]["contributions"]
    inputs = prediction.get("inputs", {})
    frame = pd.DataFrame(
        {
            "feature": list(contributions),
            "label": [FEATURE_LABELS.get(f, f) for f in contributions],
            "value": [format_input(f, inputs.get(f)) for f in contributions],
            "points": list(contributions.values()),
        }
    )
    frame["direction"] = frame["points"].map(lambda p: RAISES if p > 0 else LOWERS)
    return frame.reindex(frame["points"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def explain_in_words(prediction: dict) -> str:
    """Plain-language summary of a /predict response (no jargon, numbers from the API)."""
    token = prediction["token"]
    forecast = prediction["prediction"]
    current = prediction.get("current_mri")
    date = prediction["prediction_date"]
    if current is None:
        text = (f"{token}'s risk is forecast to be **{forecast:.1f}/100 ({risk_level(forecast)})** on {date}. "
                "Today's level could not be calculated because some inputs were unavailable.")
    else:
        change = forecast - current
        if abs(change) < 0.5:
            movement = f"stay about the same at **{forecast:.1f}**"
        else:
            movement = f"{'rise' if change > 0 else 'fall'} to **{forecast:.1f}** ({change:+.1f} points)"
        text = (f"{token}'s risk is **{current:.1f}/100 ({risk_level(current)})** today and is forecast to "
                f"{movement} by {date} - **{risk_level(forecast)}** risk.")
    frame = contributions_frame(prediction)
    frame = frame[frame["feature"] != "MRI"]  # persistence is the starting point, not a driver
    up = frame[frame["points"] > 0.05]
    down = frame[frame["points"] < -0.05]
    drivers = []
    if not up.empty:
        row = up.iloc[0]
        drivers.append(f"the main factor pushing risk up is the **{row['label'].lower()}** "
                       f"({row['value']}, {row['points']:+.1f} points)")
    if not down.empty:
        row = down.iloc[0]
        drivers.append(f"the main factor pulling it down is the **{row['label'].lower()}** "
                       f"({row['value']}, {row['points']:+.1f} points)")
    if drivers:
        text += " Compared with a typical week, " + "; ".join(drivers) + "."
    return text


def risk_scale_chart(current: float | None, forecast: float) -> alt.LayerChart:
    """0-100 scale with the four risk bands, today's value and the forecast."""
    bands = pd.DataFrame(RISK_BANDS, columns=["start", "end", "band", "colour"]).assign(y0=0.0, y1=1.0)
    x_scale = alt.Scale(domain=[0, 100], nice=False)
    y_scale = alt.Scale(domain=[0, 1.6])
    band_layer = alt.Chart(bands).mark_rect(opacity=0.35, cornerRadius=4, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("start:Q", scale=x_scale, title="Market Risk Index (0-100)"),
        x2="end:Q",
        y=alt.Y("y0:Q", scale=y_scale, axis=None),
        y2="y1:Q",
        color=alt.Color("band:N", scale=alt.Scale(domain=[b[2] for b in RISK_BANDS],
                                                  range=[b[3] for b in RISK_BANDS]), legend=None),
        tooltip=[alt.Tooltip("band:N", title="Band"), alt.Tooltip("start:Q", title="From"),
                 alt.Tooltip("end:Q", title="To")],
    )
    band_text = alt.Chart(bands).transform_calculate(mid="(datum.start + datum.end) / 2", ty="1.3").mark_text(
        fontSize=12, color="#52514e").encode(
        x=alt.X("mid:Q", scale=x_scale), y=alt.Y("ty:Q", scale=y_scale), text="band:N")
    points = [{"what": f"Forecast (+{FORECAST_HORIZON} days)", "value": forecast, "y": 0.32,
               "label": f"{forecast:.1f}"}]
    if current is not None:
        points.insert(0, {"what": "Today", "value": current, "y": 0.68, "label": f"{current:.1f}"})
    marker_frame = pd.DataFrame(points)
    shapes = ["circle", "diamond"] if current is not None else ["diamond"]
    marks = alt.Chart(marker_frame).mark_point(size=240, filled=True, color="#0b0b0b", stroke="#fcfcfb",
                                               strokeWidth=2, opacity=1).encode(
        x=alt.X("value:Q", scale=x_scale),
        y=alt.Y("y:Q", scale=y_scale),
        shape=alt.Shape("what:N", scale=alt.Scale(domain=list(marker_frame["what"]), range=shapes),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("what:N", title=""), alt.Tooltip("value:Q", title="MRI", format=".1f")],
    )
    # Streamlit fits the whole chart (axis + legend included) into this height
    return (band_layer + band_text + marks).properties(height=170)


def contributions_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Horizontal bars: how much each input moved the forecast (MRI points)."""
    bars = alt.Chart(frame).mark_bar(cornerRadius=4, height=18).encode(
        x=alt.X("points:Q", title="Effect on the forecast (MRI points)"),
        y=alt.Y("label:N", sort=list(frame["label"]), title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=list(DIRECTION_COLOURS),
                                                       range=list(DIRECTION_COLOURS.values())),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[alt.Tooltip("label:N", title="Input"), alt.Tooltip("value:N", title="Value"),
                 alt.Tooltip("points:Q", title="Effect (points)", format="+.2f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")
    return (bars + zero).properties(height=110 + 36 * len(frame))


def history_frame(history: list[dict]) -> pd.DataFrame:
    """Long format (date, component, points) for a stacked area chart."""
    frame = pd.DataFrame(history)
    frame["date"] = pd.to_datetime(frame["date"])
    long = frame.melt(id_vars=["date", "MRI"], value_vars=list(COMPONENTS), var_name="column",
                      value_name="points")
    long["component"] = long["column"].map(lambda c: COMPONENTS[c][0])
    return long


def history_chart(history: list[dict], prediction: dict | None) -> alt.LayerChart:
    """Realised MRI split into its four weighted components, plus the 7-day forecast."""
    long = history_frame(history)
    order = [label for label, _ in COMPONENTS.values()]
    colours = [colour for _, colour in COMPONENTS.values()]
    area = alt.Chart(long).mark_area(opacity=0.9).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("points:Q", stack="zero", title="Market Risk Index", scale=alt.Scale(domainMin=0)),
        color=alt.Color("component:N", sort=order, scale=alt.Scale(domain=order, range=colours),
                        legend=alt.Legend(title="Contribution to the index", orient="bottom", columns=2,
                                          labelLimit=260)),
        order=alt.Order("column_order:Q"),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("component:N", title="Component"),
                 alt.Tooltip("points:Q", title="Points", format=".1f"),
                 alt.Tooltip("MRI:Q", title="Total MRI", format=".1f")],
    ).transform_calculate(column_order=f"indexof({order!r}, datum.component)")
    layers = [area]
    if prediction is not None:
        last = long.drop_duplicates("date").iloc[-1]
        path = pd.DataFrame({
            "date": [last["date"], pd.to_datetime(prediction["prediction_date"])],
            "MRI": [last["MRI"], prediction["prediction"]],
            "label": ["Latest", f"Forecast ({prediction['prediction_date']})"],
        })
        layers.append(alt.Chart(path).mark_line(strokeDash=[6, 4], strokeWidth=2, color="#0b0b0b")
                      .encode(x="date:T", y="MRI:Q"))
        layers.append(alt.Chart(path.tail(1)).mark_point(shape="diamond", size=160, filled=True,
                                                         color="#0b0b0b").encode(
            x="date:T", y="MRI:Q",
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("MRI:Q", title="MRI", format=".1f")]))
    return alt.layer(*layers).properties(height=400)


def comparison_frame(results: list[dict | None], tokens: list[str]) -> pd.DataFrame:
    rows = []
    for token, body in zip(tokens, results, strict=True):
        if body is None:
            rows.append({"Token": token, "Name": TOKENS[token], "Today": None, "Forecast": None,
                         "Change": None, "Risk level": "Unavailable", "Model": "-"})
            continue
        current = body.get("current_mri")
        rows.append({
            "Token": token, "Name": TOKENS[token], "Today": current, "Forecast": body["prediction"],
            "Change": None if current is None else body["prediction"] - current,
            "Risk level": body["risk_level"], "Model": body["model_used"],
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("Forecast", ascending=False, na_position="last").reset_index(drop=True)


def comparison_chart(frame: pd.DataFrame) -> alt.LayerChart:
    data = frame.dropna(subset=["Forecast"])
    domain = [band for _, _, band, _ in RISK_BANDS]
    colours = [colour for *_, colour in RISK_BANDS]
    bars = alt.Chart(data).mark_bar(cornerRadius=4, height=18).encode(
        x=alt.X("Forecast:Q", title=f"Forecast MRI in {FORECAST_HORIZON} days", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Token:N", sort=list(data["Token"]), title=None),
        color=alt.Color("Risk level:N", scale=alt.Scale(domain=domain, range=colours),
                        legend=alt.Legend(orient="bottom")),
        tooltip=["Token", "Name", alt.Tooltip("Today:Q", format=".1f"), alt.Tooltip("Forecast:Q", format=".1f"),
                 alt.Tooltip("Change:Q", format="+.1f"), "Risk level"],
    )
    with_today = data.dropna(subset=["Today"])
    if with_today.empty:
        return bars.properties(height=110 + 34 * len(data))
    today = alt.Chart(with_today).mark_tick(color="#0b0b0b", thickness=2, size=22).encode(
        x="Today:Q", y=alt.Y("Token:N", sort=list(data["Token"])),
        tooltip=["Token", alt.Tooltip("Today:Q", format=".1f", title="Today's MRI")],
    )
    return (bars + today).properties(height=110 + 34 * len(data))

def price_chart(history: list[dict], token: str, end_date: str | None = None) -> alt.Chart:
    """Daily close price (USD), on a date axis that can be extended to line up with the risk forecast."""
    frame = pd.DataFrame(history)
    frame["date"] = pd.to_datetime(frame["date"])
    fmt = ",.4f" if frame["close"].max() < 1 else ",.2f"
    domain = None
    if end_date is not None:
        domain = [frame["date"].min().isoformat(), pd.Timestamp(end_date).isoformat()]
    return alt.Chart(frame).mark_line(strokeWidth=2, color="#0d366b").encode(
        x=alt.X("date:T", title=None, scale=alt.Scale(domain=domain) if domain else alt.Undefined),
        y=alt.Y("close:Q", title=f"{token} close (USD)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", title="Close (USD)", format=fmt)],
    ).properties(height=220)


def composition_frame(history: list[dict]) -> pd.DataFrame:
    """Latest day's MRI split into its four weighted components (points and share)."""
    last = history[-1]
    rows = [{"component": label, "points": float(last[col]), "colour": colour}
            for col, (label, colour) in COMPONENTS.items()]
    frame = pd.DataFrame(rows)
    total = frame["points"].sum()
    frame["share"] = frame["points"] / total if total > 0 else 0.0
    return frame


def composition_chart(frame: pd.DataFrame) -> alt.Chart:
    order = list(frame["component"])
    return alt.Chart(frame).mark_bar(height=26, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("points:Q", stack="zero", title="Contribution to today's index (MRI points)"),
        color=alt.Color("component:N", sort=order, scale=alt.Scale(domain=order, range=list(frame["colour"])),
                        legend=alt.Legend(orient="bottom", title=None, columns=2, labelLimit=260)),
        order=alt.Order("rank:Q"),
        tooltip=[alt.Tooltip("component:N", title="Component"), alt.Tooltip("points:Q", format=".2f"),
                 alt.Tooltip("share:Q", title="Share", format=".0%")],
    ).transform_calculate(rank=f"indexof({order!r}, datum.component)").properties(height=150)


def pipeline_dot(shrink: float) -> str:
    """Graphviz description of how /predict turns raw data into an explained forecast."""
    return f"""digraph P {{
rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.55;
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, color="#c3c2b7", fillcolor="#ffffff"];
edge [fontname="Helvetica", fontsize=9, color="#898781", fontcolor="#52514e"];
k [label="Kraken daily OHLC\\n(cached 15 min)", fillcolor="#eef4fc"];
a [label="Student A /features\\nReturn20"]; c [label="Student C /features\\nVolatility20, MaxPrice20, GapRisk"];
f [label="Fallback chain\\nlive → cached ≤24 h →\\nlocal recompute → missing", fillcolor="#fff7e6"];
d [label="Derived inputs\\nDR = (MaxPrice20 − Close) / MaxPrice20\\nMRI(t) = brief formula"];
q [label="All 5 inputs?", shape=diamond, fillcolor="#f6f6f3"];
p [label="Primary: LightGBM residual\\nŷ = MRI(t) + {shrink:.3f} × Δ̂", fillcolor="#e3f5e3"];
b [label="Fallback: LightGBM level\\nNaN-aware default branches", fillcolor="#fbe3e3"];
o [label="Clip 0-100 · risk band\\nSHAP contributions\\nJSON response", fillcolor="#0d366b", fontcolor="#ffffff"];
k -> d [label="close"]; a -> f; c -> f; f -> d; d -> q; q -> p [label="yes"]; q -> b [label="no"]; p -> o; b -> o;
}}"""


# ----------------------------------------------------------------- page sections
def _base_url() -> str | None:
    return api_url(STUDENT_KEY)


def _payload(token: str, **extra) -> dict:
    return {"token": token, "forecast_horizon": FORECAST_HORIZON, **extra}


def _show_failure(what: str, result: core.ApiResult) -> None:
    st.error(f"Could not load the {what}: {result.error}")
    st.caption("The API runs on a free Render instance that sleeps when idle; the first request can take "
               "about a minute. Press **Refresh data** to try again.")


def _source_label(source: str | None) -> str:
    if not source:
        return "-"
    if source.startswith("local_fallback"):
        return "recomputed locally (shared package)"
    return source.replace("_api", " API").replace("student_", "Student ").replace("derived: ", "derived - ")


def _forecast_tab(token: str, history_days: int) -> None:
    base = _base_url()
    with st.spinner("Getting the latest forecast (the first request can take up to a minute)..."):
        result = core.cached_call(base, "/predict", _payload(token))
        history = core.cached_call(base, "/risk-history", _payload(token, days=history_days))
    if not result.ok:
        _show_failure("forecast", result)
        prediction = None
    else:
        prediction = result.data
        current = prediction.get("current_mri")
        forecast = prediction["prediction"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Risk today", "n/a" if current is None else f"{current:.1f}", border=True,
                  help=f"Market Risk Index on {prediction['as_of_date']} (latest complete daily candle).")
        c2.metric(f"Forecast in {FORECAST_HORIZON} days", f"{forecast:.1f}", border=True,
                  delta=None if current is None else f"{forecast - current:+.1f}", delta_color="inverse",
                  help="Higher = riskier. A red change means risk is expected to rise.")
        c3.metric("Forecast risk level", prediction["risk_level"], border=True)
        c4.metric("Forecast date", pd.Timestamp(prediction["prediction_date"]).strftime("%d %b"), border=True,
                  help=f"{prediction['prediction_date']}: {FORECAST_HORIZON} days after the latest complete "
                       f"candle ({prediction['as_of_date']}).")
        c5.metric("Model used", prediction["model_used"].capitalize(), border=True,
                  help="Primary = residual LightGBM with all five inputs; fallback = level LightGBM that "
                       "tolerates missing inputs.")
        st.altair_chart(risk_scale_chart(current, forecast), width="stretch")
        with st.container(border=True):
            st.markdown("**Analyst summary**")
            st.markdown(explain_in_words(prediction))
        _data_quality_notes(prediction)

        core.section(f"Market inputs on {prediction['as_of_date']}")
        inputs, sources = prediction["inputs"], prediction["feature_sources"]
        cols = st.columns(4)
        for col, name in zip(cols, ["Return20", "Volatility20", "DR", "GapRisk"], strict=True):
            shown = format_input(name, inputs.get(name)).replace(" per day", "")
            col.metric(FEATURE_LABELS[name] + (" (daily)" if name == "Volatility20" else ""), shown, border=True,
                       help=f"Source: {_source_label(sources.get(name))}")

    if history.ok and history.data.get("history"):
        rows = history.data["history"]
        core.section("Risk composition")
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(composition_chart(composition_frame(rows)), width="stretch")
        with right:
            comp = composition_frame(rows).sort_values("points", ascending=False).iloc[0]
            st.markdown(f"On {rows[-1]['date']} the index was **{rows[-1]['MRI']:.1f}**. The largest part "
                        f"(**{comp['share']:.0%}**) came from **{comp['component'].split(' (')[0].lower()}**.")
            st.caption("Each component is capped, so no single factor can exceed its weight "
                       "(40 / 30 / 20 / 10 points).")
        core.section(f"Price and risk - last {len(rows)} days")
        end = prediction["prediction_date"] if prediction is not None else None
        st.altair_chart(price_chart(rows, token, end), width="stretch")
        st.altair_chart(history_chart(rows, prediction), width="stretch")
        st.caption("Top: daily close on Kraken. Bottom: realised Market Risk Index split into its four weighted "
                   "components; the dashed line joins the latest value to the model's 7-day forecast.")
    elif not history.ok:
        _show_failure("risk history", history)

    if prediction is not None:
        core.section("Explainability")
        st.markdown("#### Why the model predicts this")
        frame = contributions_frame(prediction)
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(contributions_chart(frame), width="stretch")
        with right:
            baseline = prediction["explanation"]["baseline"]
            with st.container(border=True):
                st.markdown(f"**Starting point:** {baseline:.2f}  \n"
                            f"**+ effects of the inputs:** {frame['points'].sum():+.2f}  \n"
                            f"**= forecast:** {prediction['prediction']:.2f}")
            st.caption("Effects are exact SHAP values from LightGBM's tree path attribution (`pred_contrib`) for "
                       "this prediction, so starting point + effects = forecast. "
                       + ("For the primary model the starting point is today's risk plus the shrunk average "
                          "weekly change, and each effect is scaled by the same shrink factor."
                          if prediction["model_used"] == "primary"
                          else "For the fallback model the starting point is the average training risk."))
        with st.expander("All model inputs and their sources"):
            st.dataframe(pd.DataFrame([
                {"Input": FEATURE_LABELS.get(name, name), "Feature": name, "Value": format_input(name, value),
                 "Source": _source_label(prediction["feature_sources"].get(name))}
                for name, value in prediction["inputs"].items()
            ]), hide_index=True, width="stretch")


def _data_quality_notes(prediction: dict) -> None:
    sources = prediction["feature_sources"].values()
    if prediction["model_used"] == "fallback":
        missing = ", ".join(FEATURE_LABELS.get(f, f) for f in prediction["missing_features"])
        st.warning(f"Some inputs were unavailable ({missing}), so a backup model was used. This forecast is "
                   "less reliable than usual (test error rises from 3.7 to up to 11.0 points).", icon="⚠️")
    elif any(str(s).startswith("local_fallback") for s in sources):
        st.info("Peer feature services did not cover this request, so their features were recalculated from "
                "Kraken data with the group's shared feature package - the same code the peers use, so the "
                "forecast is unaffected.", icon="ℹ️")
    elif any("cached" in str(s) for s in sources):
        st.info("A teammate's service was briefly unavailable; its most recent answer was reused.", icon="ℹ️")


def _compare_tab() -> None:
    st.markdown("Forecast risk for all eight supported tokens, riskiest first. The black tick marks today's "
                "level, so a bar extending past its tick means risk is expected to rise.")
    tokens = list(TOKENS)
    with st.spinner("Forecasting all eight tokens..."):
        results = core.cached_call_many(_base_url(), "/predict", [_payload(t) for t in tokens])
    if not any(r.ok for r in results):
        _show_failure("forecasts", results[0])
        return
    frame = comparison_frame([r.data if r.ok else None for r in results], tokens)
    valid = frame.dropna(subset=["Forecast"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average forecast risk", f"{valid['Forecast'].mean():.1f}", border=True)
    c2.metric("Riskiest", f"{valid.iloc[0]['Token']} · {valid.iloc[0]['Forecast']:.1f}", border=True)
    c3.metric("Calmest", f"{valid.iloc[-1]['Token']} · {valid.iloc[-1]['Forecast']:.1f}", border=True)
    c4.metric("Tokens with rising risk", f"{int((valid['Change'] > 0).sum())} / {len(valid)}", border=True)
    st.altair_chart(comparison_chart(frame), width="stretch")
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "Today": st.column_config.NumberColumn(format="%.1f"),
        "Forecast": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
        "Change": st.column_config.NumberColumn(format="%+.1f"),
    })
    st.caption("One pooled LightGBM model serves all eight tokens; it was trained on 5,544 token-days so that "
               "patterns learned on volatile tokens also inform the calmer ones.")
    failed = [t for t, r in zip(tokens, results, strict=True) if not r.ok]
    if failed:
        st.warning(f"No forecast for: {', '.join(failed)}. Press Refresh data to retry.")


def _model_tab() -> None:
    result = core.cached_call(_base_url(), "/model-metadata")
    if not result.ok:
        _show_failure("model information", result)
        return
    meta = result.data
    metrics = meta.get("metrics") or {}
    period = meta.get("training_period") or {}
    st.markdown(f"**{meta['algorithm']}** · version {meta['model_version']} · target: {meta['target']} "
                f"{FORECAST_HORIZON} days ahead · data {period.get('start', '?')} to {period.get('end', '?')} "
                "(Kraken daily candles, 8 tokens).")
    if metrics:
        core.section("Accuracy on the unseen test period")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average error of this model", f"{metrics['test_mae']:.2f} points", border=True,
                  help="Mean absolute error on the test period (evaluated once).")
        c2.metric("'Next week = this week' rule", f"{metrics['persistence_test_mae']:.2f} points", border=True,
                  delta=f"{metrics['test_skill_vs_persistence']:.1%} better", delta_color="normal",
                  help="Persistence baseline: assumes risk will not change.")
        c3.metric("Always predicting the average", f"{metrics['mean_baseline_test_mae']:.2f} points", border=True)
        r2 = metrics.get("test_r2")
        c4.metric("R² (test)", "-" if r2 is None else f"{r2:.2f}", border=True,
                  help="Share of the variance in next week's risk explained by the model.")

    core.section("How a prediction is made")
    st.graphviz_chart(pipeline_dot(float(meta.get("shrink") or 1.0)), width="stretch")
    st.caption("Residual-on-persistence: the model predicts the *change* in risk over the week and only a tuned "
               "fraction of it (the shrink factor) is added to today's level, because risk is highly persistent "
               "and a full correction over-reacts. The shrink factor was tuned by Optuna on cross-validation, "
               "never on the test set.")

    left, right = st.columns(2)
    importance = meta.get("permutation_importance_test_mae_increase")
    if importance:
        with left:
            st.markdown("**Which inputs matter most** (permutation importance)")
            frame = pd.DataFrame({"Input": [FEATURE_LABELS.get(k, k) for k in importance],
                                  "Error increase": list(importance.values())})
            chart = alt.Chart(frame).mark_bar(cornerRadius=4, height=16, color="#2a78d6").encode(
                x=alt.X("Error increase:Q", title="Extra error when the input is shuffled (points)"),
                y=alt.Y("Input:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                tooltip=["Input", alt.Tooltip("Error increase:Q", format=".2f")],
            ).properties(height=260)
            st.altair_chart(chart, width="stretch")
    degradation = meta.get("degradation_test_mae")
    if degradation:
        with right:
            st.markdown("**Accuracy when a peer service is down**")
            frame = pd.DataFrame({"Scenario": list(degradation), "Error": list(degradation.values())})
            chart = alt.Chart(frame).mark_bar(cornerRadius=4, height=16, color="#eb6834").encode(
                x=alt.X("Error:Q", title="Average error on the test period (points)"),
                y=alt.Y("Scenario:N", sort=list(frame["Scenario"]), title=None, axis=alt.Axis(labelLimit=220)),
                tooltip=["Scenario", alt.Tooltip("Error:Q", format=".2f")],
            ).properties(height=260)
            st.altair_chart(chart, width="stretch")

    hyper = meta.get("hyperparameters")
    if hyper:
        core.section("Hyperparameters (Optuna)")
        optuna = meta.get("optuna") or {}
        left, right = st.columns([3, 2])
        with left:
            table = pd.DataFrame(hyper).rename(columns=str.capitalize)
            st.dataframe(table.map(lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)), width="stretch")
        with right:
            if optuna:
                st.markdown(
                    f"**Validation:** {optuna.get('cv', '-')}  \n"
                    f"**Residual model:** {optuna.get('residual', {}).get('n_trials', '?')} trials, best CV MAE "
                    f"{optuna.get('residual', {}).get('best_cv_mae', float('nan')):.2f}  \n"
                    f"**Level model:** {optuna.get('level', {}).get('n_trials', '?')} trials, best CV MAE "
                    f"{optuna.get('level', {}).get('best_cv_mae', float('nan')):.2f}  \n"
                    f"**Persistence CV MAE:** {optuna.get('persistence_cv_mae', float('nan')):.2f}  \n"
                    f"**Shrink factor:** {meta.get('shrink', '-')}"
                )
            space = meta.get("search_space")
            if space:
                with st.expander("Search space"):
                    st.dataframe(pd.DataFrame({"Parameter": list(space), "Range": list(space.values())}),
                                 hide_index=True, width="stretch")

    core.section("Limitations")
    st.markdown(
        "- Risk is very persistent: most of the forecast comes from today's level. The model's edge over "
        "\"next week = this week\" is about 5 %, largest for the more volatile tokens (XLM, XMR, DOGE, SOL); "
        "for BTC and ETH it was slightly *worse* than that simple rule on the test period.\n"
        "- It was trained on about two years of data that included the 2024 rally and the calmer 2026 market "
        "(a regime shift); sudden shocks (exchange failures, regulation news) are not in the inputs.\n"
        "- The index describes price risk only - not liquidity, custody or counterparty risk.\n"
        "- Gap risk is almost always zero in 24/7 crypto markets, so it carries little information."
    )


def _api_tab(token: str) -> None:
    base = _base_url()
    st.markdown(f"Base URL: `{base}` · interactive documentation at [{base}/docs]({base}/docs)"
                if base else "The API URL is not configured.")
    health = core.health(base)
    st.markdown(
        "Status: " + (core.pill(f"Online · {health.elapsed_seconds:.2f} s", "ok") if health.ok
                      else core.pill("Waking up / unavailable", "warn")),
        unsafe_allow_html=True,
    )
    st.markdown(
        "| Method | Endpoint | Purpose |\n|---|---|---|\n"
        "| GET | `/` | Service metadata and endpoint list |\n"
        "| GET | `/health` | Model loaded, peers configured |\n"
        "| POST | `/features` | Student B's features (MA5, MA20, TrendStrength) for peers |\n"
        "| POST | `/predict` | 7-day MRI forecast with SHAP explanation and feature provenance |\n"
        "| GET | `/model-metadata` | Target, features, importance, metrics, hyperparameters |\n"
        "| POST | `/risk-history` | Realised MRI and its four components (used by the charts) |"
    )
    st.markdown("**Example request**")
    st.code(f"curl -X POST {base}/predict -H 'Content-Type: application/json' \\\n"
            f"     -d '{{\"token\": \"{token}\", \"forecast_horizon\": {FORECAST_HORIZON}}}'", language="bash")
    result = core.cached_call(base, "/predict", _payload(token))
    if result.ok:
        st.markdown(f"**Live response for {token}**")
        st.json(result.data, expanded=1)
    st.markdown("**Error contract** - invalid input returns a JSON error, never a stack trace:")
    st.code('{"error": "Unsupported cryptocurrency", "supported_tokens": ["BTC", "ETH", ...]}   # HTTP 400\n'
            '{"error": "Unsupported forecast_horizon", "supported_horizons": [7]}             # HTTP 422',
            language="text")


# ----------------------------------------------------------------- entry point
def render() -> None:
    core.inject_css()
    member = MEMBERS[STUDENT_KEY]
    core.hero(
        f"Student {member.letter} · {member.name} ({member.student_id})",
        "Market Risk Index - 7-day forecast",
        "How risky will a cryptocurrency be one week from now? The Market Risk Index scores risk from 0 (calm) "
        "to 100 (extreme) from volatility, drawdown, recent losses and price gaps. Use it to size positions, "
        "set stop-losses or decide when to stay out of the market.",
        ["LightGBM · residual-on-persistence", "All 8 tokens · pooled model", "Optuna · 100 trials",
         "SHAP explanations", "5.3 % better than persistence", "Peer-failure resilient"],
    )

    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    token = c1.selectbox("Cryptocurrency", list(TOKENS), index=list(TOKENS).index(DEFAULT_TOKEN),
                         format_func=lambda t: f"{t} - {TOKENS[t]}", key="b_token")
    history_days = c2.select_slider("History shown (days)", options=HISTORY_OPTIONS, value=90, key="b_days")
    if c3.button("Refresh data", key="b_refresh", width="stretch"):
        core.clear_cache()
        st.rerun()

    tabs = st.tabs(["Forecast", "All 8 tokens", "Model & methodology", "API"])
    with tabs[0]:
        _forecast_tab(token, history_days)
    with tabs[1]:
        _compare_tab()
    with tabs[2]:
        _model_tab()
    with tabs[3]:
        _api_tab(token)

    with st.expander("How the Market Risk Index is calculated"):
        st.latex(r"\mathrm{MRI} = 100\Big(0.4\,\min\big(\tfrac{\mathrm{Vol}_{20}}{0.10},1\big)"
                 r" + 0.3\,\min\big(\tfrac{DR}{0.40},1\big)"
                 r" + 0.2\,\min\big(\tfrac{|\min(R_{20},0)|}{0.30},1\big)"
                 r" + 0.1\,\min\big(\tfrac{GR}{0.10},1\big)\Big)")
        st.markdown("Vol20 = standard deviation of daily returns over 20 days · DR = fall from the 20-day "
                    "high · R20 = 20-day return (only losses count) · GR = gap between the previous close "
                    "and today's open. Bands: Low < 20 ≤ Moderate < 40 ≤ High < 60 ≤ Very high.")
    st.caption("This tool gives an estimate of future risk for educational purposes. It is not financial "
               "advice; crypto-assets are volatile and you can lose money.")


if __name__ == "__main__":  # allows `streamlit run students/student_b.py` on its own
    render()
