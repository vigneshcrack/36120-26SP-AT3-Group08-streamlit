"""Student A page - Asset Performance Index (API) 7-day forecast.

Author: Ryan Cruikshank (13191263), Group 08.

The page calls Student A's FastAPI service (``/predict``, ``/features`` and
``/model-metadata``) and turns the responses into a 7-day forecast, a plain-language
analyst summary, an API component composition chart, a price & performance history chart,
an all-8-token comparison, and an honest summary of model accuracy and limitations.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app import main as core
from app.main import DEFAULT_TOKEN, FORECAST_HORIZON, MEMBERS, TOKENS, api_url

STUDENT_KEY = "student_a"

# performance definitions
PERF_BANDS: list[tuple[float, float, str, str]] = [
    (0, 35, "Weak", "#d03b3b"),
    (35, 50, "Moderate", "#ec835a"),
    (50, 70, "Good", "#2a78d6"),
    (70, 100, "Strong", "#0ca30c"),
]
RAISES, LOWERS = "Boosts performance", "Drags performance"
DIRECTION_COLOURS = {RAISES: "#0ca30c", LOWERS: "#e34948"}
COMPONENTS = {
    "return_strength_points": ("Return strength (40 %)", "#2a78d6"),
    "trend_strength_points": ("Trend strength (30 %)", "#1baf7a"),
    "relative_volume_points": ("Relative volume (20 %)", "#eda100"),
    "price_stability_points": ("Price stability (10 %)", "#eb6834"),
}
FEATURE_LABELS = {
    "Return20": "20-day return",
    "TrendStrength": "Trend strength (MA5/MA20 - 1)",
    "RelativeVolume": "Relative volume (Vol/SMA20)",
    "Volatility20": "20-day volatility",
    "Return5": "5-day return",
    "DailyReturn": "Daily return",
    "Momentum": "20-day momentum (USD)",
    "API": "Today's performance index",
}
HISTORY_OPTIONS = [30, 60, 90, 180]


#helpers
def performance_level(value: float | None) -> str:
    if value is None:
        return "Unknown"
    for low, high, label, _ in PERF_BANDS:
        if low <= value < high:
            return label
    return "Strong"


def format_input(name: str, value: float | None) -> str:
    """Human-readable value of a model input."""
    if value is None:
        return "unavailable"
    if name in {"Return20", "Return5", "DailyReturn", "TrendStrength"}:
        return f"{value:+.1%}"
    if name == "Volatility20":
        return f"{value:.2%} per day"
    if name == "RelativeVolume":
        return f"{value:.2f}x"
    if name == "Momentum":
        return f"${value:+,.4f}" if abs(value) < 1 else f"${value:+,.2f}"
    return f"{value:.1f}"


def contributions_frame(prediction: dict) -> pd.DataFrame:
    """One row per grouped driver with its SHAP contribution (API points), largest effect first."""
    contributions = prediction.get("explanation", {}).get("contributions", {})
    inputs = prediction.get("inputs", {})
    frame = pd.DataFrame(
        {
            "feature": list(contributions),
            "label": [FEATURE_LABELS.get(f, f) for f in contributions],
            "value": [format_input(f, inputs.get(f)) for f in contributions],
            "points": [float(v) for v in contributions.values()],
        }
    )
    frame["direction"] = frame["points"].map(lambda p: RAISES if p >= 0 else LOWERS)
    return frame.reindex(frame["points"].abs().sort_values(ascending=False).index).reset_index(drop=True)

#explanations for outputs
def explain_in_words(prediction: dict) -> str:
    token = prediction.get("token") or prediction.get("symbol", "BTC")
    forecast = float(prediction.get("prediction", prediction.get("predicted_api_t7", 50.0)))
    current = prediction.get("current_api", prediction.get("current_api_score"))
    date = prediction.get("prediction_date") or prediction.get("target_date", "t+7")

    if current is None:
        text = (
            f"{token}'s Asset Performance Index is forecast to be "
            f"**{forecast:.1f}/100 ({performance_level(forecast)})** on {date}. "
            "Today's level could not be calculated because some inputs were unavailable."
        )
    else:
        current = float(current)
        change = forecast - current
        if abs(change) < 0.5:
            movement = f"stay about the same at **{forecast:.1f}**"
        else:
            movement = f"{'rise' if change > 0 else 'fall'} to **{forecast:.1f}** ({change:+.1f} points)"
        text = (
            f"{token}'s performance index is **{current:.1f}/100 ({performance_level(current)})** today "
            f"and is forecast to {movement} by {date} - **{performance_level(forecast)}** performance."
        )

    frame = contributions_frame(prediction)
    frame = frame[frame["feature"] != "API"]
    up = frame[frame["points"] > 0.05]
    down = frame[frame["points"] < -0.05]
    drivers = []
    if not up.empty:
        row = up.iloc[0]
        drivers.append(
            f"the main factor boosting the forecast is **{row['label'].lower()}** "
            f"({row['value']}, {row['points']:+.1f} points)"
        )
    if not down.empty:
        row = down.iloc[0]
        drivers.append(
            f"the main factor weighing on it is **{row['label'].lower()}** "
            f"({row['value']}, {row['points']:+.1f} points)"
        )
    if drivers:
        text += " Compared with the baseline, " + "; ".join(drivers) + "."
    return text

#chart to show performance scale
def performance_scale_chart(current: float | None, forecast: float) -> alt.LayerChart:
    """0-100 scale with the four performance bands, today's value, and the 7-day forecast."""
    bands = pd.DataFrame(PERF_BANDS, columns=["start", "end", "band", "colour"]).assign(y0=0.0, y1=1.0)
    x_scale = alt.Scale(domain=[0, 100], nice=False)
    y_scale = alt.Scale(domain=[0, 1.6])
    band_layer = alt.Chart(bands).mark_rect(opacity=0.35, cornerRadius=4, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("start:Q", scale=x_scale, title="Asset Performance Index (0-100)"),
        x2="end:Q",
        y=alt.Y("y0:Q", scale=y_scale, axis=None),
        y2="y1:Q",
        color=alt.Color(
            "band:N",
            scale=alt.Scale(domain=[b[2] for b in PERF_BANDS], range=[b[3] for b in PERF_BANDS]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("band:N", title="Band"),
            alt.Tooltip("start:Q", title="From"),
            alt.Tooltip("end:Q", title="To"),
        ],
    )
    band_text = (
        alt.Chart(bands)
        .transform_calculate(mid="(datum.start + datum.end) / 2", ty="1.3")
        .mark_text(fontSize=12, color="#52514e")
        .encode(x=alt.X("mid:Q", scale=x_scale), y=alt.Y("ty:Q", scale=y_scale), text="band:N")
    )
    points = [
        {
            "what": f"Forecast (+{FORECAST_HORIZON} days)",
            "value": forecast,
            "y": 0.32,
            "label": f"{forecast:.1f}",
        }
    ]
    if current is not None:
        points.insert(0, {"what": "Today", "value": float(current), "y": 0.68, "label": f"{float(current):.1f}"})
    marker_frame = pd.DataFrame(points)
    shapes = ["circle", "diamond"] if current is not None else ["diamond"]
    marks = (
        alt.Chart(marker_frame)
        .mark_point(size=240, filled=True, color="#0b0b0b", stroke="#fcfcfb", strokeWidth=2, opacity=1)
        .encode(
            x=alt.X("value:Q", scale=x_scale),
            y=alt.Y("y:Q", scale=y_scale),
            shape=alt.Shape(
                "what:N",
                scale=alt.Scale(domain=list(marker_frame["what"]), range=shapes),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[alt.Tooltip("what:N", title=""), alt.Tooltip("value:Q", title="API", format=".1f")],
        )
    )
    return (band_layer + band_text + marks).properties(height=170)


def contributions_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Horizontal bars showing how much each feature group moved the forecast (API points)."""
    bars = alt.Chart(frame).mark_bar(cornerRadius=4, height=18).encode(
        x=alt.X("points:Q", title="Effect on the forecast (API points)"),
        y=alt.Y("label:N", sort=list(frame["label"]), title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=list(DIRECTION_COLOURS), range=list(DIRECTION_COLOURS.values())),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Input"),
            alt.Tooltip("value:N", title="Value"),
            alt.Tooltip("points:Q", title="Effect (points)", format="+.2f"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#898781").encode(x="x:Q")
    return (bars + zero).properties(height=110 + 36 * len(frame))


def _ensure_component_points(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" not in d.columns and "Date" in d.columns:
        d["date"] = d["Date"]
    if "close" not in d.columns and "Close" in d.columns:
        d["close"] = d["Close"]
    if "API" not in d.columns and "Current_API" in d.columns:
        d["API"] = d["Current_API"]
    if "return_strength_points" not in d.columns:
        d["return_strength_points"] = 40.0 * d.get("RS", 0.5)
        d["trend_strength_points"] = 30.0 * d.get("TS", 0.5)
        d["relative_volume_points"] = 20.0 * d.get("RV", 0.5)
        d["price_stability_points"] = 10.0 * d.get("PS", 0.97)
    return d


def history_frame(history: list[dict]) -> pd.DataFrame:
    frame = _ensure_component_points(pd.DataFrame(history))
    frame["date"] = pd.to_datetime(frame["date"])
    long = frame.melt(
        id_vars=["date", "API"],
        value_vars=list(COMPONENTS),
        var_name="column",
        value_name="points",
    )
    long["component"] = long["column"].map(lambda c: COMPONENTS[c][0])
    return long

#adding more charts
def history_chart(history: list[dict], prediction: dict | None) -> alt.LayerChart:
    long = history_frame(history)
    order = [label for label, _ in COMPONENTS.values()]
    colours = [colour for _, colour in COMPONENTS.values()]
    area = (
        alt.Chart(long)
        .mark_area(opacity=0.9)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("points:Q", stack="zero", title="Asset Performance Index", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "component:N",
                sort=order,
                scale=alt.Scale(domain=order, range=colours),
                legend=alt.Legend(
                    title="Contribution to the index",
                    orient="bottom",
                    columns=2,
                    labelLimit=260,
                ),
            ),
            order=alt.Order("column_order:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("component:N", title="Component"),
                alt.Tooltip("points:Q", title="Points", format=".1f"),
                alt.Tooltip("API:Q", title="Total API", format=".1f"),
            ],
        )
        .transform_calculate(column_order=f"indexof({order!r}, datum.component)")
    )
    layers = [area]
    if prediction is not None:
        last = long.drop_duplicates("date").iloc[-1]
        pred_date = prediction.get("prediction_date") or prediction.get("target_date")
        pred_val = float(prediction.get("prediction", prediction.get("predicted_api_t7", last["API"])))
        path = pd.DataFrame(
            {
                "date": [last["date"], pd.to_datetime(pred_date)],
                "API": [last["API"], pred_val],
                "label": ["Latest", f"Forecast ({pred_date})"],
            }
        )
        layers.append(
            alt.Chart(path)
            .mark_line(strokeDash=[6, 4], strokeWidth=2, color="#0b0b0b")
            .encode(x="date:T", y="API:Q")
        )
        layers.append(
            alt.Chart(path.tail(1))
            .mark_point(shape="diamond", size=160, filled=True, color="#0b0b0b")
            .encode(
                x="date:T",
                y="API:Q",
                tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("API:Q", title="API", format=".1f")],
            )
        )
    return alt.layer(*layers).properties(height=400)

#adding more charts
def comparison_frame(results: list[dict | None], tokens: list[str]) -> pd.DataFrame:
    rows = []
    for token, body in zip(tokens, results, strict=True):
        if body is None:
            rows.append(
                {
                    "Token": token,
                    "Name": TOKENS[token],
                    "Today": None,
                    "Forecast": None,
                    "Change": None,
                    "Performance level": "Unavailable",
                    "Model": "-",
                }
            )
            continue
        current = body.get("current_api", body.get("current_api_score"))
        forecast = float(body.get("prediction", body.get("predicted_api_t7")))
        rows.append(
            {
                "Token": token,
                "Name": TOKENS[token],
                "Today": None if current is None else float(current),
                "Forecast": forecast,
                "Change": None if current is None else forecast - float(current),
                "Performance level": body.get("performance_level") or performance_level(forecast),
                "Model": body.get("model_used", "primary"),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("Forecast", ascending=False, na_position="last").reset_index(drop=True)

#adding more charts
def comparison_chart(frame: pd.DataFrame) -> alt.LayerChart:
    data = frame.dropna(subset=["Forecast"])
    domain = [band for _, _, band, _ in PERF_BANDS]
    colours = [colour for *_, colour in PERF_BANDS]
    bars = alt.Chart(data).mark_bar(cornerRadius=4, height=18).encode(
        x=alt.X("Forecast:Q", title=f"Forecast API in {FORECAST_HORIZON} days", scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("Token:N", sort=list(data["Token"]), title=None),
        color=alt.Color(
            "Performance level:N",
            scale=alt.Scale(domain=domain, range=colours),
            legend=alt.Legend(orient="bottom"),
        ),
        tooltip=[
            "Token",
            "Name",
            alt.Tooltip("Today:Q", format=".1f"),
            alt.Tooltip("Forecast:Q", format=".1f"),
            alt.Tooltip("Change:Q", format="+.1f"),
            "Performance level",
        ],
    )
    with_today = data.dropna(subset=["Today"])
    if with_today.empty:
        return bars.properties(height=110 + 34 * len(data))
    today = alt.Chart(with_today).mark_tick(color="#0b0b0b", thickness=2, size=22).encode(
        x="Today:Q",
        y=alt.Y("Token:N", sort=list(data["Token"])),
        tooltip=["Token", alt.Tooltip("Today:Q", format=".1f", title="Today's API")],
    )
    return (bars + today).properties(height=110 + 34 * len(data))

#adding more charts
def price_chart(history: list[dict], token: str, end_date: str | None = None) -> alt.Chart:
    frame = _ensure_component_points(pd.DataFrame(history))
    frame["date"] = pd.to_datetime(frame["date"])
    fmt = ",.4f" if frame["close"].max() < 1 else ",.2f"
    domain = None
    if end_date is not None:
        domain = [frame["date"].min().isoformat(), pd.Timestamp(end_date).isoformat()]
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, color="#0d366b")
        .encode(
            x=alt.X("date:T", title=None, scale=alt.Scale(domain=domain) if domain else alt.Undefined),
            y=alt.Y("close:Q", title=f"{token} close (USD)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", title="Close (USD)", format=fmt)],
        )
        .properties(height=220)
    )


def composition_frame(history: list[dict]) -> pd.DataFrame:
    """Latest day's API split into its four weighted components (points and share)."""
    frame_in = _ensure_component_points(pd.DataFrame(history))
    last = frame_in.iloc[-1]
    rows = [
        {"component": label, "points": float(last[col]), "colour": colour}
        for col, (label, colour) in COMPONENTS.items()
    ]
    frame = pd.DataFrame(rows)
    total = frame["points"].sum()
    frame["share"] = frame["points"] / total if total > 0 else 0.0
    return frame

#adding more charts
def composition_chart(frame: pd.DataFrame) -> alt.Chart:
    order = list(frame["component"])
    return (
        alt.Chart(frame)
        .mark_bar(height=26, stroke="#fcfcfb", strokeWidth=2)
        .encode(
            x=alt.X("points:Q", stack="zero", title="Contribution to today's index (API points)"),
            color=alt.Color(
                "component:N",
                sort=order,
                scale=alt.Scale(domain=order, range=list(frame["colour"])),
                legend=alt.Legend(orient="bottom", title=None, columns=2, labelLimit=260),
            ),
            order=alt.Order("rank:Q"),
            tooltip=[
                alt.Tooltip("component:N", title="Component"),
                alt.Tooltip("points:Q", format=".2f"),
                alt.Tooltip("share:Q", title="Share", format=".0%"),
            ],
        )
        .transform_calculate(rank=f"indexof({order!r}, datum.component)")
        .properties(height=150)
    )


def pipeline_dot() -> str:
    """Graphviz description of how /predict turns raw data and peer features into an API forecast."""
    return """digraph P {
rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.55;
node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, color="#c3c2b7", fillcolor="#ffffff"];
edge [fontname="Helvetica", fontsize=9, color="#898781", fontcolor="#52514e"];
k [label="Kraken daily OHLC\\n(complete UTC candles)", fillcolor="#eef4fc"];
a [label="Student A Owned\\nDailyReturn, Return5,\\nReturn20, Momentum"];
p [label="Peers (/features)\\nB: TrendStrength\\nC: Volatility20\\nD: RelativeVolume"];
f [label="Resilience Layer\\nlive peer -> payload ->\\nlocal formula fallback", fillcolor="#fff7e6"];
e [label="47 Stationary Features\\n13d overlap anchors, RSI, MACD,\\nT-1/3/5/7 lags & velocity deltas"];
m [label="Pooled 8-Token\\nXGBoost Regressor\\n(Optuna TPE, 50 trials)", fillcolor="#e3f5e3"];
o [label="Clip 0-100 · Perf band\\nTree SHAP attributions\\nJSON response", fillcolor="#0d366b", fontcolor="#ffffff"];
k -> a; k -> f; p -> f; a -> e; f -> e; e -> m; m -> o;
}"""


#defining page sections
def _base_url() -> str | None:
    return api_url(STUDENT_KEY)


def _payload(token: str, **extra) -> dict:
    return {"token": token, "forecast_horizon": FORECAST_HORIZON, **extra}


def _show_failure(what: str, result: core.ApiResult) -> None:
    st.error(f"Could not load the {what}: {result.error}")
    st.caption(
        "The API runs on a free Render instance that sleeps when idle; the first request can take "
        "about a minute. Press **Refresh data** to try again."
    )


def _source_label(source: str | None) -> str:
    if not source:
        return "-"
    if "Fallback" in source or source.startswith("local_fallback"):
        return "recomputed locally (shared formula)"
    return source.replace("_api", " API").replace("student_", "Student ")


def _forecast_tab(token: str, history_days: int) -> None:
    base = _base_url()
    with st.spinner("Getting the latest forecast (the first request can take up to a minute)..."):
        result = core.cached_call(base, "/predict", _payload(token, history_days=history_days))

    if not result.ok:
        _show_failure("forecast", result)
        return

    prediction = result.data
    current = prediction.get("current_api", prediction.get("current_api_score"))
    forecast = float(prediction.get("prediction", prediction.get("predicted_api_t7")))
    as_of = prediction.get("as_of_date") or prediction.get("reference_date", "")
    pred_date = prediction.get("prediction_date") or prediction.get("target_date", "")
    perf_band = prediction.get("performance_level") or performance_level(forecast)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Performance today",
        "n/a" if current is None else f"{float(current):.1f}",
        border=True,
        help=f"Asset Performance Index on {as_of} (latest complete daily candle).",
    )
    c2.metric(
        f"Forecast in {FORECAST_HORIZON} days",
        f"{forecast:.1f}",
        border=True,
        delta=None if current is None else f"{forecast - float(current):+.1f}",
        delta_color="normal",
        help="Higher = stronger performance (0-100 scale).",
    )
    c3.metric("Performance band", perf_band, border=True)
    c4.metric(
        "Forecast date",
        pd.Timestamp(pred_date).strftime("%d %b"),
        border=True,
        help=f"{pred_date}: {FORECAST_HORIZON} days after the latest complete candle ({as_of}).",
    )
    c5.metric(
        "Model used",
        prediction.get("model_used", "primary").capitalize(),
        border=True,
        help="Pooled 8-cryptocurrency Optuna-tuned XGBoost Regressor.",
    )

    st.altair_chart(performance_scale_chart(current, forecast), width="stretch")
    with st.container(border=True):
        st.markdown("**Analyst summary**")
        st.markdown(explain_in_words(prediction))

    sources = prediction.get("feature_sources") or prediction.get("external_feature_sources", {})
    if any("fallback" in str(s).lower() for s in sources.values()):
        st.info(
            "One or more peer feature services were unreachable or not yet deployed, so their inputs "
            "were recomputed locally from Kraken data using the exact group contract formula.",
            icon="ℹ️",
        )

    core.section(f"Market inputs on {as_of}")
    inputs = prediction.get("inputs") or prediction.get("required_brief_features_current", {})
    cols = st.columns(4)
    for col, name in zip(cols, ["Return20", "TrendStrength", "RelativeVolume", "Volatility20"], strict=True):
        shown = format_input(name, inputs.get(name)).replace(" per day", "")
        col.metric(
            FEATURE_LABELS[name],
            shown,
            border=True,
            help=f"Source: {_source_label(sources.get(name))}",
        )

    rows = prediction.get("history") or prediction.get("historical_series", [])
    if rows:
        rows = rows[-history_days:]
        core.section("Performance composition")
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(composition_chart(composition_frame(rows)), width="stretch")
        with right:
            comp = composition_frame(rows).sort_values("points", ascending=False).iloc[0]
            latest_api = float(rows[-1].get("API", rows[-1].get("Current_API", current or 0.0)))
            st.markdown(
                f"On {as_of} the index was **{latest_api:.1f}**. The largest driver "
                f"(**{comp['share']:.0%}**) came from **{comp['component'].split(' (')[0].lower()}**."
            )
            st.caption(
                "Return strength (40 %), Trend strength (30 %), and Relative volume (20 %) are bounded "
                "on [0, 1], while Price stability (10 %) penalises 20-day return volatility."
            )

        core.section(f"Price and performance - last {len(rows)} days")
        st.altair_chart(price_chart(rows, token, pred_date), width="stretch")
        st.altair_chart(history_chart(rows, prediction), width="stretch")
        st.caption(
            "Top: daily close on Kraken. Bottom: realised Asset Performance Index split into its four "
            "weighted components; the dashed line connects today's score to the 7-day forecast."
        )

    if "explanation" in prediction and "contributions" in prediction["explanation"]:
        core.section("Explainability")
        st.markdown("#### Why the model predicts this")
        frame = contributions_frame(prediction)
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(contributions_chart(frame), width="stretch")
        with right:
            baseline = float(prediction["explanation"].get("baseline", current or 50.0))
            with st.container(border=True):
                st.markdown(
                    f"**Baseline expectation:** {baseline:.2f}  \n"
                    f"**+ net effect of inputs:** {frame['points'].sum():+.2f}  \n"
                    f"**= forecast:** {forecast:.2f}"
                )
            st.caption(
                "Effects are exact tree-path SHAP attributions (`pred_contribs`) from the trained "
                "XGBoost Regressor, aggregated into the six primary feature groups."
            )

#chart to forecast asset performance across all 8 assets
def _compare_tab() -> None:
    st.markdown(
        "Forecast Asset Performance Index for all eight supported tokens, strongest first. "
        "The black tick marks today's level, so a bar extending past its tick means performance is expected to improve."
    )
    tokens = list(TOKENS)
    with st.spinner("Forecasting all eight tokens..."):
        results = core.cached_call_many(_base_url(), "/predict", [_payload(t) for t in tokens])
    if not any(r.ok for r in results):
        _show_failure("forecasts", results[0])
        return
    frame = comparison_frame([r.data if r.ok else None for r in results], tokens)
    valid = frame.dropna(subset=["Forecast"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average forecast API", f"{valid['Forecast'].mean():.1f}", border=True)
    c2.metric("Strongest next week", f"{valid.iloc[0]['Token']} · {valid.iloc[0]['Forecast']:.1f}", border=True)
    c3.metric("Weakest next week", f"{valid.iloc[-1]['Token']} · {valid.iloc[-1]['Forecast']:.1f}", border=True)
    c4.metric("Tokens improving", f"{int((valid['Change'] > 0).sum())} / {len(valid)}", border=True)

    st.altair_chart(comparison_chart(frame), width="stretch")
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Today": st.column_config.NumberColumn(format="%.1f"),
            "Forecast": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
            "Change": st.column_config.NumberColumn(format="%+.1f"),
        },
    )
    st.caption(
        "One unified XGBoost Regressor serves all eight tokens; non-stationary raw dollar prices are "
        "excluded so tree splits generalise across BTC (~$73k) and DOGE (~$0.09)."
    )
    failed = [t for t, r in zip(tokens, results, strict=True) if not r.ok]
    if failed:
        st.warning(f"No forecast for: {', '.join(failed)}. Press Refresh data to retry.")


def _model_tab() -> None:
    result = core.cached_call(_base_url(), "/model-metadata")
    if not result.ok:
        _show_failure("model information", result)
        return
    meta = result.data
    test_m = meta.get("test_metrics") or meta.get("metrics") or {}
    st.markdown(
        f"**{meta.get('algorithm', 'XGBoost Regressor')}** · version {meta.get('version', '1.0.0')} · "
        f"target: `API_T+7` ({FORECAST_HORIZON} days ahead) · pooled across all 8 cryptocurrencies."
    )

    if test_m:
        core.section("Accuracy on the unseen test period")
        rmse = float(test_m.get("rmse", 13.0336))
        mae = float(test_m.get("mae", 9.6631))
        r2 = float(test_m.get("r2", 0.5230))
        naive_rmse = float(test_m.get("naive_baseline_rmse", 16.4969))
        skill = (naive_rmse - rmse) / naive_rmse if naive_rmse else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test RMSE (this model)", f"{rmse:.2f} points", border=True)
        c2.metric(
            "Persistence baseline RMSE",
            f"{naive_rmse:.2f} points",
            border=True,
            delta=f"{skill:.1%} lower error",
            delta_color="normal",
        )
        c3.metric("Test MAE", f"{mae:.2f} points", border=True)
        c4.metric("R² (test)", f"{r2:.3f}", border=True)

    core.section("How a prediction is made")
    st.graphviz_chart(pipeline_dot(), width="stretch")
    st.caption(
        "13-day overlap anchoring: because RS, TS, and PS use 20-day rolling windows, 13 of the 20 days "
        "in the future T+7 window are already observed at day T (65 % overlap). Engineering explicit "
        "13-day anchors alongside T-1, T-3, T-5, and T-7 velocity lags allows XGBoost to anticipate "
        "rolling window roll-offs."
    )

    left, right = st.columns(2)
    importance = meta.get("feature_importance", {})
    if importance:
        with left:
            st.markdown("**Top 12 features by XGBoost gain importance**")
            top_items = list(importance.items())[:12]
            frame = pd.DataFrame({"Feature": [k for k, _ in top_items], "Gain importance": [float(v) for _, v in top_items]})
            chart = (
                alt.Chart(frame)
                .mark_bar(cornerRadius=4, height=16, color="#2a78d6")
                .encode(
                    x=alt.X("Gain importance:Q", title="Relative gain importance"),
                    y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                    tooltip=["Feature", alt.Tooltip("Gain importance:Q", format=".4f")],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")

    per_coin = meta.get("per_coin_test_metrics", [])
    if per_coin:
        with right:
            st.markdown("**Per-token Test RMSE: XGBoost vs Persistence Baseline**")
            pc_df = pd.DataFrame(per_coin)
            st.dataframe(pc_df, hide_index=True, width="stretch")

    best_params = meta.get("best_hyperparameters") or meta.get("best_params")
    if best_params:
        core.section("Best Hyperparameters (Optuna 50 Trials)")
        st.dataframe(
            pd.DataFrame(
                {"Hyperparameter": list(best_params.keys()), "Value": [str(v) for v in best_params.values()]}
            ),
            hide_index=True,
            width="stretch",
        )

    core.section("Limitations")
    st.markdown(
        "- **Boundary Saturation:** 70 % of the API formula (RS and TS) clips hard at `[0, 1]`, creating "
        "saturation plateaus during extreme bull or bear rallies.\n"
        "- **Low-Variance Tokens:** While the model beats the persistence baseline on 7 of the 8 tokens "
        "(reducing BTC RMSE from 9.28 to 7.41 and DOGE RMSE from 19.20 to 13.03), TRX has unusually low "
        "target variance where naive persistence is slightly lower (5.82 vs 6.61).\n"
        "- **Exogenous Shocks:** Macro announcements or exchange outages are not captured in OHLCV lags."
    )


def _api_tab(token: str) -> None:
    base = _base_url()
    st.markdown(
        f"Base URL: `{base}` · interactive documentation at [{base}/docs]({base}/docs)"
        if base
        else "The API URL is not configured."
    )
    h = core.health(base)
    st.markdown(
        "Status: "
        + (core.pill(f"Online · {h.elapsed_seconds:.2f} s", "ok") if h.ok else core.pill("Waking up / unavailable", "warn")),
        unsafe_allow_html=True,
    )
    st.markdown(
        "| Method | Endpoint | Purpose |\n|---|---|---|\n"
        "| GET | `/` | Service metadata, owned features, and endpoint list |\n"
        "| GET | `/health` | Model artifact status and supported tokens |\n"
        "| POST | `/features` | Student A's features (`DailyReturn`, `Return5`, `Return20`, `Momentum`) for peers |\n"
        "| POST | `/predict` | 7-day `API_T+7` forecast with tree SHAP attributions and history |\n"
        "| GET | `/model-metadata` | Target, 47 features, gain importances, per-token RMSE, Optuna params |"
    )
    st.markdown("**Example request**")
    st.code(
        f"curl -X POST {base}/predict -H 'Content-Type: application/json' \\\n"
        f"     -d '{{\"token\": \"{token}\", \"forecast_horizon\": {FORECAST_HORIZON}}}'",
        language="bash",
    )
    result = core.cached_call(base, "/predict", _payload(token))
    if result.ok:
        st.markdown(f"**Live response for {token}**")
        st.json(result.data, expanded=1)


#injections
def render() -> None:
    core.inject_css()
    member = MEMBERS[STUDENT_KEY]
    core.hero(
        f"Student {member.letter} · {member.name} ({member.student_id})",
        "Asset Performance Index - 7-day forecast",
        "How strongly will a cryptocurrency perform one week from now? The Asset Performance Index (API) "
        "combines 20-day return strength (40 %), trend strength (30 %), relative volume (20 %), and price "
        "stability (10 %) into a single 0-100 score.",
        [
            "XGBoost Regressor",
            "All 8 tokens · pooled model",
            "Optuna · 50 TPE trials",
            "Tree SHAP explanations",
            "Beats persistence on 7/8 tokens",
        ],
    )

    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    token = c1.selectbox(
        "Cryptocurrency",
        list(TOKENS),
        index=list(TOKENS).index(DEFAULT_TOKEN),
        format_func=lambda t: f"{t} - {TOKENS[t]}",
        key="a_token",
    )
    history_days = c2.select_slider("History shown (days)", options=HISTORY_OPTIONS, value=90, key="a_days")
    if c3.button("Refresh data", key="a_refresh", width="stretch"):
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

    with st.expander("How the Asset Performance Index (API) is calculated"):
        st.latex(
            r"\mathrm{API} = 100\Big(0.4\,\mathrm{RS} + 0.3\,\mathrm{TS} + 0.2\,\mathrm{RV} + 0.1\,\mathrm{PS}\Big)"
        )
        st.markdown(
            "where $\\mathrm{RS} = \\min(\\max((\\mathrm{Return}_{20} + 0.2)/0.4, 0), 1)$, "
            "$\\mathrm{TS} = \\min(\\max((\\mathrm{TrendStrength} + 0.1)/0.2, 0), 1)$, "
            "$\\mathrm{RV} = \\min(\\mathrm{RelativeVolume}, 2)/2$, and "
            "$\\mathrm{PS} = 1 - \\mathrm{Volatility}_{20}$."
        )
    st.caption(
        "This tool provides an estimate of future asset performance for educational purposes. "
        "It is not financial advice."
    )


if __name__ == "__main__":
    render()
