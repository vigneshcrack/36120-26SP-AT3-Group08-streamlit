"""Student D page - Trading Opportunity Rating (TOR), a five-class rating seven days ahead.

The page answers one question: *should I buy, hold or sell this coin, judged a week from now?*
It is deliberately candid about how far that answer can be trusted - the model does not beat the
persistence baseline on the contract's primary metric, and the page says so rather than hiding it
behind a single confident-looking label.

Layout mirrors the other student pages: a hero, a token picker, then four tabs
(Rating · All 8 tokens · Model & methodology · API). Pure helpers at the top are unit-tested in
``tests/test_student_d.py``; everything that touches Streamlit lives below them.
"""

from __future__ import annotations

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from app import main as core
from app.main import DEFAULT_TOKEN, FORECAST_HORIZON, MEMBERS, TOKENS, api_url

STUDENT_KEY = "student_d"

# CONTRACT.md section 5: the exact strings the API returns, ordered worst -> best.
TOR_LABELS: list[str] = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]

# Diverging red -> blue, so the direction of a rating reads without the label.
RATING_COLOURS: dict[str, str] = {
    "Strong Sell": "#b02525",
    "Sell": "#e8706f",
    "Hold": "#9a9994",
    "Buy": "#5598e7",
    "Strong Buy": "#1c5cab",
}

# What each rating means as an action, so the page is useful without the brief to hand.
RATING_MEANING: dict[str, str] = {
    "Strong Sell": "conditions are poor - reduce or avoid exposure",
    "Sell": "conditions are weakening - trim exposure",
    "Hold": "no clear edge either way - keep current exposure",
    "Buy": "conditions are favourable - consider adding",
    "Strong Buy": "conditions are strongly favourable - the clearest signal the model gives",
}

FEATURE_LABELS: dict[str, str] = {
    "Return20": "20-day return",
    "RelativeVolume": "Relative volume",
    "Volatility20": "20-day volatility",
    "MA5": "5-day moving average",
    "MA20": "20-day moving average",
    "MA5_MA20_Ratio": "Trend (MA5 / MA20)",
    "OS": "Opportunity Score today",
    "VolumeSMA20": "20-day average volume",
}

HISTORY_OPTIONS = [30, 60, 90, 180, 365]

# Confidence bands. The model's mean confidence on the test set is ~0.5, so "high" is set where the
# probability is genuinely dominant rather than at an optimistic-looking threshold.
CONFIDENCE_BANDS: list[tuple[float, str]] = [(0.70, "High"), (0.45, "Moderate"), (0.0, "Low")]


# Where a service may put each optional metadata block. CONTRACT.md section 6.5 fixes
# only the required keys at the top level, so a service is free to group its extras -
# and Student D's groups them under `performance` and `training`. Looking in every
# plausible place is what stops this page rendering a silently empty table.
METADATA_SECTIONS = ("performance", "training")

# Keys the service exposes under a different name from the training artefact.
METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "best_params": ("hyperparameters",),
}


# ----------------------------------------------------------------- pure helpers (tested)
def metadata_value(metadata: dict, key: str, default=None):
    """Read `key` from /model-metadata wherever the service happens to put it.

    Checks the top level, then each known grouping section, then any documented
    alias. Returns `default` only when the value is genuinely absent, so a caller
    can tell "not reported" apart from "nested one level down".
    """
    for name in (key, *METADATA_ALIASES.get(key, ())):
        if metadata.get(name) not in (None, {}, []):
            return metadata[name]
        for section in METADATA_SECTIONS:
            block = metadata.get(section)
            if isinstance(block, dict) and block.get(name) not in (None, {}, []):
                return block[name]
    return default


def rating_index(label: str | None) -> int | None:
    """Position of a rating on the ordered scale, or ``None`` if it is not a known rating."""
    return TOR_LABELS.index(label) if label in TOR_LABELS else None


def rating_meaning(label: str | None) -> str:
    """What a rating implies as an action, tolerant of a label off the contract's scale.

    ``RATING_MEANING[label]`` would raise, and a ``KeyError`` here takes the whole page down over
    one unexpected string - the opposite of how every other unknown is handled. A rating the scale
    does not contain is reported as unrecognised instead.
    """
    if label in RATING_MEANING:
        return RATING_MEANING[label]
    return "not one of the five ratings this page knows - treat it as unrecognised"


def rating_colour(label: str | None) -> str:
    """Colour for a rating; neutral grey for anything off the scale."""
    return RATING_COLOURS.get(label, RATING_COLOURS["Hold"])


def format_metric(value, spec: str = ".3f") -> str:
    """Format a measured figure, or ``-`` when the service did not report one.

    ``metrics.get(key, nan)`` only defaults when the key is *absent*; an explicit ``null`` - which
    is what the contract requires for a value that could not be computed - reaches the format string
    as ``None`` and raises.
    """
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def confidence_band(confidence: float | None) -> str:
    """Bucket a predicted-class probability, so the number is interpretable at a glance."""
    if confidence is None:
        return "Unknown"
    for threshold, label in CONFIDENCE_BANDS:
        if confidence >= threshold:
            return label
    return "Low"


def format_input(name: str, value: float | None) -> str:
    """Render a model input in the unit a reader expects, not as a raw float."""
    if value is None:
        return "unavailable"
    if name in {"Return20"}:
        return f"{value:+.1%}"
    if name == "Volatility20":
        return f"{value:.2%} per day"
    if name == "RelativeVolume":
        return f"{value:.2f}x average"
    if name in {"MA5", "MA20"}:
        return f"${value:,.2f}" if value >= 1 else f"${value:,.4f}"
    if name == "VolumeSMA20":
        return f"{value:,.0f}"
    return f"{value:.3f}"


def probabilities_frame(prediction: dict) -> pd.DataFrame:
    """All five ratings with their probabilities, always in scale order (worst -> best).

    The API only has to return the distribution; forcing the full ordered label set here means a
    rating the model considers impossible still appears, as an explicit zero, instead of silently
    vanishing from the chart.
    """
    probabilities = prediction.get("probabilities") or {}
    predicted = prediction.get("prediction")
    return pd.DataFrame(
        [
            {
                "rating": label,
                "probability": float(probabilities.get(label, 0.0)),
                "predicted": label == predicted,
                "meaning": rating_meaning(label),
            }
            for label in TOR_LABELS
        ]
    )


def comparison_frame(results: list[dict | None], tokens: list[str]) -> pd.DataFrame:
    """One row per token, best opportunity first; tokens whose call failed are kept and labelled.

    Keeping the failures visible matters: silently dropping a coin would make a partial outage look
    like a complete answer.
    """
    rows = []
    for token, prediction in zip(tokens, results, strict=True):
        if prediction is None:
            rows.append({"Token": token, "Rating": "Unavailable", "Confidence": None, "_rank": -1})
            continue
        label = prediction.get("prediction")
        rows.append(
            {
                "Token": token,
                "Rating": label,
                "Confidence": prediction.get("confidence"),
                "_rank": rating_index(label) if rating_index(label) is not None else -1,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["_rank", "Confidence"], ascending=[False, False], na_position="last"
    )
    return frame.drop(columns="_rank").reset_index(drop=True)


def baseline_frame(metadata: dict) -> pd.DataFrame:
    """Model vs every baseline on macro-F1, strongest first, with the model marked.

    Built from ``/model-metadata`` rather than written down, so the page cannot drift away from the
    artefact the service actually loaded.
    """
    metrics = metadata_value(metadata, "metrics", {})
    rows = [
        {
            "Approach": "This model (LightGBM + Optuna)",
            "Macro-F1": metrics.get("test_macro_f1"),
            "Balanced accuracy": metrics.get("test_balanced_accuracy"),
            "Accuracy": metrics.get("test_accuracy"),
            "is_model": True,
        }
    ]
    pretty = {
        "persistence": "Carry today's rating forward",
        "majority_class": "Always predict the most common rating",
        "stratified_random": "Random, matching the class balance",
        "lightgbm_defaults": "LightGBM with default settings",
    }
    for name, scores in metadata_value(metadata, "baselines", {}).items():
        rows.append(
            {
                "Approach": pretty.get(name, name.replace("_", " ").capitalize()),
                "Macro-F1": scores.get("macro_f1"),
                "Balanced accuracy": scores.get("balanced_accuracy"),
                "Accuracy": scores.get("accuracy"),
                "is_model": False,
            }
        )
    return pd.DataFrame(rows).sort_values("Macro-F1", ascending=False, na_position="last").reset_index(drop=True)


def beats_persistence(metadata: dict) -> bool | None:
    """Whether the model clears the baseline that matters; ``None`` if either figure is missing."""
    model = metadata_value(metadata, "metrics", {}).get("test_macro_f1")
    persistence = (metadata_value(metadata, "baselines", {}).get("persistence") or {}).get("macro_f1")
    if model is None or persistence is None:
        return None
    return model > persistence


def confusion_frame(metadata: dict) -> pd.DataFrame:
    """Long-form confusion matrix with each cell's share of its actual class."""
    block = metadata_value(metadata, "confusion_matrix", {})
    labels, matrix = block.get("labels") or [], block.get("matrix") or []
    rows = []
    for actual, counts in zip(labels, matrix, strict=True):
        total = sum(counts) or 1
        for predicted, count in zip(labels, counts, strict=True):
            rows.append(
                {"actual": actual, "predicted": predicted, "count": int(count), "share": count / total}
            )
    return pd.DataFrame(rows)


def importance_frame(metadata: dict) -> pd.DataFrame:
    """Feature importance with the owning student attached, biggest first."""
    owners = {
        "Return20": "Student A",
        "MA5": "Student B",
        "MA20": "Student B",
        "MA5_MA20_Ratio": "Student B (derived)",
        "Volatility20": "Student C",
        "RelativeVolume": "Student D",
        "OS": "All peers (derived)",
    }
    rows = [
        {
            "Input": FEATURE_LABELS.get(name, name),
            "Feature": name,
            "Share of model": share,
            "Owner": owners.get(name, "-"),
        }
        for name, share in metadata_value(metadata, "feature_importance", {}).items()
    ]
    return pd.DataFrame(rows).sort_values("Share of model", ascending=False).reset_index(drop=True)


def degradation_frame(metadata: dict) -> pd.DataFrame:
    """Macro-F1 when each peer service is unavailable, as the brief's resilience requirement asks."""
    pretty = {
        "all_peers_available": "All services available",
        "student_a_down": "Student A unavailable",
        "student_b_down": "Student B unavailable",
        "student_c_down": "Student C unavailable",
        "all_peers_down": "All peers unavailable",
    }
    rows = [
        {"Scenario": pretty.get(name, name), "Macro-F1": scores.get("macro_f1"),
         "Accuracy": scores.get("accuracy")}
        for name, scores in metadata_value(metadata, "degradation", {}).items()
    ]
    return pd.DataFrame(rows)


def history_frame(history: list[dict]) -> pd.DataFrame:
    """Tidy the ``/features`` history rows into a frame the charts can use."""
    frame = pd.DataFrame(history)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def explain_in_words(prediction: dict) -> str:
    """A short analyst note, assembled only from values the API actually returned."""
    label = prediction.get("prediction")
    confidence = prediction.get("confidence")
    token = prediction.get("token", "this token")
    as_of = prediction.get("as_of_date", "the latest complete candle")
    target_date = prediction.get("prediction_date", f"{FORECAST_HORIZON} days ahead")

    if label is None:
        return "The service did not return a rating for this request."

    lines = [
        f"On **{as_of}** the model rates **{token}** as **{label}** for **{target_date}** "
        f"({FORECAST_HORIZON} days ahead) - {rating_meaning(label)}."
    ]
    if confidence is not None:
        band = confidence_band(confidence)
        lines.append(
            f"Confidence is **{confidence:.0%}** ({band.lower()}). The rating is the single most likely of "
            f"five, not a certainty - size any decision by this number."
        )
    current = prediction.get("current_rating")
    forecast_rank, current_rank = rating_index(label), rating_index(current)
    if current_rank is not None:
        if current == label:
            lines.append(f"Today's rating is also **{current}**, so the model expects no change this week.")
        elif forecast_rank is not None:
            direction = "an improvement" if forecast_rank > current_rank else "a deterioration"
            lines.append(f"Today's rating is **{current}**, so this forecast is {direction}.")
        else:
            lines.append(f"Today's rating is **{current}**.")

    missing = prediction.get("missing_features") or []
    if missing:
        names = ", ".join(FEATURE_LABELS.get(f, f) for f in missing)
        lines.append(
            f"⚠️ Some inputs were unavailable ({names}); the rating was produced without them and is less "
            f"reliable than usual."
        )
    return "  \n".join(lines)


# ----------------------------------------------------------------- charts
def probability_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Probability across the five ratings, in scale order, predicted class outlined."""
    order = TOR_LABELS
    base = alt.Chart(frame).encode(
        y=alt.Y("rating:N", sort=order, title=None),
        x=alt.X("probability:Q", title="Probability", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip("rating:N", title="Rating"),
            alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            alt.Tooltip("meaning:N", title="Means"),
        ],
    )
    bars = base.mark_bar().encode(
        color=alt.Color("rating:N", sort=order, legend=None,
                        scale=alt.Scale(domain=order, range=[RATING_COLOURS[r] for r in order])),
        stroke=alt.condition(alt.datum.predicted, alt.value("#111"), alt.value(None)),
        strokeWidth=alt.condition(alt.datum.predicted, alt.value(1.6), alt.value(0)),
    )
    labels = base.mark_text(align="left", dx=4, fontSize=11).encode(
        text=alt.Text("probability:Q", format=".0%")
    )
    return (bars + labels).properties(height=190)


def confusion_chart(frame: pd.DataFrame) -> alt.LayerChart:
    """Heat map of actual vs predicted on the test period, shaded by row share."""
    order = TOR_LABELS
    base = alt.Chart(frame).encode(
        x=alt.X("predicted:N", sort=order, title="Predicted"),
        y=alt.Y("actual:N", sort=order, title="Actual (7 days later)"),
    )
    cells = base.mark_rect().encode(
        color=alt.Color("share:Q", title="Share of actual", scale=alt.Scale(scheme="blues", domain=[0, 1])),
        tooltip=[
            alt.Tooltip("actual:N", title="Actual"),
            alt.Tooltip("predicted:N", title="Predicted"),
            alt.Tooltip("count:Q", title="Days"),
            alt.Tooltip("share:Q", title="Share of actual", format=".0%"),
        ],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("count:Q"),
        color=alt.condition(alt.datum.share > 0.5, alt.value("white"), alt.value("#222")),
    )
    return (cells + text).properties(height=240)


def importance_chart(frame: pd.DataFrame) -> alt.Chart:
    """Which inputs the model leans on, coloured by the student who owns each."""
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            y=alt.Y("Input:N", sort="-x", title=None),
            x=alt.X("Share of model:Q", title="Share of model", axis=alt.Axis(format="%")),
            color=alt.Color("Owner:N", title="Owned by"),
            tooltip=[
                alt.Tooltip("Input:N"),
                alt.Tooltip("Feature:N", title="API key"),
                alt.Tooltip("Share of model:Q", format=".1%"),
                alt.Tooltip("Owner:N", title="Owned by"),
            ],
        )
        .properties(height=220)
    )


def volume_chart(frame: pd.DataFrame, token: str) -> alt.LayerChart:
    """Relative volume over time - the feature Student D owns and publishes to the group."""
    line = (
        alt.Chart(frame)
        .mark_line(color="#2a78d6", strokeWidth=1.8)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("RelativeVolume:Q", title="Relative volume (x 20-day average)"),
            tooltip=[alt.Tooltip("date:T", title="Date"),
                     alt.Tooltip("RelativeVolume:Q", title="Relative volume", format=".2f")],
        )
    )
    normal = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(strokeDash=[4, 4], color="#8a8984")
        .encode(y="y:Q")
    )
    return (line + normal).properties(height=200, title=f"{token} trading volume against its 20-day average")


def price_chart(frame: pd.DataFrame, token: str) -> alt.Chart:
    """Daily close, for context under the volume chart."""
    return (
        alt.Chart(frame)
        .mark_line(color="#1c5cab", strokeWidth=1.8)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("close:Q", title="Close (USD)", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("close:Q", title="Close", format=",.2f")],
        )
        .properties(height=180, title=f"{token} daily close")
    )


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
    return (source.replace("_api", " API").replace("student_", "Student ")
            .replace("derived: ", "derived - ").replace("owned by ", "owned by "))


def _rating_tab(token: str, history_days: int) -> None:
    base = _base_url()
    with st.spinner("Getting the latest rating (the first request can take up to a minute)..."):
        result = core.cached_call(base, "/predict", _payload(token))
        features = core.cached_call(base, "/features", _payload(token, history_days=history_days))

    if not result.ok:
        _show_failure("rating", result)
    else:
        prediction = result.data
        label = prediction.get("prediction")
        confidence = prediction.get("confidence")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Rating in {FORECAST_HORIZON} days", label or "-", border=True,
                  help=f"The model's rating for {prediction.get('prediction_date', 'the forecast date')}.")
        c2.metric("Confidence", "-" if confidence is None else f"{confidence:.0%}", border=True,
                  help="Probability of the predicted rating. The other four share the remainder.")
        c3.metric("Confidence level", confidence_band(confidence), border=True)
        c4.metric("Forecast date",
                  pd.Timestamp(prediction["prediction_date"]).strftime("%d %b")
                  if prediction.get("prediction_date") else "-", border=True,
                  help=f"{FORECAST_HORIZON} days after the latest complete candle "
                       f"({prediction.get('as_of_date', 'unknown')}).")

        if label:
            st.markdown(
                f'<div style="border-left:6px solid {rating_colour(label)};padding:10px 14px;'
                f'background:rgba(0,0,0,0.03);border-radius:6px">'
                f'<strong>{escape(str(label))}</strong> - {rating_meaning(label)}</div>',
                unsafe_allow_html=True,
            )

        core.section("How sure is the model?")
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(probability_chart(probabilities_frame(prediction)), width="stretch")
        with right:
            with st.container(border=True):
                st.markdown("**Analyst summary**")
                st.markdown(explain_in_words(prediction))
            st.caption("The five bars always sum to 100%. A tall single bar is a confident call; several "
                       "similar bars mean the model is genuinely unsure and the rating should carry less "
                       "weight in a decision.")

        inputs = prediction.get("inputs") or {}
        sources = prediction.get("feature_sources") or {}
        if inputs:
            core.section(f"Market inputs on {prediction.get('as_of_date', 'the latest candle')}")
            names = [n for n in ["Return20", "MA5_MA20_Ratio", "RelativeVolume", "Volatility20"] if n in inputs]
            for col, name in zip(st.columns(max(1, len(names))), names, strict=False):
                col.metric(FEATURE_LABELS.get(name, name), format_input(name, inputs.get(name)), border=True,
                           help=f"Source: {_source_label(sources.get(name))}")

        missing = prediction.get("missing_features") or []
        if missing:
            st.warning(
                "Some peer feature services were unavailable, so this rating was produced with missing "
                f"inputs ({', '.join(FEATURE_LABELS.get(f, f) for f in missing)}). The model is trained to "
                "cope with that, but see the degradation table under **Model & methodology** for what it "
                "costs.",
                icon="⚠️",
            )
        elif any(str(s).startswith("local_fallback") for s in sources.values()):
            st.info("A peer feature service did not answer, so its features were recomputed from Kraken data "
                    "with the group's shared package - the same code the peer uses, so the rating is "
                    "unaffected.", icon="ℹ️")

    if features.ok and (features.data or {}).get("history"):
        frame = history_frame(features.data["history"])
        core.section(f"Volume and price - last {len(frame)} days")
        if "RelativeVolume" in frame:
            st.altair_chart(volume_chart(frame, token), width="stretch")
        if "close" in frame:
            st.altair_chart(price_chart(frame, token), width="stretch")
        st.caption("Relative volume is Student D's own feature: today's volume divided by its 20-day average. "
                   "The dashed line is 1.0 (an average day); spikes above it mark unusually heavy trading, "
                   "which is the 20% volume term in the Opportunity Score.")
    elif not features.ok:
        _show_failure("feature history", features)


def _compare_tab() -> None:
    st.markdown("The rating for all eight supported tokens, best opportunity first. One model serves every "
                "token, so these are directly comparable - but see the per-token accuracy under "
                "**Model & methodology**: reliability varies a lot by coin.")
    tokens = list(TOKENS)
    with st.spinner("Rating all eight tokens..."):
        results = core.cached_call_many(_base_url(), "/predict", [_payload(t) for t in tokens])
    frames = [r.data if r.ok else None for r in results]
    if all(f is None for f in frames):
        _show_failure("ratings", results[0])
        return

    frame = comparison_frame(frames, tokens)
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0,
                                                          max_value=1),
        },
    )
    unavailable = int((frame["Rating"] == "Unavailable").sum())
    if unavailable:
        st.caption(f"{unavailable} of {len(tokens)} tokens could not be rated; they are listed last rather "
                   "than hidden.")


def _model_tab() -> None:
    result = core.cached_call(_base_url(), "/model-metadata")
    if not result.ok:
        _show_failure("model information", result)
        return
    meta = result.data
    metrics = metadata_value(meta, "metrics", {})
    period = metadata_value(meta, "training_period", {})

    st.markdown(f"**{meta.get('algorithm', 'LightGBM Classifier')}** · version "
                f"{meta.get('model_version', '?')} · target: {meta.get('target', 'Trading Opportunity Rating')} "
                f"{FORECAST_HORIZON} days ahead · trained on {period.get('start', '?')} to "
                f"{period.get('end', '?')} (Kraken daily candles, "
                f"{len(metadata_value(meta, 'supported_tokens') or []) or 8} tokens).")

    core.section("Accuracy on the unseen test period")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Macro-F1", format_metric(metrics.get("test_macro_f1")), border=True,
              help="Primary metric: the average F1 across all five ratings, so the rare ratings count as "
                   "much as the common ones.")
    c2.metric("Balanced accuracy", format_metric(metrics.get("test_balanced_accuracy")), border=True,
              help="Mean recall across the five ratings.")
    c3.metric("Accuracy", format_metric(metrics.get("test_accuracy")), border=True)
    c4.metric("Calibration error", format_metric(metrics.get("test_ece")), border=True,
              help="Expected calibration error: the gap between the stated confidence and how often the "
                   "model is actually right. Lower is better.")

    verdict = beats_persistence(meta)
    frame = baseline_frame(meta)
    core.section("Compared with the alternatives")
    if verdict is False:
        st.warning(
            "**Read this before acting on a rating.** On macro-F1 - the metric the group agreed as primary - "
            "this model does **not** beat simply carrying today's rating forward for a week. It is clearly "
            "better than guessing, and it is the only option here that reports a calibrated confidence, but "
            "it has not demonstrated that it predicts *change* better than assuming no change.",
            icon="⚠️",
        )
    elif verdict is True:
        st.success("The model beats every baseline, including carrying today's rating forward.", icon="✅")
    st.dataframe(
        frame.drop(columns="is_model"),
        hide_index=True,
        width="stretch",
        column_config={
            "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
            "Balanced accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    note = metadata_value(meta, "metrics_note")
    if note:
        st.caption(note)

    left, right = st.columns(2)
    with left:
        st.markdown("**Where the model gets it wrong**")
        confusion = confusion_frame(meta)
        if not confusion.empty:
            st.altair_chart(confusion_chart(confusion), width="stretch")
            st.caption("Most errors sit next to the diagonal - one rating out, which barely changes what an "
                       "investor would do. Errors two or more ratings away are the expensive ones.")
        in_test = metadata_value(meta, "labels_in_test")
        if in_test and len(in_test) < len(TOR_LABELS):
            absent = [label for label in TOR_LABELS if label not in in_test]
            st.caption(f"⚠️ {', '.join(absent)} did not occur at all in the test period, so there is **no "
                       f"evidence here** about how well the model identifies it.")
    with right:
        st.markdown("**Which inputs the model uses**")
        importance = importance_frame(meta)
        if not importance.empty:
            st.altair_chart(importance_chart(importance), width="stretch")
            st.caption("Six of the seven inputs come from a teammate's service, which is why the resilience "
                       "table below matters.")

    recall = metadata_value(meta, "per_class_recall", {})
    if recall:
        core.section("How often each rating is caught")
        st.dataframe(
            pd.DataFrame([{"Rating": label, "Recall": recall.get(label)} for label in TOR_LABELS]),
            hide_index=True, width="stretch",
            column_config={"Recall": st.column_config.ProgressColumn("Recall", format="%.2f", min_value=0,
                                                                     max_value=1)},
        )

    degradation = degradation_frame(meta)
    if not degradation.empty:
        core.section("When a teammate's service is down")
        st.dataframe(degradation, hide_index=True, width="stretch",
                     column_config={"Macro-F1": st.column_config.NumberColumn(format="%.3f"),
                                    "Accuracy": st.column_config.NumberColumn(format="%.3f")})
        st.caption("The model is trained with peer features randomly blanked, so it keeps answering when a "
                   "service is unreachable instead of failing the request.")

    per_token = metadata_value(meta, "per_token", {})
    if per_token:
        core.section("Reliability by token")
        rows = [{"Token": token, "Macro-F1": scores.get("macro_f1"), "Accuracy": scores.get("accuracy"),
                 "Test days": scores.get("rows")} for token, scores in per_token.items()]
        st.dataframe(pd.DataFrame(rows).sort_values("Macro-F1", ascending=False), hide_index=True,
                     width="stretch",
                     column_config={"Macro-F1": st.column_config.NumberColumn(format="%.3f"),
                                    "Accuracy": st.column_config.NumberColumn(format="%.3f")})
        st.caption("One model serves all eight tokens, and it is markedly weaker on some. The interface looks "
                   "the same for every coin, so this table is the honest caveat.")

    core.section("How the model was tuned")
    st.markdown(
        f"- **Search:** Optuna TPE, {metadata_value(meta, 'n_trials', '?')} trials, "
        f"seed {metadata_value(meta, 'random_seed', 42)}\n"
        f"- **Cross-validation:** {metadata_value(meta, 'cv_strategy', 'not reported')}\n"
        f"- **Best cross-validated macro-F1:** "
        f"{format_metric(metadata_value(meta, 'best_cv_macro_f1'))} "
        f"(the test figure above is lower, as it should be)\n"
        f"- **Data:** {metadata_value(meta, 'data_source', 'Kraken daily candles')}"
    )
    with st.expander("Selected hyperparameters and the space they were searched in"):
        best = metadata_value(meta, "best_params", {})
        space = metadata_value(meta, "search_space", {})
        # `Selected` mixes numbers and strings (e.g. 158 and "balanced"); Arrow cannot type a mixed
        # column, so render everything as text rather than letting Streamlit guess and warn.
        st.dataframe(
            pd.DataFrame([{"Hyperparameter": k, "Selected": f"{best.get(k)}",
                           "Searched over": space.get(k, "-")} for k in best]),
            hide_index=True, width="stretch",
        )


def _api_tab(token: str) -> None:
    base = _base_url()
    if base:
        st.markdown(f"Base URL: `{base}` · interactive documentation at [{base}/docs]({base}/docs)")
        health = core.health(base)
        st.markdown(
            "Status: " + (core.pill(f"Online · {health.elapsed_seconds:.2f} s", "ok") if health.ok
                          else core.pill("Waking up / unavailable", "warn")),
            unsafe_allow_html=True,
        )
    else:
        st.markdown("Status: " + core.pill("Not deployed yet", "off"), unsafe_allow_html=True)
        st.caption("Set `STUDENT_D_API_URL` as an environment variable or in `.streamlit/secrets.toml` once "
                   "the service is live.")

    member = MEMBERS[STUDENT_KEY]
    st.markdown(
        "| Method | Endpoint | Purpose |\n|---|---|---|\n"
        "| GET | `/` | Service metadata and endpoint list |\n"
        "| GET | `/health` | Model loaded, peers configured |\n"
        f"| POST | `/features` | Student D's features ({', '.join(member.owns)}) for peers |\n"
        "| POST | `/predict` | 7-day rating with confidence and the full probability distribution |\n"
        "| GET | `/model-metadata` | Target, features, importance, metrics, hyperparameters |"
    )
    st.markdown("**Example request**")
    st.code(f"curl -X POST {base or 'https://<service>'}/predict -H 'Content-Type: application/json' \\\n"
            f"     -d '{{\"token\": \"{token}\", \"forecast_horizon\": {FORECAST_HORIZON}}}'", language="bash")

    if base:
        result = core.cached_call(base, "/predict", _payload(token))
        if result.ok:
            st.markdown(f"**Live response for {token}**")
            st.json(result.data, expanded=1)

    st.markdown("**Error contract** - invalid input returns a JSON error, never a stack trace:")
    st.code('{"error": "Unsupported cryptocurrency", "supported_tokens": ["BTC", "ETH", ...]}   # HTTP 400\n'
            '{"error": "Unsupported forecast_horizon", "supported_horizons": [7]}             # HTTP 422\n'
            '{"error": "Market data source unavailable", "retry_after_seconds": 60}           # HTTP 503',
            language="text")


# ----------------------------------------------------------------- entry point
def render() -> None:
    core.inject_css()
    member = MEMBERS[STUDENT_KEY]
    core.hero(
        f"Student {member.letter} · {member.name} ({member.student_id})",
        "Trading Opportunity Rating - 7-day outlook",
        "Buy, hold or sell? This rates the trading opportunity for a cryptocurrency one week ahead on a "
        "five-point scale, from Strong Sell to Strong Buy, and reports how confident it is in that call.",
        ["LightGBM Classifier", "5 ordered ratings", "All 8 tokens · pooled model",
         "Calibrated confidence", "Peer-failure resilient"],
    )

    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    token = c1.selectbox("Cryptocurrency", list(TOKENS), index=list(TOKENS).index(DEFAULT_TOKEN),
                         format_func=lambda t: f"{t} - {TOKENS[t]}", key="d_token")
    history_days = c2.select_slider("History shown (days)", options=HISTORY_OPTIONS, value=90, key="d_days")
    if c3.button("Refresh data", key="d_refresh", width="stretch"):
        core.clear_cache()
        st.rerun()

    tabs = st.tabs(["Rating", "All 8 tokens", "Model & methodology", "API"])
    with tabs[0]:
        _rating_tab(token, history_days)
    with tabs[1]:
        _compare_tab()
    with tabs[2]:
        _model_tab()
    with tabs[3]:
        _api_tab(token)

    with st.expander("How the Trading Opportunity Rating is calculated"):
        st.latex(r"\mathrm{OS} = 0.40\,M + 0.30\,T + 0.20\,V - 0.10\,R")
        st.markdown(
            "- $M = (R_{20} + 0.20)/0.40$ - **momentum**, the 20-day return (Student A)\n"
            "- $T = \\mathrm{MA}_5/\\mathrm{MA}_{20}$ - **trend**, short against long moving average "
            "(Student B)\n"
            "- $V = \\min(\\mathrm{RelVol}, 2)/2$ - **volume**, today against its 20-day average "
            "(Student D)\n"
            "- $R = \\min(\\mathrm{Vol}_{20}/0.10, 1)$ - **volatility penalty** (Student C)\n\n"
            "The score is then cut into five ratings: Strong Sell below 0.20, Sell to 0.40, Hold to 0.60, "
            "Buy to 0.80, Strong Buy at 0.80 and above. The model predicts the rating that the formula will "
            "produce seven days from now, using only information available today."
        )
    st.caption("This tool gives an estimate for educational purposes. It is not financial advice; "
               "crypto-assets are volatile and you can lose money.")


if __name__ == "__main__":  # allows `streamlit run students/student_d.py` on its own
    render()
