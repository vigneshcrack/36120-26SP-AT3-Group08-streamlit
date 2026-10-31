"""Group 08 crypto forecasting web app - entry point and shared building blocks.

Run with ``streamlit run app/main.py``. As required by the AT3 brief, the repository holds
``app/main.py`` plus one page per student in ``students/student_x.py``. This file contains:

1. group configuration - team, tokens and API URLs (env vars / Streamlit secrets / defaults);
2. shared presentation helpers - page styling, hero banner, badges and status pills;
3. a shared HTTP client - timeouts, retries and caching, returning results instead of raising;
4. the Overview page - service status, market snapshot, architecture, team and methodology;
5. navigation between the Overview and the four student pages.

Student pages import the shared helpers with ``from app import main as core``. Page setup and
navigation only run when this file is executed by Streamlit (``__name__ == "__main__"``), so
importing it from a student page has no side effects.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # Streamlit puts app/ (not the repository root) on the path
    sys.path.insert(0, str(ROOT))



# ====================================================================================================
# 1. Group configuration
# ====================================================================================================

GROUP = "08"
SUBJECT = "36120 Advanced Machine Learning Application - AT3 (Spring 2026)"
FORECAST_HORIZON = 7


@dataclass(frozen=True)
class Member:
    key: str  # "student_a" ... "student_d"
    letter: str
    name: str
    student_id: str
    target: str
    task: str  # "Regression" or "Classification"
    algorithm: str
    tokens: tuple[str, ...]  # tokens this member's service forecasts
    owns: tuple[str, ...]  # features served by this member's /features endpoint
    consumes: dict[str, tuple[str, ...]]  # peer letter -> features used from that peer
    default_api_url: str | None


# symbol -> display name (the eight tokens agreed in CONTRACT.md)
TOKENS: dict[str, str] = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "XRP": "XRP",
    "SOL": "Solana",
    "DOGE": "Dogecoin",
    "XLM": "Stellar",
    "TRX": "TRON",
    "XMR": "Monero",
}
DEFAULT_TOKEN = "BTC"
ALL_TOKENS = tuple(TOKENS)

TEAM: list[Member] = [
    Member("student_a", "A", "Ryan Cruikshank", "13191263", "Asset Performance Index", "Regression",
           "XGBoost Regressor", ("BTC",), ("DailyReturn", "Return5", "Return20", "Momentum"),
           {"B": ("TrendStrength",), "C": ("Volatility20",), "D": ("RelativeVolume",)}, None),
    Member("student_b", "B", "Vignesh Selvam", "25740287", "Market Risk Index", "Regression",
           "LightGBM Regressor", ALL_TOKENS, ("MA5", "MA20", "TrendStrength"),
           {"A": ("Return20",), "C": ("Volatility20", "MaxPrice20", "GapRisk")},
           "https://group08-studentb-mri-api.onrender.com"),
    Member("student_c", "C", "Nandeeswari Kumaravelu", "25980436", "Trend Category", "Classification",
           "XGBoost Classifier", ALL_TOKENS, ("Volatility20", "MaxPrice20", "MaxDrawdown", "GapRisk"),
           {"A": ("Return20", "Return5"), "B": ("MA5", "MA20")}, None),
    # Student D's model is pooled across every supported token, not BTC only: the notebook trains on all
    # eight and models/student_d/metrics.json lists all eight under `supported_tokens`.
    Member("student_d", "D", "Vinnie Bui", "26160438", "Trading Opportunity Rating", "Classification",
           "LightGBM Classifier", ALL_TOKENS, ("VolumeSMA20", "RelativeVolume"),
           {"A": ("Return20",), "B": ("MA5", "MA20"), "C": ("Volatility20",)},
           "https://group08-studentd-tor-api.onrender.com"),
]
MEMBERS = {m.key: m for m in TEAM}
BY_LETTER = {m.letter: m for m in TEAM}


def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)  # raises if no secrets file exists at all
    except Exception:  # noqa: BLE001 - no secrets configured is a normal state
        return None
    return str(value) if value else None


def api_url(student_key: str) -> str | None:
    """Base URL of a student's API, e.g. ``api_url("student_b")``; ``None`` if not deployed yet."""
    env_name = f"{student_key.upper()}_API_URL"  # e.g. STUDENT_B_API_URL
    value = os.getenv(env_name) or _secret(env_name) or MEMBERS[student_key].default_api_url
    return value.strip().rstrip("/") if value else None

# ====================================================================================================
# 2. Presentation helpers
# ====================================================================================================

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.g8-hero {
  background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%);
  border-radius: 14px; padding: 26px 30px 22px; color: #ffffff; margin-bottom: 1.1rem;
}
.g8-hero .g8-eyebrow {font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.85;}
.g8-hero h1 {color: #ffffff; font-size: 2.05rem; line-height: 1.2; margin: 0.25rem 0 0.4rem; padding: 0;}
.g8-hero p {color: #e6eefb; font-size: 1.0rem; margin: 0 0 0.8rem; max-width: 900px;}
.g8-badge {
  display: inline-block; padding: 3px 10px; margin: 0 6px 6px 0; border-radius: 999px;
  font-size: 0.78rem; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.35);
  color: #ffffff;
}
.g8-pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;}
.g8-pill.ok {background: #e3f5e3; color: #006300;}
.g8-pill.warn {background: #fff3d6; color: #7a5200;}
.g8-pill.off {background: #efefec; color: #52514e;}
.g8-pill.bad {background: #fbe3e3; color: #a32121;}
.g8-section {font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #52514e;
  margin: 1.4rem 0 0.2rem;}
.g8-muted {color: #52514e; font-size: 0.9rem;}
.g8-mono {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85rem;}
div[data-testid="stMetric"] {background: #fcfcfb;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str, badges: list[str]) -> None:
    badge_html = "".join(f'<span class="g8-badge">{escape(b)}</span>' for b in badges)
    st.markdown(
        f'<div class="g8-hero"><div class="g8-eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1>'
        f"<p>{escape(subtitle)}</p>{badge_html}</div>",
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "off") -> str:
    """HTML for a small status pill; ``kind`` is one of ok, warn, off, bad."""
    return f'<span class="g8-pill {kind}">{escape(text)}</span>'


def section(label: str) -> None:
    st.markdown(f'<div class="g8-section">{escape(label)}</div>', unsafe_allow_html=True)

# ====================================================================================================
# 3. Shared HTTP client
# ====================================================================================================

logger = logging.getLogger("group08_app")

DEFAULT_TIMEOUT_SECONDS = 70  # a sleeping free Render instance needs ~50 s to wake up
DEFAULT_RETRIES = 1
CACHE_TTL_SECONDS = 600


@dataclass
class ApiResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    status: int | None = None
    elapsed_seconds: float = 0.0


class ApiCallError(RuntimeError):
    """Raised inside cached helpers so that failures are not stored in the cache."""

    def __init__(self, result: ApiResult):
        super().__init__(result.error)
        self.result = result


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if isinstance(body, dict) and body.get("error"):
        return f"{body['error']} (HTTP {response.status_code})"
    return f"HTTP {response.status_code}"


def call_api(
    base_url: str | None,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
) -> ApiResult:
    """GET ``base_url + path`` (or POST ``payload`` as JSON) and never raise.

    Connection errors, timeouts and 5xx responses are retried ``retries`` times; 4xx
    responses are returned immediately because retrying a bad request cannot help.
    """
    if not base_url:
        return ApiResult(ok=False, error="This API has not been deployed or configured yet.")
    url = f"{base_url.rstrip('/')}{path}"
    start = time.perf_counter()
    last = ApiResult(ok=False, error="Unknown error")
    for attempt in range(retries + 1):
        try:
            if payload is None:
                response = requests.get(url, timeout=timeout)
            else:
                response = requests.post(url, json=payload, timeout=timeout)
        except requests.Timeout:
            last = ApiResult(ok=False, error="The service took too long to respond (it may be waking up).")
        except requests.ConnectionError:
            last = ApiResult(ok=False, error="The service could not be reached.")
        else:
            if response.ok:
                try:
                    data = response.json()
                except ValueError:
                    last = ApiResult(ok=False, error="The service returned an invalid response.",
                                     status=response.status_code)
                else:
                    return ApiResult(ok=True, data=data, status=response.status_code,
                                     elapsed_seconds=time.perf_counter() - start)
            else:
                last = ApiResult(ok=False, error=_error_message(response), status=response.status_code)
                if response.status_code < 500:
                    break
        logger.warning("API call %s failed (attempt %d): %s", url, attempt + 1, last.error)
        if attempt < retries:
            time.sleep(1.0 * (attempt + 1))
    last.elapsed_seconds = time.perf_counter() - start
    return last


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached(base_url: str | None, path: str, payload: dict | None) -> dict:
    result = call_api(base_url, path, payload)
    if not result.ok:
        raise ApiCallError(result)
    return result.data


def cached_call(base_url: str | None, path: str, payload: dict | None = None) -> ApiResult:
    """Like :func:`call_api` but successful responses are cached for 10 minutes."""
    try:
        return ApiResult(ok=True, data=_cached(base_url, path, payload))
    except ApiCallError as exc:
        return exc.result


def cached_call_many(base_url: str | None, path: str, payloads: list[dict]) -> list[ApiResult]:
    """Run several cached calls concurrently (e.g. one forecast per token)."""
    ctx = get_script_run_ctx()

    def run(payload: dict) -> ApiResult:
        if ctx is not None:  # lets st.cache_data work inside worker threads
            add_script_run_ctx(threading.current_thread(), ctx)
        return cached_call(base_url, path, payload)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(payloads)))) as pool:
        return list(pool.map(run, payloads))


def clear_cache() -> None:
    _cached.clear()


def health(base_url: str | None) -> ApiResult:
    """Uncached ``GET /health`` with a short timeout (used for status badges and warm-up)."""
    return call_api(base_url, "/health", timeout=15, retries=0)

# ====================================================================================================
# 4. Overview page
# ====================================================================================================

STATUS_TTL_SECONDS = 120
RISK_COLOURS = {"Low": "#0ca30c", "Moderate": "#fab219", "High": "#ec835a", "Very high": "#d03b3b"}


# ------------------------------------------------------------------ data
@st.cache_data(ttl=STATUS_TTL_SECONDS, show_spinner=False)
def service_status() -> list[dict]:
    """Ping every configured /health in parallel (this also wakes sleeping Render instances)."""
    urls = [api_url(m.key) for m in TEAM]
    with ThreadPoolExecutor(max_workers=len(TEAM)) as pool:
        results = list(pool.map(lambda u: health(u) if u else None, urls))
    rows = []
    for member, url, result in zip(TEAM, urls, results, strict=True):
        if url is None:
            state, kind, latency = "Not deployed", "off", None
        elif result.ok:
            state, kind, latency = "Online", "ok", result.elapsed_seconds
        else:
            state, kind, latency = "Waking up / unavailable", "warn", None
        rows.append({"letter": member.letter, "state": state, "kind": kind, "latency": latency, "url": url})
    return rows


def risk_snapshot() -> pd.DataFrame | None:
    """Current and 7-day-ahead Market Risk Index for all eight tokens (Student B service)."""
    tokens = list(ALL_TOKENS)
    payloads = [{"token": t, "forecast_horizon": FORECAST_HORIZON} for t in tokens]
    results = cached_call_many(api_url("student_b"), "/predict", payloads)
    rows = [
        {"Token": t, "Name": TOKENS[t], "Today": r.data.get("current_mri"), "Forecast": r.data["prediction"],
         "Risk level": r.data["risk_level"]}
        for t, r in zip(tokens, results, strict=True) if r.ok
    ]
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    frame["Change"] = frame["Forecast"] - frame["Today"]
    return frame.sort_values("Forecast", ascending=False).reset_index(drop=True)


def architecture_dot(status: dict[str, str]) -> str:
    """Graphviz description of the data flow: Kraken -> four services <-> each other -> this app."""
    colour = {"ok": "#0ca30c", "warn": "#eda100", "off": "#b5b4ad", "bad": "#d03b3b"}
    lines = [
        "digraph G {", 'rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=1.1; splines=true;',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13, color="#c3c2b7", '
        'fillcolor="#ffffff", penwidth=1.4, margin="0.18,0.10"];',
        'edge [fontname="Helvetica", fontsize=11, color="#898781", fontcolor="#52514e", penwidth=1.2];',
        'kraken [label="Kraken public API\ndaily OHLC (USD)\n720 candles", fillcolor="#eef4fc"];',
        'app [label="Streamlit app\n(this website)", fillcolor="#0d366b", fontcolor="#ffffff", color="#0d366b"];',
        "subgraph cluster_apis {",
        'label="FastAPI services on Render  ·  peers exchange features via POST /features"; fontname="Helvetica"; '
        'fontsize=11; fontcolor="#52514e"; style="rounded,dashed"; color="#c3c2b7";',
    ]
    for m in TEAM:
        scope = "8 tokens" if len(m.tokens) == len(ALL_TOKENS) else ", ".join(m.tokens)
        lines.append(
            f'{m.letter} [label="Student {m.letter}: {m.target}\n{m.algorithm} · {scope}", '
            f'color="{colour[status.get(m.letter, "off")]}", penwidth=2.4];'
        )
    lines.append("}")
    for m in TEAM:
        lines.append(f"kraken -> {m.letter};")
        lines.append(f"{m.letter} -> app [label=\"/predict\"];")
    lines.append("}")
    return "\n".join(lines)


def feature_exchange() -> pd.DataFrame:
    """Which engineered features each service serves to which peer (from the group contract)."""
    rows = []
    for consumer in TEAM:
        for letter, feats in consumer.consumes.items():
            for feat in feats:
                rows.append({"Feature": feat, "Served by": f"Student {letter}",
                             "Used by": f"Student {consumer.letter} ({consumer.target})"})
    frame = pd.DataFrame(rows)
    return (frame.groupby(["Feature", "Served by"], sort=False)["Used by"].agg(", ".join)
            .reset_index().sort_values(["Served by", "Feature"]).reset_index(drop=True))


def snapshot_chart(frame: pd.DataFrame) -> alt.LayerChart:
    order = list(frame["Token"])
    bars = alt.Chart(frame).mark_bar(cornerRadius=4, height=16).encode(
        x=alt.X("Forecast:Q", scale=alt.Scale(domain=[0, 100]), title=f"Market Risk Index in {FORECAST_HORIZON} days"),
        y=alt.Y("Token:N", sort=order, title=None),
        color=alt.Color("Risk level:N", scale=alt.Scale(domain=list(RISK_COLOURS), range=list(RISK_COLOURS.values())),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Token", "Name", alt.Tooltip("Today:Q", format=".1f"), alt.Tooltip("Forecast:Q", format=".1f"),
                 alt.Tooltip("Change:Q", format="+.1f"), "Risk level"],
    )
    ticks = alt.Chart(frame.dropna(subset=["Today"])).mark_tick(color="#0b0b0b", thickness=2, size=20).encode(
        x="Today:Q", y=alt.Y("Token:N", sort=order), tooltip=["Token", alt.Tooltip("Today:Q", format=".1f")])
    return (bars + ticks).properties(height=110 + 30 * len(frame))


# ------------------------------------------------------------------ page
def render_overview() -> None:
    inject_css()
    hero(
        f"Group {GROUP} · {SUBJECT}",
        "Crypto Market Intelligence Platform",
        "Four independently deployed machine-learning services forecast performance, risk, trend and trading "
        f"opportunity {FORECAST_HORIZON} days ahead from live Kraken market data. The services exchange "
        "engineered features through a shared API contract and keep working when a peer is unavailable.",
        ["4 FastAPI microservices on Render", "XGBoost & LightGBM", "Live Kraken OHLC data",
         f"{FORECAST_HORIZON}-day horizon", "Optuna-tuned · SHAP-explained"],
    )

    with st.spinner("Checking the four services (a sleeping service can take up to a minute to wake)..."):
        status = service_status()
    online = sum(r["state"] == "Online" for r in status)
    deployed = sum(r["url"] is not None for r in status)
    covered = {t for m in TEAM for t in m.tokens}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Services online", f"{online} / {len(TEAM)}", help=f"{deployed} deployed so far")
    c2.metric("Tokens covered", f"{len(covered)}", help="Bitcoin by every service; all eight by the "
              "Market Risk service")
    c3.metric("Forecast horizon", f"{FORECAST_HORIZON} days")
    c4.metric("Data source", "Kraken", help="Public OHLC endpoint, daily candles in USD, no API key")

    section("Live market risk snapshot")
    st.markdown("#### Which cryptocurrencies will be riskiest next week?")
    with st.spinner("Forecasting risk for all eight tokens..."):
        snapshot = risk_snapshot()
    if snapshot is None:
        st.info("The Market Risk service is waking up - refresh the page in a minute to see the snapshot.")
    else:
        left, right = st.columns([3, 2])
        with left:
            st.altair_chart(snapshot_chart(snapshot), width="stretch")
        with right:
            riskiest, calmest = snapshot.iloc[0], snapshot.iloc[-1]
            rising = snapshot.dropna(subset=["Change"]).sort_values("Change", ascending=False).iloc[0]
            st.metric("Riskiest next week", f"{riskiest['Token']} · {riskiest['Forecast']:.1f}",
                      riskiest["Risk level"], delta_color="off", delta_arrow="off", border=True)
            st.metric("Calmest next week", f"{calmest['Token']} · {calmest['Forecast']:.1f}",
                      calmest["Risk level"], delta_color="off", delta_arrow="off", border=True)
            st.metric("Largest expected rise", rising["Token"], f"{rising['Change']:+.1f} points",
                      delta_color="inverse", border=True)
        st.caption("Bars: forecast Market Risk Index (0-100) for the coming week; black ticks: today's level. "
                   "Source: Student B's LightGBM service - open **Market Risk (B)** for explanations.")

    section("System architecture")
    st.markdown("#### How a forecast is produced")
    st.graphviz_chart(architecture_dot({r["letter"]: r["kind"] for r in status}), width="stretch")
    st.caption("Border colour = live status (green online, amber waking up, grey not deployed).")
    steps = [
        ("1 · Data", "Each service downloads daily candles from the Kraken public API, drops the incomplete "
                     "current day and caches them for 15 minutes."),
        ("2 · Feature exchange", "Each service owns some engineered features and serves them on `POST /features`; "
                                 "peers call it in parallel with timeouts, retries and caching."),
        ("3 · Prediction", "`POST /predict` runs the tuned model and returns the forecast, its date and an "
                           "explanation; classifiers also return a confidence."),
        ("4 · Resilience", "If a peer is down, a service reuses a cached answer, recomputes the feature locally or "
                           "predicts with missing values - it never fails the request."),
        ("5 · Presentation", "This app calls each `/predict`, caches successful answers for 10 minutes and "
                             "explains the result in plain language."),
    ]
    for col, (title, text) in zip(st.columns(len(steps)), steps, strict=True):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.caption(text)

    with st.expander("Feature exchange between services (group API contract)"):
        st.dataframe(feature_exchange(), hide_index=True, width="stretch")
        st.caption("Each feature is computed once, by its owner, and shared - so all four models see identical "
                   "definitions (single source of truth: the shared package mykrml_25740287).")

    section("Team and services")
    st.markdown("#### Four models, one contract")
    cols = st.columns(4)
    status_by_letter = {r["letter"]: r for r in status}
    for col, member in zip(cols, TEAM, strict=True):
        s = status_by_letter[member.letter]
        latency = f" · {s['latency']:.1f} s" if s["latency"] is not None else ""
        scope = "All 8 tokens" if len(member.tokens) == len(ALL_TOKENS) else ", ".join(member.tokens)
        consumes = "; ".join(f"{', '.join(f)} ({p})" for p, f in member.consumes.items())
        with col.container(border=True):
            st.markdown(f"**Student {member.letter}** &nbsp; {pill(s['state'] + latency, s['kind'])}",
                        unsafe_allow_html=True)
            st.markdown(f"#### {member.target}")
            st.markdown(
                f"{member.name} · {member.student_id}  \n"
                f"**Model:** {member.algorithm} ({member.task.lower()})  \n"
                f"**Coverage:** {scope}  \n"
                f"**Serves:** {', '.join(member.owns)}  \n"
                f"**Uses:** {consumes}"
            )

    section("Coverage")
    st.markdown("#### What is forecast for each token")
    coverage = pd.DataFrame(
        {m.target: ["✓" if t in m.tokens else "-" for t in ALL_TOKENS] for m in TEAM},
        index=[f"{t} · {TOKENS[t]}" for t in ALL_TOKENS],
    )
    st.dataframe(coverage, width="stretch")
    st.caption("The group product covers Bitcoin end-to-end. The Market Risk service extends the analysis to all "
               "eight tokens: one pooled LightGBM model trained on 5,544 token-days, with the peer features it "
               "needs recomputed from Kraken through the shared feature package when a peer does not cover a token.")

    section("Shared methodology")
    st.markdown("#### Conventions every model follows")
    st.markdown(
        "| Area | Agreed approach |\n|---|---|\n"
        "| Data | Kraken daily OHLC (USD), last 720 complete days; same source for training and serving |\n"
        "| Validation | Chronological train / validation / test split with a 7-day purge gap; never random |\n"
        "| Tuning | Optuna (TPE, seed 42) with 5-fold `TimeSeriesSplit(gap=7)` on the training period only |\n"
        "| Baselines | Mean / majority class **and** persistence (today's value as the 7-day forecast) |\n"
        "| Metrics | Regression: MAE, RMSE, R² · Classification: macro-F1, balanced accuracy, log-loss |\n"
        "| Tracking | Every trial logged to Weights & Biases; test set evaluated once |\n"
        "| Serving | FastAPI with Pydantic validation, `{\"error\": ...}` responses, `/health`, `/model-metadata` |"
    )
    st.caption("Educational project - forecasts are estimates, not financial advice.")


def render_placeholder(student_key: str) -> None:
    """Temporary page for a student whose page is not built yet (replaced by that student)."""
    member = MEMBERS[student_key]
    inject_css()
    hero(f"Student {member.letter} · {member.name} ({member.student_id})", f"{member.target} - 7-day forecast",
         f"{member.algorithm} forecasting the {member.target.lower()} for {', '.join(member.tokens)}.",
         [member.algorithm, member.task, f"Serves: {', '.join(member.owns)}"])
    st.info(f"This page is being developed by {member.name}.", icon=":material/construction:")
    url = api_url(student_key)
    if url is None:
        st.caption("API not deployed yet.")
    elif st.button("Check API", key=f"{student_key}_check"):
        result = health(url)
        st.write("Online" if result.ok else f"Unavailable: {result.error}")

# ====================================================================================================
# 5. Navigation
# ====================================================================================================

def main() -> None:
    """Configure the page and route between the Overview and the four student pages."""
    # Use the importable module ``app.main`` (not this ``__main__`` copy), so the Overview and the student
    # pages share one set of caches and helpers. Imported here to avoid an import cycle.
    from app import main as core
    from students import student_a, student_b, student_c, student_d

    st.set_page_config(page_title="Group 08 - Crypto Forecasts", page_icon=":material/monitoring:", layout="wide")
    pages = [
        st.Page(core.render_overview, title="Overview", icon=":material/home:", url_path="overview", default=True),
        st.Page(student_a.render, title="Asset Performance (A)", icon=":material/trending_up:",
                url_path="performance"),
        st.Page(student_b.render, title="Market Risk (B)", icon=":material/warning:", url_path="market-risk"),
        st.Page(student_c.render, title="Trend Category (C)", icon=":material/timeline:", url_path="trend"),
        st.Page(student_d.render, title="Trading Opportunity (D)", icon=":material/target:",
                url_path="opportunity"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
