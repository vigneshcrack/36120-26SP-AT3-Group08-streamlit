# Group 08 – Crypto Forecasts (Streamlit app)

**Subject:** 36120 Advanced Machine Learning Application, Spring 2026 (UTS) – AT3 Data Product with Machine Learning
**Group:** 08
**Live app:** _Streamlit Community Cloud URL – add after deployment_

| Member | Role | Student ID | Page |
|---|---|---|---|
| Ryan Cruikshank | Student A – Asset Performance Index (XGBoost regression) | 13191263 | `students/student_a.py` |
| Vignesh Selvam | Student B – Market Risk Index (LightGBM regression) | 25740287 | `students/student_b.py` |
| Nandeeswari Kumaravelu | Student C – Trend Category (XGBoost classification) | 25980436 | `students/student_c.py` |
| Vinnie Bui | Student D – Trading Opportunity Rating (LightGBM classification) | 26160438 | `students/student_d.py` |

The app forecasts four indicators **7 days ahead** from live Kraken data. Each indicator comes from its own
FastAPI service on Render; this app only calls those services and presents the results (no model runs
inside the app). The group product covers **Bitcoin** end-to-end; the Market Risk service (Student B)
extends the analysis to all eight tokens (BTC, ETH, XRP, SOL, DOGE, XLM, TRX, XMR) with one pooled model.

## Pages

| Page | What it shows |
|---|---|
| Overview | Product summary, live status of all four services (also wakes sleeping Render instances), an all-token market-risk snapshot, the system architecture, team cards, token coverage and the shared methodology |
| Market Risk (B) | **Forecast** (KPIs, risk scale, analyst summary, market inputs with provenance, risk composition, price and risk history, SHAP explanation) · **All 8 tokens** (ranked comparison) · **Model & methodology** (test accuracy vs baselines, prediction pipeline, permutation importance, peer-outage accuracy, Optuna hyperparameters, limitations) · **API** (endpoints, live status, example request and live JSON response, error contract) |
| A, C, D | Each student's own page (placeholders until implemented) |

## Project structure

The repository follows the structure required by the AT3 brief exactly:

```
project/
├── app/
│   └── main.py          # entry point + shared building blocks (see below)
├── students/
│   ├── student_a.py     # Student A page – Asset Performance Index
│   ├── student_b.py     # Student B page – Market Risk Index (all 8 tokens)
│   ├── student_c.py     # Student C page – Trend Category
│   └── student_d.py     # Student D page – Trading Opportunity Rating
├── requirements.txt     # pinned versions (Streamlit Cloud and Docker)
├── pyproject.toml       # uv project definition (Python 3.12.13)
├── Dockerfile           # container image
├── github.txt           # links to all group repositories and deployments
└── README.md
```

Supporting files: `tests/` (pytest suite), `.streamlit/config.toml` (theme) and
`.streamlit/secrets.toml.example`, `uv.lock`, `.python-version`, `.gitignore`, `.dockerignore`.

`app/main.py` is organised in five sections: (1) group configuration – team, tokens and API URLs;
(2) presentation helpers – styling, hero banner, status pills; (3) a shared HTTP client – timeouts,
retries, caching, never raises; (4) the Overview page; (5) navigation. Each student page exposes
`render()` and imports the shared helpers with `from app import main as core`; the page setup and
navigation only run when Streamlit executes the file, so importing it has no side effects.

## Configuration – API URLs

Each page finds its API through `core.api_url("student_x")`, which reads, in order:

1. the environment variable `STUDENT_X_API_URL` (e.g. `STUDENT_B_API_URL`),
2. the same key in Streamlit secrets (`.streamlit/secrets.toml` locally, **App settings → Secrets** on
   Streamlit Cloud – see `.streamlit/secrets.toml.example`),
3. the default in `app/main.py` (Student B's live URL is preset).

No code change is needed when a teammate deploys: add their URL as a secret.

## Run locally

Requirements: Git and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/vigneshcrack/36120-26SP-AT3-Group08-streamlit.git
cd 36120-26SP-AT3-Group08-streamlit
uv python install 3.12.13
uv sync
uv run streamlit run app/main.py          # opens http://localhost:8501
```

Without uv: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, then
`streamlit run app/main.py`.

## Tests

```bash
uv run pytest      # 28 tests, no network needed (the APIs are replaced by canned responses)
uv run ruff check .
```

They cover the HTTP client (success, 4xx not retried, timeouts and 5xx retried, invalid JSON, failures
never cached), every chart/explanation helper, and the Student B page end-to-end with Streamlit's
`AppTest` (normal forecast, all four tabs, API outage, fallback-model warning, changing token) and the
Overview page (normal and outage).

## Deploy

### Streamlit Community Cloud (used for submission)
1. Push this repository to GitHub (private).
2. https://share.streamlit.io → **Create app** → *Deploy a public app from GitHub* → choose this repository,
   branch `main`, main file path **`app/main.py`**.
3. **Advanced settings** → Python version **3.12**; paste the API URLs into **Secrets**
   (format of `.streamlit/secrets.toml.example`).
4. **Deploy**. Private repositories are supported once Streamlit is authorised for them on GitHub.

### Docker
Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running).
```bash
docker build -t group08-streamlit .
docker run --rm -p 8501:8501 group08-streamlit                  # http://localhost:8501
# optional: point a page at a different API
docker run --rm -p 8501:8501 -e STUDENT_A_API_URL=https://<student-a>.onrender.com group08-streamlit
```
The image uses `python:3.12.13-slim`, installs the pinned `requirements.txt`, runs as a non-root user
and has a health check on Streamlit's `/_stcore/health` endpoint (`docker ps` shows `healthy`).

## Failure handling

| Situation | What the user sees |
|---|---|
| API asleep (Render free tier) | Spinner; the first request waits up to 70 s and is retried once |
| API down / timeout | A clear error with a hint and a **Refresh data** button – never a stack trace |
| API returns a 4xx error | The API's own `error` message |
| A peer feature service is down | A notice explaining which inputs were missing and how accuracy changes |
| One token fails in the comparison | The other tokens are still shown; the failed ones are listed |

Successful responses are cached for 10 minutes (`st.cache_data`); failures are not cached.

## AI usage acknowledgement

Generative AI (Claude) was used to support the design, coding, testing and documentation of this app,
in line with the subject's AI usage policy. All code was reviewed, executed and tested by the students,
who can explain and justify every design decision.
