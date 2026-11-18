# Ahead of the Curve

A personal Fantasy Premier League scouting dashboard. It ranks players on underlying
output (xG, xA, xGC, defensive contributions), minutes and fixtures, and explains each
ranking. It is a transparent decision aid, not a points predictor.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first load fetches about 450 player histories from the public FPL API, which takes
roughly a minute. After that, data comes from the local cache in `data/`, and **Refresh
data** in the sidebar forces a new fetch.

Other modes:

```bash
AOTC_DEMO=1 streamlit run app.py      # fictional demo data, no network
AOTC_OFFLINE=1 streamlit run app.py   # last good snapshot only
python -m pytest -q                   # run the checks
```

On Windows PowerShell, set the variable first: `$env:AOTC_DEMO="1"; streamlit run app.py`.

## How the code is laid out

| File | Job | Edit it to... |
|---|---|---|
| `aotc/config.py` | Every default number and threshold | change weights, windows, haircuts, DefCon thresholds |
| `aotc/fpl_data.py` | HTTP, retries, cache, snapshot | change timeouts, TTLs, concurrency |
| `aotc/transform.py` | Raw JSON to tidy tables; season/GW detection | add a new API field |
| `aotc/metrics.py` | All football maths | change any formula (see the docstring map at the top) |
| `aotc/explain.py` | Reason/risk sentences | add or reword explanations |
| `app.py` | Streamlit layout only | change the dashboard |
| `aotc/demo_data.py` | Fictional data with DGW/blank/postponed cases | test edge cases |
| `tests/` | Checks for per-90, shrinkage, missing data, blanks/doubles, caching | — |

## Key ideas

* **Opportunity score (0–100)** is a percentile within position, so it's best compared
  within a position. It is not expected points.
* **Data confidence** describes how much data stands behind a row (sample, minutes
  stability, input coverage). It is not a statistical interval.
* **Ownership** only decides who counts as a differential. It never changes a score.
* Unavailable values show as blank or "—" and are never treated as zero.

Full formulas, defaults and limitations are on the dashboard's **Methodology** tab.

## Data and terms

Uses the public, undocumented FPL API, which needs no key. The Premier League's terms allow
private, personal use. The API can change without notice; `transform.py` coerces missing
fields to "unavailable" so that a schema change degrades gracefully instead of crashing.
