# DPFP — Data-Pipeline Failure Prediction
### Complete project — start here

This is the **single, complete project folder**. It replaces every earlier scattered copy
you have (old pipeline, old notebook, old CSVs). From now on, work only inside this folder.

The project predicts **data-pipeline failures one day ahead** from the pipeline's own
data-quality signals, for an Open-Meteo air-quality pipeline running for two cities
(Pune + Delhi). It is built to **run in the present**: you train and test on all past data,
then a daily job keeps collecting new data and retraining so the model stays current.

---

## What's in this folder

| Item | What it is |
|---|---|
| `src/` | the pipeline code (ingestion, validation, transformation, feature enrichment, runner) |
| `config/config.yaml`, `config/config_delhi.yaml` | Pune & Delhi settings (data range 2022-08-05 → 2035) |
| `run_two_cities.py` | **backfill** — builds the full historical dataset for both cities |
| `run_daily.py` | **daily job** — fetches one new day per city; retrains on Sundays |
| `retrain_weekly.py` | retrains on all data, saves the model, logs performance over time |
| `DPFP_Phase2_Modelling.ipynb` | the **analysis notebook** — all models, results, charts for your thesis |
| `requirements.txt` | Python packages to install |
| `.github/workflows/daily.yml` | **GitHub Actions automation** — daily run + Sunday retrain (see Step 4) |
| `DEPLOY_PYTHONANYWHERE.md` | older alternative: run the daily job on a free PythonAnywhere server |
| `logs/`, `data/`, `database/` | the datasets and run history (committed, so the automation can continue them) |
| `outputs/` | trained model, performance history, notebook figures |
| `Documents/Research_Findings_47_Papers.xlsx` | your 47-paper research-gap analysis |
| `Documents/Project_Novelty_Objectives_Methodology.docx` | objectives, methodology, research problem |
| `Data pipeline failure prediction novelty.md` | the novelty write-up |
| `Project validation and ratings.md` | the Q2/Q3 publication rating |

---

## The plan (past → present)

```
STEP 1  install
STEP 2  build the dataset from the PAST   (run_two_cities.py)  ← "test on past data"
STEP 3  run the analysis notebook         (the models + results for your thesis)
STEP 4  schedule the DAILY job            (run_daily.py)       ← "come to present & stay"
```

---

## STEP 1 — Install (once)

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```
(Python 3.10+ and internet access to `air-quality-api.open-meteo.com`.)

---

## STEP 2 — Build the dataset from past data

This is the "run the pipeline again" part. It fetches every day from **2022-08-05 up to today**
for both cities and writes two logs:

```bash
python run_two_cities.py
```

This produces:
```
logs/api_runs/pipeline_runs.csv      <- Pune  dataset
logs/delhi_runs/pipeline_runs.csv    <- Delhi dataset
```

Notes:
- It is **safe to stop and re-run** — days already done are skipped, so you can resume anytime.
- It takes a while (thousands of API calls), but it's just network, not heavy computing.
- Quick test first (optional), to confirm everything works before the full run:
  ```bash
  python run_two_cities.py --start 2026-09-01 --end 2026-09-30
  ```

**You do not need to re-run the full history.** The finished datasets are already in the repo
(`logs/api_runs/` and `logs/delhi_runs/`, 2022-08-05 onward). Step 2 is only for rebuilding from scratch.

---

## STEP 3 — Run the analysis notebook

Open `DPFP_Phase2_Modelling.ipynb` (Jupyter / VS Code) and **Run All**.
At the top, Section 0 points to the two CSV files. It then does everything:

- leakage-free next-day prediction, model + imbalance selection, calibration
- infra-vs-data ablation (proves data signals win)
- per-failure-type predictability map (the headline result)
- lead-time, leakage-gap demo, cross-city validation, explainability
- `predict_failure(city, date)` to score any date
- Section 15 = the results summary table for your thesis

Re-run this notebook any time your dataset grows and all results update.

---

## STEP 4 — Come to the present and stay there (automation)

This is automated with **GitHub Actions** (`.github/workflows/daily.yml`). Nothing to run by hand:

- **Every day, 08:00 India time:** fetches yesterday for Pune and Delhi. If any earlier day was
  missed, it is filled in too (up to 60 days back).
- **Every Sunday (India time):** retrains the model on all data, saves `outputs/dpfp_model.joblib`
  and appends one row to `outputs/performance_history.csv`.
- The new data is committed back to the repo, so the repo always holds the latest dataset and model.
- To run it now: GitHub → **Actions** → *DPFP daily pipeline* → **Run workflow**
  (tick *force_retrain* to retrain outside Sunday).

Manual run on your own machine is the same script: `python run_daily.py`.

`outputs/performance_history.csv` feeds **Section 14b of the notebook**, a
"model improving over time" chart. (The notebook also needs `pip install -r requirements-notebook.txt`.)

---

## The one concept that makes this project correct (keep this for your thesis)

A run is a **FAILURE only when the data/pipeline is actually broken** — negative values, stuck
sensors, impossible jumps, missing/short data, API timeouts. **Genuinely high real-world
pollution (e.g. Delhi) is NOT a failure** — it is recorded as a *warning* and fed to the model
as a signal, never as the thing being predicted. This is why the results are honest: the model
predicts broken data, not dirty air.

---

## Quick FAQ

**Do I merge these files into my old project?** No — use this folder and set the old copies aside.
Everything here is the latest version.

**Do I have to run the pipeline again?** Only if you want a fresh dataset (Step 2). If you're happy
using `seed_data_backup/`, you can skip straight to Step 3. Running fresh is recommended if you
want data all the way to today.

**Order to do things:** Step 1 → 2 → 3 for the thesis results; add Step 4 to keep it live.

---

## STEP 5 — The dashboard (Pipeline Radar)

`dashboard/index.html` is a live dashboard for data engineers: every watched pipeline ranked by
tomorrow's failure risk, why the risk is high, an alert log (came true / false alarm) and the
health signals (missing rows, data drift, source speed, sudden jumps). Click a pipeline to open it.

- **Rebuilt every morning** by the GitHub Action (`python build_dashboard.py`, after `run_daily.py`).
- **Live link:** https://vrushablanjekar.github.io/DPFP-Project/ (published by `.github/workflows/pages.yml`
  after every daily run). Offline: double-click `dashboard/index.html` (all data is inside the file).
- **Honest history:** each day's forecast is saved as it is made in `outputs/forecast_log.csv`.
- **Add a pipeline:** add one entry to `PIPELINES` in `build_dashboard.py` (its run log needs the
  same columns as `logs/*/pipeline_runs.csv`). The dashboard's "+ Add pipeline" screen checks a CSV.
- Pipelines with fewer than 30 past failures are marked **LOW CONF.** (Pune today).
