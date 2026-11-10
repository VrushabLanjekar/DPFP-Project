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
| `DEPLOY_PYTHONANYWHERE.md` | how to put the daily job on a free server so it runs by itself |
| `seed_data_backup/` | a **backup** of already-generated datasets (fallback — see Step 2) |
| `Research_Findings_47_Papers.xlsx` | your 47-paper research-gap analysis |
| `Project_Novelty_Objectives_Methodology.docx` | objectives, methodology, research problem |
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

**If you don't want to re-run the full history:** the folder `seed_data_backup/` already contains
finished datasets (through 2026-10-27, hazard-free). To use them instead of running Step 2, copy
them into place:
```bash
mkdir -p logs/api_runs logs/delhi_runs
cp seed_data_backup/pipeline_runs_p.csv logs/api_runs/pipeline_runs.csv
cp seed_data_backup/pipeline_runs_d.csv logs/delhi_runs/pipeline_runs.csv
```
Either way (fresh run or backup), you end up with the same two CSV files in the same place.

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

So the project keeps working day by day without you running anything by hand, schedule the
daily job. On your own machine this is one cron line; on a free server follow
`DEPLOY_PYTHONANYWHERE.md` (Open-Meteo is already allowed on the free tier).

```bash
# runs once a day: fetches yesterday for both cities, retrains every Sunday
python run_daily.py
```

Each run adds one new day per city. Every Sunday it retrains and appends a row to
`outputs/performance_history.csv`, which **Section 14b of the notebook** turns into a
"model improving over time" chart.

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
