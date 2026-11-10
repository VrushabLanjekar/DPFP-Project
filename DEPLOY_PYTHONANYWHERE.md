# Deploying the daily pipeline on PythonAnywhere (continuous learning)

This sets up your pipeline so it **runs by itself every day** — fetching one new day of
air-quality data for Pune and Delhi, appending it to the logs, and **retraining the model
every Sunday** so it keeps learning as data grows.

The whole thing runs on a **free PythonAnywhere account**. You do not need a paid plan to start.

---

## 0. Why PythonAnywhere works here (the one thing people get wrong)

Free PythonAnywhere accounts can only reach an **allow-listed set of websites**. Good news:
the exact endpoint this pipeline calls —

> **`air-quality-api.open-meteo.com`** — is already on the allow-list.

So the Open-Meteo air-quality API works on the free tier with no request and no upgrade.
(You can confirm it any time at <https://www.pythonanywhere.com/whitelist/>.)

Two free-tier limits to know:
- **One scheduled task**, running **once per day**. That's exactly what `run_daily.py` needs
  (it does the daily fetch and, on Sundays, the weekly retrain — one task covers both).
- A modest CPU-seconds budget per day. Fetching two cities for one day is tiny, well inside it.

If you later outgrow the free tier (more cities, hourly runs), a **$5/month "Hacker" plan**
removes the website allow-list and gives more scheduled tasks — but you do **not** need it now.

---

## 1. Make a free account

1. Go to <https://www.pythonanywhere.com/> → **Pricing & signup** → **Create a Beginner account** (free).
2. Confirm your email and log in. You land on the **Dashboard**.

---

## 2. Upload your project

**Option A — upload the zip (simplest):**
1. Dashboard → **Files** tab.
2. Under *Directories*, create a folder named `dpfp` (type `dpfp/` in the "New directory" box).
3. Enter `dpfp`, then **Upload a file** → upload `dpfp_deploy.zip` (the zip delivered with this guide).
4. Open a **Bash console** (Dashboard → **Consoles** → **Bash**) and unzip:
   ```bash
   cd ~/dpfp
   unzip dpfp_deploy.zip
   ls           # you should see run_daily.py, retrain_weekly.py, src/, config/, requirements.txt
   ```

**Option B — pull from GitHub (if you push the project there):**
```bash
cd ~
git clone https://github.com/<you>/<repo>.git dpfp
cd dpfp
```

---

## 3. Install the dependencies

In the Bash console:
```bash
cd ~/dpfp
pip install --user -r requirements.txt
```
`--user` installs into your account without needing admin rights. This also installs
scikit-learn and joblib, which the weekly retrain needs.

---

## 4. Seed the historical data (once)

So the model has history to learn from on day one, upload your existing logs — the two CSVs
you already generated — into `~/dpfp/`:

- `pipeline_runs_d.csv`  (Delhi)  → the retrainer also reads `pipeline_runs_delhi.csv`; either name works
- `pipeline_runs_p.csv`  (Pune)

Or, if you prefer to rebuild the full history on the server, run a backfill once. Your data
range is **2022-08-05 up to today**, which is a lot of days — on the free tier's daily CPU
budget, do it in shorter chunks (any range length works; these are just examples) and
resume across days if needed:
```bash
cd ~/dpfp
python run_two_cities.py --start 2022-08-05 --end 2022-12-31
python run_two_cities.py --start 2023-01-01 --end 2023-12-31
python run_two_cities.py --start 2024-01-01 --end 2024-12-31
python run_two_cities.py --start 2025-01-01 --end 2025-12-31
python run_two_cities.py --start 2026-01-01   # no --end = up to today
```
Already-completed days are skipped automatically, so it is safe to stop and resume.

---

## 5. Test the daily job by hand

Before scheduling, run it once and read the output:
```bash
cd ~/dpfp
python run_daily.py                 # fetches yesterday for both cities
python run_daily.py --force-retrain # also do a retrain now, to confirm it works
```
You should see lines like `[Delhi] 2026-11-02: SUCCESS` and, after the forced retrain,
`Retrained on N rows ... Model + history saved.` Check that
`~/dpfp/outputs/performance_history.csv` now exists.

---

## 6. Schedule the daily task

1. Dashboard → **Tasks** tab.
2. Under **Scheduled tasks**, pick a time (UTC). Pick something a couple of hours after
   midnight **in your timezone** so "yesterday" is fully complete — e.g. for IST (UTC+5:30),
   03:00 IST ≈ **21:30 UTC the previous day**. Any early-morning local time is fine.
3. In the command box enter (use YOUR username in the path):
   ```bash
   python3 ~/dpfp/run_daily.py
   ```
4. Click **Create**.

That one task now:
- **every day** → fetches yesterday's data for Pune + Delhi and appends it, and
- **every Sunday** → also retrains the model and logs a new performance row.

The free tier runs a scheduled task roughly at the chosen time (it can be delayed a little
under load — that's normal and harmless here).

---

## 7. Watch it learn

- **Logs:** Tasks tab → click your task → **View log** to see each day's output.
- **Data growing:** `~/dpfp/pipeline_runs_*.csv` gain one row per city per day.
- **Model improving:** open `~/dpfp/outputs/performance_history.csv` — one row per Sunday
  retrain (date, dataset size, held-out PR-AUC/ROC-AUC). Download it and re-run
  **section 14b of the modelling notebook** to redraw the "performance over time" chart for
  your thesis.

---

## 8. Keeping the account awake (free-tier housekeeping)

Free accounts ask you to click a **"Run until 3 months from today"** button for scheduled
tasks every three months (PythonAnywhere emails you a reminder). Just log in and click it;
otherwise the task pauses. Set a recurring reminder so your data collection never stops.

---

## 9. Moving to a real server later (optional)

When you want more than one run a day, more cities, or a public dashboard, the same code runs
unchanged on any always-on host (a $5 PythonAnywhere plan, a small VPS, or a cloud VM) under
`cron`:
```cron
# daily fetch at 02:00, retrain is handled inside run_daily.py on Sundays
0 2 * * *  cd /home/you/dpfp && /usr/bin/python3 run_daily.py >> logs/daily.log 2>&1
```
Nothing about the model or the pipeline changes — only where the daily job is scheduled.

---

### Files involved
| File | Role |
|---|---|
| `run_daily.py` | the scheduled job: daily fetch (both cities) + Sunday retrain |
| `retrain_weekly.py` | retrains on all data, saves the model, appends to `performance_history.csv` |
| `outputs/dpfp_model.joblib` | latest trained model (features list included) |
| `outputs/performance_history.csv` | the learning-over-time record for your thesis chart |
