# Social Media Fraud Detection - Data Mining Project

Detect fake / bot accounts on social media. A full data-mining pipeline with a Streamlit dashboard
that shows every step (to present to the professor).

## 1. Setup (VS Code terminal, in this folder)

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

The app opens at http://localhost:8501. With no dataset in `data/raw/` it uses a built-in NOISY
demo dataset, so it always runs.

## 2. Use a REAL dataset (recommended for the professor)

Recommended: **Cresci-2017** (Twitter fake accounts / bots; one folder per account type).

1. Download it (Kaggle or the Bot Repository) and unzip it.
2. Merge it into one labelled CSV:
   ```bash
   python build_cresci_dataset.py "C:\path\to\cresci-2017"
   ```
   This writes `data/raw/cresci_2017_users.csv` (label `is_fake`: 0 genuine, 1 fake/bot).
3. Restart / refresh the app: choose the file in the sidebar ("File from data/raw/").

Any other CSV works too (Kaggle Instagram fake accounts, TwiBot-20 ...): drop it in `data/raw/`
or use "Upload a CSV", then pick the **target column** and the value meaning **fraud** in the sidebar.
Remove identifier / leakage columns in "Columns to drop".

## 3. What is covered (and where)

| Requirement | Where |
|---|---|
| Missing values (mean / median / mode / KNN / ffill) | `src/cleaning.py`, sidebar 3, tab Cleaning |
| Outliers: Z-score, IQR (remove or cap) | `src/cleaning.py`, tab Cleaning |
| Inconsistencies + duplicates | `src/cleaning.py`, tab Cleaning |
| Normalise / standardise (standard, min-max, robust) | `src/transform.py`, tab Transformation |
| Encoding (one-hot, ordinal, frequency) | `src/transform.py` |
| Aggregation + discretisation | `src/transform.py`, tab Transformation |
| Feature engineering | `src/features.py` (ratios, text stats, date age, missing flags) |
| Filter (correlation, mutual information) | `src/selection.py`, tab Feature selection |
| Wrapper (RFE) | `src/selection.py` |
| Embedded (LASSO, Decision Tree importance) | `src/selection.py` |
| PCA / t-SNE / Autoencoder | `src/reduction.py`, tab Dimensionality reduction |
| EDA (distribution, heatmap, bars, trend) | tab EDA |
| Train / test split (stratified, no leakage) | `src/pipeline.py`, tab Train/Test split |
| Model + metrics | `src/model.py`, tab Model |
| Before vs After + CSV download | tab Before vs After |

## 4. Important design notes (good to say in the presentation)

* The raw file is never modified: everything happens in memory.
* Split first; imputers, outlier bounds, scalers, encoders, selectors and PCA are fitted on TRAIN only
  (no data leakage). Outlier rows are removed from train only.
* In fraud detection, extreme values are often the signal. Compare the model with and without
  outlier removal in the Model tab.
* The autoencoder is a scikit-learn neural network (input -> hidden -> bottleneck -> hidden -> input),
  so no TensorFlow install is needed.
* Use precision / recall / F1 / ROC-AUC, not only accuracy (imbalanced classes).

## 5. Project structure

```
app/app.py                  Streamlit dashboard
build_cresci_dataset.py     merge Cresci-2017 folders into one CSV
generate_demo_data.py       save the noisy demo dataset to data/raw
requirements.txt
data/raw/                   your datasets (never modified)
src/
  data_loader.py            loading + noisy demo data generator
  cleaning.py               inconsistencies, duplicates, outliers, imputation
  features.py               column typing + feature engineering
  transform.py              log, discretisation, aggregation, encoding, scaling
  pipeline.py               end-to-end preprocessing (stage A + stage B)
  selection.py              filter / wrapper / embedded selection
  reduction.py              PCA, t-SNE, autoencoder
  model.py                  models + metrics
```

## 6. MLOps part (use case 2): project structure + data versioning with DVC

New files:

| File | Role |
|---|---|
| `params.yaml` | every setting in one place (data version, cleaning, selection, model) |
| `dvc.yaml` | the pipeline: `1-data_validation` -> `2-prepare_data` -> `3-train` |
| `src/data/data_validation.py` | stage 1: checks the raw data (rows, target, classes) |
| `src/data/prepare_data.py` | stage 2: cleaning + transformation -> `data/processed/` |
| `src/models/train_model.py` | stage 3: feature selection + model -> `models/`, `reports/metrics.json` |
| `src/data/update_dataDVC.py` | creates a new dataset version (dvc add + git commit + git tag `v.N`) |
| `src/config.py` | reads `params.yaml` |

Every script ends with `if __name__ == "__main__":` so it can be run alone
(`python src/data/prepare_data.py`) or imported by another script without running.

Quick start (in the project folder; Git must be installed):

```bash
git init
dvc init
dvc add data/raw            # BEFORE the first "git add ."
git add .
git commit -m "init project"
dvc repro                   # runs the 3 stages (only what changed)
dvc dag                     # draws the pipeline
dvc remote add -d myremote C:/dvc-storage      # local storage (or a Google Drive URL)
dvc push
```

New dataset version: edit `params.yaml` (increase `data_version`, change e.g. `out_method`), then

```bash
dvc repro
dvc params diff
dvc metrics diff
python src/data/update_dataDVC.py        # add + commit + tag v.2  (--push to also push, --dry-run to preview)
```

The Streamlit app has a last tab, "MLOps / Versioning", that shows params.yaml, dvc.yaml and the results.


## 7. Use case 3: versioning data from the Streamlit app

* The app lives in `app/app.py` and finds the project folder from its own location, so
  `streamlit run app/app.py` works from any folder.
* `src/versioning.py` contains `version_data(...)`, which runs `dvc add`, `git add`, `git commit` and `git tag`
  from Python (plus an optional `dvc push` / `git push`), and records the parameters in
  `reports/versions/<tag>.json` (tracked by Git).
* Tab **"Data versions"**: save the current result as a new version (tag + commit message), browse every
  version, see the parameters that produced it, download it, compare two versions.
* Sidebar source **"Tracked dataset version (DVC tag)"** loads any past version as the app's dataset.

Manual part (Task 0 and Task 1), once, in the terminal:

```bash
git init
dvc init
git config user.name "Your Name"
git config user.email you@mail.com
copy data\raw\cresci_2017_users.csv data\dataset.csv        # macOS/Linux: cp
dvc remote add -d localremote C:/dvc-storage                  # a folder OUTSIDE the project
dvc add data/dataset.csv
git add .
git commit -m "Add initial dataset"
git tag -a v1.0 -m "Initial dataset"
dvc push
```
