# Data and methods

## Source and attribution

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
**Heart Disease**. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X. License: **CC BY 4.0**.

The downloader uses the original UCI archive and the four explicitly named
processed files: Cleveland, Hungarian, Switzerland and VA Long Beach. No patient
records are collected for this restoration. Data are downloaded locally, not
bundled in Git. Each run records the SHA-256 of each input file, record IDs,
class counts, missingness and site counts. Record IDs are source-file line
references, not patient identifiers.

The binary target is **presence of the dataset's recorded disease label**
(`num > 0`), not future cardiovascular events, a diagnosis made by this tool,
or a calibrated personal risk estimate.

## Feature contract

| Field | Meaning / original UCI encoding |
|---|---|
| age | Age in years |
| sex | UCI binary coding: 0 female, 1 male |
| cp | Type of chest pain: 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| trestbps | Resting blood pressure, mm Hg |
| chol | Serum cholesterol, mg/dL |
| fbs | Fasting blood sugar > 120 mg/dL: 0 no, 1 yes |
| restecg | 0 normal, 1 ST-T abnormality, 2 probable/definite LV hypertrophy |
| thalach | Maximum achieved heart rate |
| exang | Exercise-induced angina: 0 no, 1 yes |
| oldpeak | Exercise-induced ST depression relative to rest |
| slope | 1 upsloping, 2 flat, 3 downsloping |
| ca | Number of major vessels colored by fluoroscopy: 0-3 |
| thal | Thallium stress-test result: 3 normal, 6 fixed defect, 7 reversible defect |

These codes are shared by the CLI and bot. In particular, `cp` is not an ordinal
pain severity scale, and `thal` does not mean thalassemia. See the
[UCI variable documentation](https://archive.ics.uci.edu/dataset/45/heart+disease).

## Cleaning and missing data

- `?` becomes a missing value. Invalid or absent target labels cause an error.
- Nonpositive `trestbps`, `chol` and `thalach` measurements are treated as missing;
  the number changed is recorded in the report.
- Exact duplicate feature vectors are reduced to their first occurrence before
  splitting. The removal count is reported. This prevents identical vectors
  appearing on both sides but cannot establish patient-level independence when
  original patient identifiers are unavailable.
- Mean imputation and scaling are fitted inside each training CV fold, never
  on the holdout. Unknown demo values must be explicit JSON `null`; more than
  three unknown demo values are rejected. This is a software guard, not a
  clinically established eligibility criterion.

## Frozen protocols

| Protocol | Data | Holdout | Purpose |
|---|---|---|---|
| `cleveland` | Original processed Cleveland file | Stratified 20%, seed 42 | Reconstruction of the published notebook procedure |
| `multisite300` | Four processed UCI sites, deduplicated | Exactly 300 rows, stratified by target, seed 42 | New retrospective public-data experiment |

The multisite holdout is drawn from the same mixture of centers as training.
It **is not external-site validation**. Strong missingness differences across
centers may act as proxies for site and disease prevalence. Generalization to
new hospitals, populations or current clinical practice has not been established.

## Model selection

The original notebook's 243-combination XGBoost grid is retained:

- estimators: 100, 200, 300;
- maximum depth: 2, 3, 4;
- learning rate: 0.01, 0.05, 0.1;
- L1 regularization: 0, 0.1, 0.5;
- L2 regularization: 1, 5, 10.

Ten-fold stratified CV without shuffling selects parameters using training-only
ROC-AUC. The tree method is explicitly `hist`; XGBoost runs one thread per fit.
The final selected model is evaluated on the untouched holdout. The classification
threshold is fixed at 0.5 before training. A logistic-regression baseline uses
the same training rows, preprocessing and holdout, with no holdout-based tuning.

`--search smoke` is a deliberately small functionality check. Its outputs are
labeled `smoke_test_only` and are not the reported full experiment. Do not choose
between smoke/full configurations by their holdout scores. No seed sweep, target
AUC objective, synthetic holdout enlargement or post-hoc threshold selection is
part of either protocol.

## Evaluation

Reports include ROC-AUC, 95% percentile bootstrap intervals (1,000 resamples,
fixed seed), accuracy, sensitivity, specificity, precision, Brier score,
confusion matrices, calibration plots and per-record predictions. Intervals
reflect this holdout's sampling variability, not all model-selection uncertainty.

The historical notebook had already exposed its Cleveland holdout. The rerun is
therefore a reproducibility exercise, not a newly blind validation study.

Run outputs also contain the entire training-only CV table and source-file
hashes. Committed reports contain only predictions for public UCI examples;
they contain no new patient data.
