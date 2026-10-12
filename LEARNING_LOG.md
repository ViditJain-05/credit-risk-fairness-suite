# 📓 Learning Log — Credit Risk Explainability & Fairness Suite

This log documents what was built in each phase, the core ML/Regulatory concept behind it, and the primary engineering "gotcha" encountered and solved.

---

## Phase 0: Project Setup, Packaging & Centralized Configuration
- **What Was Built:** Created the `src/creditfair/` package layout, `pyproject.toml`, `.gitignore`, pinned `requirements.txt`, and `src/creditfair/config.py` using `pathlib.Path(__file__).resolve().parents[2]`.
- **Key Concept:** **Editable Package Layout (`pip install -e .`) vs. `sys.path` Hacks.** Installing `creditfair` as a local editable package registers a pointer inside `.venv/Lib/site-packages/` so notebooks, tests, FastAPI, and Streamlit pages can `import creditfair` cleanly from any working directory.
- **One Gotcha:** Hardcoding relative strings like `"../data/raw"` breaks depending on whether a script is launched from `notebooks/`, `app/`, or the repo root. Anchoring all paths to `Path(__file__).resolve().parents[2]` inside `config.py` guarantees OS-independent path resolution.

---

## Phase 1: Data Hygiene, Quirk Cleaning & Leakage-Safe Splitting
- **What Was Built:** `src/creditfair/data.py` and `src/creditfair/features.py`, plus `notebooks/01_eda.ipynb`. Downloaded the UCI Default of Credit Card Clients dataset ($N = 30,000$), fixed `PAY_0` $\rightarrow$ `PAY_1`, remapped undocumented `EDUCATION` (`0, 5, 6` $\rightarrow$ `4` = Others, 345 rows) and `MARRIAGE` (`0` $\rightarrow$ `3` = Others, 54 rows), derived `AGE_GROUP` (`21-29`, `30-39`, `40-49`, `50+`), and created stratified 70/15/15 splits (`21,000 / 4,500 / 4,500`) with protected attributes (`SEX`, `AGE_GROUP`, `MARRIAGE`) separated into `S_train`, `S_val`, and `S_test`.
- **Key Concept:** **Demographic Base Rates & Impossibility Theorems.** In this dataset, Male applicants have a higher observed base default rate (~24.2%) than Female applicants (~20.8%), and younger/older age brackets differ in base rates. Because base rates differ across groups, a well-calibrated model cannot simultaneously satisfy Demographic Parity (equal denial rates) and Equalized Odds (equal TPR and FPR across groups) — known as the **Chouldechova / Kleinberg-Mullainathan-Raghavan Impossibility Theorem**.
- **One Gotcha:** The raw UCI Excel sheet has two header rows (`X1..X23` on row 0 and human names on row 1) and jumps from `PAY_0` directly to `PAY_2` (there is no `PAY_1`). Standardizing `PAY_0` $\rightarrow$ `PAY_1` immediately on ingestion prevents off-by-one bugs in feature descriptions and reason codes.

---

## Phase 2: Baseline Models, Credit-Industry Metrics & Probability Calibration
- **What Was Built:** `src/creditfair/evaluate.py`, `src/creditfair/train.py`, and `notebooks/02_baseline_models.ipynb`. Trained Scaled Logistic Regression, HistGradientBoosting, Class-Weighted XGBoost, and Isotonic-Calibrated XGBoost. Computed ROC-AUC, PR-AUC, Gini ($2 \times \text{AUC} - 1$), Kolmogorov-Smirnov (KS) statistic, and Brier Score. Selected the cost-optimal decision cutoff on the validation set using a 5:1 False-Negative to False-Positive business cost ratio.
- **Key Concept:** **Ranking Power (AUC/KS/Gini) vs. Probability Calibration (Brier Score).** Setting `scale_pos_weight` in XGBoost to counter the 22% class imbalance preserves ranking order (ROC-AUC stays ~0.78), but multiplies the odds by ~3.5x—pushing predicted probabilities artificially high (poor Brier score). Post-hoc Isotonic Regression fitted on the validation set restores true empirical probabilities so threshold selection reflects real default risk.
- **One Gotcha:** Never select the decision threshold on the test set, and never fit a probability calibrator on the training set (where tree models overfit) or test set (which leaks test labels). Both calibration and threshold optimization must happen strictly on the held-out validation split (`X_val, y_val`).

---

## Phase 3: Explainability (SHAP, Adverse Action Reason Codes & Proxy Leakage)
- **What Was Built:** `src/creditfair/explain.py` and `notebooks/03_explainability.ipynb`. Built global SHAP feature importance, local `explain_applicant(row)` generating Top-4 plain-English Adverse Action Notice reason codes, nearest-neighbor explanation stability testing, and a Proxy Leakage Audit training XGBoost models to predict `SEX` and `AGE_GROUP` purely from allowed financial features.
- **Key Concept:** **Why "Fairness Through Unawareness" Fails (Proxy Leakage).** Even though `SEX` and `AGE` were completely excluded from `X_train`, our proxy classifier predicts `SEX` with ROC-AUC = `0.6166` and `AGE_GROUP` with Macro ROC-AUC = `0.6393` using only `LIMIT_BAL`, `EDUCATION`, and bill/payment histories. Features like credit limit encode age and gender correlations, proving that dropping protected columns does not make a model blind to demographics.
- **One Gotcha:** `shap.TreeExplainer` explains the raw margin (log-odds) of the underlying `XGBClassifier` trees rather than the post-hoc `IsotonicRegression` wrapper. Because Isotonic Regression is a monotonic non-decreasing transformation, the ranking and signs of feature contributions remain faithful to the final calibrated probability.

---

## Phase 4: Subgroup Fairness Audit (`Fairlearn MetricFrame` & Bootstrap CIs)
- **What Was Built:** `src/creditfair/fairness.py` and `notebooks/04_fairness_audit.ipynb`. Computed per-group accuracy, denial selection rate, approval rate, TPR, FPR, and precision across `SEX`, `AGE_GROUP`, `MARRIAGE`, and intersectional `SEX x AGE_GROUP`, along with 200-iteration 95% bootstrap confidence intervals.
- **Key Concept:** **Point Estimates vs. Statistical Uncertainty in Small Subgroups.** Small subgroups (such as `MARRIAGE = Others` or `Male | 50+`) exhibit wider bootstrap confidence intervals. Reporting a raw point-estimate gap without confidence intervals can cause teams to chase statistical noise rather than genuine systematic bias.
- **One Gotcha:** In Fairlearn, `selection_rate` measures the fraction of `y_pred == 1`. Since `1 = Default (Deny Credit)` in the UCI dataset, the **favorable outcome** for an applicant is `y_pred == 0` (Approve Credit). For the US EEOC Four-Fifths Rule ($\text{DI} \ge 0.80$), Disparate Impact must be evaluated on the **favorable outcome (Approval Rate)** as well as monitoring disparities in Denial Rate and False Positive Rate (wrongful denial of good borrowers).

---

## Phase 5: Algorithmic Bias Mitigation & The Legal/Accuracy Trade-Off
- **What Was Built:** `src/creditfair/mitigate.py`. Implemented and benchmarked:
  1. **Pre-processing:** Kamiran & Calders (2012) Sample Reweighing
  2. **In-processing:** `fairlearn.reductions.ExponentiatedGradient` (`DemographicParity` & `EqualizedOdds`)
  3. **Post-processing:** `fairlearn.postprocessing.ThresholdOptimizer` (`demographic_parity` & `equalized_odds`)
- **Key Concept:** **Disparate Impact vs. Disparate Treatment at Inference Time.** `ThresholdOptimizer` achieves near-zero Demographic Parity and Equalized Odds gaps on the validation/test sets, but it **requires the applicant's sensitive attribute (`SEX` or `AGE_GROUP`) at inference time** so it can apply different score cutoffs to different demographic groups. Under fair-lending laws (e.g., US ECOA), explicitly applying different credit score thresholds based on sex or age is **Disparate Treatment**, which is legally restricted. Pre-processing (Reweighing) and In-processing (`ExponentiatedGradient`) do not require protected attributes at prediction time.
- **One Gotcha:** `ThresholdOptimizer` outputs randomized binary predictions (`0` or `1`) rather than continuous probabilities, so ROC-AUC does not apply directly to its post-processed binary output; instead, we compare binary Accuracy, F1, Expected Business Cost per Applicant, and Fairness Gaps on equal footing.

---

## Phase 6–8: Multi-Page Streamlit App, Pytest Suite, Model Card & FastAPI
- **What Was Built:** `app/Home.py`, `1_Underwriter_View.py`, `2_Compliance_View.py`, `3_Model_Card.py`, `api/main.py`, `tests/test_creditfair.py`, `Dockerfile`, `MODEL_CARD.md`, and `README.md`.
- **Key Concept:** **Artifact-Driven UI & Caching Separation (`@st.cache_resource` vs. `@st.cache_data`).** Precomputing all heavy training, bootstrap CIs, and `ExponentiatedGradient` reductions into `artifacts/*.json` and `*.parquet` allows the Streamlit Cloud app to boot in <2 seconds on free-tier CPU.
- **One Gotcha:** In multi-page Streamlit apps (`app/pages/*.py`), running `streamlit run app/Home.py` sets the working directory to the repository root, but importing local packages can fail on cloud runners unless `sys.path` includes `src/` or `pip install -e .` is run. Adding a 3-line `Path(__file__).resolve()` fallback in each page guarantees zero `ModuleNotFoundError` issues on Streamlit Community Cloud.
