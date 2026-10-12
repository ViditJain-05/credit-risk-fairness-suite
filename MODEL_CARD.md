# Model Card — Credit Risk Explainability & Fairness Suite

*Following the Model Card framework introduced by Mitchell et al. (2019), "Model Cards for Model Reporting".*

> **⚠️ Educational & Regulatory Disclaimer:** This model and auditing suite are built strictly for **educational and portfolio demonstration purposes** using historical 2005 data from Taiwan. They are **not** validated or approved for real-world credit underwriting or lending decisions.

---

## 1. Model Details
- **Developer:** Vidit Jain (B.Tech CSE — Data Science, DJSCE Mumbai)
- **Model Version:** `creditfair v0.1.0` (`RANDOM_SEED = 42`)
- **Primary Model Architecture:** Regularized Gradient Boosted Decision Trees (`XGBClassifier`, `max_depth=4`, `n_estimators=250`, `learning_rate=0.04`) wrapped with validation-fitted **Isotonic Regression probability calibration**.
- **Baseline Benchmarks Compared:** Scaled L2 Logistic Regression & scikit-learn `HistGradientBoostingClassifier`.
- **Decision Rule:** Deny credit (`y_pred = 1`) if calibrated default probability $P(\text{default}) \ge 0.1250$, where the cutoff $0.1250$ was selected on the held-out validation split (`N = 4,500`) using a 5:1 False-Negative to False-Positive business cost matrix.

---

## 2. Intended Use & Out-of-Scope Uses
### Intended Use
- Demonstrating end-to-end **regulator-facing ML auditing** in consumer credit risk:
  1. Per-applicant **SHAP Adverse Action Notice reason codes** (mirroring US ECOA / Regulation B and EU GDPR Article 22 explainability concepts).
  2. Subgroup fairness auditing via `fairlearn.metrics.MetricFrame` with **95% bootstrap confidence intervals**.
  3. Quantifying the accuracy-fairness trade-off across Pre-processing (Sample Reweighing), In-processing (`ExponentiatedGradient`), and Post-processing (`ThresholdOptimizer`).

### Out-of-Scope Uses
- **Production credit scoring or automated loan rejection:** The dataset dates from April–September 2005 from a single Taiwanese bank. Economic regimes, inflation, credit bureau reporting, and legal jurisdictions do not transfer to modern lending.

---

## 3. Training & Evaluation Data
- **Source:** UCI Machine Learning Repository — *Default of Credit Card Clients* (Taiwan, 2005, ID=350).
- **Total Records:** `30,000` credit card holders; overall default rate = `22.12%` (`6,636` defaults, `23,364` non-defaults).
- **Stratified Split (70 / 15 / 15):**
  - **Train:** `21,000` rows (default rate `22.12%`)
  - **Validation:** `4,500` rows (default rate `22.11%`) — used *only* for probability calibration and threshold selection.
  - **Test:** `4,500` rows (default rate `22.13%`) — evaluated *once* for final reporting.
- **Documented Data Quirks Cleaned:**
  - Renamed `PAY_0` $\rightarrow$ `PAY_1` for sequential alignment (`PAY_1`..`PAY_6`).
  - Remapped `345` rows with undocumented `EDUCATION` codes (`0, 5, 6`) into `4` (*Others*).
  - Remapped `54` rows with undocumented `MARRIAGE` code (`0`) into `3` (*Others*).
- **Sensitive Attribute Handling ("Fairness Through Unawareness" Baseline):**
  - `SEX`, `AGE`, `AGE_GROUP` (`21-29`, `30-39`, `40-49`, `50+`), and `MARRIAGE` were stripped from the model feature matrix `X` and stored in separate audit frames (`S_train`, `S_val`, `S_test`).

---

## 4. Overall Quantitative Performance (Held-Out Test Set, $N = 4,500$)

| Model | ROC-AUC | PR-AUC | Gini ($2\cdot\text{AUC}-1$) | KS Statistic | Brier Score $\downarrow$ | Accuracy (@ $t=0.12$) | Recall / TPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Scaled)** | `0.7492` | `0.4969` | `0.4984` | `0.4007` | `0.1429` | `0.5424` | `0.8213` |
| **HistGradientBoosting** | `0.7770` | `0.5554` | `0.5541` | `0.4192` | `0.1353` | `0.5564` | `0.8444` |
| **XGBoost (Calibrated — Primary)** | **`0.7830`** | **`0.5515`** | **`0.5660`** | **`0.4346`** | **`0.1343`** | **`0.6109`** | **`0.8163`** |

---

## 5. Subgroup Disaggregated Performance (`Fairlearn MetricFrame`)

### 5.1 By `SEX` (Unmitigated Calibrated XGBoost)
| Group | $N$ | Base Default Rate | Denial Rate (Selection) | Approval Rate | TPR | FPR (Wrongful Denial) | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Female | 2,734 | 20.04% | 51.32% | 48.68% | 0.8303 | 0.4337 | 0.3243 | 0.6192 |
| Male | 1,766 | 25.37% | 55.38% | 44.62% | 0.7991 | 0.4704 | 0.3661 | 0.5980 |

- **Demographic Parity Difference:** `0.0406` (95% Bootstrap CI: `[0.0094, 0.0706]`)
- **Approval Disparate Impact Ratio:** `0.9165` (Four-Fifths Rule $\ge 0.80$: **PASS**)
- **Equal Opportunity Difference (TPR Gap):** `0.0312`
- **Equalized Odds Difference:** `0.0367`

### 5.2 By `AGE_GROUP` (Unmitigated Calibrated XGBoost)
| Group | $N$ | Base Default Rate | Denial Rate (Selection) | Approval Rate | TPR | FPR (Wrongful Denial) | Precision | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 21-29 | 1,414 | 22.21% | 56.36% | 43.64% | 0.8567 | 0.4800 | 0.3375 | 0.5948 |
| 30-39 | 1,700 | 20.24% | 49.53% | 50.47% | 0.8081 | 0.4159 | 0.3302 | 0.6294 |
| 40-49 | 975 | 22.67% | 52.62% | 47.38% | 0.7919 | 0.4483 | 0.3411 | 0.6062 |
| 50+ | 411 | 28.47% | 55.72% | 44.28% | 0.7778 | 0.4694 | 0.3974 | 0.6010 |

- **Demographic Parity Difference:** `0.0684` (95% Bootstrap CI: `[0.0442, 0.1157]`)
- **Approval Disparate Impact Ratio:** `0.8646` (Four-Fifths Rule $\ge 0.80$: **PASS**)
- **Equalized Odds Difference:** `0.0789`

---

## 6. Proxy Leakage Audit ("Fairness Through Unawareness" Evaluation)
Even though `SEX` and `AGE_GROUP` were removed from the training features:
- A proxy `XGBClassifier` trained on the 26 allowed financial features predicts `SEX` with **ROC-AUC = `0.6166`** (vs. `0.5000` random chance).
- A proxy `XGBClassifier` predicts `AGE_GROUP` with **Macro OvR ROC-AUC = `0.6393`** (driven primarily by `LIMIT_BAL` and `EDUCATION`).
- **Takeaway:** Removing sensitive columns does not prevent demographic encoding when correlated financial proxies remain.

---

## 7. Bias Mitigation & Accuracy/Fairness Trade-Off

### 7.1 Mitigation Comparison on `SEX`
| Method | Paradigm | Uses Protected Attr at Inference? | Accuracy | F1 | Expected Cost/App $\downarrow$ | DP Diff $\downarrow$ | EO Diff $\downarrow$ | Equalized Odds Diff $\downarrow$ | Approval DI Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Unmitigated Baseline (XGBoost) | Baseline | No | 0.6109 | 0.4815 | 0.5518 | 0.0406 | 0.0312 | 0.0367 | 0.9165 |
| Pre-Processing: Sample Reweighing | Pre-processing | No | 0.5700 | 0.4686 | 0.5571 | 0.0345 | 0.0312 | 0.0312 | 0.9190 |
| In-Processing: ExponentiatedGradient (Demographic Parity) | In-processing | No | 0.7958 | 0.3316 | 0.8869 | 0.0096 | 0.0388 | 0.0388 | 0.9896 |
| In-Processing: ExponentiatedGradient (Equalized Odds) | In-processing | No | 0.7940 | 0.3439 | 0.8753 | 0.0115 | 0.0337 | 0.0337 | 0.9874 |
| Post-Processing: ThresholdOptimizer (Demographic Parity) | Post-processing | Yes | 0.8180 | 0.4356 | 0.7864 | 0.0070 | 0.0817 | 0.0817 | 0.9922 |
| Post-Processing: ThresholdOptimizer (Equalized Odds) | Post-processing | Yes | 0.8189 | 0.4504 | 0.7696 | 0.0001 | 0.0740 | 0.0740 | 0.9999 |

### 7.2 Mitigation Comparison on `AGE_GROUP`
| Method | Paradigm | Uses Protected Attr at Inference? | Accuracy | F1 | Expected Cost/App $\downarrow$ | DP Diff $\downarrow$ | EO Diff $\downarrow$ | Equalized Odds Diff $\downarrow$ | Approval DI Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Unmitigated Baseline (XGBoost) | Baseline | No | 0.6109 | 0.4815 | 0.5518 | 0.0684 | 0.0789 | 0.0789 | 0.8646 |
| Pre-Processing: Sample Reweighing | Pre-processing | No | 0.5687 | 0.4672 | 0.5602 | 0.0809 | 0.0603 | 0.0671 | 0.8185 |
| In-Processing: ExponentiatedGradient (Demographic Parity) | In-processing | No | 0.8029 | 0.3784 | 0.8424 | 0.0252 | 0.0528 | 0.0528 | 0.9725 |
| In-Processing: ExponentiatedGradient (Equalized Odds) | In-processing | No | 0.8038 | 0.3804 | 0.8407 | 0.0253 | 0.0401 | 0.0401 | 0.9725 |
| Post-Processing: ThresholdOptimizer (Demographic Parity) | Post-processing | Yes | 0.8178 | 0.4619 | 0.7547 | 0.0229 | 0.1001 | 0.1001 | 0.9743 |
| Post-Processing: ThresholdOptimizer (Equalized Odds) | Post-processing | Yes | 0.8180 | 0.4212 | 0.8024 | 0.0349 | 0.1262 | 0.1262 | 0.9622 |

---

## 8. Ethical Considerations, Limitations & Legal Nuance
1. **No Universal "Fairness":** Because base default rates differ across groups (e.g., Male `24.17%` vs. Female `20.78%`), forcing exact Demographic Parity requires denying creditworthy individuals in higher-risk groups or approving riskier individuals, creating tension with calibration and Equalized Odds.
2. **Disparate Treatment vs. Disparate Impact:** Post-processing (`ThresholdOptimizer`) requires protected attributes at inference time to set group-specific thresholds. In regulated credit markets, explicitly using sex or age at decision time can constitute **Disparate Treatment**, making Pre-processing (Sample Reweighing) or In-processing (`ExponentiatedGradient`) preferable in practice.
