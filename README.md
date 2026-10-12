# ⚖️ Credit Risk Explainability & Fairness Auditing Suite

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange.svg)](https://shap.readthedocs.io/)
[![Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn%200.15-purple.svg)](https://fairlearn.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

> **⚠️ Educational & Portfolio Disclaimer:** This repository is an **educational demonstration** simulating the regulator-facing explainability and fairness auditing pipeline used in consumer credit risk. It is trained on the historical 2005 UCI Taiwan Credit Card Default dataset ($N = 30,000$) and **must never be used for real-world lending decisions**. Nothing in this repository constitutes legal or regulatory compliance advice.

---

## 🎯 Problem & Regulatory Context

In consumer lending, training a high-AUC classifier is only half the engineering challenge. Financial institutions operate under strict transparency and fair-lending expectations—such as **Adverse Action Notice** requirements under the US Equal Credit Opportunity Act (ECOA / Regulation B), automated profiling safeguards under **EU GDPR Article 22**, and fair-practice guidelines for digital lenders in India.

Specifically, a production-ready credit risk system must answer three questions:
1. **Individual Explainability (Underwriter & Applicant View):** *Why was this specific applicant denied credit, expressed in plain-English Adverse Action reason codes?*
2. **Subgroup Fairness & Proxy Auditing (Compliance View):** *Does removing protected attributes (`SEX`, `AGE_GROUP`, `MARRIAGE`) make the model unbiased, or do allowed financial variables act as demographic proxies?*
3. **Bias Mitigation Trade-Offs:** *How much predictive accuracy or business cost do Pre-processing (Sample Reweighing), In-processing (`ExponentiatedGradient`), and Post-processing (`ThresholdOptimizer`) trade off to reduce group disparities?*

---

## 🏗️ System Architecture

```
UCI Credit Card Default (30,000 rows, 22.12% default rate)
   │
   ├──► Phase 1: Quirk Cleaning (PAY_0→PAY_1, EDUCATION/MARRIAGE codes)
   │             + Stratified 70/15/15 Split (21,000 / 4,500 / 4,500)
   │             + Strict Separation: Model Features (X) vs. Audit Frame (S)
   │
   ├──► Phase 2: Baseline Models (Scaled LogReg, HistGBDT, XGBoost)
   │             + Validation Isotonic Probability Calibration
   │             + Cost-Optimal Thresholding (5:1 FN:FP Business Cost Matrix)
   │
   ├──► Phase 3: TreeSHAP Explainability + Plain-English Adverse Action Codes
   │             + k-NN Explanation Stability Audit + Proxy Leakage Detector
   │
   ├──► Phase 4: Fairlearn MetricFrame Subgroup Audit (SEX, AGE_GROUP, MARRIAGE,
   │             SEX × AGE_GROUP) + 200-Round 95% Bootstrap Confidence Intervals
   │
   ├──► Phase 5: Bias Mitigation Comparison
   │             ├── Pre-processing: Sample Reweighing (Kamiran & Calders, 2012)
   │             ├── In-processing: Fairlearn ExponentiatedGradient (DP & EO)
   │             └── Post-processing: Fairlearn ThresholdOptimizer (DP & EO)
   │
   └──► Phase 6–8: Multi-Page Streamlit App + FastAPI Service + Pytest Suite
```

---

## 📊 Key Results (100% Generated from `artifacts/`)

### 1. Held-Out Test Set Model Comparison ($N = 4,500$, Cost-Optimal Cutoff $t = 0.1250$)

| Model | ROC-AUC | PR-AUC | Gini ($2\cdot\text{AUC}-1$) | KS Statistic | Brier Score $\downarrow$ | Accuracy | Recall (TPR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Scaled)** | `0.7492` | `0.4969` | `0.4984` | `0.4007` | `0.1429` | `0.5424` | `0.8213` |
| **HistGradientBoosting** | `0.7770` | `0.5554` | `0.5541` | `0.4192` | `0.1353` | `0.5564` | `0.8444` |
| **XGBoost (Weighted, Uncalibrated)** | `0.7811` | `0.5606` | `0.5623` | `0.4268` | `0.1794` | `0.7502` | `0.6416` |
| **XGBoost (Calibrated — Primary)** | **`0.7830`** | **`0.5515`** | **`0.5660`** | **`0.4346`** | **`0.1343`** | **`0.6109`** | **`0.8163`** |

### 2. Global SHAP Feature Importance & Explanation Stability
| Rank | Feature | Description | Mean `|SHAP|` (Log-Odds) |
| :---: | :--- | :--- | :---: |
| 1 | `MAX_DELINQUENCY_6M` | Worst repayment delay (months late) over past 6 months | `0.3567` |
| 2 | `PAY_1` | Repayment status in most recent month (Month 1 / Sept) | `0.3019` |
| 3 | `MONTHS_DELINQUENT_6M` | Total number of months with late payment in past 6 months | `0.1716` |
| 4 | `UTILIZATION_RATIO_1` | Recent credit line utilization ratio (Bill 1 / Credit Limit) | `0.1064` |
| 5 | `LIMIT_BAL` | Credit limit amount (NT$) | `0.0891` |
| 6 | `BILL_AMT1` | Bill statement balance in most recent month (NT$) | `0.0871` |

- **Explanation Stability ($k=5$ Nearest Neighbors):** Mean Spearman rank correlation = **`0.6300`**, Top-4 Adverse Reason Jaccard overlap = **`59.3%`**.
- **Proxy Leakage Audit ("Fairness Through Unawareness"):** Even with `SEX` and `AGE` excluded from `X`, allowed features predict `SEX` with **ROC-AUC = `0.6166`** and `AGE_GROUP` with **Macro ROC-AUC = `0.6393`**.

### 3. Bias Mitigation Comparison (`SEX`)
| Method | Paradigm | Protected Attr at Inference? | Accuracy | F1 | Expected Cost/App $\downarrow$ | DP Diff $\downarrow$ | Equalized Odds Diff $\downarrow$ | Approval DI Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unmitigated Baseline (XGBoost)** | Baseline | No ✅ | `0.6109` | `0.4815` | `0.5518` | `0.0406` | `0.0367` | `0.9165` |
| **Pre-Processing: Sample Reweighing** | Pre-processing | No ✅ | `0.5700` | `0.4686` | `0.5571` | `0.0345` | `0.0312` | `0.9190` |
| **In-Processing: ExponentiatedGradient (Demographic Parity)** | In-processing | No ✅ | `0.7958` | `0.3316` | `0.8869` | `0.0096` | `0.0388` | `0.9896` |
| **In-Processing: ExponentiatedGradient (Equalized Odds)** | In-processing | No ✅ | `0.7940` | `0.3439` | `0.8753` | `0.0115` | `0.0337` | `0.9874` |
| **Post-Processing: ThresholdOptimizer (Demographic Parity)** | Post-processing | Yes ⚠️ | `0.8180` | `0.4356` | `0.7864` | `0.0070` | `0.0817` | `0.9922` |
| **Post-Processing: ThresholdOptimizer (Equalized Odds)** | Post-processing | Yes ⚠️ | `0.8189` | `0.4504` | `0.7696` | `0.0001` | `0.0740` | `0.9999` |

### 4. Bias Mitigation Comparison (`AGE_GROUP`)
| Method | Paradigm | Protected Attr at Inference? | Accuracy | F1 | Expected Cost/App $\downarrow$ | DP Diff $\downarrow$ | Equalized Odds Diff $\downarrow$ | Approval DI Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unmitigated Baseline (XGBoost)** | Baseline | No ✅ | `0.6109` | `0.4815` | `0.5518` | `0.0684` | `0.0789` | `0.8646` |
| **Pre-Processing: Sample Reweighing** | Pre-processing | No ✅ | `0.5687` | `0.4672` | `0.5602` | `0.0809` | `0.0671` | `0.8185` |
| **In-Processing: ExponentiatedGradient (Demographic Parity)** | In-processing | No ✅ | `0.8029` | `0.3784` | `0.8424` | `0.0252` | `0.0528` | `0.9725` |
| **In-Processing: ExponentiatedGradient (Equalized Odds)** | In-processing | No ✅ | `0.8038` | `0.3804` | `0.8407` | `0.0253` | `0.0401` | `0.9725` |
| **Post-Processing: ThresholdOptimizer (Demographic Parity)** | Post-processing | Yes ⚠️ | `0.8178` | `0.4619` | `0.7547` | `0.0229` | `0.1001` | `0.9743` |
| **Post-Processing: ThresholdOptimizer (Equalized Odds)** | Post-processing | Yes ⚠️ | `0.8180` | `0.4212` | `0.8024` | `0.0349` | `0.1262` | `0.9622` |

---

## 📂 Repository Structure

```text
credit-risk-fairness-suite/
├── README.md
├── LEARNING_LOG.md
├── MODEL_CARD.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── data/
│   ├── raw/                        # Gitignored; cached raw UCI dataset
│   └── processed/                  # Stratified 70/15/15 train/val/test parquet splits
├── notebooks/
│   ├── 01_eda.ipynb                # Quirk cleaning, demographic base rates & leakage checks
│   ├── 02_baseline_models.ipynb    # LogReg, HistGBDT, XGBoost, KS/Gini, calibration & cost cutoff
│   ├── 03_explainability.ipynb     # Global/local SHAP, Adverse Action reasons & proxy leakage
│   └── 04_fairness_audit.ipynb     # Fairlearn MetricFrame, bootstrap CIs & mitigation trade-offs
├── src/creditfair/
│   ├── config.py                   # Paths, seeds, feature definitions & cost ratios
│   ├── data.py                     # Ingestion, quirk cleaning & stratified splitting
│   ├── features.py                 # Row-wise domain feature engineering
│   ├── train.py                    # Model training & validation Isotonic calibration
│   ├── evaluate.py                 # ROC-AUC, PR-AUC, KS, Gini, Brier & cost threshold
│   ├── explain.py                  # TreeSHAP, Adverse Action codes, stability & proxy audit
│   ├── fairness.py                 # MetricFrame subgroup audit & bootstrap 95% CIs
│   ├── mitigate.py                 # Reweighing, ExponentiatedGradient & ThresholdOptimizer
│   └── report.py                   # End-to-end artifact generator
├── artifacts/                      # Precomputed models, JSON metrics & SHAP parquet samples
├── app/
│   ├── Home.py                     # Overview, dataset hygiene & calibration curves
│   └── pages/
│       ├── 1_Underwriter_View.py   # Applicant simulator, SHAP waterfall & reason codes
│       ├── 2_Compliance_View.py    # MetricFrame, bootstrap CIs & mitigation dashboard
│       └── 3_Model_Card.py         # Interactive Model Card viewer
├── api/
│   └── main.py                     # FastAPI /predict, /explain, and /health service
└── tests/
    └── test_creditfair.py          # Pytest unit & API integration tests
```

---

## 🚀 Quickstart & Running Locally

### 1. Create Virtual Environment & Install Package
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### 2. Regenerate All Data Splits & Artifacts (Optional — Precomputed in `artifacts/`)
```powershell
python -m creditfair.report
```

### 3. Launch the Multi-Page Streamlit Application
```powershell
streamlit run app/Home.py
```

### 4. Launch the FastAPI Service (Stretch Goal)
```powershell
uvicorn api.main:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `GET http://localhost:8000/health`
- Prediction: `POST http://localhost:8000/predict`
- SHAP Adverse Action Explanation: `POST http://localhost:8000/explain`

### 5. Run the Automated Test Suite
```powershell
pytest -v
```

---

## ☁️ Deployment (Streamlit Community Cloud & Docker)

### Streamlit Community Cloud
1. Push this repository to GitHub (the precomputed `artifacts/` and `data/processed/` are small and tracked so the cloud app starts in `<2 seconds`).
2. On [share.streamlit.io](https://share.streamlit.io), select your repository and set **Main file path** to:
   ```text
   app/Home.py
   ```
3. Click **Deploy**.

### Docker (Render / Cloud Run / Local)
```bash
docker build -t credit-risk-fairness-suite .
docker run -p 8501:8501 credit-risk-fairness-suite
```

---

## ⚠️ Honest Limitations
1. **Historical 2005 Taiwan Data:** Collected from a single bank in Taiwan in 2005; demographic base rates and credit behaviors do not generalize to modern lending markets.
2. **Fairness Metric Impossibility:** Because base default rates differ across sex and age groups, satisfying Demographic Parity, Equalized Odds, and Calibration simultaneously is mathematically impossible.
3. **Disparate Treatment Constraint:** Post-processing (`ThresholdOptimizer`) requires protected attributes at inference time, which is legally restricted under fair-lending statutes; only pre-processing and in-processing methods operate without inference-time demographic inputs.
