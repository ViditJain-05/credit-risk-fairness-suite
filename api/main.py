"""Phase 8 (Stretch): FastAPI Service for Credit Risk Prediction & SHAP Reason Codes.

Exposes:
- `GET /health`: Liveness & model metadata check
- `POST /predict`: Validated credit default probability & binary underwriting decision
- `POST /explain`: Full SHAP breakdown + Top-4 plain-English Adverse Action reason codes

Uses Pydantic v2 models (`Field` constraints) for strict input validation and
calls the exact same `creditfair` functions used by the Streamlit app.
"""

from __future__ import annotations

from typing import Any, Dict, List

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from creditfair import config
from creditfair.explain import explain_applicant

app = FastAPI(
    title="Credit Risk Explainability & Fairness API",
    description=(
        "Educational REST API for calibrated credit-default prediction and "
        "SHAP Adverse Action Notice reason code generation."
    ),
    version="0.1.0",
)

_MODEL_BUNDLE: Dict[str, Any] | None = None


def get_model_bundle() -> Dict[str, Any]:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = joblib.load(config.get_artifact_path("model.joblib"))
    return _MODEL_BUNDLE


class ApplicantInput(BaseModel):
    """Pydantic request schema validating the 20 non-sensitive applicant features."""

    LIMIT_BAL: float = Field(120000.0, ge=1000.0, le=2000000.0, description="Credit limit (NT$)")
    EDUCATION: int = Field(2, ge=1, le=4, description="1=Grad, 2=Univ, 3=High School, 4=Others")
    PAY_1: int = Field(0, ge=-2, le=8, description="Repayment status in Month 1 (Sept)")
    PAY_2: int = Field(0, ge=-2, le=8, description="Repayment status in Month 2 (Aug)")
    PAY_3: int = Field(0, ge=-2, le=8, description="Repayment status in Month 3 (July)")
    PAY_4: int = Field(0, ge=-2, le=8, description="Repayment status in Month 4 (June)")
    PAY_5: int = Field(0, ge=-2, le=8, description="Repayment status in Month 5 (May)")
    PAY_6: int = Field(0, ge=-2, le=8, description="Repayment status in Month 6 (April)")
    BILL_AMT1: float = Field(35000.0, description="Statement balance Month 1 (NT$)")
    BILL_AMT2: float = Field(32000.0, description="Statement balance Month 2 (NT$)")
    BILL_AMT3: float = Field(30000.0, description="Statement balance Month 3 (NT$)")
    BILL_AMT4: float = Field(28000.0, description="Statement balance Month 4 (NT$)")
    BILL_AMT5: float = Field(25000.0, description="Statement balance Month 5 (NT$)")
    BILL_AMT6: float = Field(22000.0, description="Statement balance Month 6 (NT$)")
    PAY_AMT1: float = Field(3000.0, ge=0.0, description="Payment amount Month 1 (NT$)")
    PAY_AMT2: float = Field(3000.0, ge=0.0, description="Payment amount Month 2 (NT$)")
    PAY_AMT3: float = Field(2500.0, ge=0.0, description="Payment amount Month 3 (NT$)")
    PAY_AMT4: float = Field(2500.0, ge=0.0, description="Payment amount Month 4 (NT$)")
    PAY_AMT5: float = Field(2000.0, ge=0.0, description="Payment amount Month 5 (NT$)")
    PAY_AMT6: float = Field(2000.0, ge=0.0, description="Payment amount Month 6 (NT$)")


class PredictResponse(BaseModel):
    default_probability: float
    threshold: float
    decision: str
    disclaimer: str


class ReasonItem(BaseModel):
    rank: int
    feature: str
    feature_description: str
    value: float
    shap_value: float
    reason_text: str


class ExplainResponse(BaseModel):
    default_probability: float
    threshold: float
    decision: str
    base_value: float
    top_adverse_reasons: List[ReasonItem]
    top_protective_reasons: List[ReasonItem]
    shap_values: Dict[str, float]


@app.get("/health")
def health_check() -> Dict[str, Any]:
    bundle = get_model_bundle()
    return {
        "status": "ok",
        "model": "XGBoost (Isotonic Calibrated)",
        "optimal_threshold": float(bundle["optimal_threshold"]),
        "feature_count": len(bundle["feature_names"]),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(applicant: ApplicantInput) -> PredictResponse:
    bundle = get_model_bundle()
    result = explain_applicant(applicant.model_dump(), model_bundle=bundle, top_k=1)
    return PredictResponse(
        default_probability=result["default_probability"],
        threshold=result["threshold"],
        decision=result["decision"],
        disclaimer="Educational demo only — not for real lending decisions.",
    )


@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(applicant: ApplicantInput) -> ExplainResponse:
    bundle = get_model_bundle()
    result = explain_applicant(applicant.model_dump(), model_bundle=bundle, top_k=4)
    return ExplainResponse(
        default_probability=result["default_probability"],
        threshold=result["threshold"],
        decision=result["decision"],
        base_value=result["base_value"],
        top_adverse_reasons=[ReasonItem(**r) for r in result["top_adverse_reasons"]],
        top_protective_reasons=[ReasonItem(**r) for r in result["top_protective_reasons"]],
        shap_values=result["shap_values"],
    )
