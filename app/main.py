"""Prediction service for MP071 - Housing Society Water Demand (Nagpur).

    POST /predict  -> litres of water the society will use in the day
    GET  /health   -> {"status": "ok"}

Run it locally from the repository root:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000/docs to try it in the browser.
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Repository root. Building paths from here means the app works whichever
# folder it is started from: a laptop, GitHub Actions, or /code in Docker.
ROOT = Path(__file__).resolve().parent.parent

# The same settings file the training code reads, so the API and training
# always agree on the model file and the feature names.
with open(ROOT / "params.yaml") as f:
    PARAMS = yaml.safe_load(f)

FEATURES = PARAMS["features"]["numeric"] + PARAMS["features"]["categorical"]

# Load the champion model ONCE, when the app starts (not on every request).
# It is a whole scikit-learn Pipeline: median imputer + one-hot encoder +
# regressor. So the raw request values can go straight into .predict().
MODEL = joblib.load(ROOT / PARAMS["model_path"])


class WaterDemandRequest(BaseModel):
    """The request body: exactly the 8 fields from PROBLEM.md section 5.

    Each Field(...) rule is checked by FastAPI before predict() runs. A request
    that breaks a rule gets status 422 and never reaches the model.
    """

    # Client rule 1: more than 500 occupied flats is impossible -> le=500.
    flats_occupied: int = Field(..., ge=0, le=500,
                                description="Flats with people living")
    # Client rule 2: a temperature below 4 deg C is a typing mistake -> ge=4.
    temp_c: float = Field(..., ge=4, description="Maximum temperature, deg C")

    # Our own common-sense limits (not client rules): no negative areas or
    # counts, a percentage stays within 0-100, a day has at most 24 hours.
    # A negative supply_hours also stops an old -999 sentinel getting in.
    garden_area_m2: float = Field(..., ge=0,
                                  description="Garden area watered that day, m2")
    tank_level_morning_pct: float = Field(..., ge=0, le=100,
                                          description="Overhead tank level at 6 am, %")
    leak_complaints: int = Field(..., ge=0, description="Open leak complaints")
    supply_hours: float = Field(..., ge=0, le=24,
                                description="Hours of municipal supply")

    # Only the spellings the model was trained on. Without this, a typo like
    # "tower b" would not fail: the encoder would silently ignore it and the
    # prediction would quietly get worse.
    supply_source: Literal["municipal", "borewell", "tanker"]
    tower: Literal["Tower A", "Tower B", "Tower C"]


class PredictionResponse(BaseModel):
    prediction: float  # predicted litres_per_day
    unit: str = "litres_per_day"


app = FastAPI(
    title="MP071 Housing Society Water Demand API",
    description="Predicts the litres of water a Nagpur housing society uses in a day.",
    version="1.0",
)


@app.get("/health")
def health():
    """Liveness/readiness check. Kubernetes probes call this."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: WaterDemandRequest):
    # One-row table with the columns in the same order as in training.
    row = pd.DataFrame([request.model_dump()], columns=FEATURES)
    litres = float(MODEL.predict(row)[0])
    return PredictionResponse(prediction=round(litres, 1))
