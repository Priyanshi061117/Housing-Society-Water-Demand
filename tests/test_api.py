"""Tests for the prediction service (app/main.py).

They use only the committed model file and a request written below, so they
do not need the CSV files (those are tracked by DVC, not stored in git).
Run from the repository root with:  pytest
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

# A realistic request: record R50001 from holdout_features.csv, with only the
# 8 fields the API takes (no record_id, no date, no leaking meter column).
VALID_REQUEST = {
    "flats_occupied": 373,
    "temp_c": 8.7,
    "garden_area_m2": 890,
    "tank_level_morning_pct": 65,
    "leak_complaints": 14,
    "supply_hours": 11.8,
    "supply_source": "municipal",
    "tower": "Tower C",
}


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_litres():
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["unit"] == "litres_per_day"
    assert isinstance(body["prediction"], float)
    assert body["prediction"] > 0


def test_rule_1_rejects_more_than_500_flats():
    response = client.post("/predict", json={**VALID_REQUEST, "flats_occupied": 501})
    assert response.status_code == 422


def test_rule_2_rejects_temperature_below_4():
    response = client.post("/predict", json={**VALID_REQUEST, "temp_c": 3.9})
    assert response.status_code == 422


def test_rule_edges_500_flats_and_4_degrees_are_allowed():
    response = client.post(
        "/predict", json={**VALID_REQUEST, "flats_occupied": 500, "temp_c": 4}
    )
    assert response.status_code == 200


def test_missing_field_is_rejected():
    request = dict(VALID_REQUEST)
    del request["tower"]
    response = client.post("/predict", json=request)
    assert response.status_code == 422


def test_unknown_tower_is_rejected():
    response = client.post("/predict", json={**VALID_REQUEST, "tower": "Tower D"})
    assert response.status_code == 422


def test_sentinel_supply_hours_is_rejected():
    response = client.post("/predict", json={**VALID_REQUEST, "supply_hours": -999})
    assert response.status_code == 422
