# Housing Society Water Demand Prediction

**Group ID:** G017  
**Problem ID:** MP071  
**Course:** SCSE3040 MLOps  
**Batch:** 52

## Team Members

| Enrollment Number | Name |
|---|---|
| S24CSEU0983 | Priyanshi Aggarwal |
| S24CSEU1026 | Anvi Tyagi |
| S24CSEU1477 | Trishika Gupta |
| S24CSEU1021 | Anubhav |

## Project Description

This project focuses on predicting daily water demand in housing societies in Nagpur using machine learning and an MLOps pipeline.

## Data Versioning

DVC is used to track the original training and later datasets. Original data files are preserved and are not manually modified.

## Prediction Service

`app/main.py` is a FastAPI service that loads the champion model from `models/model.joblib`.

| Endpoint | What it does |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` |
| `POST /predict` | Takes the 8 input fields and returns `{"prediction": <litres>, "unit": "litres_per_day"}` |

The client's two rules are in the request schema, so a bad request gets status 422:
`flats_occupied` above 500 is rejected, and `temp_c` below 4 is rejected.

Run locally (from the repository root):

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload      # then open http://127.0.0.1:8000/docs
pytest                             # runs tests/test_api.py
```

Run in Docker:

```bash
docker build -t mp071-api:1.0 .
docker run -p 8000:8000 mp071-api:1.0
```

## Repository

https://github.com/Priyanshi061117/Housing-Society-Water-Demand.git
