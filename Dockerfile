# Prediction service image for MP071.
#   Build:  docker build -t mp071-api:1.0 .
#   Run:    docker run -p 8000:8000 mp071-api:1.0
#   Try:    curl http://localhost:8000/health
FROM python:3.12-slim

# Send logs straight to the screen (no buffering); skip writing .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

# 1. Install the libraries BEFORE copying our code. Docker caches this layer,
#    so editing app/main.py does not reinstall every library on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy only what the service needs: settings, the champion model, the app.
COPY params.yaml .
COPY models/ models/
COPY app/ app/

EXPOSE 8000

# 0.0.0.0 so the server is reachable from outside the container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
