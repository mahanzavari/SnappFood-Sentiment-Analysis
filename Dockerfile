FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends -qq build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=1000 -r requirements.docker.txt
COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app.main:app"]
