# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# EasyOCR runtime deps (opencv/torch may require these libs at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first (smaller + more stable for deploy)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY ocr/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Warm-up: pre-download default EasyOCR models (id,en) to avoid slow first request
RUN python -c "import easyocr; easyocr.Reader(['id','en'], gpu=False)"

COPY ocr ./ocr

EXPOSE 8001

CMD ["sh", "-c", "uvicorn ocr.main:app --host 0.0.0.0 --port ${PORT:-8001}"]

