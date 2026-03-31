# Use Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for EasyOCR/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models (id, en) to speed up first request
# This avoids slow startup when the first OCR request is made
RUN python -c "import easyocr; reader = easyocr.Reader(['id', 'en'], gpu=False)"

# Copy application code
COPY . .

# Expose port (Railway will provide this via $PORT environment variable)
EXPOSE 8000

# Start the application
# We use 0.0.0.0 to bind to all interfaces
# Railway provides the port via the PORT environment variable
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
