# Use Python 3.11 slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

# Set working directory
WORKDIR /app

# System deps: needed when some Python wheels are unavailable and pip must build from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --retries 10 --timeout 100

# Copy application code
COPY . .

# Expose ports (Railway commonly routes to 8080 for Docker deployments)
EXPOSE 8000
EXPOSE 8080

# Start the application
# Railway provides the port via the PORT environment variable
# Start via Python to avoid shell quoting / expansion issues
CMD ["python", "-m", "app.run"]
