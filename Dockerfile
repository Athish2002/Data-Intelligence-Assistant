# ==============================================================================
# Data Intelligence Assistant (Fullstack Edition)
# Production Container Image
# ==============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered streaming logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DIA_HOST=0.0.0.0 \
    DIA_PORT=8000

WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and frontend assets
COPY dia/ dia/
COPY api/ api/
COPY frontend/ frontend/
COPY run_fullstack.py .
COPY pyproject.toml .

# Create non-root system user for least privilege security
RUN useradd -u 10001 -m -s /bin/bash appuser && \
    mkdir -p /home/appuser/.dia && \
    chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

# Health check against root /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run_fullstack.py"]
