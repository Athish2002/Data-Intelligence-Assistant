# ==============================================================================
# Data Intelligence Assistant (Fullstack Enterprise Edition)
# Production Hardened Multi-Stage Container Image
# ==============================================================================

# ─── Stage 1: Build Dependencies & Wheels ─────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


# ─── Stage 2: Hardened Runtime Container ──────────────────────────────────────
FROM python:3.12-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DIA_HOST=0.0.0.0 \
    DIA_PORT=8000 \
    DIA_STORAGE_PATH=/data/dia

WORKDIR /app

# Install runtime utilities (curl for Kubernetes/Docker container probes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-compiled dependencies from builder
COPY --from=builder /opt/venv /opt/venv

# Create unprivileged system user and group dia:dia (UID/GID 10001)
# Create persistent storage mount path and ensure non-root ownership
RUN groupadd -g 10001 dia && \
    useradd -u 10001 -g dia -m -s /bin/bash dia && \
    mkdir -p /data/dia /home/dia/.dia /app && \
    chown -R dia:dia /data/dia /home/dia /app

# Copy application source code and frontend assets with dia:dia ownership
COPY --chown=dia:dia dia/ dia/
COPY --chown=dia:dia api/ api/
COPY --chown=dia:dia frontend/ frontend/
COPY --chown=dia:dia run_fullstack.py .
COPY --chown=dia:dia pyproject.toml .

# Enforce least-privilege non-root execution context
USER dia:dia

# Dynamic port binding
EXPOSE ${DIA_PORT}

# Production Health & Readiness check against the readiness probe endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${DIA_PORT}/api/v1/ready || exit 1

CMD ["python", "run_fullstack.py"]
