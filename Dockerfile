# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — builder
# Installs dependencies into an isolated location, using pip's build cache.
# This stage is discarded from the final image; only the installed packages
# and the app code below are kept, which is what makes the final image small.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

WORKDIR /app

# Build tools needed to compile some ML libraries' C/C++ extensions
# (e.g. lightgbm, xgboost) if no pre-built wheel is available for this
# platform. Kept only in this stage, not in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

# --prefix isolates installed packages under /install, which the final
# stage copies wholesale — avoids reinstalling on every code change since
# this layer is only invalidated when requirements-api.txt changes.
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ---------------------------------------------------------------------------
# Stage 2 — final runtime image
# Starts fresh from a clean slim image; only copies what's needed to run
# the API, not the build tools or pip cache from stage 1.
# ---------------------------------------------------------------------------
FROM python:3.13-slim

WORKDIR /app

# Run as a non-root user — standard security practice for containers that
# will be exposed on a network (Render, HuggingFace Spaces, etc.).
RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY models/ ./models/

USER appuser

EXPOSE 8000

# Basic container-level healthcheck, backed by the API's own /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# No --reload here: --reload is a development convenience (auto-restarts on
# file changes) and adds overhead that has no place in a production image.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]