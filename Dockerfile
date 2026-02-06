# Railway deployment Dockerfile - Optimized for size
# Uses multi-stage build and CPU-only PyTorch

# =============================================================================
# Stage 1: Builder - Install dependencies with build tools
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libpq-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only dependency files first (for better caching)
COPY pyproject.toml README.md ./

# Install CPU-only PyTorch FIRST (saves ~1.5GB vs full torch)
# This satisfies torch requirement before sentence-transformers
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install neo4j driver explicitly (required for Cognee graph storage)
RUN pip install --no-cache-dir "neo4j>=5.20.0"

# Copy source and install main package
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Verify critical dependencies
RUN python -c "import neo4j; print(f'neo4j {neo4j.__version__}')" && \
    python -c "import cognee; print(f'cognee {cognee.__version__}')" && \
    python -c "import torch; print(f'torch {torch.__version__} (CPU)')"

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/ ./src/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (Railway sets PORT dynamically)
EXPOSE ${PORT:-8000}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run the application
CMD ["sh", "-c", "python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
