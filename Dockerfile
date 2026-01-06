# E2E Testing Agent - Railway Optimized Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for caching)
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e . || pip install \
    "anthropic>=0.75.0" \
    "langgraph>=1.0.5" \
    "langchain-anthropic>=1.3.0" \
    "langchain-core>=1.2.5" \
    "httpx>=0.27.0" \
    "pydantic>=2.9.0" \
    "pydantic-settings>=2.5.0" \
    "fastapi>=0.115.0" \
    "uvicorn>=0.32.0" \
    "structlog>=24.4.0" \
    "python-dotenv>=1.0.0" \
    "rich>=13.9.0" \
    "pillow>=10.4.0"

# Copy application code
COPY src/ /app/src/

# Create output directories
RUN mkdir -p /app/test-results /app/baselines

# Expose port (Railway sets PORT env var)
EXPOSE 8000

# Use shell form to expand $PORT variable
CMD uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-8000}
