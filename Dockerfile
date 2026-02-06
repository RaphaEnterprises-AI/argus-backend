# Railway deployment Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Cache-bust argument - Railway passes build timestamp
ARG CACHEBUST=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libpq-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
# 1. First install neo4j driver explicitly (before cognee to ensure it's available)
# 2. Then install main package (which includes cognee[postgres,neo4j])
# 3. Verify neo4j is importable
RUN echo "CACHEBUST: ${CACHEBUST}" && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "neo4j>=5.20.0" && \
    pip install --no-cache-dir . && \
    python -c "import neo4j; print(f'neo4j {neo4j.__version__} installed successfully')" && \
    python -c "import cognee; print(f'cognee {cognee.__version__} installed successfully')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (Railway sets PORT dynamically)
EXPOSE ${PORT:-8000}

# Run the application using shell form to expand $PORT
CMD python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
