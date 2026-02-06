# Railway deployment Dockerfile - Lightweight
FROM python:3.12-slim

WORKDIR /app

# Install only runtime dependencies (no build tools in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies (no ML/PyTorch - saves ~2GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "neo4j>=5.20.0" && \
    pip install --no-cache-dir . && \
    python -c "import neo4j; print(f'neo4j {neo4j.__version__}')" && \
    python -c "import cognee; print(f'cognee {cognee.__version__}')"

# Cleanup build dependencies to reduce image size
RUN apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (Railway sets PORT dynamically)
EXPOSE ${PORT:-8000}

# Run the application
CMD ["sh", "-c", "python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
