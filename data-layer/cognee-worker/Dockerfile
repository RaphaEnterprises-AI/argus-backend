# Cognee Kafka Worker
# Event-driven knowledge graph builder for Argus
#
# Build: docker build -t ghcr.io/heyargus/cognee-worker:latest .
# Run:   docker run -e KAFKA_BOOTSTRAP_SERVERS=... ghcr.io/heyargus/cognee-worker:latest

FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Runtime stage
FROM python:3.12-slim AS runtime

# Create non-root user
RUN groupadd -r cognee && useradd -r -g cognee cognee

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./

# Set ownership
RUN chown -R cognee:cognee /app

# Switch to non-root user
USER cognee

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    HEALTH_CHECK_PORT=8080

# Expose health check port
EXPOSE 8080

# Run the worker
CMD ["python", "worker.py"]
