FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
EXPOSE ${PORT:-8000}
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}
