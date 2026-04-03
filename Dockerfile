FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY api ./api
COPY config ./config
COPY pipeline ./pipeline
COPY utils ./utils
COPY src ./src
COPY core ./core
COPY database ./database

# Explicitly copy artifacts with verbose output for debugging
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]