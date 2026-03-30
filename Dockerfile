###############################
# ✅ Stage 1 — Build environment
###############################
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies for numpy, scipy, tensorflow, faiss, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python3-dev \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install dependencies to a target folder
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


###############################
# ✅ Stage 2 — Runtime container
###############################
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_PORT=${APP_PORT:-5000} \
    API_ENV=${API_ENV:-production}

# Copy Python dependencies
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY . .

# ⚠️ DO NOT train model inside the image (best practice)
# Model should be trained in CI/CD or manually, then mounted or baked into image.
# If you really must train inside image, uncomment:
# RUN python pipeline/training_pipeline.py

EXPOSE $APP_PORT

CMD ["python", "application.py"]