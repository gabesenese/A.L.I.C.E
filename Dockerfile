# A.L.I.C.E - Production Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models

# Create non-root user
RUN useradd -m -u 1000 alice && \
    chown -R alice:alice /app

USER alice

# Health check the CLI process. ALICE does not expose an HTTP server by default.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from pathlib import Path; cmd = Path('/proc/1/cmdline').read_bytes(); raise SystemExit(0 if (b'app/main.py' in cmd or b'app.main' in cmd) else 1)"

# Expose ports
# 8000: Main API
# 9090: Prometheus metrics
EXPOSE 8000 9090

# Run application
CMD ["python", "app/main.py"]
