FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps for Playwright
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2 libxshmfence1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    playwright install chromium

# Run as non-root user for security
RUN useradd -m -s /bin/bash appuser

COPY src/ src/
COPY agent/ agent/
COPY config.yaml .
COPY pyproject.toml .

RUN mkdir -p data && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# Default: run scan
CMD ["python", "src/main_intel.py", "--days", "7"]
