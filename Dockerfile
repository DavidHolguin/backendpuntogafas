FROM python:3.11-slim

WORKDIR /opt/worker

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ app/

# Health check: verify process is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD pgrep -f "python -m app.worker" || exit 1

# Run
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "app.run"]
