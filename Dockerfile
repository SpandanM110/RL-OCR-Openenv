FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        fontconfig \
        fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "app.py"]
