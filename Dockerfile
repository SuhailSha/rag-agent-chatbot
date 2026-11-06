FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose standard port (7860 for Hugging Face Spaces, dynamically overridden by Cloud Run / Render)
EXPOSE 7860

# Start command supporting both fixed port (HF Spaces) and $PORT environment variable (Cloud Run / Render)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
