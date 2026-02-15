FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (CPU only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download CodeBERT model to bake it into a cached layer
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('microsoft/codebert-base'); AutoModel.from_pretrained('microsoft/codebert-base')"

# Copy the rest of the application
COPY . .

EXPOSE 8080
# Use the PORT variable provided by Railway, defaulting to 8080
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]