FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch to save ~700MB and speed up build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download CodeBERT
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('microsoft/codebert-base'); AutoModel.from_pretrained('microsoft/codebert-base')"

COPY . .
CMD ["python", "app.py"]
