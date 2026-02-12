FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install CPU-only torch to save ~700MB and speed up build
# Explicitly require v2.6.0+ to fix CVE-2025-32434 security requirement
# Pre-install typing-extensions to resolve metadata name mismatch in the PyTorch index
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "typing-extensions>=4.10.0" && \
    pip install --no-cache-dir "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download CodeBERT
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('microsoft/codebert-base'); AutoModel.from_pretrained('microsoft/codebert-base')"

COPY . .
CMD ["python", "app.py"]
