FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download CodeBERT
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('microsoft/codebert-base'); AutoModel.from_pretrained('microsoft/codebert-base')"

COPY . .

CMD ["python", "app.py"]