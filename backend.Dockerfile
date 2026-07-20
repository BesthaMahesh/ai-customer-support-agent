FROM python:3.11-slim

# System libraries for FAISS compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir tf-keras

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000}"]
