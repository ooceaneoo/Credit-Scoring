FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir --default-timeout=300 -r requirements-api.txt

COPY api/ api/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]