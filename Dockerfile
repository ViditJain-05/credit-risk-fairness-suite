FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by XGBoost / light scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

COPY data/processed/ ./data/processed/
COPY artifacts/ ./artifacts/
COPY app/ ./app/
COPY api/ ./api/
COPY MODEL_CARD.md README.md ./

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
