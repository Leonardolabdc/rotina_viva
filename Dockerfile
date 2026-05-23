# syntax=docker/dockerfile:1
# Streamlit + DuckDB + ChromaDB + RAG (langchain-text-splitters) — PyTorch CPU (menos GB que pacotes NVIDIA no C:)
FROM python:3.12-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    ROTINA_DATA_DIR=/data \
    CHROMA_PERSIST_DIR=/data/vector_db \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Cache de pip no BuildKit (fica no “data root” do Docker — mover Docker Desktop para D: reduz uso em C:)
# Torch CPU primeiro evita baixar ~1GB+ de wheels CUDA no build.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4,<3" \
    && pip install -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none"]
