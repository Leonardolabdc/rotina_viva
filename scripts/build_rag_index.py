#!/usr/bin/env python3
"""
Pré-constrói o índice Chroma (RAG) antes do deploy no Streamlit Cloud.

Uso (na raiz, com .env / OpenRouter configurado):
  python scripts/build_rag_index.py

Depois faça commit de data/vector_db/ para evitar indexação longa no 1.º login:
  git add -f data/vector_db/
  git commit -m "chore: índice RAG para Streamlit Cloud"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from core.cloud_bootstrap import apply_cloud_bootstrap

apply_cloud_bootstrap()

from core.database import DATA_DIR
from modules.rag_index import CHROMA_DIR, INDEX_PROFILE, get_chroma_collection, rag_will_run_full_document_ingest


def main() -> None:
    print(f"DATA_DIR={DATA_DIR}")
    print(f"CHROMA_DIR={CHROMA_DIR}")
    print(f"INDEX_PROFILE={INDEX_PROFILE}")
    if not rag_will_run_full_document_ingest(CHROMA_DIR, DATA_DIR):
        col = get_chroma_collection(str(CHROMA_DIR), str(DATA_DIR), INDEX_PROFILE)
        print(f"Índice já actualizado — {col.count()} chunks.")
        return
    print("A indexar PDFs (pode demorar vários minutos e consumir tokens OpenRouter)...")
    col = get_chroma_collection(str(CHROMA_DIR), str(DATA_DIR), INDEX_PROFILE)
    print(f"Concluído — {col.count()} chunks em {CHROMA_DIR}")


if __name__ == "__main__":
    main()
