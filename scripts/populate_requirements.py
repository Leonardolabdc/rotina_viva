#!/usr/bin/env python3
"""
Instala as dependências do Rotina Viva e grava `requirements.txt` com `pip freeze` completo.

Use em um ambiente virtual limpo na raiz do projeto:
  python scripts/populate_requirements.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "requirements.txt"

PACKAGES = [
    "streamlit>=1.32,<2",
    "python-dotenv>=1.0,<2",
    "duckdb>=1.0,<2",
    "chromadb>=0.5.5,<2",
    "ollama>=0.3,<1",
    "txtai>=6.0,<10",
    "pypdf>=4.0,<6",
    "httpx>=0.27,<1",
]


def main() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", *PACKAGES])
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    OUT.write_text(frozen, encoding="utf-8")
    print(f"Escrito {OUT} ({len(frozen.splitlines())} linhas).")


if __name__ == "__main__":
    main()
