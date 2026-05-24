"""Flags de funcionalidades opcionais (produção vs desenvolvimento)."""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


# Produção: defina false para imagem mais leve (sem CrewAI / laboratório FLAML).
ROTINA_ENABLE_CREWAI = _env_bool("ROTINA_ENABLE_CREWAI", default=True)
ROTINA_ENABLE_ML_LAB = _env_bool("ROTINA_ENABLE_ML_LAB", default=True)
