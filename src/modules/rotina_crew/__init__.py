"""Orquestração multi-agente (CrewAI) opcional para o chat Rotina Viva."""

from __future__ import annotations

from modules.rotina_crew.runner import CrewChatResult, crewai_import_ok, run_rotina_crew_chat

__all__ = ["CrewChatResult", "crewai_import_ok", "run_rotina_crew_chat"]
