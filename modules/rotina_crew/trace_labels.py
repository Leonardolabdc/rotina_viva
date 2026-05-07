"""Rótulos curtos para logs e Langfuse (saídas `TaskOutput` da CrewAI)."""

from __future__ import annotations

from typing import Any

CREW_TASK_LOG_LABELS: dict[str, str] = {
    "recepcao": "Receção",
    "analista_dados": "Dados",
    "especialista_ml": "Emoções",
    "especialista_rag": "Agente RAG",
    "redatora_final": "Redação",
}


def trace_agent_label_for_task_output(item: Any) -> str:
    """CrewAI ≥0.86: `TaskOutput.agent` é str (role); o nome está em `TaskOutput.name`."""
    tn = getattr(item, "name", None)
    if isinstance(tn, str) and tn in CREW_TASK_LOG_LABELS:
        return CREW_TASK_LOG_LABELS[tn]
    if isinstance(tn, str):
        tl = tn.strip().lower()
        for key, label in CREW_TASK_LOG_LABELS.items():
            if tl.startswith(key.lower()):
                return label
    agent_val = getattr(item, "agent", None)
    if isinstance(agent_val, str) and agent_val.strip():
        return agent_val.strip()
    agent_obj = agent_val
    if agent_obj is not None and hasattr(agent_obj, "role"):
        return str(getattr(agent_obj, "role", "") or "agente")
    return "agente"
