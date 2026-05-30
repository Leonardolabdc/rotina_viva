"""
Arranque para Streamlit Community Cloud: secrets → os.environ, utilizadores demo, defaults CrewAI.
Deve correr antes de importar `modules.ai_engine` e restantes módulos que leem env no import.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def _is_truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _apply_secret_value(key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        os.environ.setdefault(key, "true" if value else "false")
        return
    if isinstance(value, (int, float)):
        os.environ.setdefault(key, str(value))
        return
    if isinstance(value, str) and value.strip():
        os.environ.setdefault(key, value.strip())


def _walk_streamlit_secrets(prefix: str, node: Any) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            _walk_streamlit_secrets(key, v)
        return
    env_key = prefix.upper()
    _apply_secret_value(env_key, node)


def apply_streamlit_secrets_to_environ() -> None:
    """Copia st.secrets para os.environ (Cloud não tem ficheiro .env)."""
    try:
        import streamlit as st
    except ImportError:
        return
    try:
        secrets = st.secrets
    except Exception:
        return
    try:
        items = secrets.items()
    except Exception:
        return
    for key, value in items:
        if isinstance(value, dict):
            _walk_streamlit_secrets(str(key), value)
        else:
            _apply_secret_value(str(key).upper(), value)


def ensure_rotina_users_file(data_dir: Path) -> None:
    """Garante login demo se rotina_users.json não existir (Cloud, disco efémero)."""
    users_path = data_dir / "rotina_users.json"
    if users_path.is_file():
        return
    example = data_dir / "rotina_users.example.json"
    if example.is_file():
        try:
            shutil.copyfile(example, users_path)
            return
        except OSError:
            pass
    # Fallback mínimo embutido (mesmos perfis do example)
    demo = {
        "gestao.demo": {
            "password": "demo123",
            "role": "gestao",
            "display_name": "Gestão (edição CSV)",
        },
        "professor.demo": {
            "password": "demo123",
            "role": "educador",
            "display_name": "Prof. Demonstração (só leitura nos dados)",
        },
        "pai.demo": {
            "password": "demo123",
            "role": "familia",
            "display_name": "Responsável (Rafael Souza)",
            "id_aluno": 1,
        },
    }
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        users_path.write_text(
            json.dumps(demo, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def apply_transcribe_env() -> None:
    """
    STT externo (escalável): alias ROTINA_TRANSCRIBE_* → OPENAI_TRANSCRIBE_*.
    Remove URLs Docker internas inválidas fora do contentor.
    """
    svc = (os.getenv("ROTINA_TRANSCRIBE_SERVICE_URL") or "").strip().rstrip("/")
    if svc and not os.getenv("OPENAI_TRANSCRIBE_BASE_URL", "").strip():
        os.environ["OPENAI_TRANSCRIBE_BASE_URL"] = svc

    tkey = (os.getenv("ROTINA_TRANSCRIBE_API_KEY") or "").strip()
    if tkey and not os.getenv("OPENAI_TRANSCRIBE_API_KEY", "").strip():
        os.environ["OPENAI_TRANSCRIBE_API_KEY"] = tkey

    transcribe = os.getenv("OPENAI_TRANSCRIBE_BASE_URL", "").strip().lower()
    if "whisper:" in transcribe or transcribe.endswith("//whisper:9000/v1"):
        os.environ["OPENAI_TRANSCRIBE_BASE_URL"] = svc or ""


def apply_cloud_runtime_defaults() -> None:
    """Defaults seguros para Community Cloud com agentes activos."""
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("ROTINA_ENABLE_CREWAI", "true")
    # FLAML/torch estoura RAM no tier free; agentes CrewAI são o foco do deploy.
    os.environ.setdefault("ROTINA_ENABLE_ML_LAB", "false")
    os.environ.setdefault("ROTINA_LANGFUSE_ENABLED", "false")
    # F5 no Community Cloud: token de login na URL + ficheiro do chat (ver auth_manager).
    os.environ.setdefault("ROTINA_SESSION_IN_URL", "true")

    apply_transcribe_env()

    # OpenRouter por defeito se houver chave mas provider em ollama
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        if not os.getenv("ROTINA_CHAT_PROVIDER", "").strip():
            os.environ.setdefault("ROTINA_CHAT_PROVIDER", "openrouter")
        if not os.getenv("ROTINA_EMBED_PROVIDER", "").strip():
            os.environ.setdefault("ROTINA_EMBED_PROVIDER", "openrouter")
        os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        # STT OpenRouter (JSON base64) — mesma chave; evita VM Whisper na Fase 0
        if not os.getenv("OPENAI_TRANSCRIBE_BASE_URL", "").strip():
            os.environ.setdefault("OPENAI_TRANSCRIBE_BASE_URL", "https://openrouter.ai/api/v1")
        os.environ.setdefault("OPENAI_TRANSCRIBE_MODEL", "openai/whisper-1")

    if _is_truthy(os.getenv("ROTINA_LANGFUSE_ENABLED")):
        os.environ.setdefault("LITELLM_SUCCESS_CALLBACKS", "langfuse")
        os.environ.setdefault("LITELLM_FAILURE_CALLBACKS", "langfuse")


def apply_cloud_bootstrap() -> None:
    apply_streamlit_secrets_to_environ()
    apply_cloud_runtime_defaults()
    data_dir = Path(os.getenv("ROTINA_DATA_DIR", "data")).resolve()
    ensure_rotina_users_file(data_dir)
