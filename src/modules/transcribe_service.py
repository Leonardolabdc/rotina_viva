"""
Transcrição de voz (STT) — facade escalável.

Backends via OPENAI_TRANSCRIBE_BASE_URL:
  - OpenRouter: https://openrouter.ai/api/v1  (JSON + base64, mesma OPENROUTER_API_KEY)
  - Whisper VM / Docker: http://host:9000/v1  (multipart OpenAI-compatível)
  - OpenAI API: https://api.openai.com/v1
"""

from __future__ import annotations

from modules import ai_engine


def transcribe_base_url() -> str:
    return (ai_engine.OPENAI_TRANSCRIBE_BASE_URL or "").strip().rstrip("/")


def is_transcribe_configured() -> bool:
    return bool(transcribe_base_url())


def transcribe_backend_label() -> str:
    base = transcribe_base_url().lower()
    if not base:
        return "não configurado"
    if "openrouter.ai" in base:
        return "OpenRouter Whisper STT"
    if "api.openai.com" in base:
        return "OpenAI Whisper API"
    if "127.0.0.1" in base or "localhost" in base:
        return "Whisper local"
    return "Whisper (serviço externo)"


def transcribe_setup_hint() -> str:
    if is_transcribe_configured():
        return f"Voz: **{transcribe_backend_label()}**."
    return (
        "Voz **desactivada** — defina `OPENAI_TRANSCRIBE_BASE_URL=https://openrouter.ai/api/v1` "
        "(Cloud) ou Whisper local. Ver `docs/DEPLOY_WHISPER.md`."
    )


def transcribe_voice_bytes(audio_bytes: bytes, filename: str) -> tuple[str | None, str | None]:
    """Delega para o motor; ponto único para trocar por HTTP/gRPC na Fase 1."""
    return ai_engine.transcribe_voice_bytes(audio_bytes, filename)
