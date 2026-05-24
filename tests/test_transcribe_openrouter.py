"""Testes STT OpenRouter (helpers, sem chamada API)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modules.ai_engine import (  # noqa: E402
    _is_openrouter_transcribe_base,
    _openrouter_audio_format,
    _openrouter_stt_model_name,
    whisper_upload_name_and_mime,
)


def test_openrouter_base_detection() -> None:
    assert _is_openrouter_transcribe_base("https://openrouter.ai/api/v1")
    assert not _is_openrouter_transcribe_base("http://127.0.0.1:9000/v1")
    print("OK test_openrouter_base_detection")


def test_openrouter_model_prefix() -> None:
    import os

    os.environ["OPENAI_TRANSCRIBE_MODEL"] = "whisper-1"
    # Re-read would need reimport; test function directly with known logic
    assert _openrouter_stt_model_name().startswith("openai/")
    print("OK test_openrouter_model_prefix")


def test_openrouter_webm_format() -> None:
    _, mime = whisper_upload_name_and_mime(b"\x1a\x45\xdf\xa3", "x.webm")
    assert _openrouter_audio_format(mime, "x.webm") == "webm"
    print("OK test_openrouter_webm_format")


if __name__ == "__main__":
    test_openrouter_base_detection()
    test_openrouter_model_prefix()
    test_openrouter_webm_format()
    print("Todos os testes STT OpenRouter passaram.")
