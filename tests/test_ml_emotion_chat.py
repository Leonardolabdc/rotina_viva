"""Testes das heurísticas de emoção ML (sem carregar .pkl)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (_SRC, _ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modules.ml_emotion_chat import (
    apply_pedagogical_emotion_overrides,
    expand_emotion_lines,
    predictive_message_looks_emotional,
)


def test_expand_splits_logo_and_comma() -> None:
    text = "O aluno machucou o pé e chorou, logo não voltou a brincar"
    parts = expand_emotion_lines([text])
    assert len(parts) >= 2
    assert any("chorou" in p for p in parts)
    assert any("brincar" in p for p in parts)
    print("OK test_expand_splits_logo_and_comma")


def test_override_joy_to_sadness_injury() -> None:
    user = ["O aluno machucou o pé e chorou, logo não voltou a brincar"]
    names, notes = apply_pedagogical_emotion_overrides(user, ["joy"])
    assert names[0] == "sadness"
    assert notes[0]
    print("OK test_override_joy_to_sadness_injury")


def test_predictive_detects_machucou() -> None:
    assert predictive_message_looks_emotional(
        "O aluno machucou o pé e chorou, logo não voltou a brincar"
    )
    print("OK test_predictive_detects_machucou")


def main() -> None:
    test_expand_splits_logo_and_comma()
    test_override_joy_to_sadness_injury()
    test_predictive_detects_machucou()
    print("\nTestes ML emoção OK.")


if __name__ == "__main__":
    main()
