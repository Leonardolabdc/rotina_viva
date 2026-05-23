"""
Coloca cache/telemetria do DeepEval em `tests/.deepeval` em vez da raiz do repo.

Deve ser importado/corrido antes de qualquer `import deepeval` (o `deepeval/__init__.py`
carrega telemetria com `HIDDEN_DIR` fixo na importação).
"""

from __future__ import annotations

import os
import site
import sys
import types
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
DEEPEVAL_HOME = _TESTS_DIR / ".deepeval"


def configure_deepeval_home() -> Path:
    DEEPEVAL_HOME.mkdir(parents=True, exist_ok=True)
    home = str(DEEPEVAL_HOME.resolve())
    os.environ.setdefault("DEEPEVAL_CACHE_FOLDER", home)

    if "deepeval.constants" in sys.modules:
        sys.modules["deepeval.constants"].HIDDEN_DIR = home  # type: ignore[attr-defined]
        return DEEPEVAL_HOME

    deepeval_path: Path | None = None
    for base in site.getsitepackages():
        cand = Path(base) / "deepeval"
        if cand.is_dir():
            deepeval_path = cand
            break
    if deepeval_path is None:
        return DEEPEVAL_HOME

    if "deepeval" not in sys.modules:
        pkg = types.ModuleType("deepeval")
        pkg.__path__ = [str(deepeval_path)]
        sys.modules["deepeval"] = pkg

    constants = types.ModuleType("deepeval.constants")
    constants.HIDDEN_DIR = home
    constants.KEY_FILE = ".deepeval"
    constants.LOGIN_PROMPT = ""
    constants.PYTEST_RUN_TEST_NAME = "CONFIDENT_AI_RUN_TEST_NAME"
    sys.modules["deepeval.constants"] = constants
    return DEEPEVAL_HOME
