"""
Alinha `data/diario_estruturado.csv` ao modelo de sono do relatório:
- Cada linha: duração exatamente 10, 35 ou 60 min (conforme a categoria inferida do texto).
- `qualidade_sono` canônico: Dormiu pouco | Dormiu normal (20–40 min) | Dormiu bastante.

Executar na raiz do projeto: python scripts/fix_diario_sono_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "diario_estruturado.csv"

SONO_QUAL_BASTANTE = "Dormiu bastante"
SONO_QUAL_POUCO = "Dormiu pouco"
# Mesmo texto que `SONO_QUAL_PADRAO` em app.py (travessão U+2013).
SONO_QUAL_PADRAO = "Dormiu normal (20\u201340 min)"

TARGET_MIN = {
    SONO_QUAL_POUCO: 10,
    SONO_QUAL_PADRAO: 35,
    SONO_QUAL_BASTANTE: 60,
}

CAP = 60


def parse_clock(s) -> int | None:
    if pd.isna(s):
        return None
    t = str(s).strip()
    if not t:
        return None
    parts = t.replace("h", ":").split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0].strip())
        m = int(parts[1].strip()[:2])
        return max(0, min(23, h)) * 60 + max(0, min(59, m))
    except ValueError:
        return None


def sleep_minutes(a: int, b: int) -> int | None:
    d = (b - a) % (24 * 60)
    if d == 0:
        return None
    d_short = min(d, 24 * 60 - d)
    if d_short <= 0:
        return None
    if d_short > 240:
        return None
    return int(d_short)


def add_minutes(start_mins: int, add: int) -> str:
    total = (start_mins + add) % (24 * 60)
    h, m = divmod(total, 60)
    return f"{h:02d}:{m:02d}"


def classify_from_text_only(raw: str) -> str:
    t = (raw or "").strip().lower()
    if t in ("—", "-", "", "nan", "none"):
        t = ""

    exact = {
        SONO_QUAL_BASTANTE.lower(): SONO_QUAL_BASTANTE,
        SONO_QUAL_POUCO.lower(): SONO_QUAL_POUCO,
        SONO_QUAL_PADRAO.lower(): SONO_QUAL_PADRAO,
        "dormiu normal (20-40 min)": SONO_QUAL_PADRAO,
        "dormiu normal (20\u201340 min)": SONO_QUAL_PADRAO,
    }
    if t in exact:
        return exact[t]
    if t.startswith("sono no padrão") or t.startswith("sono no padrao"):
        return SONO_QUAL_PADRAO

    if any(
        k in t
        for k in (
            "acordou agit",
            "muito agit",
            "sono agit",
            "interrup",
            "chorou muito",
            "acordou cedo",
            "não dormiu",
            "nao dormiu",
            "quase não dorm",
        )
    ):
        return SONO_QUAL_POUCO
    if any(
        k in t
        for k in (
            "dormiu pouco",
            "pouco sono",
            "sono curto",
            "descanso curto",
            "dormiu mal",
        )
    ):
        return SONO_QUAL_POUCO
    if any(
        k in t
        for k in (
            "dormiu bastante",
            "bastante sono",
            "sono longo",
            "dormiu bem",
            "dormiu tranquilo",
            "sono tranquilo",
            "bom sono",
            "descanso bom",
            "dormiu o dia",
        )
    ):
        return SONO_QUAL_BASTANTE
    if any(
        k in t
        for k in (
            "dormiu normal",
            "no padrão",
            "no padrao",
            "padrão da",
            "padrao da",
            "média da semana",
            "media da semana",
            "sono normal",
            "sono moderado",
            "sono regular",
            "sono leve",
            "padrão (~60",
            "padrao (~60",
        )
    ):
        return SONO_QUAL_PADRAO
    return "—"


def classify_from_minutes(m: float) -> str:
    l1, l2, cap = 20.0, 40.0, 60.0
    x = max(0.0, min(float(m), cap))
    if x < l1:
        return SONO_QUAL_POUCO
    if x < l2:
        return SONO_QUAL_PADRAO
    return SONO_QUAL_BASTANTE


def main() -> int:
    if not CSV_PATH.is_file():
        print(f"Arquivo não encontrado: {CSV_PATH}", file=sys.stderr)
        return 1
    df = pd.read_csv(CSV_PATH)
    for col in ("hora_sono_inicio", "hora_sono_fim"):
        if col not in df.columns:
            print(f"Coluna ausente: {col}", file=sys.stderr)
            return 1

    for i, row in df.iterrows():
        a = parse_clock(row["hora_sono_inicio"])
        b = parse_clock(row["hora_sono_fim"])
        d_raw = sleep_minutes(a, b) if a is not None and b is not None else None

        cat = classify_from_text_only(str(row.get("qualidade_sono", "")))
        if cat == "—":
            if d_raw is not None and d_raw > 0:
                cat = classify_from_minutes(min(d_raw, CAP))
            else:
                continue

        tgt = TARGET_MIN[cat]
        if a is None:
            continue
        df.at[i, "hora_sono_fim"] = add_minutes(a, tgt)
        df.at[i, "qualidade_sono"] = cat

    df.to_csv(CSV_PATH, index=False)
    print(f"Atualizado: {CSV_PATH} ({len(df)} linhas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
