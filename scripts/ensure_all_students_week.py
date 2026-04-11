"""
Garante 7 dias corridos de diário (com sono válido) para cada aluno em info_alunos.csv,
na semana que termina na data mais recente do arquivo diario_estruturado.csv.

Uso (na raiz do repo): python scripts/ensure_all_students_week.py
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INFO_CSV = ROOT / "data" / "info_alunos.csv"
DIARIO_CSV = ROOT / "data" / "diario_estruturado.csv"

ATIVIDADES = [
    "Brincadeiras livres",
    "Coordenação motora",
    "Contação de histórias",
    "Musicalização",
    "Pintura",
]
INTERACOES = [
    "Preferiu brincar sozinho",
    "Brincou com os colegas",
    "Interagiu bem com colegas",
]
RECADOS = [
    "Participou bem das atividades.",
    "Muito ativo durante o dia.",
    "Apresentou leve sonolência.",
    "Mostrou-se mais quieto que o normal.",
]
SONO_SLOTS = [
    ("12:30", "14:00"),
    ("13:00", "15:00"),
    ("12:00", "13:30"),
    ("13:30", "15:00"),
    ("12:15", "14:15"),
    ("14:00", "15:30"),
    ("12:45", "14:30"),
]


def _parse_clock_to_minutes(s: object) -> int | None:
    raw = str(s or "").strip()
    if not raw:
        return None
    raw = raw.replace(".", ":")
    parts = raw.split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0].strip())
        m = int(parts[1].strip()[:2])
        return max(0, min(23, h)) * 60 + max(0, min(59, m))
    except ValueError:
        return None


def sleep_minutes(start: object, end: object) -> int | None:
    a = _parse_clock_to_minutes(start)
    b = _parse_clock_to_minutes(end)
    if a is None or b is None:
        return None
    d = b - a
    if d < 0:
        d += 24 * 60
    if d <= 0:
        return None
    return d


def meal_combo(n: int) -> tuple[str, str, str, str]:
    opts = [
        ("Comeu bem", "Comeu bem", "Comeu pouco", "Comeu bem"),
        ("Comeu pouco", "Comeu pouco", "Comeu bem", "Recusou"),
        ("Comeu bem", "Comeu pouco", "Comeu pouco", "Comeu pouco"),
        ("Recusou", "Comeu bem", "Comeu bem", "Comeu pouco"),
        ("Comeu pouco", "Comeu bem", "Recusou", "Comeu bem"),
        ("Comeu bem", "Recusou", "Comeu bem", "Comeu pouco"),
        ("Comeu bem", "Comeu bem", "Comeu bem", "Recusou"),
    ]
    return opts[n % len(opts)]


def synth_row(id_aluno: int, date_str: str, seq: int) -> dict:
    c1, c2, c3, c4 = meal_combo(seq)
    hi, hf = SONO_SLOTS[seq % len(SONO_SLOTS)]
    qual = ["Sono leve", "Dormiu tranquilo", "Dormiu pouco", "Acordou agitado"][
        seq % 4
    ]
    return {
        "id_aluno": id_aluno,
        "data": date_str,
        "cafe_manha": c1,
        "almoco": c2,
        "lanche_tarde": c3,
        "jantar_extra": c4,
        "trocas_banheiro": 2 + (seq % 3),
        "evacuacao": "Normal" if seq % 2 else "Pastoso",
        "medicamentos": "Nenhum" if seq % 3 else "Vitamina 10ml 09:00",
        "hora_sono_inicio": hi,
        "hora_sono_fim": hf,
        "qualidade_sono": qual,
        "atividade_dia": ATIVIDADES[seq % len(ATIVIDADES)],
        "interacao_social": INTERACOES[seq % len(INTERACOES)],
        "recado_professora": RECADOS[seq % len(RECADOS)],
    }


def main() -> None:
    if not INFO_CSV.exists() or not DIARIO_CSV.exists():
        raise SystemExit(f"Esperado {INFO_CSV} e {DIARIO_CSV}")

    info = pd.read_csv(INFO_CSV)
    df = pd.read_csv(DIARIO_CSV)
    all_ids = sorted(int(x) for x in info["id_aluno"].unique())

    df["data_dt"] = pd.to_datetime(df["data"], errors="coerce")
    global_end = df["data_dt"].max()
    if pd.isna(global_end):
        raise SystemExit("Nenhuma data válida no diário.")
    global_end = global_end.normalize()
    norm_dates = [global_end - pd.Timedelta(days=6 - i) for i in range(7)]
    norm_set = set(norm_dates)

    df["_nd"] = df["data_dt"].dt.normalize()
    week_mask = df["id_aluno"].isin(all_ids) & df["_nd"].isin(norm_set)
    df_out = df[~week_mask].drop(columns=["data_dt", "_nd"], errors="ignore")

    week_chunk: list[dict] = []
    for aid in all_ids:
        for seq, nd in enumerate(norm_dates):
            ds = nd.strftime("%Y-%m-%d")
            sub = df[(df["id_aluno"] == aid) & (df["_nd"] == nd)]
            chosen: dict | None = None
            for _, r in sub.iterrows():
                sm = sleep_minutes(r.get("hora_sono_inicio"), r.get("hora_sono_fim"))
                if sm:
                    chosen = {
                        "id_aluno": int(r["id_aluno"]),
                        "data": ds,
                        "cafe_manha": r["cafe_manha"],
                        "almoco": r["almoco"],
                        "lanche_tarde": r["lanche_tarde"],
                        "jantar_extra": r["jantar_extra"],
                        "trocas_banheiro": int(r["trocas_banheiro"])
                        if pd.notna(r["trocas_banheiro"])
                        else 2,
                        "evacuacao": r["evacuacao"],
                        "medicamentos": r["medicamentos"],
                        "hora_sono_inicio": r["hora_sono_inicio"],
                        "hora_sono_fim": r["hora_sono_fim"],
                        "qualidade_sono": r["qualidade_sono"],
                        "atividade_dia": r["atividade_dia"],
                        "interacao_social": r["interacao_social"],
                        "recado_professora": r["recado_professora"],
                    }
                    break
            if chosen is None:
                chosen = synth_row(aid, ds, seq + aid * 13)
            week_chunk.append(chosen)

    df_week = pd.DataFrame(week_chunk)
    out = pd.concat([df_out, df_week], ignore_index=True)
    out = out.sort_values(by=["id_aluno", "data"], kind="mergesort")
    out["id_registro"] = range(1, len(out) + 1)

    cols = [
        "id_registro",
        "id_aluno",
        "data",
        "cafe_manha",
        "almoco",
        "lanche_tarde",
        "jantar_extra",
        "trocas_banheiro",
        "evacuacao",
        "medicamentos",
        "hora_sono_inicio",
        "hora_sono_fim",
        "qualidade_sono",
        "atividade_dia",
        "interacao_social",
        "recado_professora",
    ]
    out = out[cols]
    out.to_csv(DIARIO_CSV, index=False, lineterminator="\n", encoding="utf-8-sig")
    print(
        f"Semana global: {norm_dates[0].date()} a {norm_dates[-1].date()} "
        f"({len(all_ids)} alunos × 7 dias = {len(df_week)} linhas na semana)."
    )
    print(f"Total de linhas no diário: {len(out)}. Gravado {DIARIO_CSV}")


if __name__ == "__main__":
    main()
