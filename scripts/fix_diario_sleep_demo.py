"""
Ajusta diario_estruturado.csv de demo:
- hora_sono_inicio == hora_sono_fim -> acrescenta 90 min no fim (duração zero vira soneca plausível);
- garante uma semana completa para id_aluno 1 (Rafael Souza) em torno de 2026-04-15.

Rodar na raiz: python scripts/fix_diario_sleep_demo.py
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "diario_estruturado.csv"


def norm_clock(s: object) -> str:
    return str(s or "").strip().replace(".", ":")


def bump_end_90(start_raw: object) -> str:
    s = norm_clock(start_raw)
    parts = s.split(":")
    if len(parts) < 2:
        return str(start_raw)
    try:
        h = int(parts[0].strip())
        m = int(parts[1].strip()[:2])
    except ValueError:
        return str(start_raw)
    total = (h * 60 + m + 90) % (24 * 60)
    return f"{total // 60:d}:{total % 60:02d}"


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Arquivo não encontrado: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    fixed = 0
    for i in df.index:
        a, b = norm_clock(df.at[i, "hora_sono_inicio"]), norm_clock(df.at[i, "hora_sono_fim"])
        if a and b and a == b:
            df.at[i, "hora_sono_fim"] = bump_end_90(a)
            fixed += 1
    print(f"Linhas com início=fim ajustadas (+90 min no fim): {fixed}")

    sub1 = df[df["id_aluno"] == 1]
    have = set(sub1["data"].astype(str))
    apr03 = sub1[sub1["data"].astype(str) == "2026-04-03"]
    if apr03.empty:
        print("Sem linha modelo 2026-04-03 para id 1; pulando inclusão de dias.")
    else:
        template = apr03.iloc[0].to_dict()
        new_rows: list[dict] = []
        nid = int(df["id_registro"].max()) + 1
        for ds in pd.date_range("2026-04-09", "2026-04-14"):
            s = ds.strftime("%Y-%m-%d")
            if s in have:
                continue
            r = {k: template[k] for k in template}
            r["id_registro"] = nid
            r["id_aluno"] = 1
            r["data"] = s
            r["hora_sono_inicio"] = "12:30"
            r["hora_sono_fim"] = "14:00"
            r["qualidade_sono"] = [
                "Sono leve",
                "Dormiu tranquilo",
                "Dormiu pouco",
                "Acordou agitado",
                "Sono leve",
                "Dormiu tranquilo",
            ][len(new_rows) % 6]
            meals = [
                ("Comeu bem", "Comeu bem", "Comeu pouco", "Comeu bem"),
                ("Comeu pouco", "Comeu pouco", "Comeu bem", "Recusou"),
                ("Comeu bem", "Comeu pouco", "Comeu pouco", "Comeu pouco"),
                ("Recusou", "Comeu bem", "Comeu bem", "Comeu pouco"),
                ("Comeu pouco", "Comeu bem", "Recusou", "Comeu bem"),
                ("Comeu bem", "Recusou", "Comeu bem", "Comeu pouco"),
            ][len(new_rows) % 6]
            r["cafe_manha"], r["almoco"], r["lanche_tarde"], r["jantar_extra"] = meals
            r["trocas_banheiro"] = 2 + (len(new_rows) % 3)
            r["evacuacao"] = "Normal" if len(new_rows) % 2 else "Pastoso"
            new_rows.append(r)
            nid += 1
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            print(f"Linhas adicionadas (id 1, semana 2026-04-09..14): {len(new_rows)}")

    try:
        df.to_csv(CSV_PATH, index=False, lineterminator="\n", encoding="utf-8-sig")
        print(f"Gravado {CSV_PATH} ({len(df)} linhas).")
    except PermissionError:
        alt = CSV_PATH.with_name("diario_estruturado_write_pending.csv")
        df.to_csv(alt, index=False, lineterminator="\n", encoding="utf-8-sig")
        print(
            f"Não foi possível sobrescrever {CSV_PATH} (arquivo em uso). "
            f"Gravado {alt} — pare o container ou feche o app e copie/renomeie para diario_estruturado.csv."
        )


if __name__ == "__main__":
    main()
