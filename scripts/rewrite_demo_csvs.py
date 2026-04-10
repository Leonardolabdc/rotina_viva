"""
Reescreve info_alunos.csv e diario_estruturado.csv do demo:
- nomes completos únicos (mesmo id_aluno / turma / alergias preservados por linha);
- telefones únicos por aluno;
- remove registros duplicados no diário (mesmo id_aluno + mesma data), mantendo o primeiro.
Execute a partir da raiz do projeto: python scripts/rewrite_demo_csvs.py
"""
from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INFO = DATA / "info_alunos.csv"
DIARIO = DATA / "diario_estruturado.csv"

FIRST = [
    "Ana",
    "Beatriz",
    "Bruno",
    "Carlos",
    "Daniela",
    "Eduardo",
    "Fernanda",
    "Gabriel",
    "Helena",
    "Igor",
    "Julia",
    "Karina",
    "Lucas",
    "Mariana",
    "Nicolas",
    "Otávio",
    "Patrícia",
    "Quésia",
    "Rafael",
    "Sabrina",
    "Thiago",
    "Ulisses",
    "Vitória",
    "William",
    "Yasmin",
]
LAST = [
    "Almeida",
    "Alves",
    "Araújo",
    "Barbosa",
    "Cardoso",
    "Carvalho",
    "Castro",
    "Correia",
    "Costa",
    "Cruz",
    "Dias",
    "Duarte",
    "Fernandes",
    "Freitas",
    "Gomes",
    "Lima",
    "Martins",
    "Mendes",
    "Monteiro",
    "Moreira",
    "Moura",
    "Nascimento",
    "Nogueira",
    "Oliveira",
    "Pereira",
    "Ramos",
    "Reis",
    "Ribeiro",
    "Rodrigues",
    "Santos",
    "Silva",
    "Souza",
    "Teixeira",
    "Vieira",
]


def unique_names(n: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for f, l in itertools.product(FIRST, LAST):
        s = f"{f} {l}"
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"só geramos {len(out)} nomes únicos, precisamos {n}")
    return out


def main() -> None:
    if not INFO.exists() or not DIARIO.exists():
        raise SystemExit(f"Esperado {INFO} e {DIARIO}")

    with INFO.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        info_rows = list(reader)
    n = len(info_rows)
    pool = [x for x in unique_names(n + 30) if x != "Rafael Souza"]
    random.seed(42)
    random.shuffle(pool)
    it = iter(pool)
    info_rows.sort(key=lambda r: int(r["id_aluno"]))
    for row in info_rows:
        iid = int(row["id_aluno"])
        row["nome"] = "Rafael Souza" if iid == 1 else next(it)
        row["contato_pais"] = str(11999900000 + iid)

    with INFO.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id_aluno", "nome", "turma", "alergias", "contato_pais"],
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(info_rows)

    with DIARIO.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        drows = list(reader)

    seen_key: set[tuple[str, str]] = set()
    kept: list[dict[str, str]] = []
    for r in drows:
        k = (r["id_aluno"], r["data"])
        if k in seen_key:
            continue
        seen_key.add(k)
        kept.append(r)

    fieldnames = list(reader.fieldnames or [])
    for i, r in enumerate(kept, start=1):
        r["id_registro"] = str(i)

    with DIARIO.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(kept)

    print(f"info_alunos: {n} linhas, nomes e telefones únicos.")
    print(f"diario_estruturado: {len(drows)} -> {len(kept)} linhas (duplicatas id+data removidas).")


if __name__ == "__main__":
    main()
