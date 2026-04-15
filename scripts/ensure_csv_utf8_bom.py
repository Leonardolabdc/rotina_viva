"""
Regrava CSVs de dados em UTF-8 com BOM para o Excel (Windows) abrir acentos corretamente.

Também tenta corrigir texto já "partido" (UTF-8 lido como Latin-1), ex.: Ã§ Ã£ → ç ã.

Uso na raiz do projeto:
  python scripts/ensure_csv_utf8_bom.py
  python scripts/ensure_csv_utf8_bom.py data/diario_estruturado.csv
"""
from __future__ import annotations

import csv
import io
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _try_fix_mojibake(text: str) -> str:
    if "Ã" not in text and "Â" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def _clean_cell(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    return s


def process_file(path: Path, dest: Path | None = None) -> None:
    raw = path.read_bytes()
    enc = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    text = raw.decode(enc)
    text = _try_fix_mojibake(text)

    buf = io.StringIO(text)
    rows = list(csv.reader(buf))

    if not rows:
        raise ValueError("CSV vazio")
    ncols = len(rows[0])
    out_rows: list[list[str]] = []
    for i, row in enumerate(rows):
        if len(row) != ncols:
            raise ValueError(
                f"Linha {i}: {len(row)} colunas, esperado {ncols}"
            )
        out_rows.append([_clean_cell(c) for c in row])

    target = dest if dest is not None else path
    if dest is not None:
        with target.open("w", encoding="utf-8-sig", newline="") as f_out:
            csv.writer(f_out, lineterminator="\n").writerows(out_rows)
        print(f"OK: {path} -> {target} ({len(out_rows)} linhas, UTF-8 com BOM)")
        return

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f_out:
        csv.writer(f_out, lineterminator="\n").writerows(out_rows)
    tmp.replace(path)
    print(f"OK: {path} ({len(out_rows)} linhas, UTF-8 com BOM)")


def main() -> None:
    argv = sys.argv[1:]
    dest: Path | None = None
    if "--out" in argv:
        i = argv.index("--out")
        dest = Path(argv[i + 1]).resolve()
        argv = argv[:i] + argv[i + 2 :]

    paths = [Path(p) for p in argv] if argv else [
        ROOT / "data" / "diario_estruturado.csv",
        ROOT / "data" / "info_alunos.csv",
    ]
    if len(paths) != 1 and dest is not None:
        raise SystemExit("Use --out só com um ficheiro de entrada.")
    for p in paths:
        p = p.resolve()
        if not p.is_file():
            print(f"Pulando (não existe): {p}")
            continue
        process_file(p, dest=dest)


if __name__ == "__main__":
    main()
