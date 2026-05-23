"""
Validação SQL com AST (sqlparse) + fallback regex em `core.database`.
"""

from __future__ import annotations

import re
from typing import FrozenSet

_ALLOWED_TABLES = frozenset({"info_alunos", "diario_estruturado"})

_SELECT_FORBIDDEN = frozenset(
    {
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "attach",
        "detach",
        "pragma",
        "copy",
        "call",
        "truncate",
        "replace",
    }
)

_MUTATION_FORBIDDEN = frozenset(
    {
        "drop",
        "alter",
        "create",
        "attach",
        "detach",
        "pragma",
        "copy",
        "call",
        "truncate",
    }
)


def _normalize_table_name(raw: str) -> str:
    t = (raw or "").strip().lower().strip('`"[]')
    if "." in t:
        t = t.rsplit(".", 1)[-1]
    return t


def _statement_strings(stmt) -> list[str]:  # noqa: ANN001
    out: list[str] = []
    for tok in stmt.flatten():
        if tok.is_whitespace or tok.is_group:
            continue
        v = (tok.value or "").strip().lower()
        if v:
            out.append(v)
    return out


def _extract_table_names(stmt) -> set[str]:  # noqa: ANN001
    """Tabelas em FROM / JOIN / UPDATE / INSERT INTO / DELETE FROM."""
    import sqlparse
    from sqlparse.sql import Identifier, IdentifierList

    tables: set[str] = set()
    from_seen = False
    for tok in stmt.tokens:
        if tok.is_whitespace:
            continue
        val = (tok.value or "").lower()
        if val in ("from", "join", "into", "update"):
            from_seen = val != "update" or True
            if val == "update":
                from_seen = "update_target"
            continue
        if from_seen == "update_target" and isinstance(tok, Identifier):
            tables.add(_normalize_table_name(tok.get_real_name() or tok.get_name() or ""))
            from_seen = False
            continue
        if from_seen and isinstance(tok, (Identifier, IdentifierList)):
            ids = tok.get_identifiers() if isinstance(tok, IdentifierList) else [tok]
            for ident in ids:
                if isinstance(ident, Identifier):
                    tables.add(
                        _normalize_table_name(
                            ident.get_real_name() or ident.get_name() or ""
                        )
                    )
            from_seen = False
    # fallback: regex na string inteira
    raw = str(stmt).lower()
    for m in re.finditer(
        r"\b(?:from|join|into|update)\s+([`\"\w.]+)", raw, re.IGNORECASE
    ):
        tables.add(_normalize_table_name(m.group(1)))
    for m in re.finditer(r"\bdelete\s+from\s+([`\"\w.]+)", raw, re.IGNORECASE):
        tables.add(_normalize_table_name(m.group(1)))
    return {t for t in tables if t}


def _single_statement(sql: str):
    import sqlparse

    text = (sql or "").strip().rstrip(";")
    parts = [p for p in sqlparse.parse(text) if str(p).strip()]
    if len(parts) != 1:
        return None
    return parts[0]


def validate_select_ast(
    sql: str,
    allowed_tables: FrozenSet[str] | None = None,
) -> tuple[bool, str]:
    allowed = allowed_tables or _ALLOWED_TABLES
    try:
        import sqlparse  # noqa: F401
    except ImportError:
        return False, "sqlparse indisponível"

    inner = (sql or "").strip().rstrip(";")
    if ";" in inner:
        return False, "múltiplas instruções"

    stmt = _single_statement(sql)
    if stmt is None:
        return False, "não é uma única instrução"

    stype = (stmt.get_type() or "").upper()
    if stype != "SELECT" and not (sql or "").strip().lower().startswith("select"):
        return False, f"tipo {stype or '?'} não é SELECT"

    tokens = _statement_strings(stmt)
    for i, t in enumerate(tokens):
        if t in _SELECT_FORBIDDEN:
            return False, f"palavra proibida: {t}"
        if t == "into" and i > 0 and tokens[i - 1] not in ("insert",):
            return False, "SELECT INTO não permitido"

    tables = _extract_table_names(stmt)
    if tables and not tables.issubset(allowed):
        bad = tables - allowed
        return False, f"tabela não permitida: {', '.join(sorted(bad))}"

    return True, ""


def validate_mutation_ast(
    sql: str,
    allowed_tables: FrozenSet[str] | None = None,
) -> tuple[bool, str]:
    allowed = allowed_tables or _ALLOWED_TABLES
    try:
        import sqlparse  # noqa: F401
    except ImportError:
        return False, "sqlparse indisponível"

    stmt = _single_statement(sql)
    if stmt is None:
        return False, "não é uma única instrução"

    stype = (stmt.get_type() or "").upper()
    if stype not in ("INSERT", "UPDATE", "DELETE"):
        low = (sql or "").strip().lower()
        if low.startswith("select"):
            return False, "SELECT não é mutação"
        if not any(
            low.startswith(k) for k in ("insert", "update", "delete")
        ):
            return False, f"tipo {stype or '?'} inválido"

    tokens = _statement_strings(stmt)
    joined = " ".join(tokens)
    if "insert or replace" in joined or "replace into" in joined:
        return False, "REPLACE não permitido"
    for t in tokens:
        if t in _MUTATION_FORBIDDEN:
            return False, f"palavra proibida: {t}"

    tables = _extract_table_names(stmt)
    if not tables:
        return False, "tabela não identificada"
    if not tables.issubset(allowed):
        bad = tables - allowed
        return False, f"tabela não permitida: {', '.join(sorted(bad))}"

    return True, ""
