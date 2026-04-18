"""
Persistência: DuckDB (CSVs em memória), gravação de CSV e ficheiros JSON em `ROTINA_DATA_DIR` (pasta `data/` por defeito).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("ROTINA_DATA_DIR", "data")).resolve()

ROTINA_CHAT_SESSION_SUBDIR = ".rotina_chat"
ROTINA_BROWSER_SESSION_SUBDIR = ".rotina_browser_sessions"
ROTINA_DIRECT_CHAT_FILE = "chat_familia_educadores.json"
ROTINA_USERS_FILENAME = "rotina_users.json"
ROTINA_REPORT_PHASES = frozenset({"idle", "ask_name", "result"})

# read_csv_auto falha silenciosamente com linhas “curtas” (menos vírgulas que o cabeçalho):
# devolve uma única coluna com o texto inteiro da linha. null_padding alinha às colunas do header.
_READ_CSV_ROTINA_OPTS = (
    "header = true, delim = ',', quote = '\"', escape = '\"', "
    "null_padding = true, ignore_errors = false, max_line_size = 16777216, "
    "auto_detect = true"
)


def _normalize_diario_columns_for_partial_inserts(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """
    O `read_csv` pode inferir hora_sono_* como TIME/DATE. INSERT com '' falha
    (Conversion Error: time field value out of range). Texto aceita vazio e horários parciais.
    """
    # data: permite CURRENT_DATE no INSERT e '' sem falha de tipo; relatório usa to_datetime.
    for col in ("hora_sono_inicio", "hora_sono_fim", "data"):
        try:
            conn.execute(
                f"""
                ALTER TABLE diario_estruturado
                ALTER COLUMN {col} TYPE VARCHAR
                USING CASE
                    WHEN {col} IS NULL THEN ''
                    ELSE trim(cast({col} AS VARCHAR))
                END
                """
            )
        except Exception:
            continue


def _is_safe_chat_session_id(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _chat_session_dir() -> Path:
    d = DATA_DIR / ROTINA_CHAT_SESSION_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _browser_session_dir() -> Path:
    d = DATA_DIR / ROTINA_BROWSER_SESSION_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_browser_session_token(token: str, username: str) -> None:
    p = _browser_session_dir() / f"{token}.json"
    try:
        p.write_text(
            json.dumps({"username": username.strip(), "v": 1}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _delete_browser_session_file(token: str) -> None:
    if not _is_safe_chat_session_id(token):
        return
    try:
        (_browser_session_dir() / f"{token}.json").unlink(missing_ok=True)
    except OSError:
        pass


def _direct_chat_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / ROTINA_DIRECT_CHAT_FILE


def _legacy_direct_chat_path() -> Path:
    return (DATA_DIR / ".familia_educador_chat") / "mensagens.json"


def _normalize_direct_chat_sender_role(raw: str) -> str | None:
    """`familia` ↔ `educador` (gestão conta como lado escola)."""
    r = (raw or "").strip().lower()
    if r == "familia":
        return "familia"
    if r in ("educador", "gestao"):
        return "educador"
    return None


def _load_direct_chat_store() -> dict[str, list[dict[str, str]]]:
    p = _direct_chat_path()
    if not p.is_file():
        lp = _legacy_direct_chat_path()
        if lp.is_file():
            p = lp
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, list[dict[str, str]]] = {}
    for key, msgs in raw.items():
        if not isinstance(key, str) or not isinstance(msgs, list):
            continue
        parsed: list[dict[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            txt = str(m.get("content") or "").strip()
            sender = str(m.get("sender") or "").strip()
            role = _normalize_direct_chat_sender_role(str(m.get("sender_role") or ""))
            if not txt or role is None:
                continue
            parsed.append({"content": txt, "sender": sender, "sender_role": role})
        clean[key] = parsed[-300:]
    return clean


def _persist_direct_chat_store(store: dict[str, list[dict[str, str]]]) -> None:
    try:
        _direct_chat_path().write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return


def _ensure_direct_chat_store_file() -> None:
    p = _direct_chat_path()
    if p.is_file():
        return
    _persist_direct_chat_store({})


def load_rotina_users() -> dict[str, Any]:
    p = DATA_DIR / ROTINA_USERS_FILENAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _coerce_stored_chat_messages_list(data: list[Any]) -> list[dict[str, str]] | None:
    out: list[dict[str, str]] = []
    for m in data:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            return None
        out.append({"role": str(role), "content": content})
    return out


def _report_blob_for_disk() -> dict[str, Any] | None:
    """Estado mínimo do relatório sono/refeições para restaurar após F5 (gráficos refeitos no DuckDB)."""
    phase = st.session_state.get("sleep_rep_phase", "idle")
    if phase not in ROTINA_REPORT_PHASES:
        phase = "idle"
    if phase == "idle":
        return None
    blob: dict[str, Any] = {
        "phase": phase,
        "query_name": str(st.session_state.get("sleep_rep_query_name") or ""),
        "resolved_label": str(st.session_state.get("sleep_rep_resolved_label") or ""),
    }
    if phase == "ask_name":
        blob["nome_field"] = str(st.session_state.get("sleep_rep_nome_field") or "")
    return blob


def _apply_report_blob_from_disk(blob: Any) -> None:
    if not isinstance(blob, dict):
        return
    phase = blob.get("phase")
    if phase not in ROTINA_REPORT_PHASES:
        return
    qn = str(blob.get("query_name") or "").strip()
    if phase == "result" and not qn:
        st.session_state.sleep_rep_phase = "idle"
        return
    if phase == "idle":
        st.session_state.sleep_rep_phase = "idle"
        return
    st.session_state.sleep_rep_phase = phase
    st.session_state.sleep_rep_query_name = str(blob.get("query_name") or "")
    st.session_state.sleep_rep_resolved_label = str(blob.get("resolved_label") or "")
    if phase == "ask_name" and "nome_field" in blob:
        st.session_state.sleep_rep_nome_field = str(blob.get("nome_field") or "")


def _parse_session_file_payload(data: Any) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Ficheiro legado: lista só de mensagens. Novo: `{\"messages\": [...], \"report\": {...}}`."""
    if isinstance(data, list):
        parsed = _coerce_stored_chat_messages_list(data)
        return (parsed or [], None)
    if isinstance(data, dict) and "messages" in data:
        raw_m = data["messages"]
        if not isinstance(raw_m, list):
            return ([], None)
        parsed = _coerce_stored_chat_messages_list(raw_m)
        if parsed is None:
            return ([], None)
        rep = data.get("report")
        if isinstance(rep, dict):
            return (parsed, rep)
        return (parsed, None)
    return ([], None)


def _rotina_session_serial_current() -> str:
    payload = {
        "messages": st.session_state.get("messages") or [],
        "report": _report_blob_for_disk(),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def sync_rotina_chat_from_disk(chat_id: str) -> None:
    """Carrega mensagens e estado do relatório quando a sessão Streamlit é nova ou o id da URL mudou."""
    if st.session_state.get("_chat_disk_synced_for") == chat_id:
        return
    path = _chat_session_dir() / f"{chat_id}.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            msgs, rep = _parse_session_file_payload(data)
            st.session_state.messages = msgs
            if rep is not None:
                _apply_report_blob_from_disk(rep)
        except (OSError, json.JSONDecodeError):
            pass
    st.session_state._chat_disk_synced_for = chat_id
    st.session_state._rotina_session_serial = _rotina_session_serial_current()


def persist_rotina_chat_to_disk(chat_id: str) -> None:
    """Grava conversa + relatório em `ROTINA_DATA_DIR/.rotina_chat/<uuid>.json`."""
    msgs: list[dict[str, str]] = st.session_state.get("messages") or []
    rep = _report_blob_for_disk()
    payload = {"messages": msgs, "report": rep}
    serial = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if st.session_state.get("_rotina_session_serial") == serial:
        return
    st.session_state._rotina_session_serial = serial
    path = _chat_session_dir() / f"{chat_id}.json"
    if not msgs and rep is None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    try:
        path.write_text(serial, encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# DuckDB / CSV
# ---------------------------------------------------------------------------


def _sql_cell_text(v: Any) -> str:
    return str(v).replace("|", "·").replace("\n", " ").strip()


def format_sql_rows(rows: list[Any], columns: list[str], max_rows: int = 80) -> str:
    if not rows:
        return (
            "(nenhuma linha retornada)\n"
            "→ Não invente dados de aluno/diário; informe que a consulta não retornou linhas."
        )
    show = rows[:max_rows]
    head_cells = ["linha"] + [_sql_cell_text(c) for c in columns]
    lines = [
        "=== Resultado SQL (literal) — use só estas linhas/células; não crie outras ===",
        f"Total de linhas retornadas: {len(rows)} (tabela abaixo: até {len(show)}).",
        "",
        "| " + " | ".join(head_cells) + " |",
        "| " + " | ".join(["---"] * len(head_cells)) + " |",
    ]
    for i, r in enumerate(show, start=1):
        cells = [str(i)] + [_sql_cell_text(r[j]) for j in range(len(columns))]
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > max_rows:
        lines.append(
            f"\n({len(rows) - max_rows} linhas omitidas — não invente o conteúdo delas.)"
        )
    return "\n".join(lines)


def validate_sql(sql: str) -> bool:
    if not sql or not isinstance(sql, str):
        return False
    t = sql.strip()
    if ";" in t.rstrip().rstrip(";"):
        return False
    t = t.rstrip().rstrip(";")
    low = t.lower()
    if not low.startswith("select"):
        return False
    banned = re.compile(
        r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|copy|call|truncate|replace)\b",
        re.IGNORECASE,
    )
    return banned.search(low) is None


def validate_mutation_sql(sql: str) -> bool:
    """Permite uma instrução INSERT/UPDATE/DELETE nas tabelas CSV (perfil gestão)."""
    if not sql or not isinstance(sql, str):
        return False
    t = sql.strip().rstrip(";")
    if ";" in t:
        return False
    low = t.lower()
    if not any(low.startswith(p) for p in ("insert", "update", "delete")):
        return False
    banned = re.compile(
        r"\b(drop|alter|create|attach|detach|pragma|copy|call|truncate|replace|into\s+sqlite|from\s+sqlite)\b",
        re.IGNORECASE,
    )
    if banned.search(low):
        return False
    if "diario_estruturado" not in low and "info_alunos" not in low:
        return False
    return True


def _resolve_info_alunos_csv(base: Path) -> Path:
    for name in ("info_alunos.csv", "info_alunos_v2.csv"):
        p = base / name
        if p.exists():
            return p
    return base / "info_alunos.csv"


def _ensure_utf8_bom_csv(path: Path) -> None:
    """Excel no Windows abre CSV sem BOM como ANSI — o BOM marca UTF-8 (acentos corretos)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return
    if raw.startswith(b"\xef\xbb\xbf"):
        return
    try:
        path.write_bytes(b"\xef\xbb\xbf" + raw)
    except OSError:
        pass


def persist_duckdb_tables_to_csv(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """Grava tabelas em memória de volta aos CSVs (após mutação)."""
    info_csv = _resolve_info_alunos_csv(data_dir)
    diario_csv = data_dir / "diario_estruturado.csv"
    ip = str(info_csv.resolve())
    dp = str(diario_csv.resolve())
    _csv_copy_opts = "FORMAT CSV, HEADER, DELIMITER ',', QUOTE '\"', ESCAPE '\"'"
    conn.execute(
        f"COPY (SELECT * FROM info_alunos) TO ? ({_csv_copy_opts});",
        [ip],
    )
    conn.execute(
        f"COPY (SELECT * FROM diario_estruturado) TO ? ({_csv_copy_opts});",
        [dp],
    )
    _ensure_utf8_bom_csv(info_csv)
    _ensure_utf8_bom_csv(diario_csv)


def _extract_first_values_tuple(sql: str) -> str | None:
    """Conteúdo interno do primeiro VALUES (...) num INSERT (uma linha)."""
    m = re.search(r"VALUES\s*\(", sql, re.IGNORECASE)
    if not m:
        return None
    i = m.end()
    depth = 1
    while i < len(sql) and depth:
        c = sql[i]
        if c == "'":
            i += 1
            while i < len(sql):
                if sql[i] == "'":
                    if i + 1 < len(sql) and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return sql[m.end() : i]
        i += 1
    return None


def _split_sql_value_list(inner: str) -> list[str]:
    """Divide valores de um VALUES (…) sem nested calls (caso típico do cadastro)."""
    parts: list[str] = []
    cur: list[str] = []
    i = 0
    in_q = False
    depth = 0
    while i < len(inner):
        c = inner[i]
        if in_q:
            if c == "'":
                if i + 1 < len(inner) and inner[i + 1] == "'":
                    cur.append("''")
                    i += 2
                    continue
                in_q = False
                cur.append("'")
                i += 1
                continue
            cur.append(c)
            i += 1
            continue
        if c == "'":
            in_q = True
            cur.append(c)
            i += 1
            continue
        if c == "(":
            depth += 1
            cur.append(c)
            i += 1
            continue
        if c == ")":
            if depth > 0:
                depth -= 1
            cur.append(c)
            i += 1
            continue
        if c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        if c.isspace():
            i += 1
            continue
        cur.append(c)
        i += 1
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _sql_unquote_string_token(raw: str) -> str | None:
    s = raw.strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].replace("''", "'")
    return None


def _parse_info_alunos_insert(sql: str) -> dict[str, str] | None:
    if not re.search(r"INSERT\s+INTO\s+info_alunos\b", sql, re.IGNORECASE):
        return None
    inner = _extract_first_values_tuple(sql)
    if not inner:
        return None
    parts = _split_sql_value_list(inner)
    mcols = re.search(
        r"INSERT\s+INTO\s+info_alunos\s*\(\s*([^)]+?)\s*\)\s*VALUES",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if mcols:
        cols = [c.strip().lower() for c in mcols.group(1).split(",")]
    else:
        cols = ["id_aluno", "nome", "turma", "alergias", "contato_pais"]
    out: dict[str, str] = {}
    for i, col in enumerate(cols):
        if i >= len(parts):
            break
        raw = parts[i].strip()
        uq = _sql_unquote_string_token(raw)
        out[col] = uq if uq is not None else raw
    # Fallback robusto: quando o 1º valor é subconsulta complexa, garanta os 4 campos textuais.
    # Padrão esperado em info_alunos: (id_expr, 'nome', 'turma', 'alergias', 'contato_pais')
    if "nome" not in out or "contato_pais" not in out:
        lits = re.findall(r"'((?:[^']|'')*)'", inner)
        if len(lits) >= 4:
            out["nome"] = lits[-4].replace("''", "'")
            out["turma"] = lits[-3].replace("''", "'")
            out["alergias"] = lits[-2].replace("''", "'")
            out["contato_pais"] = lits[-1].replace("''", "'")
    return out or None


def _parse_info_alunos_update(sql: str) -> dict[str, str] | None:
    if not re.search(r"UPDATE\s+info_alunos\b", sql, re.IGNORECASE):
        return None
    out: dict[str, str] = {}
    for field in ("nome", "turma", "alergias", "contato_pais"):
        pat = rf"\b{field}\s*=\s*('(?:[^']|'')*')"
        m = re.search(pat, sql, re.IGNORECASE)
        if m:
            lit = m.group(1)
            out[field] = lit[1:-1].replace("''", "'")
    mw = re.search(r"WHERE\s+id_aluno\s*=\s*(\d+)", sql, re.IGNORECASE)
    if mw:
        out["_where_id_aluno"] = mw.group(1)
    return out if out else None


def _duplicate_info_alunos_warning_before_write(
    conn: duckdb.DuckDBPyConnection,
    fields: dict[str, str],
    exclude_id_aluno: int | None,
) -> str | None:
    """
    Verifica se nome ou contacto já existem no cadastro (antes de INSERT/UPDATE persistir).
    """
    lines: list[str] = []
    nome = (fields.get("nome") or "").strip()
    contato = (fields.get("contato_pais") or "").strip()
    excl = exclude_id_aluno

    if nome:
        q = (
            "SELECT id_aluno, nome, turma FROM info_alunos "
            "WHERE TRIM(COALESCE(nome,'')) ILIKE TRIM(?)"
        )
        params: list[Any] = [nome]
        if excl is not None:
            q += " AND CAST(id_aluno AS INTEGER) != ?"
            params.append(excl)
        q += " LIMIT 8"
        try:
            rows = conn.execute(q, params).fetchall()
        except Exception:
            rows = []
        if rows:
            bits = [f"id_aluno={r[0]} ({r[1]} | {r[2]})" for r in rows]
            lines.append(
                f"Nome «{nome}» já consta no cadastro: " + "; ".join(bits)
            )

    trivial_ct = ("", "nenhum", "nenhuma", "-", "—", "na", "n/a")
    if contato and contato.lower() not in trivial_ct:
        q = (
            "SELECT id_aluno, nome, contato_pais FROM info_alunos "
            "WHERE TRIM(COALESCE(contato_pais,'')) ILIKE TRIM(?)"
        )
        params = [contato]
        if excl is not None:
            q += " AND CAST(id_aluno AS INTEGER) != ?"
            params.append(excl)
        q += " LIMIT 8"
        try:
            rows = conn.execute(q, params).fetchall()
        except Exception:
            rows = []
        if rows:
            bits = [f"id_aluno={r[0]} ({r[1]})" for r in rows]
            lines.append(
                f"Contacto «{contato}» já consta no cadastro: " + "; ".join(bits)
            )

    if not lines:
        return None
    return (
        "**Atenção**: " + " ".join(lines) + " "
        "A alteração foi bloqueada para evitar duplicidade; confirme com o utilizador se era intencional."
    )


def _precheck_duplicate_info_alunos(
    conn: duckdb.DuckDBPyConnection, sql: str
) -> str | None:
    low = sql.strip().lower()
    if re.search(r"insert\s+into\s+info_alunos\b", low):
        parsed = _parse_info_alunos_insert(sql)
        if parsed:
            return _duplicate_info_alunos_warning_before_write(conn, parsed, None)
    if re.search(r"update\s+info_alunos\b", low):
        parsed = _parse_info_alunos_update(sql)
        if parsed:
            wid = parsed.pop("_where_id_aluno", None)
            excl = int(wid) if wid is not None else None
            if parsed:
                return _duplicate_info_alunos_warning_before_write(conn, parsed, excl)
    return None


def _format_mutation_persist_error(e: BaseException) -> str:
    """Mensagem legível; destaca bloqueio por Excel/CSV aberto no Windows."""
    raw = str(e)
    el = raw.lower()
    if isinstance(e, PermissionError) or "permission denied" in el or "errno 13" in el:
        return (
            "Não foi possível **gravar** os CSV: permissão negada ou ficheiro bloqueado. "
            "Se `info_alunos.csv` ou `diario_estruturado.csv` estiver aberto no **Excel** (ou outro programa), "
            "**feche o ficheiro** e tente novamente. "
            f"(Detalhe técnico: {raw})"
        )
    if (
        "being used by another process" in el
        or "cannot access the file" in el
        or "another program" in el
    ):
        return (
            "Não foi possível gravar os CSV: o ficheiro está **em uso** por outro programa. "
            "Feche o Excel ou o editor que tiver o CSV aberto e tente de novo. "
            f"(Detalhe: {raw})"
        )
    return f"Erro ao aplicar alteração: {raw}"


def _parse_insert_column_names_after_table(sql: str, table: str) -> list[str] | None:
    """
    Se `INSERT INTO <table> (a, b, ...) VALUES`, devolve os nomes das colunas em minúsculas.
    Se for `INSERT INTO <table> VALUES` (sem lista), devolve None.
    """
    m = re.search(rf"INSERT\s+INTO\s+{re.escape(table)}\s*", sql, re.IGNORECASE)
    if not m:
        return None
    i = m.end()
    while i < len(sql) and sql[i].isspace():
        i += 1
    if i >= len(sql) or sql[i] != "(":
        return None
    depth = 0
    j = i
    while j < len(sql):
        c = sql[j]
        if c == "'":
            j += 1
            while j < len(sql):
                if sql[j] == "\\" and j + 1 < len(sql):
                    j += 2
                    continue
                if sql[j] == "'":
                    if j + 1 < len(sql) and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                inner = sql[i + 1 : j]
                tail = sql[j + 1 :].lstrip()
                if not re.match(r"values\b", tail, re.IGNORECASE):
                    return None
                names: list[str] = []
                for part in inner.split(","):
                    p = part.strip()
                    if not p:
                        continue
                    mm = re.match(r'^[`"]?(\w+)', p, re.IGNORECASE)
                    if mm:
                        names.append(mm.group(1).lower())
                return names
        j += 1
    return None


def _mutation_insert_required_columns_error(sql: str) -> str | None:
    """
    Evita INSERT no diário/cadastro sem `id_aluno` (o DuckDB aceita colunas em falta como NULL
    e o CSV fica com célula vazia).
    """
    t = sql.strip()
    low = t.lower()
    if not low.startswith("insert"):
        return None
    if not re.search(r"\bvalues\b", low):
        return None
    if re.search(r"\binsert\s+into\s+diario_estruturado\b", low):
        cols = _parse_insert_column_names_after_table(t, "diario_estruturado")
        if cols is None:
            return (
                "INSERT em `diario_estruturado` sem lista de colunas não é aceite. "
                "Indique explicitamente **`id_registro`** e **`id_aluno`**; pode listar só mais as colunas que "
                "quiser preencher (as outras ficam vazias). Exemplo mínimo: "
                "`INSERT INTO diario_estruturado (id_registro, id_aluno, data, cafe_manha) VALUES "
                "((SELECT COALESCE(MAX(id_registro), 0) + 1 FROM diario_estruturado), 121, '2026-04-22', 'Comeu bem')` — "
                "o id do aluno tem de ir na coluna **`id_aluno`**."
            )
        cset = frozenset(cols)
        if "id_aluno" not in cset:
            return (
                "O INSERT no diário tem de incluir a coluna **`id_aluno`** com o id do aluno "
                "(ex.: `121`). Sem essa coluna o registo grava com `id_aluno` vazio."
            )
        if "id_registro" not in cset:
            return (
                "O INSERT no diário tem de incluir a coluna **`id_registro`** com o próximo id, por exemplo: "
                "`(SELECT COALESCE(MAX(id_registro), 0) + 1 FROM diario_estruturado)`."
            )
        return None
    if re.search(r"\binsert\s+into\s+info_alunos\b", low):
        cols = _parse_insert_column_names_after_table(t, "info_alunos")
        if cols is not None and "id_aluno" not in frozenset(cols):
            return (
                "O INSERT em `info_alunos` tem de incluir a coluna **`id_aluno`** "
                "(em geral `(SELECT COALESCE(MAX(id_aluno), 0) + 1 FROM info_alunos)`)."
            )
        return None
    return None


def run_mutation_and_persist(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    data_dir: Path,
    *,
    allow_delete: bool = True,
) -> tuple[str, bool, str | None]:
    low = sql.strip().lower()
    if not allow_delete and re.match(r"^\s*delete\b", low):
        return (
            "DELETE não é permitido para este perfil: apenas **Gestão** pode apagar registos nos CSV.",
            False,
            None,
        )
    if not validate_mutation_sql(sql):
        return "Instrução de alteração rejeitada (tabelas ou tipo não permitidos).", False, None
    _col_err = _mutation_insert_required_columns_error(sql)
    if _col_err:
        return _col_err, False, None
    dup_warn: str | None = None
    try:
        dup_warn = _precheck_duplicate_info_alunos(conn, sql)
    except Exception:
        dup_warn = None
    if dup_warn:
        return (
            "Alteração **não gravada**: o texto/nome/contacto informado já existe no cadastro.\n\n"
            + dup_warn
            + "\n\nNada foi alterado no CSV. Ajuste os dados e tente novamente.",
            False,
            dup_warn,
        )
    try:
        conn.execute(sql)
        persist_duckdb_tables_to_csv(conn, data_dir)
        msg = "Alteração aplicada e CSVs atualizados."
        return msg, True, None
    except Exception as e:
        return _format_mutation_persist_error(e), False, None


def _verification_selects_after_mutation(mut_sql: str) -> list[str]:
    """SELECTs só de leitura para o LLM ver o estado pós-mutação (evita resposta genérica de recusa)."""
    low = (mut_sql or "").lower()
    out: list[str] = []
    if "info_alunos" in low:
        # Amostra pequena: reduz tentação de o modelo copiar a lista inteira para a resposta.
        # O total exato vem de `CONTAGEM_OFICIAL_ALUNOS` em `post_mutation_verification_block`, não da tabela formatada.
        out.append(
            "SELECT id_aluno, nome, turma, alergias, contato_pais FROM info_alunos ORDER BY id_aluno DESC LIMIT 12"
        )
    if "diario_estruturado" in low:
        out.append(
            "SELECT id_registro, id_aluno, data, cafe_manha, almoco, recado_professora "
            "FROM diario_estruturado ORDER BY id_registro DESC LIMIT 12"
        )
    if not out:
        out.append(
            "SELECT 'info_alunos' AS tabela, COUNT(*)::BIGINT AS n FROM info_alunos "
            "UNION ALL SELECT 'diario_estruturado', COUNT(*)::BIGINT FROM diario_estruturado"
        )
    return out


def _official_new_aluno_id_block(conn: duckdb.DuckDBPyConnection) -> str | None:
    """
    Após INSERT em info_alunos (ids monótonos), o maior id_aluno é o registo criado.
    Texto plano para o LLM não confundir com células da tabela markdown nem inventar id+1.
    """
    try:
        r = conn.execute(
            "SELECT id_aluno, nome, turma, alergias, contato_pais "
            "FROM info_alunos ORDER BY id_aluno DESC LIMIT 1"
        ).fetchone()
        if not r:
            return None
        aid = int(r[0])
        return (
            "=== Aluno com maior id_aluno no cadastro (em INSERT recente, é o registo novo) ===\n"
            f"CONFIRME_id_aluno={aid}\n"
            f"nome={r[1]}\n"
            f"turma={r[2]}\n"
            f"alergias={r[3]}\n"
            f"contato_pais={r[4]}\n"
            "Ao confirmar o cadastro, repita **só** o inteiro em CONFIRME_id_aluno (não use id+1, id−1 nem outro número)."
        )
    except Exception:
        return None


def _official_info_alunos_count_block(conn: duckdb.DuckDBPyConnection) -> str | None:
    """Uma linha inequívoca com COUNT(*) real (evita o modelo inventar ou ler mal tabelas markdown)."""
    try:
        r = conn.execute(
            "SELECT COUNT(*) FROM info_alunos WHERE TRIM(COALESCE(nome, '')) <> ''"
        ).fetchone()
        if not r:
            return None
        n = int(r[0])
        return (
            "=== TOTAL NO CADASTRO (apenas alunos com nome preenchido) ===\n"
            f"CONTAGEM_OFICIAL_ALUNOS={n}\n"
            "Ao dizer quantos alunos existem, use **exatamente** o inteiro acima (não some linhas da amostra nem estime)."
        )
    except Exception:
        return None


def run_safe_select(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple[str, bool]:
    if not validate_sql(sql):
        return "Consulta SQL rejeitada (apenas SELECT nas tabelas permitidas).", False
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return format_sql_rows(rows, cols), True
    except Exception as e:
        return f"Erro ao executar SQL: {e}", False


def post_mutation_verification_block(
    conn: duckdb.DuckDBPyConnection, mut_sql: str
) -> str:
    """Texto tabular para anexar ao contexto do chat após mutação bem-sucedida."""
    mut_low = (mut_sql or "").lower()
    note = (
        "Estas linhas reflectem o cadastro/diário **depois** da mutação aplicada **nesta mensagem**. "
        "Se o pedido foi inserir um registo, a linha com esses dados é em geral o **novo** registo, não prova de que já existia antes."
    )
    if "insert" in mut_low:
        note += (
            " (Pedido inclui INSERT: confirme o sucesso; para **id_aluno** use só a linha `CONFIRME_id_aluno=...` "
            "se existir, não interprete como duplicata por defeito.)"
        )
    if "delete" in mut_low:
        note += (
            " **DELETE:** a lista abaixo é só uma **amostra** (últimos ids); após apagar, o nome pode **não** "
            "aparecer — isso indica sucesso. Para o total de alunos use só **CONTAGEM_OFICIAL_ALUNOS** no fim; **não** reproduza a tabela na resposta."
        )
    parts: list[str] = [
        "=== Verificação pós-mutação (estado atual dos CSV após esta operação) ===",
        note,
    ]
    for sel in _verification_selects_after_mutation(mut_sql):
        block, ok = run_safe_select(conn, sel)
        parts.append(block)
        if not ok:
            break
    if "insert" in mut_low and "info_alunos" in mut_low:
        _nid = _official_new_aluno_id_block(conn)
        if _nid:
            parts.append(_nid)
    if "info_alunos" in mut_low:
        _cnt = _official_info_alunos_count_block(conn)
        if _cnt:
            parts.append(_cnt)
    return "\n\n".join(parts)


def _duckdb_csv_reload_token(data_dir: Path) -> str:
    """Muda quando os CSVs são alterados, para o cache do DuckDB recarregar."""
    info_csv = _resolve_info_alunos_csv(data_dir)
    diario_csv = data_dir / "diario_estruturado.csv"
    parts: list[str] = []
    for p in (info_csv, diario_csv):
        try:
            parts.append(str(p.stat().st_mtime_ns))
        except OSError:
            parts.append("0")
    return "|".join(parts)


@st.cache_resource
def get_duckdb_connection(data_dir_str: str, _csv_token: str) -> duckdb.DuckDBPyConnection:
    data_dir = Path(data_dir_str)
    info_csv = _resolve_info_alunos_csv(data_dir)
    diario_csv = data_dir / "diario_estruturado.csv"
    if not info_csv.exists():
        raise FileNotFoundError(
            f"CSV de alunos não encontrado em {data_dir} (esperado info_alunos.csv ou info_alunos_v2.csv)."
        )
    if not diario_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {diario_csv}")

    conn = duckdb.connect(database=":memory:")
    conn.execute(
        f"CREATE OR REPLACE TABLE info_alunos AS SELECT * FROM read_csv(?, {_READ_CSV_ROTINA_OPTS});",
        [str(info_csv.resolve())],
    )
    conn.execute(
        f"CREATE OR REPLACE TABLE diario_estruturado AS SELECT * FROM read_csv(?, {_READ_CSV_ROTINA_OPTS});",
        [str(diario_csv.resolve())],
    )
    _normalize_diario_columns_for_partial_inserts(conn)
    return conn
