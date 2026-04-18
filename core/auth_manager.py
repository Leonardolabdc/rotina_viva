"""
Autenticação (rotina_users.json) e RBAC: perfis gestão, educador e família.
Mantém as mesmas chaves em st.session_state usadas pelo app Streamlit.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import streamlit as st

from core.database import (
    DATA_DIR,
    _browser_session_dir,
    _delete_browser_session_file,
    _is_safe_chat_session_id,
    _save_browser_session_token,
    load_rotina_users,
)

ROTINA_CHAT_QUERY_PARAM = "rotina_chat"
ROTINA_BROWSER_SESSION_QUERY_PARAM = "rotina_session"

VALID_ROTINA_ROLES = frozenset({"gestao", "educador", "familia"})


def _query_param_first(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        return str(v[0]) if v else None
    return str(v)


def _issue_browser_session_token() -> str:
    return str(uuid.uuid4())


def check_password(rec: dict[str, Any], password: str) -> bool:
    """Comparação com o campo `password` em texto plano do JSON de utilizadores."""
    return isinstance(rec, dict) and rec.get("password") == password


def login_user(username_key: str, password: str) -> dict[str, Any] | None:
    """
    Valida utilizador e senha contra `load_rotina_users()`.
    Retorna o registo do JSON ou None se falhar.
    """
    users = load_rotina_users()
    if not isinstance(users, dict):
        return None
    rec = users.get(username_key)
    if not isinstance(rec, dict) or not check_password(rec, password):
        return None
    return rec


def _clear_browser_session_query_param() -> None:
    if ROTINA_BROWSER_SESSION_QUERY_PARAM not in st.query_params:
        return
    try:
        del st.query_params[ROTINA_BROWSER_SESSION_QUERY_PARAM]
    except Exception:
        pass


def try_restore_rotina_browser_session() -> bool:
    """
    Após F5 o session_state do Streamlit reinicia; restaura login se a URL tiver
    `?rotina_session=<uuid>` e existir o ficheiro em disco (token opaco).
    """
    if st.session_state.get("rotina_authenticated"):
        return True
    raw = _query_param_first(st.query_params.get(ROTINA_BROWSER_SESSION_QUERY_PARAM))
    if not raw or not _is_safe_chat_session_id(raw):
        return False
    path = _browser_session_dir() / f"{raw}.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    username = str(data.get("username") or "").strip()
    if not username:
        return False
    users = load_rotina_users()
    rec = users.get(username) if isinstance(users, dict) else None
    if not isinstance(rec, dict):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    role = str(rec.get("role", "")).strip().lower()
    if role not in VALID_ROTINA_ROLES:
        return False
    st.session_state.rotina_authenticated = True
    st.session_state.rotina_role = role
    st.session_state.rotina_user_label = str(rec.get("display_name") or username).strip()
    if role == "familia":
        try:
            st.session_state.rotina_parent_id_aluno = int(rec.get("id_aluno"))
        except (TypeError, ValueError):
            st.session_state.rotina_authenticated = False
            return False
    else:
        st.session_state.rotina_parent_id_aluno = None
    st.session_state.setdefault("rotina_sidebar_screen", "assistant")
    st.session_state.setdefault("rotina_direct_chat_student", None)
    return True


def _direct_chat_viewer_side(session_role: str) -> str | None:
    """`familia` ↔ `educador` (gestão conta como lado escola)."""
    sr = (session_role or "").strip().lower()
    if sr == "familia":
        return "familia"
    if sr in ("educador", "gestao"):
        return "educador"
    return None


def _planner_suffix_gestao() -> str:
    return (
        'Inclua no JSON o campo opcional "mutacao": null ou UMA string SQL com INSERT, UPDATE ou DELETE '
        "apenas nas tabelas `info_alunos` e `diario_estruturado` (colunas do esquema acima). "
        'Use "mutacao" só se o utilizador pedir para criar, alterar ou apagar registos; caso contrário "mutacao": null. '
        "Para **criar ou alterar**, **\"mutacao\"** deve conter o INSERT/UPDATE — um SELECT em `\"sql\"` sozinho **não grava** no CSV. "
        "**Novos alunos (`INSERT` em `info_alunos`):** use **sempre** lista explícita de colunas, **começando por** "
        "`id_aluno`, por exemplo: `INSERT INTO info_alunos (id_aluno, nome, turma, alergias, contato_pais) VALUES "
        "((SELECT COALESCE(MAX(id_aluno), 0) + 1 FROM info_alunos), 'Nome Completo', 'Infantil 2', 'Nenhuma', '41999999999')`. "
        "**Não** omita `id_aluno` na lista nem use só `INSERT INTO info_alunos VALUES (...)` sem nomes de colunas; "
        "não invente um id fixo nem reutilize ids existentes. "
        "**Novas linhas de diário (`INSERT` em `diario_estruturado`):** use **sempre** a lista de colunas entre parênteses "
        "incluindo **`id_registro`** e **`id_aluno`** (nunca omita `id_aluno` — use o número que o utilizador indicar, ex.: 121); "
        "`id_registro` com `(SELECT COALESCE(MAX(id_registro), 0) + 1 FROM diario_estruturado)`. "
        "Pode omitir colunas que não forem preenchidas — as restantes ficam vazias (NULL). "
        "Após mutação bem-sucedida o servidor corre SELECTs de verificação (estado **final** dos CSV); "
        "na resposta, confirme o sucesso do pedido — não trate a linha inserida como duplicata pré-existente. "
        "Antes de gravar, o servidor pode avisar se **nome** ou **contacto** já existiam noutra linha — repita esse aviso ao utilizador. "
        'Pode omitir "sql" no JSON ou devolver só um SELECT complementar. '
        'Formato: {"fontes": [...], "sql": null ou "SELECT ...", "mutacao": null ou "DELETE ..."}.'
    )


def _planner_suffix_educador_no_delete() -> str:
    return (
        'RBAC — Perfil **Educador** (leitura + escrita nos CSV, sem apagar): '
        "**Leitura:** inclua o campo **\"sql\"** com um **SELECT** sempre que a pergunta for consultar listas, "
        "cadastro, diário, turmas, alergias, etc. "
        "**Escrita:** quando o utilizador pedir **criar ou alterar** dados (novo aluno, linha de diário, atualizar cadastro), "
        '**"mutacao"** tem de trazer **obrigatoriamente** o INSERT ou UPDATE — **não** resolva só com `"sql"` SELECT; '
        "SELECT sozinho **não grava** no CSV. "
        "Em **INSERT em `info_alunos`** (novo aluno), liste **sempre** as colunas entre parênteses **incluindo** "
        "**`id_aluno` em primeiro lugar**, por exemplo: `INSERT INTO info_alunos (id_aluno, nome, turma, alergias, contato_pais) VALUES "
        "((SELECT COALESCE(MAX(id_aluno), 0) + 1 FROM info_alunos), 'Nome', 'Turma', 'Nenhuma' ou alergia, 'contacto')`. "
        "Nunca omita `id_aluno` nem use `INSERT INTO info_alunos VALUES (...)` sem nomes de colunas. "
        "Em **INSERT no diário**, liste as colunas que for preencher e inclua sempre **`id_aluno`** (ex.: 121) e **`id_registro`** "
        "(próximo id); nunca omita `id_aluno`. As colunas não listadas ficam vazias. "
        "**Não** use **DELETE** em `\"mutacao\"` — apagar é **exclusivo da Gestão**; para pedidos de apagar, "
        'responda que só a gestão pode e use `"mutacao": null`. '
        'Formato: {"fontes": [...], "sql": null ou "SELECT ...", "mutacao": null ou "INSERT ..." ou "UPDATE ..."}.'
    )


def educador_rotina_csv_access() -> dict[str, Any]:
    """
    Parâmetros do chat Rotina para o perfil **educador**: leitura (SELECT via `sql`) e escrita
    (`mutacao` com INSERT/UPDATE). DELETE fica só para gestão (`allow_delete_mutations=False`).
    """
    return {
        "read_only_db": False,
        "allow_mutations": True,
        "allow_delete_mutations": False,
        "planner_extra": _planner_suffix_educador_no_delete(),
    }


def _planner_suffix_familia(id_aluno: int, nome: str) -> str:
    return (
        f"RBAC — Perfil Família (só leitura): o responsável vê apenas o aluno **{nome}** (id_aluno={id_aluno}). "
        'Não inclua "mutacao". Todas as consultas SQL devem restringir-se a esse aluno.'
    )


def _chat_system_familia(id_aluno: int, nome: str) -> str:
    return (
        f"O utilizador é um responsável (perfil leitura). Para dados de cadastro ou diário, aborde apenas o aluno "
        f"**{nome}** (id_aluno={id_aluno}). Não revele dados de outras crianças."
    )


def render_login() -> None:
    _pad_l, _center, _pad_r = st.columns([1, 2, 1])
    with _center:
        _il, _inner, _ir = st.columns([1, 2, 1])
        with _inner:
            _logo_path = DATA_DIR / "logo_rotina_viva.png"
            if _logo_path.is_file():
                st.image(str(_logo_path), use_container_width=True)
            else:
                st.title("Rotina Viva")
            with st.form("rotina_login_form"):
                username = st.text_input("Usuário")
                password = st.text_input("Senha", type="password")
                submitted = st.form_submit_button("Entrar", use_container_width=True)
            if submitted:
                key = (username or "").strip()
                rec = login_user(key, password)
                if rec is not None:
                    role = str(rec.get("role", "")).strip().lower()
                    if role not in VALID_ROTINA_ROLES:
                        st.error(
                            "Perfil inválido: use `gestao`, `educador` ou `familia` no ficheiro de utilizadores."
                        )
                        return
                    st.session_state.rotina_authenticated = True
                    st.session_state.rotina_role = role
                    st.session_state.rotina_user_label = str(
                        rec.get("display_name") or key
                    ).strip()
                    if role == "familia":
                        try:
                            st.session_state.rotina_parent_id_aluno = int(
                                rec.get("id_aluno")
                            )
                        except (TypeError, ValueError):
                            st.error(
                                "Para o perfil Família é obrigatório um campo numérico `id_aluno`."
                            )
                            return
                    else:
                        st.session_state.rotina_parent_id_aluno = None
                    st.session_state.rotina_sidebar_screen = "assistant"
                    st.session_state.rotina_direct_chat_student = None
                    st.session_state.messages = []
                    st.session_state.pop("_rotina_session_serial", None)
                    st.session_state.pop("_chat_disk_synced_for", None)
                    _tok = _issue_browser_session_token()
                    _save_browser_session_token(_tok, key)
                    st.query_params[ROTINA_BROWSER_SESSION_QUERY_PARAM] = _tok
                    st.query_params[ROTINA_CHAT_QUERY_PARAM] = str(uuid.uuid4())
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos.")


def render_auth_sidebar() -> None:
    label = st.session_state.get("rotina_user_label") or "—"
    role = st.session_state.get("rotina_role") or ""
    role_lc = str(role).strip().lower()
    st.markdown(f"**{label}**")
    if role_lc == "gestao":
        st.caption("Perfil: Gestão")
    elif role_lc == "educador":
        st.caption("Perfil: Educador (ler e gravar; sem apagar)")
    elif role_lc == "familia":
        st.caption("Perfil: Família")
    _left, _right = st.columns([1, 1], gap="small")
    with _left:
        if st.button("IA", key="rotina_sidebar_assistente_btn", help="Abrir Assistente IA"):
            st.session_state.rotina_sidebar_screen = "assistant"
            st.rerun()
    with _right:
        _direct_btn_label = (
            "Chat direto escola"
            if role_lc == "familia"
            else "Chat direto família"
        )
        _direct_btn_help = (
            "Abrir mensagens com a escola (educadores / gestão)."
            if role_lc == "familia"
            else "Abrir mensagens com as famílias (por aluno)."
        )
        if st.button(
            _direct_btn_label,
            key="rotina_sidebar_direct_chat_btn",
            help=_direct_btn_help,
        ):
            st.session_state.rotina_sidebar_screen = "direct_chat"
            st.rerun()
    if st.button("Sair", key="rotina_logout_btn"):
        _ltok = _query_param_first(
            st.query_params.get(ROTINA_BROWSER_SESSION_QUERY_PARAM)
        )
        if _ltok:
            _delete_browser_session_file(_ltok)
        _clear_browser_session_query_param()
        st.session_state.rotina_authenticated = False
        st.session_state.rotina_role = None
        st.session_state.rotina_user_label = ""
        st.session_state.rotina_parent_id_aluno = None
        st.session_state.rotina_sidebar_screen = "assistant"
        st.session_state.rotina_direct_chat_student = None
        st.session_state.messages = []
        st.session_state.pop("_rotina_session_serial", None)
        st.session_state.pop("_chat_disk_synced_for", None)
        st.query_params[ROTINA_CHAT_QUERY_PARAM] = str(uuid.uuid4())
        st.rerun()
    st.divider()
