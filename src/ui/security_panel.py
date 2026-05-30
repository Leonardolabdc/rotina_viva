"""Painel de segurança na sidebar (quota LLM + auditoria de mutações)."""

from __future__ import annotations

import streamlit as st

from core.database import DATA_DIR
from core.security import (
    ROTINA_LLM_DAILY_MESSAGES_PER_USER,
    check_llm_message_quota,
    read_recent_audit_lines,
)


def render_security_sidebar_panel() -> None:
    """Expander na sidebar — visível para utilizadores autenticados."""
    with st.expander("Segurança", expanded=False):
        user = str(st.session_state.get("rotina_login_username") or "").strip()
        ok, used, limit = check_llm_message_quota(user, DATA_DIR)
        if limit > 0:
            st.caption("Mensagens com IA (hoje)")
            st.progress(min(1.0, used / limit) if limit else 0.0)
            st.write(f"**{used}** / **{limit}**")
            if not ok:
                st.warning("Limite diário atingido.")
        else:
            st.caption("Quota diária desactivada (`ROTINA_LLM_DAILY_MESSAGES_PER_USER=0`).")

        st.caption("Últimas alterações nos CSV (auditoria local)")
        entries = read_recent_audit_lines(DATA_DIR, limit=8)
        if not entries:
            st.write("_Nenhuma mutação registada ainda._")
        else:
            for e in entries:
                ts = str(e.get("ts") or "")[:19].replace("T", " ")
                who = str(e.get("username") or "?")
                role = str(e.get("role") or "")
                status = "ok" if e.get("ok") else "falhou"
                sql_preview = str(e.get("sql") or "")[:120].replace("\n", " ")
                st.markdown(
                    f"- `{ts}` **{who}** ({role}) — **{status}**  \n"
                    f"  `{sql_preview}{'…' if len(str(e.get('sql') or '')) > 120 else ''}`"
                )

        st.caption(
            "Guardrails: scanners de entrada (injection, jailbreak, toxicidade, tópicos proibidos) "
            "e de saída (PII, conteúdo clínico/jurídico). Desligar: `ROTINA_GUARDRAILS_ENABLED=false`."
        )
        st.caption(
            "Backups: `data/.rotina_csv_backups/` · Log: `data/.rotina_audit/mutations.jsonl`"
        )
