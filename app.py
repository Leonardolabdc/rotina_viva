"""
Rotina Viva — assistente para rotinas em escolas infantis (AI Factory / PUCPR).
Streamlit + DuckDB (CSVs) + ChromaDB (RAG) + txtai (segmentação) + LLM/embeddings (API ou Ollama local).
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.auth_manager import render_login, try_restore_rotina_browser_session
from ui.components import init_session_state, render_educador, render_familia, render_gestao
from ui.styles import apply_styles


def main() -> None:
    st.set_page_config(
        page_title="Rotina Viva",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()
    init_session_state()
    try_restore_rotina_browser_session()
    if not st.session_state.get("rotina_authenticated"):
        render_login()
        return
    role = st.session_state.get("rotina_role")
    if role == "gestao":
        render_gestao()
    elif role == "educador":
        render_educador()
    elif role == "familia":
        render_familia()
    else:
        st.session_state.rotina_authenticated = False
        st.session_state.rotina_login_username = ""
        st.error("Sessão inválida. Entre novamente.")
        render_login()


if __name__ == "__main__":
    main()
