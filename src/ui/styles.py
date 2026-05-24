"""CSS global da aplicação Streamlit (Rotina Viva)."""

from __future__ import annotations

import streamlit as st


def apply_styles() -> None:
    """Espaço no fundo do conteúdo + tema escuro fixo + barra fixa (microfone + chat)."""
    st.markdown(
        """
<style>
/* Tema escuro fixo (Cloud: evita flash claro quando o browser guardou Light). */
:root {
    --background-color: #0e1117;
    --secondary-background-color: #1a1d24;
    --text-color: #fafafa;
    --primary-color: #4caf50;
}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section.main {
    background-color: #0e1117 !important;
    color: #fafafa !important;
}
[data-testid="stHeader"] {
    background-color: rgba(14, 17, 23, 0.92) !important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #161b22 !important;
    color: #fafafa !important;
}
section.main div.block-container {
    padding-bottom: 5.85rem !important;
    background-color: transparent !important;
    color: #fafafa !important;
}
/*
 * Barra fixa: bases alinhadas — fundo do áudio com o fundo do campo de texto (flex-end).
 * Evitamos mexer em largura/flex das colunas para não quebrar o WaveSurfer.
 */
.rotina-chat-footer-row {
    position: fixed !important;
    bottom: 0 !important;
    z-index: 1002 !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-end !important;
    justify-content: center !important;
    gap: 0.5rem !important;
    background: var(
        --secondary-background-color,
        var(--widget-background-color, var(--background-color))
    ) !important;
    border: none !important;
    padding: 0.35rem 1rem 0.55rem 1rem !important;
    padding-bottom: calc(0.55rem + env(safe-area-inset-bottom, 0px)) !important;
    /* Sombra só para baixo — evita “linha” acima do rodapé (antes: offset Y negativo). */
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    left: 0;
    width: 100%;
    overflow: visible !important;
}
/*
 * Linha interna [chat | áudio]: alinhar pela base — o chat costuma ficar visualmente “mais acima”
 * sem align-items no bloco horizontal filho.
 */
.rotina-chat-footer-row div[data-testid="stHorizontalBlock"] {
    align-items: flex-end !important;
}
.rotina-chat-footer-row div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-end !important;
}
/* Garante que o widget de texto encosta ao fundo da coluna (mesma linha do botão de áudio). */
.rotina-chat-footer-row [data-testid="stChatInput"] {
    margin-top: auto !important;
    margin-bottom: 0 !important;
}
/* Separador visual do expander "Gerar Relatório" (só área principal; sidebar não é section.main). */
section.main div[data-testid="stExpander"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
section.main div[data-testid="stExpander"] details {
    border: none !important;
    box-shadow: none !important;
}
section.main div[data-testid="stExpander"] summary {
    border-bottom: none !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )
