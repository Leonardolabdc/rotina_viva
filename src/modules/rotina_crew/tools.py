"""Ferramentas CrewAI com permissões explícitas (DuckDB só leitura, RAG, ML emoções)."""

from __future__ import annotations

from typing import Any, Callable

import duckdb
from langchain_core.tools import tool

from core.database import run_safe_select, validate_sql
from modules import ai_engine
from modules import ml_emotion_chat
from modules.rag_index import retrieve_rag_context_and_chunks


def make_duckdb_select_tool(conn: duckdb.DuckDBPyConnection) -> Any:
    """SELECT validado nas tabelas permitidas (mesma regra que o chat clássico)."""

    @tool
    def consultar_tabelas_escolares(sql: str) -> str:
        """
        Executa **uma** consulta SELECT em `info_alunos` e/ou `diario_estruturado`.
        Proibido INSERT, UPDATE, DELETE. Use nomes de colunas reais do esquema.
        """
        s = (sql or "").strip()
        if not validate_sql(s):
            return "SQL rejeitado: só SELECT nas tabelas permitidas."
        block, ok = run_safe_select(conn, s)
        if not ok:
            return block
        return block

    return consultar_tabelas_escolares


def make_rag_tool(collection: Any, default_question: str) -> Any:
    """RAG sobre a coleção Chroma já carregada na sessão."""

    @tool
    def consultar_documentos_indexados(pergunta: str) -> str:
        """
        Pesquisa trechos nos PDFs indexados (ChromaDB). Formule a pergunta em português,
        citando nomes ou palavras que devem aparecer nos documentos.
        """
        q = (pergunta or "").strip() or default_question
        block, _chunks = retrieve_rag_context_and_chunks(
            collection, q, k=ai_engine.rag_context_chunks_top_k()
        )
        return block

    return consultar_documentos_indexados


def make_ml_emotion_tool(data_dir: Any, predictive_session: bool) -> Any:
    """Inferência local TF-IDF + FLAML (addon Markdown) — não substitui a LLM final."""

    @tool
    def classificar_emocoes_ml(texto_utilizador: str) -> str:
        """
        Classifica emoções (dataset dair-ai/emotion) com o bundle `.pkl` em `DATA_DIR/ml_models/`.
        Passe o texto a classificar (uma ou mais frases). Devolve Markdown com tabela CSV de resultados.
        """
        raw = (texto_utilizador or "").strip()
        if not raw:
            return "(sem texto para classificar)"
        return ml_emotion_chat.build_emotion_ml_llm_addon(
            raw,
            data_dir,
            predictive_session=predictive_session,
        ) or "(modelo ML indisponível ou texto vazio após parse)"

    return classificar_emocoes_ml
