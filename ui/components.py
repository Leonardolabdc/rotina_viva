"""Componentes visuais Streamlit (Rotina Viva)."""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from typing import Any, Generator

import duckdb
import streamlit as st
import streamlit.components.v1 as components

from core.auth_manager import (
    ROTINA_CHAT_QUERY_PARAM,
    _chat_system_familia,
    _direct_chat_viewer_side,
    _planner_suffix_familia,
    _planner_suffix_gestao,
    _query_param_first,
    educador_rotina_csv_access,
    render_auth_sidebar,
)
from core.database import (
    DATA_DIR,
    _mensagem_csv_aberto_simples,
    _chat_session_dir,
    _direct_chat_path,
    _ensure_direct_chat_store_file,
    _is_safe_chat_session_id,
    _load_direct_chat_store,
    _normalize_direct_chat_sender_role,
    _persist_direct_chat_store,
    get_duckdb_connection,
    persist_rotina_chat_to_disk,
    post_mutation_verification_block,
    run_mutation_and_persist,
    run_safe_select,
    sync_rotina_chat_from_disk,
    promote_plan_sql_mutation_field,
    validate_mutation_sql,
    _duckdb_csv_reload_token,
)
from modules import ai_engine
from modules import ml_emotion_chat
from modules.chat_service import (
    _processing_status_rag_line,
    _processing_status_sql_line,
    _reply_delete_not_persisted_no_mutation,
    _student_label_for_chat,
    _user_requests_student_delete,
    try_gestao_delete_by_name_intent,
    apply_infer_sql_to_plan,
    apply_parent_sql_scope,
    apply_user_data_source_mode,
    augment_cadastro_question_with_history,
    augment_question_for_parent_rag,
    build_mutation_direct_reply,
    is_rag_nutrition_meals_scope_question,
    normalize_plan,
)
from modules.rag_index import (
    CHROMA_DIR,
    INDEX_PROFILE,
    ROTINA_API_PLAN_TO_CHAT_DELAY_SEC,
    get_chroma_collection,
    reset_rotina_chroma_persist,
    retrieve_rag_context_and_chunks,
)
from modules.services import (
    ROTINA_SONO_FAIXA_LIMITE_1,
    ROTINA_SONO_FAIXA_LIMITE_2,
    ROTINA_SONO_MAX_MIN,
    build_sleep_meal_report_dataframe,
    meal_intake_stacked_bar_altair,
    sleep_line_chart_altair,
    sleep_meal_report_summary_md,
    sleep_reference_table_df,
)

def render_sleep_meal_report_section(
    conn: duckdb.DuckDBPyConnection | None,
    chat_session_id: str,
    parent_lock: tuple[int, str] | None = None,
) -> None:
    """
    Conteúdo do relatório (chamado dentro do expander centralizado abaixo do título).
    Fluxo: Gerar relatório → nome do aluno → gráfico e resumo (só CSV / DuckDB).
    Com parent_lock, o relatório fica restrito ao aluno vinculado (perfil Família).
    """
    phase = st.session_state.get("sleep_rep_phase", "idle")

    if phase == "idle":
        if parent_lock:
            aid, anome = parent_lock
            if st.button(
                f"Ver relatório de {anome}",
                type="secondary",
                use_container_width=True,
                key="sleep_rep_open_parent_btn",
                help="Sono e refeições — apenas os dados do seu filho neste cadastro.",
            ):
                st.session_state.sleep_rep_query_name = anome
                st.session_state.sleep_rep_resolved_label = anome
                st.session_state.sleep_rep_phase = "result"
                persist_rotina_chat_to_disk(chat_session_id)
                st.rerun()
            return
        if st.button(
            "Gerar relatório",
            type="secondary",
            use_container_width=True,
            key="sleep_rep_open_btn",
            help="Tendência de sono e padrões de alimentação com base nos registros da escola.",
        ):
            st.session_state.sleep_rep_phase = "ask_name"
            persist_rotina_chat_to_disk(chat_session_id)
            st.rerun()
        return

    if phase == "ask_name":
        if parent_lock:
            st.session_state.sleep_rep_phase = "idle"
            persist_rotina_chat_to_disk(chat_session_id)
            st.rerun()
            return
        st.caption(
            "Use o **nome completo** como consta no **cadastro da escola**. "
            "O relatório considera os **últimos sete dias corridos** a partir da data mais recente "
            "registrada na rotina da criança. **Sono:** horas por dia (média do intervalo início–fim). "
            "**Alimentação:** café da manhã, almoço e lanche por dia."
        )
        with st.form("sleep_rep_form"):
            nome_in = st.text_input(
                "Nome do aluno",
                placeholder="Ex.: Rafael Souza",
                key="sleep_rep_nome_field",
            )
            g1, g2 = st.columns(2)
            with g1:
                sub = st.form_submit_button("Gerar gráfico")
            with g2:
                cancel = st.form_submit_button("Cancelar")

        if cancel:
            st.session_state.sleep_rep_phase = "idle"
            persist_rotina_chat_to_disk(chat_session_id)
            st.rerun()

        if sub:
            nome_val = (st.session_state.get("sleep_rep_nome_field") or "").strip()
            if conn is None:
                st.warning("Não foi possível carregar os dados da rotina.")
            elif not nome_val:
                st.warning("Digite o nome do aluno.")
            else:
                q = nome_val
                df, _meals, err, resolved, _aw, _per = build_sleep_meal_report_dataframe(
                    conn, q
                )
                if err:
                    st.warning(err)
                else:
                    st.session_state.sleep_rep_query_name = q
                    st.session_state.sleep_rep_resolved_label = resolved
                    st.session_state.sleep_rep_phase = "result"
                    persist_rotina_chat_to_disk(chat_session_id)
                    st.rerun()
        return

    # phase == "result"
    st.markdown("##### Sono e alimentação")
    st.caption(
        "Gráficos e texto elaborados com base em **dados institucionais da escola**, "
        "confidenciais e **protegidos por direitos autorais** e pela legislação de proteção de dados aplicável."
    )
    if conn is None:
        st.info("Não foi possível carregar os dados da rotina. Tente novamente mais tarde.")
        if st.button("Fechar", key="sleep_rep_close_na"):
            st.session_state.sleep_rep_phase = "idle"
            persist_rotina_chat_to_disk(chat_session_id)
            st.rerun()
        return

    qn = st.session_state.get("sleep_rep_query_name") or ""
    resolved = st.session_state.get("sleep_rep_resolved_label") or ""
    df, daily_meals, err, resolved2, aviso_sem, periodo = (
        build_sleep_meal_report_dataframe(conn, qn)
    )
    label = resolved or resolved2
    if err or df is None:
        st.warning(err or "Sem dados.")
        if st.button("Fechar", key="sleep_rep_close_err"):
            st.session_state.sleep_rep_phase = "idle"
            persist_rotina_chat_to_disk(chat_session_id)
            st.rerun()
        return

    if aviso_sem:
        st.warning(aviso_sem)

    j_ini, j_fim = (periodo or ("", ""))

    sono_chart = sleep_line_chart_altair(df)
    bar_chart, ref_meals = meal_intake_stacked_bar_altair(daily_meals)
    tbl_sono = sleep_reference_table_df(df)

    _gcol_sono, _gcol_meal = st.columns(2, gap="medium")
    with _gcol_sono:
        st.markdown("**Sono** — tendência (horas por dia)")
        st.caption(
            "Curva: **horas por dia** (média do dia; cada registro **no máximo** "
            f"{int(round(ROTINA_SONO_MAX_MIN))} min). Linha tracejada: **teto** ({int(round(ROTINA_SONO_MAX_MIN))} min). "
            "Cores = **classificação** (mesma da tabela): "
            f"pouco abaixo de {int(round(ROTINA_SONO_FAIXA_LIMITE_1))} min, "
            f"normal entre {int(round(ROTINA_SONO_FAIXA_LIMITE_1))} e {int(round(ROTINA_SONO_FAIXA_LIMITE_2))} min, "
            f"bastante acima de {int(round(ROTINA_SONO_FAIXA_LIMITE_2))} min."
        )
        st.altair_chart(sono_chart, use_container_width=True)
    with _gcol_meal:
        st.markdown("**Alimentação** — café, almoço e lanche (distribuição na semana)")
        st.caption(
            "Cada coluna é um dia. **De baixo para cima:** café da manhã, almoço e lanche. "
            "As **cores** seguem a classificação da ingestão (legenda abaixo do gráfico). "
            "Passe o cursor sobre os segmentos para ver o texto completo registrado pela escola."
        )
        st.altair_chart(bar_chart, use_container_width=True)
    st.markdown("**Referências por dia**")
    _t_sono, _t_meal = st.columns(2, gap="medium")
    with _t_sono:
        st.markdown("##### Sono")
        st.caption(
            "**Minutos:** média diária pelos horários do CSV (**cortada em** "
            f"{int(round(ROTINA_SONO_MAX_MIN))} min). **Classificação:** `qualidade_sono` do CSV quando houver; senão, pelas faixas. "
            f"“**Dormiu normal**” = entre **{int(round(ROTINA_SONO_FAIXA_LIMITE_1))}** e "
            f"**{int(round(ROTINA_SONO_FAIXA_LIMITE_2))} min** (sobre 0–{int(round(ROTINA_SONO_MAX_MIN))} min)."
        )
        st.dataframe(
            tbl_sono,
            hide_index=True,
            use_container_width=True,
            height=min(320, 40 + max(1, len(tbl_sono)) * 38),
        )
    with _t_meal:
        st.markdown("##### Refeições (texto registrado)")
        st.caption("Café da manhã, almoço e lanche — mesmo período do gráfico de barras.")
        st.dataframe(
            ref_meals,
            hide_index=True,
            use_container_width=True,
            height=min(420, 40 + max(1, len(ref_meals)) * 35),
        )
    st.markdown(sleep_meal_report_summary_md(df, label, j_ini or None, j_fim or None))
    _br_new, _br_close = st.columns(2, gap="small")
    with _br_new:
        if not parent_lock and st.button(
            "Gerar Novo Relatório",
            key="sleep_rep_new_report",
            use_container_width=True,
            help="Volta ao formulário para informar outro aluno.",
        ):
            st.session_state.sleep_rep_phase = "ask_name"
            persist_rotina_chat_to_disk(chat_session_id)
            st.rerun()
    with _br_close:
        if st.button("Fechar", key="sleep_rep_close_ok", use_container_width=True):
            st.session_state.sleep_rep_phase = "idle"
            persist_rotina_chat_to_disk(chat_session_id)
            st.rerun()


def render_direct_family_educator_chat(conn: duckdb.DuckDBPyConnection | None) -> None:
    _ensure_direct_chat_store_file()
    role_lc = str(st.session_state.get("rotina_role") or "").strip().lower()
    user_label = str(st.session_state.get("rotina_user_label") or "Utilizador")
    viewer_side = _direct_chat_viewer_side(role_lc)
    thread_aid: int | None = None

    st.subheader("Chat direto Família ↔ Educadores")
    st.caption(f"Histórico salvo em `{_direct_chat_path().name}` dentro de `data`.")

    if role_lc == "familia":
        aid = st.session_state.get("rotina_parent_id_aluno")
        if isinstance(aid, int):
            thread_aid = aid
            st.info(
                f"Canal direto com educadores para o aluno `{_student_label_for_chat(conn, aid)}`."
            )
        else:
            st.warning("Não foi possível identificar o aluno associado ao perfil Família.")
            return
    elif role_lc in ("educador", "gestao"):
        if conn is None:
            st.warning("DuckDB indisponível para carregar alunos do chat.")
            return
        try:
            _students = conn.execute(
                "SELECT id_aluno, nome, turma FROM info_alunos ORDER BY nome"
            ).fetchall()
        except Exception as ex:
            st.warning(str(ex))
            return
        if not _students:
            st.info("Não há alunos cadastrados para abrir conversas.")
            return
        options = [int(r[0]) for r in _students]
        labels = {
            int(r[0]): (
                f"{str(r[1]).strip() or f'id_aluno={int(r[0])}'}"
                + (f" ({str(r[2]).strip()})" if str(r[2] or "").strip() else "")
            )
            for r in _students
        }
        cur = st.session_state.get("rotina_direct_chat_student")
        if not isinstance(cur, int) or cur not in options:
            cur = options[0]
            st.session_state.rotina_direct_chat_student = cur
        thread_aid = int(
            st.selectbox(
                "Para qual aluno enviar a mensagem?",
                options=options,
                index=options.index(cur),
                format_func=lambda v: labels[int(v)],
                key="rotina_direct_chat_select_aluno",
            )
        )
        st.session_state.rotina_direct_chat_student = thread_aid
    else:
        st.info("Faça login com um perfil válido para usar o chat direto.")
        return

    if thread_aid is None:
        return
    thread_key = str(thread_aid)
    store = _load_direct_chat_store()
    messages = store.get(thread_key, [])

    if role_lc == "gestao":
        _lbl = _student_label_for_chat(conn, thread_aid)
        st.caption(
            f"**Gestão:** pode apagar todo o histórico deste aluno (`{_lbl}`) — não afeta outras conversas."
        )
        if st.button(
            "Limpar conversa deste aluno",
            key=f"rotina_direct_chat_clear_{thread_key}",
            type="secondary",
        ):
            store.pop(thread_key, None)
            _persist_direct_chat_store(store)
            st.success("Histórico deste aluno foi apagado.")
            st.rerun()

    for msg in messages:
        speaker = msg.get("sender") or msg.get("sender_role") or "Utilizador"
        sender_side = _normalize_direct_chat_sender_role(str(msg.get("sender_role") or ""))
        if sender_side is None or viewer_side is None:
            continue
        bubble_role = "user" if sender_side == viewer_side else "assistant"
        with st.chat_message(bubble_role):
            st.caption(speaker)
            st.markdown(msg.get("content") or "")

    text = st.chat_input("Escreva sua mensagem para a outra parte…")
    if text and text.strip():
        if viewer_side is None:
            return
        new_msg = {
            "content": text.strip(),
            "sender": user_label,
            "sender_role": viewer_side,
        }
        store.setdefault(thread_key, []).append(new_msg)
        store[thread_key] = store[thread_key][-300:]
        _persist_direct_chat_store(store)
        st.rerun()

def ensure_rotina_chat_session_id() -> str:
    """Garante `?rotina_chat=<uuid>` na URL; o mesmo id reaparece após F5 e liga ao ficheiro em disco."""
    raw = _query_param_first(st.query_params.get(ROTINA_CHAT_QUERY_PARAM))
    if raw and _is_safe_chat_session_id(raw):
        return raw
    st.query_params[ROTINA_CHAT_QUERY_PARAM] = str(uuid.uuid4())
    st.rerun()


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("data_source_mode", "auto")
    st.session_state.setdefault("last_rag_chunks", [])
    st.session_state.setdefault("last_rag_question", "")
    st.session_state.setdefault("sleep_rep_phase", "idle")
    st.session_state.setdefault("rotina_voice_hash", "")
    st.session_state.setdefault("rotina_voice_input_key", 0)
    st.session_state.setdefault("rotina_authenticated", False)
    st.session_state.setdefault("rotina_role", None)
    st.session_state.setdefault("rotina_user_label", "")
    st.session_state.setdefault("rotina_parent_id_aluno", None)
    st.session_state.setdefault("rotina_sidebar_screen", "assistant")
    st.session_state.setdefault("rotina_direct_chat_student", None)
    st.session_state.setdefault("rotina_login_username", "")
    st.session_state.setdefault("rotina_predictive_ml", False)
    st.session_state.setdefault("rotina_crewai_mode", False)

def pin_chat_footer_row() -> None:
    """Fixa a linha com st.chat_input no rodapé e alinha à largura da área principal."""
    components.html(
        """
<script>
(function () {
  const doc = window.parent.document;
  if (!doc) return;
  function pin() {
    let rows;
    try {
      rows = doc.querySelectorAll(
        'div[data-testid="stHorizontalBlock"]:has([data-testid="stChatInput"])'
      );
    } catch (e) {
      return;
    }
    if (!rows.length) return;
    /* Primeira linha com chat = wrapper [margem | chat+áudio | margem ]; a última seria só o par interno. */
    const row = rows[0];
    row.classList.add("rotina-chat-footer-row");
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    const w = sb ? Math.round(sb.getBoundingClientRect().width) : 0;
    row.style.left = w + "px";
    row.style.width = "calc(100% - " + w + "px)";
  }
  var pinTimer = null;
  function debouncedPin() {
    clearTimeout(pinTimer);
    pinTimer = setTimeout(pin, 40);
  }
  pin();
  [80, 200, 500, 1200, 2500].forEach(function (t) { setTimeout(pin, t); });
  const sb = doc.querySelector('[data-testid="stSidebar"]');
  if (sb && window.ResizeObserver) {
    new ResizeObserver(debouncedPin).observe(sb);
  }
  window.parent.addEventListener("resize", debouncedPin);
})();
</script>
        """,
        height=1,
        width=0,
    )

def render_rag_sidebar_body(body: Any) -> None:
    """Preenche o painel RAG depois que `last_rag_chunks` foi atualizado (mesmo run do Streamlit)."""
    if body is None:
        return
    with body.container():
        _chunks = st.session_state.get("last_rag_chunks") or []
        _lq = (st.session_state.get("last_rag_question") or "").strip()
        if not _chunks:
            st.caption(
                "Aparece aqui o trecho de PDF mais relevante usado na última resposta que consultou documentos."
            )
        else:
            if _lq:
                st.caption(f"Pergunta: {_lq[:180]}{'…' if len(_lq) > 180 else ''}")
            st.caption(
                "Trechos enviados ao modelo (mesma ordem do contexto RAG). "
                "O primeiro costumava ser o único visível; se a resposta citar outro PDF, abra os expanders abaixo."
            )
            for idx, ch in enumerate(_chunks, start=1):
                src = str(ch.get("source", "?"))
                ck = ch.get("chunk")
                dist = ch.get("distance")
                meta_bits: list[str] = []
                if ck not in (None, "", "?"):
                    meta_bits.append(f"índice {ck}")
                if isinstance(dist, (int, float)):
                    meta_bits.append(f"distância {float(dist):.4f}")
                txt = (ch.get("text") or "").strip()
                if len(txt) > 4000:
                    txt = txt[:4000] + "…"
                label = f"{idx}. {src}"
                with st.expander(label, expanded=(idx == 1)):
                    if meta_bits:
                        st.caption(" · ".join(meta_bits))
                    st.text(txt)


def _rotina_workspace_radio_changed() -> None:
    """Gestão/educador: 4.ª opção abre o laboratório ML; as outras mantêm o chat e a fonte de dados."""
    v = st.session_state.get("rotina_assistant_view_choice")
    if v == "ml_traditional":
        st.session_state.rotina_sidebar_screen = "ml_traditional"
    else:
        st.session_state.rotina_sidebar_screen = "assistant"
        if v in ("auto", "structured", "documents"):
            st.session_state.data_source_mode = v


def render_chat_sidebar_internals() -> Any:
    """Controlos de fonte + limpar conversa + placeholder dos trechos RAG."""
    st.subheader("Fonte da resposta")
    _mode_choices: tuple[tuple[str, str], ...] = (
        ("auto", "Automático (a IA escolhe SQL e/ou documentos)"),
        ("structured", "Só dados estruturados (DuckDB — cadastro e diário)"),
        ("documents", "Só documentos (ChromaDB — PDFs indexados)"),
    )
    _vals = [m[0] for m in _mode_choices]
    _labels = {m[0]: m[1] for m in _mode_choices}
    role_lc = str(st.session_state.get("rotina_role") or "").strip().lower()

    if role_lc in ("gestao", "educador"):
        _staff_opts = ("auto", "structured", "documents", "ml_traditional")
        _staff_labels = {
            **_labels,
            "ml_traditional": "ML clássico (laboratório FLAML — treino e exportar .pkl)",
        }
        if "rotina_assistant_view_choice" not in st.session_state:
            st.session_state.rotina_assistant_view_choice = st.session_state.get(
                "data_source_mode", "auto"
            )
        _svc = st.session_state.rotina_assistant_view_choice
        if _svc not in _staff_opts:
            st.session_state.rotina_assistant_view_choice = st.session_state.get(
                "data_source_mode", "auto"
            )
        st.radio(
            "O que usar nesta sessão",
            options=list(_staff_opts),
            key="rotina_assistant_view_choice",
            on_change=_rotina_workspace_radio_changed,
            format_func=lambda v: _staff_labels[str(v)],
        )
        if str(st.session_state.get("rotina_assistant_view_choice")) == "documents":
            st.caption(
                "Pedidos para **gravar** cadastro ou diário (CSV) continuam possíveis: "
                "descreva no chat o que quer inserir ou alterar. "
                "Para consultar só tabelas, prefira **Só dados estruturados** ou **Automático**."
            )
    else:
        cur = st.session_state.data_source_mode
        if cur not in _vals:
            cur = "auto"
            st.session_state.data_source_mode = cur
        sel = st.radio(
            "O que usar nesta sessão",
            options=_vals,
            index=_vals.index(cur),
            format_func=lambda v: _labels[str(v)],
        )
        st.session_state.data_source_mode = str(sel)
        if str(sel) == "documents":
            st.caption(
                "Pedidos para **gravar** cadastro ou diário (CSV) continuam possíveis: "
                "descreva no chat o que quer inserir ou alterar. "
                "Para consultar só tabelas, prefira **Só dados estruturados** ou **Automático**."
            )
    st.divider()
    st.markdown("**CrewAI — multi-agente (paralelo)**")
    try:
        from modules.rotina_crew.runner import crewai_import_ok as _rotina_crew_dep_ok

        _crew_dep = _rotina_crew_dep_ok()
    except Exception:
        _crew_dep = False
    if not ai_engine.use_openai_compatible_chat():
        st.caption(
            "CrewAI neste modo usa LangChain OpenAI: defina `ROTINA_CHAT_PROVIDER=openai` ou `openrouter` e chave API. "
            "Com **Ollama**, mantenha esta opção desligada."
        )
    elif not _crew_dep:
        st.caption("Instale: `pip install crewai langchain-openai`.")
    st.checkbox(
        "Orquestrar resposta com **CrewAI**",
        key="rotina_crewai_mode",
        disabled=not (_crew_dep and ai_engine.use_openai_compatible_chat()),
        help=(
            "Receção primeiro; especialistas relevantes seguem **em paralelo**; síntese final no fim. "
            "Mais lento e mais caro em tokens. Por agente: logs (`rotina.crew`, ex. `docker logs -f`). "
            "O planeamento SQL/RAG/mutações mantém-se antes da crew."
        ),
    )
    if st.button("Limpar conversa", key="rotina_clear_chat_btn"):
        _old_cid = _query_param_first(st.query_params.get(ROTINA_CHAT_QUERY_PARAM))
        if _old_cid and _is_safe_chat_session_id(_old_cid):
            try:
                (_chat_session_dir() / f"{_old_cid}.json").unlink(missing_ok=True)
            except OSError:
                pass
        st.session_state.messages = []
        st.session_state.last_rag_chunks = []
        st.session_state.last_rag_question = ""
        st.session_state.rotina_predictive_ml = False
        st.session_state.pop("rotina_voice_preview_bytes", None)
        st.session_state.rotina_voice_hash = ""
        st.session_state.pop("rotina_voice_unlock_mic_after_reply", None)
        st.session_state.pop("_rotina_session_serial", None)
        st.session_state.pop("_chat_disk_synced_for", None)
        st.session_state.sleep_rep_phase = "idle"
        st.session_state.sleep_rep_query_name = ""
        st.session_state.sleep_rep_resolved_label = ""
        st.session_state.sleep_rep_nome_field = ""
        st.query_params[ROTINA_CHAT_QUERY_PARAM] = str(uuid.uuid4())
        st.session_state.rotina_voice_input_key = (
            int(st.session_state.get("rotina_voice_input_key", 0)) + 1
        )
        st.rerun()

    st.divider()
    st.markdown("**Trechos (RAG)**")
    return st.empty()


def _render_rotina_logo_header() -> None:
    """Logo centrada (mesmo caminho `DATA_DIR`); usada acima da tabela em Gestão/Educador."""
    _logo_path = DATA_DIR / "logo_rotina_viva.png"
    _lg_l, _lg_m, _lg_r = st.columns([1, 1, 1])
    with _lg_m:
        if _logo_path.is_file():
            st.image(str(_logo_path), use_container_width=True)
        else:
            st.warning(
                f"Logo não encontrada: `{_logo_path.name}`. "
                "Coloque o arquivo em `ROTINA_DATA_DIR` (ex.: pasta `data/`)."
            )


def _rotina_append_user_chat_message(content: str) -> None:
    """Guarda o estado do interruptor «IA preditiva» no momento do envio (evita desincronizar com o Streamlit)."""
    st.session_state.messages.append(
        {
            "role": "user",
            "content": content,
            "predictive_ml": bool(st.session_state.get("rotina_predictive_ml")),
        }
    )


def render_rotina_chat(
    chat_id: str,
    conn: duckdb.DuckDBPyConnection | None,
    collection: Any,
    rag_sidebar_body: Any,
    *,
    read_only_db: bool,
    allow_mutations: bool,
    allow_delete_mutations: bool = True,
    parent_scope: tuple[int, str] | None,
    planner_extra: str,
    chat_extra_system: str | None,
    report_parent_lock: tuple[int, str] | None,
    show_logo: bool = True,
) -> None:
    """Área principal: logo (opcional), relatório, chat (texto + áudio), processamento SQL/RAG/mutação."""
    _rep_phase = st.session_state.get("sleep_rep_phase", "idle")
    _exp_relatorio = _rep_phase != "idle"
    if show_logo:
        _render_rotina_logo_header()
    with st.expander(
        "Gerar Relatório de Rotina",
        expanded=_exp_relatorio,
    ):
        render_sleep_meal_report_section(
            conn, chat_id, parent_lock=report_parent_lock
        )

    _ve = st.session_state.pop("rotina_voice_error", None)
    if _ve:
        st.warning(_ve)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", "") if isinstance(msg, dict) else msg["content"])

    _rotina_voice_spinner_slot = st.empty()

    _msgs = st.session_state.messages
    _last_is_user = bool(_msgs and _msgs[-1]["role"] == "user")

    _gutter_l, _center_wrap, _gutter_r = st.columns([1, 2.2, 1], gap="small")
    _voice_blob = None
    with _center_wrap:
        _pred_col, _icol, _vcol = st.columns([1.45, 4.55, 1], gap="small")
        with _pred_col:
            _p_on = bool(st.session_state.get("rotina_predictive_ml"))
            if _p_on:
                st.markdown(
                    '<div style="background:linear-gradient(145deg,#2e7d32,#1b5e20);color:#fff;'
                    "padding:9px 6px;border-radius:10px;font-size:12px;font-weight:700;text-align:center;"
                    'line-height:1.3;box-shadow:0 1px 4px rgba(0,0,0,.18);">IA preditiva<br/>'
                    '<span style="font-weight:600;font-size:11px;opacity:.95">ligada</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:linear-gradient(145deg,#6d4c41,#4e342e);color:#efebe9;'
                    "padding:9px 6px;border-radius:10px;font-size:12px;font-weight:700;text-align:center;"
                    'line-height:1.3;box-shadow:0 1px 4px rgba(0,0,0,.12);">IA preditiva<br/>'
                    '<span style="font-weight:600;font-size:11px;opacity:.92">desligada</span></div>',
                    unsafe_allow_html=True,
                )
            st.toggle(
                "IA preditiva",
                key="rotina_predictive_ml",
                help=(
                    "Ligado: **cada mensagem** é tratada como classificação de emoções (ML) até desligar; "
                    "não grava no diário/cadastro sem pedido explícito na mensagem. Desligado: chat normal."
                ),
                label_visibility="collapsed",
            )
        with _icol:
            _chat_ph = (
                "Frases para classificar com ML (modo ligado). Desligue o interruptor ao terminar."
                if bool(st.session_state.get("rotina_predictive_ml"))
                else "Pergunte sobre rotinas, alunos ou documentos da escola…"
            )
            prompt = st.chat_input(_chat_ph)
        with _vcol:
            _voice_preview = st.session_state.get("rotina_voice_preview_bytes")
            if _voice_preview is not None:
                st.audio(
                    _voice_preview, format=ai_engine.rotina_st_audio_format(_voice_preview)
                )
            elif hasattr(st, "audio_input"):
                _vk = int(st.session_state.get("rotina_voice_input_key", 0))
                _voice_blob = st.audio_input(
                    "🔊",
                    help=(
                        "Grave a pergunta; ao concluir, o áudio vira texto. "
                        "Se o som estiver fraco ou vazio na reprodução aqui, o Windows pode estar a usar "
                        "outro microfone do que o Chrome/Edge: no ícone do cadeado ou da barra de endereço, "
                        "abra as permissões do site e escolha o microfone certo (o mesmo do teste em Som)."
                    ),
                    key=f"rotina_chat_voice_{_vk}",
                )
            else:
                st.caption("Atualize o Streamlit (≥ 1.40) para gravar por voz.")

    pin_chat_footer_row()
    st.caption(ml_emotion_chat.emotion_command_help_caption())

    if _voice_blob is not None:
        _raw = _voice_blob.getvalue()
        if _raw:
            _vh = hashlib.sha256(_raw).hexdigest()
            if st.session_state.get("rotina_voice_hash") != _vh:
                _vname = getattr(_voice_blob, "name", None) or "gravacao.wav"
                with _rotina_voice_spinner_slot.container():
                    with st.spinner("Processando áudio…"):
                        _vtxt, _ver = ai_engine.transcribe_voice_bytes(_raw, _vname)
                if _vtxt:
                    st.session_state.rotina_voice_preview_bytes = _raw
                    st.session_state.rotina_voice_hash = _vh
                    _rotina_append_user_chat_message(_vtxt)
                    st.session_state.rotina_voice_unlock_mic_after_reply = True
                    persist_rotina_chat_to_disk(chat_id)
                else:
                    # Sem texto: não manter preview (senão o microfone fica oculto atrás do st.audio).
                    st.session_state.pop("rotina_voice_preview_bytes", None)
                    st.session_state.rotina_voice_hash = ""
                    st.session_state.rotina_voice_input_key = (
                        int(st.session_state.get("rotina_voice_input_key", 0)) + 1
                    )
                    if _ver == "__EMPTY_TRANSCRIPT__":
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "Não foi detectada fala neste áudio (silêncio ou volume muito baixo). "
                                    "Grave de novo; se o problema continuar, confira o microfone nas permissões do site."
                                ),
                            }
                        )
                        persist_rotina_chat_to_disk(chat_id)
                    else:
                        st.session_state.rotina_voice_error = _ver or (
                            "Não foi possível entender o áudio. Tente falar mais claro ou mais perto do microfone."
                        )
                st.rerun()

    if prompt:
        st.session_state.pop("rotina_voice_preview_bytes", None)
        st.session_state.rotina_voice_hash = ""
        st.session_state.pop("rotina_voice_unlock_mic_after_reply", None)
        st.session_state.rotina_voice_input_key = (
            int(st.session_state.get("rotina_voice_input_key", 0)) + 1
        )
        _rotina_append_user_chat_message(prompt)
        persist_rotina_chat_to_disk(chat_id)
        st.rerun()

    if _last_is_user:
        _last_user_msg = _msgs[-1]
        user_text = _last_user_msg["content"]
        _pred_ml = bool(_last_user_msg.get("predictive_ml"))
        mode_ds = st.session_state.data_source_mode

        if conn is None:
            err = "DuckDB indisponível. Verifique os CSVs em `ROTINA_DATA_DIR`."
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            history_for_model = _msgs[:-1]

            with st.chat_message("assistant"):
                _handled_name, _ok_name, _msg_name = try_gestao_delete_by_name_intent(
                    user_text,
                    DATA_DIR,
                    session_role=st.session_state.get("rotina_role"),
                    allow_delete_mutations=allow_delete_mutations,
                )
                if _handled_name:
                    if _ok_name:
                        st.cache_data.clear()
                        try:
                            get_duckdb_connection.clear()
                        except Exception:
                            pass
                        _conn_reload = get_duckdb_connection(
                            str(DATA_DIR), _duckdb_csv_reload_token(DATA_DIR)
                        )
                        _vblock = post_mutation_verification_block(
                            _conn_reload, "DELETE FROM info_alunos"
                        )
                        _full_name = build_mutation_direct_reply(
                            mut_sql="DELETE FROM info_alunos",
                            ok=True,
                            result_message=_msg_name,
                            duplicate_warn="",
                            duck_block=_vblock,
                        )
                        st.markdown(_full_name)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": _full_name}
                        )
                        persist_rotina_chat_to_disk(chat_id)
                        render_rag_sidebar_body(rag_sidebar_body)
                        return
                    st.error(_msg_name)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": _msg_name}
                    )
                    persist_rotina_chat_to_disk(chat_id)
                    render_rag_sidebar_body(rag_sidebar_body)
                    return

                ml_addon = ml_emotion_chat.build_emotion_ml_llm_addon(
                    user_text, DATA_DIR, predictive_session=_pred_ml
                )

                plan_force: str | None = None
                if mode_ds == "structured":
                    plan_force = "sql_only"
                elif mode_ds == "documents":
                    plan_force = "rag_only"

                progress_ui = st.empty()
                with progress_ui.container():
                    with st.status("Processando sua pergunta…", expanded=True) as proc:
                        proc.write("Analisando a pergunta e planejando consultas…")
                        if "```csv" in ml_addon:
                            proc.write(
                                "Modelo ML (emoções): inferência local concluída "
                                "(resultados vão para o contexto do assistente)."
                            )
                        if collection is None and mode_ds in ("auto", "documents"):
                            proc.write(
                                "Aviso: **ChromaDB / PDFs** indisponível — se a pergunta depender de documentos, "
                                "a parte de PDF fica vazia; **cadastro e diário (CSV)** continuam a funcionar."
                            )
                        planning_user_text = augment_cadastro_question_with_history(
                            user_text,
                            history_for_model,
                            parent_scope=parent_scope,
                        )
                        plan = normalize_plan(
                            ai_engine.llm_plan_sources(
                                planning_user_text,
                                force=plan_force,
                                history=history_for_model,
                                extra_planner_suffix=planner_extra,
                            ),
                            planning_user_text,
                        )
                        plan = apply_user_data_source_mode(plan, mode_ds)
                        plan = promote_plan_sql_mutation_field(plan)
                        plan = apply_infer_sql_to_plan(plan, planning_user_text)
                        if is_rag_nutrition_meals_scope_question(user_text):
                            # Cardápio/refeições: priorizar PDF nutricional e evitar SQL do diário
                            # (que só tem "comeu bem/pouco/recusou" e não o menu por dia).
                            plan = dict(plan)
                            plan["fontes"] = ["rag"]
                            plan["sql"] = None
                            proc.write(
                                "Escopo nutricional detectado: a resposta será baseada no PDF de planejamento nutricional (RAG)."
                            )
                        elif ml_emotion_chat.predictive_message_looks_emotional(user_text):
                            # Mesmo com IA preditiva desligada, incidentes emocionais/comportamentais
                            # precisam de base documental (saúde/segurança/convivência), não só SQL.
                            plan = dict(plan)
                            plan["fontes"] = ["rag"]
                            plan["sql"] = None
                            proc.write(
                                "Incidente emocional/comportamental detectado: priorizando documentos (RAG)."
                            )
                        if ml_emotion_chat.chat_round_suppress_csv_mutations(
                            user_text, predictive_session=_pred_ml
                        ):
                            plan = dict(plan)
                            plan["mutacao"] = None
                            _strip_sql = plan.get("sql")
                            if (
                                isinstance(_strip_sql, str)
                                and _strip_sql.strip()
                                and validate_mutation_sql(_strip_sql.strip())
                            ):
                                plan["sql"] = None
                        if read_only_db:
                            plan["mutacao"] = None
                        fontes = plan.get("fontes") or ["rag"]
                        if isinstance(fontes, str):
                            fontes = [fontes]

                        mutation_ok = False
                        mut_sql_done = ""
                        mutation_fail_detail = ""
                        mutation_duplicate_warn = ""
                        mutation_attempted = False
                        mutation_result_msg = ""
                        _conn = conn
                        mut_sql = plan.get("mutacao") if allow_mutations else None
                        # Planejador por vezes coloca INSERT/UPDATE/DELETE no campo "sql" em vez de "mutacao".
                        _planned_sql = plan.get("sql")
                        if allow_mutations and isinstance(_planned_sql, str) and _planned_sql.strip():
                            _pst = _planned_sql.strip()
                            if validate_mutation_sql(_pst):
                                _ms = (
                                    mut_sql.strip()
                                    if isinstance(mut_sql, str) and mut_sql.strip()
                                    else ""
                                )
                                if not _ms:
                                    mut_sql = _pst
                                    plan["sql"] = None
                                elif validate_mutation_sql(_ms):
                                    plan["sql"] = None
                        _mut_delete_blocked = False
                        if (
                            isinstance(mut_sql, str)
                            and mut_sql.strip()
                            and not allow_delete_mutations
                            and re.search(
                                r"\bdelete\s+from\b",
                                mut_sql.strip(),
                                re.IGNORECASE,
                            )
                        ):
                            mutation_attempted = True
                            mutation_fail_detail = (
                                "O perfil **Educador** não pode apagar no cadastro — só **Gestão** pode executar DELETE. "
                                "O CSV **não foi alterado**."
                            )
                            _mut_delete_blocked = True
                        if (
                            not _mut_delete_blocked
                            and isinstance(mut_sql, str)
                            and mut_sql.strip()
                            and _conn is not None
                        ):
                            mutation_attempted = True
                            proc.write("Aplicando alteração nos dados (CSV)…")
                            _mmsg, mok, _dup_w = run_mutation_and_persist(
                                _conn,
                                mut_sql.strip(),
                                DATA_DIR,
                                allow_delete=allow_delete_mutations,
                            )
                            mutation_result_msg = _mmsg
                            proc.write(_mmsg)
                            if _dup_w:
                                mutation_duplicate_warn = _dup_w
                            if mok:
                                mutation_ok = True
                                mut_sql_done = mut_sql.strip()
                                if re.search(
                                    r"\bdelete\s+from\b",
                                    mut_sql_done,
                                    re.IGNORECASE,
                                ):
                                    st.cache_data.clear()
                                _conn = get_duckdb_connection(
                                    str(DATA_DIR), _duckdb_csv_reload_token(DATA_DIR)
                                )
                            else:
                                mutation_fail_detail = _mmsg

                        duck_block = "(nenhuma consulta SQL executada)"
                        if "sql" in fontes:
                            sql = plan.get("sql")
                            if isinstance(sql, str) and sql.strip():
                                sql_use = sql.strip()
                                if parent_scope is not None:
                                    sql_use = apply_parent_sql_scope(
                                        sql_use, parent_scope[0]
                                    )
                                proc.write(
                                    _processing_status_sql_line(user_text, sql_use)
                                )
                                duck_block, ok = run_safe_select(_conn, sql_use)
                                if not ok:
                                    duck_block = (
                                        f"{duck_block}\n"
                                        "(Tente reformular com nome do aluno ou data, se aplicável.)"
                                    )
                            else:
                                proc.write("CSV — preparando consulta nas tabelas…")
                                duck_block = (
                                    "Nenhuma consulta SQL válida foi gerada para esta pergunta."
                                )

                        if mutation_fail_detail:
                            if _mut_delete_blocked:
                                duck_block = (
                                    "=== Remoção não aplicada (permissão / perfil) ===\n"
                                    f"{mutation_fail_detail}\n\n"
                                    "---\n\n"
                                    + duck_block
                                )
                            else:
                                if mutation_fail_detail.strip() == _mensagem_csv_aberto_simples():
                                    duck_block = (
                                        "=== A alteração aos dados NÃO foi gravada nos CSV ===\n"
                                        f"{mutation_fail_detail}\n\n---\n\n"
                                        + duck_block
                                    )
                                else:
                                    duck_block = (
                                        "=== A alteração aos dados NÃO foi gravada nos CSV ===\n"
                                        f"{mutation_fail_detail}\n\n"
                                        "Esta falha costuma ocorrer quando o ficheiro está aberto no Excel ou noutro editor. "
                                        "Feche o CSV, guarde se necessário, e volte a pedir a alteração no chat.\n\n"
                                        "---\n\n"
                                        + duck_block
                                    )

                        if mutation_ok and mut_sql_done and _conn is not None:
                            proc.write("Verificando estado após alteração nos CSV…")
                            vblock = post_mutation_verification_block(
                                _conn, mut_sql_done
                            )
                            if mutation_duplicate_warn:
                                vblock = (
                                    "=== Aviso de duplicado (nome ou contacto já existia no cadastro antes desta gravação) ===\n"
                                    f"{mutation_duplicate_warn}\n\n"
                                    "---\n\n"
                                    + vblock
                                )
                            _db_placeholder = duck_block.strip().startswith(
                                "(nenhuma"
                            ) or "Nenhuma consulta SQL válida" in duck_block
                            if _db_placeholder:
                                duck_block = vblock
                            else:
                                duck_block = (
                                    vblock
                                    + "\n\n---\n\n=== Consulta adicional do plano ===\n\n"
                                    + duck_block
                                )

                        rag_question = user_text
                        if parent_scope is not None:
                            rag_question = augment_question_for_parent_rag(
                                user_text, parent_scope[0], parent_scope[1]
                            )

                        rag_block = "(busca em documentos não solicitada)"
                        if "rag" in fontes and collection is not None:
                            proc.write(_processing_status_rag_line(user_text))
                            rag_block, _rag_chunks = retrieve_rag_context_and_chunks(
                                collection, rag_question, k=ai_engine.rag_context_chunks_top_k()
                            )
                            st.session_state.last_rag_chunks = _rag_chunks
                            st.session_state.last_rag_question = user_text
                        else:
                            st.session_state.last_rag_chunks = []
                            st.session_state.last_rag_question = ""
                            if "rag" in fontes:
                                proc.write(
                                    "PDF — indisponível no momento (índice ou ambiente)."
                                )

                        if (
                            ai_engine.use_openai_compatible_chat()
                            and ROTINA_API_PLAN_TO_CHAT_DELAY_SEC > 0
                        ):
                            time.sleep(ROTINA_API_PLAN_TO_CHAT_DELAY_SEC)

                _extra_chat = (chat_extra_system or "").strip()
                if ml_addon.strip():
                    _extra_chat = (
                        (_extra_chat + "\n\n") if _extra_chat else ""
                    ) + ml_addon.strip()
                if mutation_ok:
                    _extra_chat = (
                        (_extra_chat + "\n\n") if _extra_chat else ""
                    ) + ai_engine.system_mutation_applied()
                    if mutation_duplicate_warn:
                        _extra_chat += "\n\n" + ai_engine.system_duplicate_cadastro()
                elif mutation_fail_detail:
                    _extra_chat = (
                        (_extra_chat + "\n\n") if _extra_chat else ""
                    ) + (
                        ai_engine.system_duplicate_cadastro()
                        if mutation_duplicate_warn
                        else ai_engine.system_mutation_failed()
                    )

                if mutation_attempted and isinstance(mut_sql, str) and mut_sql.strip():
                    full = build_mutation_direct_reply(
                        mut_sql=mut_sql.strip(),
                        ok=mutation_ok,
                        result_message=mutation_result_msg or mutation_fail_detail,
                        duplicate_warn=mutation_duplicate_warn,
                        duck_block=duck_block,
                    )
                    st.markdown(full)
                    progress_ui.empty()
                    st.session_state.messages.append({"role": "assistant", "content": full})
                    persist_rotina_chat_to_disk(chat_id)
                    render_rag_sidebar_body(rag_sidebar_body)
                    return

                _planned_mut_raw = plan.get("mutacao")
                _has_planned_mutacao = isinstance(_planned_mut_raw, str) and bool(
                    _planned_mut_raw.strip()
                )
                if (
                    _user_requests_student_delete(user_text)
                    and not _has_planned_mutacao
                ):
                    full = _reply_delete_not_persisted_no_mutation(
                        perfil_educador=not allow_delete_mutations
                    )
                    st.markdown(full)
                    progress_ui.empty()
                    st.session_state.messages.append({"role": "assistant", "content": full})
                    persist_rotina_chat_to_disk(chat_id)
                    render_rag_sidebar_body(rag_sidebar_body)
                    return

                _use_crew = bool(st.session_state.get("rotina_crewai_mode")) and (
                    ai_engine.use_openai_compatible_chat()
                )
                _crew_ok = False
                try:
                    from modules.rotina_crew.runner import crewai_import_ok as _crew_imp_ok

                    _crew_ok = _crew_imp_ok()
                except Exception:
                    _crew_ok = False
                if _use_crew and _crew_ok:
                    from modules.rotina_crew.runner import run_rotina_crew_chat as _run_rotina_crew

                    try:
                        with st.spinner("CrewAI a orquestrar agentes…"):
                            _cr_out = _run_rotina_crew(
                                user_text=user_text,
                                duck_block=duck_block,
                                rag_block=rag_block,
                                ml_addon=ml_addon or "",
                                predictive_ml=_pred_ml,
                                conn=_conn,
                                data_dir=DATA_DIR,
                                collection=collection,
                            )
                        full = _cr_out.final_markdown
                        st.markdown(full)
                    except Exception as _crew_exc:
                        st.warning(f"CrewAI falhou ({_crew_exc!s}); a usar resposta em **streaming**.")
                        def _gen() -> Generator[str, None, None]:
                            yield from ai_engine.processar_resposta_chat_stream(
                                user_text,
                                duck_block,
                                rag_block,
                                history_for_model,
                                extra_system=_extra_chat or None,
                            )

                        _streamed = st.write_stream(_gen()) or ""
                        full = (
                            _streamed
                            if isinstance(_streamed, str)
                            else "".join(str(x) for x in _streamed)
                        )
                else:
                    if _use_crew and not _crew_ok:
                        st.caption(
                            "Modo CrewAI ligado, mas o pacote `crewai` não está instalado — "
                            "a usar resposta em streaming único. `pip install crewai langchain-openai`."
                        )

                    def _gen() -> Generator[str, None, None]:
                        yield from ai_engine.processar_resposta_chat_stream(
                            user_text,
                            duck_block,
                            rag_block,
                            history_for_model,
                            extra_system=_extra_chat or None,
                        )

                    _streamed = st.write_stream(_gen()) or ""
                    full = (
                        _streamed
                        if isinstance(_streamed, str)
                        else "".join(str(x) for x in _streamed)
                    )
                progress_ui.empty()

            _asst_msg: dict[str, Any] = {"role": "assistant", "content": full}
            st.session_state.messages.append(_asst_msg)

    if st.session_state.get("rotina_voice_unlock_mic_after_reply") and (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "assistant"
    ):
        st.session_state.pop("rotina_voice_unlock_mic_after_reply", None)
        st.session_state.pop("rotina_voice_preview_bytes", None)
        st.session_state.rotina_voice_hash = ""
        st.session_state.rotina_voice_input_key = (
            int(st.session_state.get("rotina_voice_input_key", 0)) + 1
        )
        st.rerun()

    persist_rotina_chat_to_disk(chat_id)
    render_rag_sidebar_body(rag_sidebar_body)

def _render_gestao_ou_educador(
    *,
    allow_mutations: bool,
    read_only_db: bool,
    allow_delete_mutations: bool = True,
    planner_extra: str,
) -> None:
    _chat_id = ensure_rotina_chat_session_id()
    sync_rotina_chat_from_disk(_chat_id)
    _screen = st.session_state.get("rotina_sidebar_screen", "assistant")
    with st.sidebar:
        render_auth_sidebar()
        rag_sidebar_body = (
            render_chat_sidebar_internals()
            if _screen == "assistant"
            else st.empty()
        )

    if _screen == "ml_traditional":
        from ui.ml_traditional_page import render_ml_traditional_page

        _render_rotina_logo_header()
        render_ml_traditional_page()
        return

    try:
        conn = get_duckdb_connection(str(DATA_DIR), _duckdb_csv_reload_token(DATA_DIR))
    except Exception as e:
        st.error(f"Falha ao carregar DuckDB: {e}")
        conn = None

    collection = None
    _need_chroma = st.session_state.data_source_mode in ("auto", "documents")
    if conn is not None and _need_chroma:
        try:
            collection = get_chroma_collection(
                str(CHROMA_DIR),
                str(DATA_DIR),
                INDEX_PROFILE,
            )
            if collection.count() == 0:
                st.warning(
                    "Índice Chroma vazio: adicione os PDFs em `ROTINA_DATA_DIR` e reinicie para indexar."
                )
        except Exception as e:
            st.error(f"Falha ao preparar ChromaDB/embeddings: {e}")

    if _screen == "direct_chat":
        render_direct_family_educator_chat(conn)
        return

    _render_rotina_logo_header()
    if st.session_state.get("rotina_role") == "gestao":
        with st.expander("Resetar banco de conhecimento (RAG)", expanded=False):
            st.caption(
                "Apaga o índice ChromaDB em disco e força nova leitura dos PDFs em ROTINA_DATA_DIR. "
                "Use após trocar ou substituir documentos."
            )
            if st.button(
                "Resetar Banco de Conhecimento",
                key="rotina_reset_rag_knowledge",
                type="primary",
            ):
                reset_rotina_chroma_persist(CHROMA_DIR)
                try:
                    get_chroma_collection.cache_clear()
                except Exception:
                    pass
                st.success("Índice RAG removido. A página vai recarregar.")
                st.rerun()
    st.subheader("Alunos cadastrados")
    if conn is not None:
        try:
            _df_alunos = conn.execute(
                "SELECT id_aluno, nome, turma, alergias FROM info_alunos ORDER BY nome"
            ).fetchdf()
            # ~metade das linhas visíveis vs. altura por omissão (~10 linhas → ~5)
            _n = len(_df_alunos)
            _row_px = 36
            _header_px = 44
            _max_vis = 5
            _vis = max(2, min(_n, _max_vis)) if _n else 2
            _table_h = min(320, _header_px + _vis * _row_px)
            st.dataframe(
                _df_alunos,
                use_container_width=True,
                hide_index=True,
                height=_table_h,
            )
        except Exception as ex:
            st.warning(str(ex))
    else:
        st.info("Sem ligação ao DuckDB — verifique os CSVs.")

    st.divider()
    render_rotina_chat(
        _chat_id,
        conn,
        collection,
        rag_sidebar_body,
        read_only_db=read_only_db,
        allow_mutations=allow_mutations,
        allow_delete_mutations=allow_delete_mutations,
        parent_scope=None,
        planner_extra=planner_extra,
        chat_extra_system=None,
        report_parent_lock=None,
        show_logo=False,
    )


def render_gestao() -> None:
    _render_gestao_ou_educador(
        allow_mutations=True,
        read_only_db=False,
        allow_delete_mutations=True,
        planner_extra=_planner_suffix_gestao(),
    )


def render_educador() -> None:
    _edu = educador_rotina_csv_access()
    _render_gestao_ou_educador(
        read_only_db=_edu["read_only_db"],
        allow_mutations=_edu["allow_mutations"],
        allow_delete_mutations=_edu["allow_delete_mutations"],
        planner_extra=_edu["planner_extra"],
    )


def render_familia() -> None:
    _chat_id = ensure_rotina_chat_session_id()
    sync_rotina_chat_from_disk(_chat_id)
    _screen = st.session_state.get("rotina_sidebar_screen", "assistant")
    with st.sidebar:
        render_auth_sidebar()
        rag_sidebar_body = (
            render_chat_sidebar_internals()
            if _screen == "assistant"
            else st.empty()
        )

    if _screen == "ml_traditional":
        st.session_state.rotina_sidebar_screen = "assistant"
        st.warning("O laboratório ML clássico está disponível apenas para gestão e educadores.")
        st.rerun()

    try:
        conn = get_duckdb_connection(str(DATA_DIR), _duckdb_csv_reload_token(DATA_DIR))
    except Exception as e:
        st.error(f"Falha ao carregar DuckDB: {e}")
        conn = None

    aid = st.session_state.get("rotina_parent_id_aluno")
    nome_filho = "—"
    if conn is not None and isinstance(aid, int):
        try:
            r = conn.execute(
                "SELECT nome FROM info_alunos WHERE id_aluno = ? LIMIT 1",
                [aid],
            ).fetchone()
            if r:
                nome_filho = str(r[0])
        except Exception:
            pass

    if _screen == "direct_chat":
        render_direct_family_educator_chat(conn)
        return

    st.info(
        f"Consulta restrita ao aluno **{nome_filho}** (id_aluno={aid}). "
        "Não é possível alterar o cadastro ou o diário a partir deste perfil."
    )
    st.divider()

    collection = None
    _need_chroma = st.session_state.data_source_mode in ("auto", "documents")
    if conn is not None and _need_chroma:
        try:
            collection = get_chroma_collection(
                str(CHROMA_DIR),
                str(DATA_DIR),
                INDEX_PROFILE,
            )
            if collection.count() == 0:
                st.warning(
                    "Índice Chroma vazio: adicione os PDFs em `ROTINA_DATA_DIR` e reinicie para indexar."
                )
        except Exception as e:
            st.error(f"Falha ao preparar ChromaDB/embeddings: {e}")

    _ps: tuple[int, str] | None = None
    if isinstance(aid, int):
        _ps = (aid, nome_filho)

    render_rotina_chat(
        _chat_id,
        conn,
        collection,
        rag_sidebar_body,
        read_only_db=True,
        allow_mutations=False,
        allow_delete_mutations=False,
        parent_scope=_ps,
        planner_extra=_planner_suffix_familia(aid, nome_filho)
        if isinstance(aid, int)
        else "",
        chat_extra_system=_chat_system_familia(aid, nome_filho)
        if isinstance(aid, int)
        else None,
        report_parent_lock=_ps,
    )
