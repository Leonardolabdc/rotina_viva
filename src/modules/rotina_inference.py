"""
Inferência do chat Rotina Viva sem Streamlit (testes DeepEval, CLI, scripts).

Replica o fluxo principal de `ui/components.render_rotina_chat`: planeamento,
DuckDB, RAG (Chroma) e resposta via CrewAI ou streaming (`ai_engine`).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

import duckdb

from core.auth_manager import _planner_suffix_gestao
from core.database import (
    DATA_DIR,
    _duckdb_csv_reload_token,
    _mensagem_csv_aberto_simples,
    get_duckdb_connection,
    post_mutation_verification_block,
    promote_plan_sql_mutation_field,
    run_mutation_and_persist,
    run_safe_select,
    validate_mutation_sql,
)
from modules import ai_engine
from modules import ml_emotion_chat
from modules.chat_service import (
    _processing_status_sql_line,
    _reply_delete_not_persisted_no_mutation,
    _user_requests_student_delete,
    apply_infer_sql_to_plan,
    apply_parent_sql_scope,
    familia_student_query_blocked_message,
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
    retrieve_rag_context_and_chunks,
)


def _env_truthy(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _finalize_assistant_reply(text: str, duck_block: str = "") -> str:
    """Aplica pipeline de saída (guardrails + PII)."""
    from core.guardrails import run_output_guardrails

    safe, _ = run_output_guardrails(text or "", duck_block=duck_block or "")
    return safe


def run_rotina_chat_inference(
    user_text: str,
    *,
    predictive_ml: bool = False,
    data_source_mode: str | None = None,
    use_crewai: bool | None = None,
    parent_scope: tuple[int, str] | None = None,
    history: list[dict[str, str]] | None = None,
    chat_extra_system: str | None = None,
    planner_extra: str | None = None,
    allow_mutations: bool = False,
    allow_delete_mutations: bool = True,
    read_only_db: bool = True,
) -> str:
    """
    Devolve o Markdown/texto final da resposta do assistente (como no Streamlit).

    Por defeito não executa mutações CSV (`allow_mutations=False`) — adequado a golden tests.
    CrewAI só corre se `use_crewai=True` ou `ROTINA_EVAL_CREWAI_MODE=true`.
    """
    um = (user_text or "").strip()
    if not um:
        return ""

    from core.guardrails import run_input_guardrails

    _recent = [
        str(m.get("content") or "")
        for m in (history or [])
        if m.get("role") == "user" and (m.get("content") or "").strip()
    ]
    _guard = run_input_guardrails(um, recent_user_messages=_recent)
    if not _guard.allowed:
        return _guard.user_message or "Mensagem bloqueada por política de segurança."

    history_for_model = history or []
    mode_ds = (data_source_mode or os.getenv("ROTINA_EVAL_DATA_SOURCE_MODE") or "auto").strip()
    use_crew = (
        bool(use_crewai) if use_crewai is not None else _env_truthy("ROTINA_EVAL_CREWAI_MODE")
    )
    _planner = planner_extra if planner_extra is not None else _planner_suffix_gestao()

    try:
        conn = get_duckdb_connection(str(DATA_DIR), _duckdb_csv_reload_token(DATA_DIR))
    except Exception as e:
        return (
            "DuckDB indisponível. Verifique os CSVs em `ROTINA_DATA_DIR` e se não há erro de caminho.\n\n"
            f"_(detalhe técnico: {e})_"
        )

    collection = None
    if mode_ds in ("auto", "documents"):
        try:
            collection = get_chroma_collection(str(CHROMA_DIR), str(DATA_DIR), INDEX_PROFILE)
        except Exception:
            collection = None

    ml_addon = ml_emotion_chat.build_emotion_ml_llm_addon(
        um, DATA_DIR, predictive_session=predictive_ml
    )

    plan_force: str | None = None
    if mode_ds == "structured":
        plan_force = "sql_only"
    elif mode_ds == "documents":
        plan_force = "rag_only"

    planning_user_text = augment_cadastro_question_with_history(
        um, history_for_model, parent_scope=parent_scope
    )
    if parent_scope is not None:
        _fam_block = familia_student_query_blocked_message(um, parent_scope)
        if _fam_block:
            return _finalize_assistant_reply(_fam_block, "")
    plan = normalize_plan(
        ai_engine.llm_plan_sources(
            planning_user_text,
            force=plan_force,
            history=history_for_model,
            extra_planner_suffix=_planner,
        ),
        planning_user_text,
    )
    plan = apply_user_data_source_mode(plan, mode_ds)
    plan = promote_plan_sql_mutation_field(plan)
    plan = apply_infer_sql_to_plan(plan, um)

    # Perguntas de cardápio/refeições devem privilegiar o PDF nutricional no RAG.
    # Se deixarmos SQL aqui, a Crew tende a responder "comeu bem/pouco/recusou" (diário),
    # que não responde "o que tem de almoço na Quarta-feira".
    if is_rag_nutrition_meals_scope_question(um):
        plan = dict(plan)
        plan["fontes"] = ["rag"]
        plan["sql"] = None

    # Incidentes socioemocionais/comportamentais ("machucou", "chorou", etc.)
    # não devem ficar só em SQL do diário; precisam de base documental (procedimentos).
    if ml_emotion_chat.predictive_message_looks_emotional(um):
        plan = dict(plan)
        plan["fontes"] = ["rag"]
        plan["sql"] = None

    if ml_emotion_chat.chat_round_suppress_csv_mutations(um, predictive_session=predictive_ml):
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

    _planned_sql = plan.get("sql")
    if allow_mutations and isinstance(_planned_sql, str) and _planned_sql.strip():
        _pst = _planned_sql.strip()
        if validate_mutation_sql(_pst):
            _ms = mut_sql.strip() if isinstance(mut_sql, str) and mut_sql.strip() else ""
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
        and re.search(r"\bdelete\s+from\b", mut_sql.strip(), re.IGNORECASE)
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
        and allow_mutations
    ):
        mutation_attempted = True
        _mmsg, mok, _dup_w = run_mutation_and_persist(
            _conn,
            mut_sql.strip(),
            DATA_DIR,
            allow_delete=allow_delete_mutations,
        )
        mutation_result_msg = _mmsg
        if _dup_w:
            mutation_duplicate_warn = _dup_w
        if mok:
            mutation_ok = True
            mut_sql_done = mut_sql.strip()
            _conn = get_duckdb_connection(str(DATA_DIR), _duckdb_csv_reload_token(DATA_DIR))
        else:
            mutation_fail_detail = _mmsg

    duck_block = "(nenhuma consulta SQL executada)"
    if "sql" in fontes:
        sql = plan.get("sql")
        if isinstance(sql, str) and sql.strip():
            sql_use = sql.strip()
            if parent_scope is not None:
                sql_use = apply_parent_sql_scope(sql_use, parent_scope[0])
            _ = _processing_status_sql_line(um, sql_use)
            duck_block, ok = run_safe_select(_conn, sql_use)
            if not ok:
                duck_block = (
                    f"{duck_block}\n(Tente reformular com nome do aluno ou data, se aplicável.)"
                )
        else:
            duck_block = "Nenhuma consulta SQL válida foi gerada para esta pergunta."

    if mutation_fail_detail:
        if _mut_delete_blocked:
            duck_block = (
                "=== Remoção não aplicada (permissão / perfil) ===\n"
                f"{mutation_fail_detail}\n\n---\n\n"
                + duck_block
            )
        elif mutation_fail_detail.strip() == _mensagem_csv_aberto_simples():
            duck_block = (
                "=== A alteração aos dados NÃO foi gravada nos CSV ===\n"
                f"{mutation_fail_detail}\n\n---\n\n"
                + duck_block
            )
        else:
            duck_block = (
                "=== A alteração aos dados NÃO foi gravada nos CSV ===\n"
                f"{mutation_fail_detail}\n\n"
                "Esta falha costuma ocorrer quando o ficheiro está aberto no Excel ou noutro editor.\n\n"
                "---\n\n"
                + duck_block
            )

    if mutation_ok and mut_sql_done and _conn is not None:
        vblock = post_mutation_verification_block(_conn, mut_sql_done)
        if mutation_duplicate_warn:
            vblock = (
                "=== Aviso de duplicado (nome ou contacto já existia no cadastro antes desta gravação) ===\n"
                f"{mutation_duplicate_warn}\n\n---\n\n"
                + vblock
            )
        _db_placeholder = duck_block.strip().startswith("(nenhuma") or "Nenhuma consulta SQL válida" in duck_block
        if _db_placeholder:
            duck_block = vblock
        else:
            duck_block = (
                vblock + "\n\n---\n\n=== Consulta adicional do plano ===\n\n" + duck_block
            )

    rag_question = um
    if parent_scope is not None:
        rag_question = augment_question_for_parent_rag(um, parent_scope[0], parent_scope[1])

    rag_block = "(busca em documentos não solicitada)"
    if "rag" in fontes and collection is not None:
        rag_block, _chunks = retrieve_rag_context_and_chunks(
            collection, rag_question, k=ai_engine.rag_context_chunks_top_k()
        )
    elif "rag" in fontes and collection is None:
        pass

    if ai_engine.use_openai_compatible_chat() and ROTINA_API_PLAN_TO_CHAT_DELAY_SEC > 0:
        time.sleep(ROTINA_API_PLAN_TO_CHAT_DELAY_SEC)

    _extra_chat = (chat_extra_system or "").strip()
    if ml_addon.strip():
        _extra_chat = ((_extra_chat + "\n\n") if _extra_chat else "") + ml_addon.strip()
    if mutation_ok:
        _extra_chat = ((_extra_chat + "\n\n") if _extra_chat else "") + ai_engine.system_mutation_applied()
        if mutation_duplicate_warn:
            _extra_chat += "\n\n" + ai_engine.system_duplicate_cadastro()
    elif mutation_fail_detail:
        _extra_chat = ((_extra_chat + "\n\n") if _extra_chat else "") + (
            ai_engine.system_duplicate_cadastro()
            if mutation_duplicate_warn
            else ai_engine.system_mutation_failed()
        )

    if mutation_attempted and isinstance(mut_sql, str) and mut_sql.strip():
        return build_mutation_direct_reply(
            mut_sql=mut_sql.strip(),
            ok=mutation_ok,
            result_message=mutation_result_msg or mutation_fail_detail,
            duplicate_warn=mutation_duplicate_warn,
            duck_block=duck_block,
        )

    _planned_mut_raw = plan.get("mutacao")
    _has_planned_mutacao = isinstance(_planned_mut_raw, str) and bool(_planned_mut_raw.strip())
    if _user_requests_student_delete(um) and not _has_planned_mutacao:
        return _reply_delete_not_persisted_no_mutation(perfil_educador=not allow_delete_mutations)

    _crew_ok = False
    try:
        from modules.rotina_crew.runner import crewai_import_ok as _crew_imp_ok

        _crew_ok = _crew_imp_ok()
    except Exception:
        _crew_ok = False

    if use_crew and _crew_ok:
        from modules.rotina_crew.runner import run_rotina_crew_chat

        try:
            _cr_out = run_rotina_crew_chat(
                user_text=um,
                duck_block=duck_block,
                rag_block=rag_block,
                ml_addon=ml_addon or "",
                predictive_ml=predictive_ml,
                conn=_conn,
                data_dir=DATA_DIR,
                collection=collection,
            )
            return _finalize_assistant_reply((_cr_out.final_markdown or "").strip(), duck_block)
        except Exception:
            pass

    return _finalize_assistant_reply(
        "".join(
            ai_engine.processar_resposta_chat_stream(
                um,
                duck_block,
                rag_block,
                history_for_model,
                extra_system=_extra_chat or None,
            )
        ).strip(),
        duck_block,
    )
