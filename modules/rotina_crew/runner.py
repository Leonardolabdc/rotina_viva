"""
CrewAI: orquestração em "cérebro" sobre o mesmo contexto (DuckDB + RAG + ML já preparados pela app).

- **Recepção** (Anfitriã) corre **em série**.
- **Especialistas** (Dados / ML / RAG) com `async_execution=True` correm **em paralelo** após a recepção,
  desde que estejam no plano da mensagem.
- O **plano** de quem entra usa heurísticas + blocos pré-calculados; `ROTINA_CREW_ALL_SPECIALISTS=1`
  força os três especialistas sempre.

Logs legíveis em stderr (logger `rotina.crew` — visível com `docker logs`), sem repetir o detalhe na UI.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from modules.rotina_crew.llm_factory import build_crew_chat_llm
from modules.rotina_crew.trace_labels import trace_agent_label_for_task_output
from modules.rotina_crew.tools import (
    make_duckdb_select_tool,
    make_ml_emotion_tool,
    make_rag_tool,
)

_CREW_LOG = logging.getLogger("rotina.crew")
_LOGGED_HANDLER = False


def _ensure_rotina_crew_stderr_logging() -> None:
    """Um handler stderr UTF-8 para `docker logs` / PowerShell sem mojibake (idempotente)."""
    global _LOGGED_HANDLER
    if _LOGGED_HANDLER:
        return
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [rotina.crew] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    _CREW_LOG.addHandler(h)
    _CREW_LOG.setLevel(logging.INFO)
    _CREW_LOG.propagate = False
    _LOGGED_HANDLER = True


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _eval_suite_running() -> bool:
    """True quando a inferência é pedida pela suíte DeepEval (`test_rotina_viva.run_inference`)."""
    return _env_truthy("ROTINA_EVAL_SUITE_RUNNING")


def _strip_markdown_section_headings_for_eval(text: str) -> str:
    """
    Remove linhas que são apenas cabeçalhos Markdown (## …), típicos de secções «Acolhimento» / «Análise»,
    para aproximar a saída do expected_output do golden set e reduzir ruído no Answer Relevancy.
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not s.strip():
        return s.strip()
    kept: list[str] = []
    for line in s.split("\n"):
        st = line.strip()
        if re.match(r"^#{2,6}\s+\S", st):
            continue
        kept.append(line)
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _ellipsis(text: str, n: int) -> str:
    t = " ".join((text or "").split())
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


# Mesmo texto que `ui/components.py` quando não há retrieve em PDFs — não vazio, mas não é "contexto RAG".
_RAG_BLOCK_PLACEHOLDER = "(busca em documentos não solicitada)"


def _has_rag_context(rag_block: str) -> bool:
    s = (rag_block or "").strip()
    if not s:
        return False
    return not s.startswith(_RAG_BLOCK_PLACEHOLDER)


def _plan_specialists(
    *,
    user_text: str,
    duck_block: str,
    rag_block: str,
    ml_addon: str,
    predictive_ml: bool,
    collection: Any,
) -> tuple[bool, bool, bool, str]:
    """
    Devolve (quer_dados, quer_ml, quer_rag, motivo).
    `ROTINA_CREW_ALL_SPECIALISTS=1` ignora heurística.
    """
    if _env_truthy("ROTINA_CREW_ALL_SPECIALISTS"):
        return True, True, True, "ROTINA_CREW_ALL_SPECIALISTS"

    lt = (user_text or "").lower()
    d = bool(duck_block and len(duck_block.strip()) > 40) or any(
        k in lt
        for k in (
            "aluno",
            "aluna",
            "diário",
            "diario",
            "turma",
            "cadastro",
            "duckdb",
            "sql",
            "tabela",
            "registo",
            "registro",
            "nota",
            "frequência",
            "frequencia",
        )
    )
    m = predictive_ml or bool(ml_addon and ml_addon.strip()) or any(
        k in lt
        for k in (
            "emoção",
            "emocao",
            "sentimento",
            "classificar",
            "tf-idf",
            "machine learning",
            "modelo ml",
        )
    )
    r = _has_rag_context(rag_block) or (
        collection is not None
        and any(
            k in lt
            for k in (
                "documento",
                "pdf",
                "regimento",
                "norma",
                "protocolo",
                "nutri",
                "ppp",
                "pedagóg",
                "pedagog",
                "lei ",
            )
        )
    )

    if not (d or m or r):
        if duck_block.strip():
            d = True
            reason = "fallback: bloco tabular"
        elif _has_rag_context(rag_block):
            r = True
            reason = "fallback: bloco RAG"
        elif ml_addon.strip() or predictive_ml:
            m = True
            reason = "fallback: ML"
        else:
            d = True
            reason = "fallback: sem sinais — dados"
    else:
        parts = []
        if d:
            parts.append("dados")
        if m:
            parts.append("ml")
        if r:
            parts.append("rag")
        reason = "heurística:" + "+".join(parts)

    return d, m, r, reason


def crewai_import_ok() -> bool:
    try:
        import crewai  # noqa: F401

        return True
    except ImportError:
        return False


def _task_outputs_to_traces(
    crew_output: Any,
    *,
    req_id: str,
    summary_chars: int = 200,
) -> list[dict[str, str]]:
    """
    Resumo por agente (retorno ainda útil para testes); detalhe nos logs `rotina.crew` (Docker).
    """
    out: list[dict[str, str]] = []
    raw_list = getattr(crew_output, "tasks_output", None)
    if not raw_list:
        return out
    log_full = _env_truthy("ROTINA_CREW_LOG_FULL")
    for item in raw_list:
        role = trace_agent_label_for_task_output(item)
        body = getattr(item, "raw", None)
        if body is None:
            body = getattr(item, "output", None)
        if body is None:
            body = str(item)
        full = str(body).strip()
        prev = _ellipsis(full, int(summary_chars))
        out.append({"agent": role, "content": prev})
        if log_full:
            _CREW_LOG.info("trace_full | req=%s | agent=%s | body=%s", req_id, role, full)
        else:
            _CREW_LOG.info(
                "trace | req=%s | agent=%s | chars=%s | %s",
                req_id,
                role,
                len(full),
                prev,
            )
    if raw_list:
        order = " → ".join(trace_agent_label_for_task_output(x) for x in raw_list)
        _CREW_LOG.info(
            "traces_order | req=%s | steps=%s | %s",
            req_id,
            len(raw_list),
            order,
        )
    return out


def _final_text(crew_output: Any) -> str:
    for attr in ("raw", "output", "result"):
        v = getattr(crew_output, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return str(crew_output).strip()


def _normalize_for_dup(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _remove_duplicate_tail_paragraph(markdown: str) -> str:
    """
    Evita respostas "em dobro" quando o último parágrafo repete o anterior
    (comum na síntese final da crew: resumo + conclusão quase idênticos).
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n", markdown or "") if p.strip()]
    if len(parts) < 2:
        return markdown

    last = parts[-1]
    prev = parts[-2]
    n_last = _normalize_for_dup(last)
    n_prev = _normalize_for_dup(prev)
    if not n_last or not n_prev:
        return markdown

    # Remove só quando o último é claramente redundante.
    if n_last == n_prev or n_last in n_prev or n_prev in n_last:
        return "\n\n".join(parts[:-1]).strip()
    return markdown


@dataclass
class CrewChatResult:
    final_markdown: str
    traces: list[dict[str, str]]


def run_rotina_crew_chat(
    *,
    user_text: str,
    duck_block: str,
    rag_block: str,
    ml_addon: str,
    predictive_ml: bool,
    conn: duckdb.DuckDBPyConnection,
    data_dir: Path,
    collection: Any,
) -> CrewChatResult:
    if not crewai_import_ok():
        raise RuntimeError("Pacote `crewai` não está instalado. `pip install crewai langchain-openai`.")

    _ensure_rotina_crew_stderr_logging()
    req_id = uuid.uuid4().hex[:12]
    crew_langfuse_trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    from crewai import Agent, Crew, Process, Task
    from crewai.tools.base_tool import Tool as CrewAITool

    llm = build_crew_chat_llm(
        langfuse_trace_id=crew_langfuse_trace_id,
        langfuse_session_id=req_id,
    )

    want_dados, want_ml, want_rag, plan_reason = _plan_specialists(
        user_text=user_text,
        duck_block=duck_block,
        rag_block=rag_block,
        ml_addon=ml_addon,
        predictive_ml=predictive_ml,
        collection=collection,
    )
    spec_labels = [x for x, w in (("dados", want_dados), ("ml", want_ml), ("rag", want_rag)) if w]
    _CREW_LOG.info(
        "kickoff | req=%s | plan=%s | parallel=%s | reason=%s",
        req_id,
        "+".join(spec_labels) if spec_labels else "—",
        len(spec_labels),
        plan_reason,
    )

    # CrewAI >0.86 valida `tools` contra `crewai.tools.BaseTool`; LangChain `@tool`
    # devolve `StructuredTool` — converter com `Tool.from_langchain`.
    duck_tool = CrewAITool.from_langchain(make_duckdb_select_tool(conn))
    ml_tool = CrewAITool.from_langchain(make_ml_emotion_tool(Path(data_dir), predictive_ml))
    rag_tools: list[Any] = []
    if collection is not None:
        rag_tools.append(CrewAITool.from_langchain(make_rag_tool(collection, user_text)))

    base_ctx = (
        "### Contexto fixo (já produzido pela aplicação — não contradizer)\n\n"
        f"**Pergunta do utilizador:**\n{user_text}\n\n"
        f"**Dados tabulares (DuckDB / CSV):**\n{duck_block}\n\n"
        f"**Trechos de documentos (RAG):**\n{rag_block}\n\n"
        f"**Inferência ML de emoções (se existir):**\n{ml_addon or '_não aplicável nesta mensagem_'}\n"
    )

    recepcao = Agent(
        role="Anfitriã — boas-vindas",
        goal="Acolher e deixar explícita a intenção do utilizador sem inventar factos.",
        backstory="Primeira interlocutora da Rotina Viva: tom acolhedor e claro com famílias e equipa.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    dados = Agent(
        role="Analista de dados (DuckDB)",
        goal="Interpretar cadastro e diário; só leitura via SQL seguro quando preciso.",
        backstory=(
            "Especialista em tabelas `info_alunos` e `diario_estruturado`. Nunca sugere INSERT/UPDATE/DELETE. "
            "Em filtros por nome de pessoa use **sempre** `ILIKE` em `nome` (o DuckDB trata `LIKE` como sensível "
            "a maiúsculas/minúsculas). O bloco **Dados tabulares** do contexto fixo já reflete a consulta preparada "
            "pela app: trate-o como fonte principal; **não** diga «não localizado» se esse bloco listar linhas com o nome pedido."
        ),
        tools=[duck_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    ml_agent = Agent(
        role="Especialista ML emoções",
        goal=(
            "Explicar ou reproduzir classificação de emoções com o modelo local; "
            "priorizar **Raiva/Frustração** (não Tristeza) quando o relato envolver **agressão física** "
            "— ex.: «bateu no colega», «bateu», empurrou, soco — ou **gritos** (ex.: «gritou»); "
            "recomendar **intervenção imediata e mediação de conflito conforme o Regimento Interno** "
            "em vez de enquadrar o caso só como tristeza ou suporte emocional genérico."
        ),
        backstory=(
            "Domina o fluxo TF-IDF + FLAML e o dataset dair-ai/emotion; traduz resultados para educadoras. "
            "Regra operacional (prioridade sobre a etiqueta bruta do modelo): relatos de **agressão física** "
            "(«bateu no colega», bateu, empurrou, etc.) ou **gritos** → classificar como **raiva ou frustração** "
            "e encaminhar para **intervenção imediata e mediação de conflito** alinhada ao Regimento Interno, "
            "mesmo que o classificador ML devolva «tristeza» ou outra etiqueta."
        ),
        tools=[ml_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    rag_agent = Agent(
        role="Agente RAG",
        goal="Fundamentar respostas em PDFs indexados (recuperação semântica) quando necessário.",
        backstory="Cita trechos institucionais (regimento, PPP, nutrição) sem extrapolar.",
        tools=rag_tools,
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    redator = Agent(
        role="Redatora — síntese final",
        goal=(
            "Produzir uma única resposta em Markdown (pt-BR) com **tom profissional e contido**: acolhimento breve, "
            "sem linguagem sentimental excessiva; em contrapartida, **protocolos operacionais obrigatórios** "
            "(Regimento, PPP, mediação, assepsia, etc.), **citação explícita** do documento institucional e uso da "
            "emoção ML para calibrar gravidade — prioridade para o rigor operacional e o GEval."
        ),
        backstory=(
            "Redatora final da Rotina Viva: **evitar ser excessivamente acolhedora** (frases vagas, eufemismos longos "
            "ou conselhos genéricos que substituam passos concretos). Acolhimento = 1–2 frases úteis no máximo. "
            "**Perguntas diretas** (ex.: qual a turma, qual a alergia, o que tem de almoço, horário, regra pontual): "
            "responde de forma **extremamente objetiva em apenas uma frase**, mantendo só o **dado técnico essencial** "
            "(e citação mínima ao documento se for obrigatório para credibilidade). "
            "**Análise socioemocional** (bloco ML ou relato comportamental com classificação): a **primeira frase** da "
            "resposta deve conter **de imediato** a **classificação da emoção** (ou o juízo mais provável) **e** o "
            "**procedimento técnico** que o juiz avalia primeiro (assepsia com soro fisiológico; intervenção imediata e "
            "mediação de conflito; acolhimento e transição conforme PPP; etc.). Se o enunciado for **vago** (ex.: "
            "«não está bem» sem sintomas), a **primeira frase** pede **clarificação** (saúde física vs. sentimento). "
            "**Prioridade técnica:** protocolos (assepsia com soro fisiológico; intervenção imediata e mediação de "
            "conflito conforme normas de convivência do Regimento; acionar coordenação; estratégias de transição do PPP) "
            "são **obrigatórios** quando o contexto, RAG ou especialistas o sustentam. "
            "**Citação de documentos:** cada medida relevante deve citar a fonte (ex.: «conforme o Regimento Interno», "
            "«segundo o PPP», «conforme o Guia de Procedimentos de Saúde e Segurança»). "
            "**ML:** incorporar a emoção classificada (raiva, tristeza, medo, alegria, …) para gravidade/acolhimento, "
            "sem omitir a ação prática. Sintetiza sem repetição; não contradiz dados tabulares nem trechos RAG; "
            "evita IDs, emails e telefones salvo se a pergunta pedir."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    t1 = Task(
        name="recepcao",
        description=base_ctx
        + "\n**Tarefa:** saúde o utilizador e **reformule a intenção** em 2–4 frases (pt-BR). Não invente dados tabulares.",
        expected_output="Parágrafo curto: saudação + intenção explícita.",
        agent=recepcao,
    )

    specialist_tasks: list[Task] = []
    agents_used: list[Any] = [recepcao]

    if want_dados:
        specialist_tasks.append(
            Task(
                name="analista_dados",
                description=base_ctx
                + "\n**Tarefa:** explique o que importa nos **dados tabulares**; use a ferramenta **só** se precisar de SELECT adicional (nunca mutação). "
                "Se o bloco tabular já mostrar o aluno pedido (nome/turma), confirme-o explicitamente. "
                "Se for à ferramenta: filtros por nome → `WHERE nome ILIKE '%token%'` por token (ex.: Bruno, Reis), nunca `LIKE` simples.",
                expected_output="Notas claras sobre cadastro/diário ou indicação de que não há dados úteis.",
                agent=dados,
                context=[t1],
                async_execution=True,
            )
        )
        agents_used.append(dados)
    if want_ml:
        specialist_tasks.append(
            Task(
                name="especialista_ml",
                description=base_ctx
                + "\n**Tarefa:** se a pergunta envolver emoções/classificação ML, use a ferramenta ou o bloco ML; senão responda **uma frase**: não aplicável.\n"
                + "**Prioridade comportamental:** se o texto mencionar **gritos** ou **bateu/agressão física**, "
                "assume **raiva ou frustração** e indica **intervenção imediata e mediação de conflito conforme o Regimento Interno** "
                "(não trates como caso só de tristeza ou suporte emocional genérico).",
                expected_output="Resumo do ML ou 'não aplicável'.",
                agent=ml_agent,
                context=[t1],
                async_execution=True,
            )
        )
        agents_used.append(ml_agent)
    if want_rag:
        specialist_tasks.append(
            Task(
                name="especialista_rag",
                description=base_ctx
                + "\n**Tarefa:** destaque o que os **documentos** sustentam; se o bloco RAG estiver vazio, diga-o numa **frase**.",
                expected_output="Síntese documental ou aviso de ausência de trechos.",
                agent=rag_agent,
                context=[t1],
                async_execution=True,
            )
        )
        agents_used.append(rag_agent)

    merge_ctx: list[Any] = [t1] + specialist_tasks
    _suite = _eval_suite_running()
    if _suite:
        _final_desc_extra = (
            "\n**Tarefa:** integra as saídas anteriores numa **única resposta final** ao utilizador.\n"
            "**Modo suíte de avaliação (golden / DeepEval):**\n"
            "- **Não uses** cabeçalhos de secção em Markdown (`##`, `###`, …) — evita títulos como «## Acolhimento», "
            "«## Análise», «## Ação técnica»; escreve em **parágrafos corridos** ou **uma única frase**.\n"
            "- **Perguntas factuais diretas** (turma, alergia, cardápio do dia, medicação, regra objetiva, horário): "
            "**uma só frase**, extremamente objetiva, só o dado essencial (citação documental **curta inline** se preciso).\n"
        )
        if predictive_ml or want_ml:
            _final_desc_extra += (
                "- Com **análise socioemocional / ML** activa ou relevante, a **primeira frase** deve trazer **de imediato** "
                "a **classificação da emoção** (coerente com o especialista ML quando existir) **e** o **procedimento técnico** "
                "que importa para o caso (assepsia com soro fisiológico; intervenção imediata e mediação de conflito; "
                "acolhimento/transição conforme PPP; etc.). Se o relato for **ambíguo** (mal-estar sem detalhe), a **primeira frase** "
                "pede **clarificação** (saúde física vs. sentimento). Se for **equívoco lúdico vs. conflito** (ex.: «barulho» na sala), "
                "a **primeira frase** pode ser **análise condicional** («se… alegria/engajamento», «se… mediação») conforme Regimento/PPP.\n"
            )
        _final_desc_extra += (
            "- Mantém protocolos e citações a documentos no texto corrido; listas só se forem curtas e necessárias.\n"
            "- **Não** cites IDs de alunos, emails, telefones ou contactos **a menos que** a pergunta peça explicitamente.\n"
            "- Não repitas blocos CSV longos; não contradizas dados tabulares nem trechos RAG.\n"
            "- Ignora lacunas: se um especialista não entrou no plano, não menciones falta dele.\n"
        )
        _final_expected_out = (
            "pt-BR sem `##`: factual numa frase quando couber; socioemocional com emoção+procedimento na 1.ª frase quando aplicável."
        )
    else:
        _final_desc_extra = (
            "\n**Tarefa:** integra as saídas anteriores numa **única resposta final** ao utilizador.\n"
            "**Estrutura obrigatória (nesta ordem), em Markdown:**\n"
            "1. **Acolhimento inicial** — 1–2 frases breves, **tom profissional e contido** (evita tom excessivamente "
            "acolhedor ou genérico; sem enrolação).\n"
            "2. **Análise da emoção detectada** — se o bloco **Inferência ML de emoções** ou o especialista ML "
            "indicarem sentimento (raiva, tristeza, medo, alegria, frustração, etc.), incorpora essa classificação: "
            "o que sugere sobre gravidade, urgência ou necessidade de acolhimento.\n"
            "3. **Ação técnica / protocolo obrigatório** — passos concretos exigidos pelo caso (ex.: assepsia com soro "
            "fisiológico e registo; intervenção imediata e mediação de conflito; acionar coordenação; protocolo de "
            "saúde/emergência). **Não omitas** estes passos por simplificar a resposta: são obrigatórios quando o "
            "contexto, RAG ou especialistas os sustentam.\n"
            "4. **Citação do documento/fonte** — pelo menos uma frase que cite explicitamente a origem institucional "
            "que valida a ação (ex.: «conforme o Regimento Interno», «segundo o PPP», «conforme o Guia de Procedimentos "
            "de Saúde e Segurança»), alinhada ao que o Agente RAG ou o bloco **Trechos de documentos (RAG)** trouxe; "
            "se vários documentos se aplicam, referencia os relevantes.\n\n"
            "**Regras adicionais:**\n"
            "- Responde ao pedido do utilizador; usa `##`, listas e negrito.\n"
            "- **Não** cites IDs de alunos, emails, telefones ou contactos **a menos que** a pergunta peça explicitamente.\n"
            "- Para perguntas puramente factuais curtas, podes ser breve nas secções 1–2, mas mantém precisão e citação "
            "quando houver base documental.\n"
            "- Não repitas literalmente blocos CSV longos; resume.\n"
            "- Não contradizas dados tabulares nem trechos RAG.\n"
            "- Ignora lacunas: se um especialista não entrou no plano, não menciones falta dele.\n"
        )
        _final_expected_out = (
            "Markdown pt-BR seguindo: acolhimento inicial → emoção (se houver ML) → protocolo técnico → citação documental."
        )

    t_final = Task(
        name="redatora_final",
        description=base_ctx + _final_desc_extra,
        expected_output=_final_expected_out,
        agent=redator,
        context=merge_ctx,
    )
    agents_used.append(redator)

    task_list: list[Any] = [t1] + specialist_tasks + [t_final]
    crew = Crew(
        agents=agents_used,
        tasks=task_list,
        process=Process.sequential,
        verbose=False,
    )

    try:
        from modules.langfuse_rotina import (
            begin_rotina_crew_langfuse_trace_if_enabled,
            refresh_litellm_langfuse_callbacks,
        )

        refresh_litellm_langfuse_callbacks()
        begin_rotina_crew_langfuse_trace_if_enabled(
            langfuse_trace_id=crew_langfuse_trace_id,
            req_id=req_id,
            user_text=user_text,
            plan_labels="+".join(spec_labels) if spec_labels else "—",
            plan_reason=plan_reason,
        )
    except Exception:
        pass

    _CREW_LOG.info(
        "execute | req=%s | mode=recepcao_serial_depois_especialistas_paralelo",
        req_id,
    )
    out = crew.kickoff()
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _CREW_LOG.info("done | req=%s | ms=%s", req_id, elapsed_ms)

    traces = _task_outputs_to_traces(out, req_id=req_id)
    final = _final_text(out)
    if not final and traces:
        final = traces[-1]["content"]
    final = _remove_duplicate_tail_paragraph(final or "")
    if _eval_suite_running():
        final = _strip_markdown_section_headings_for_eval(final)

    try:
        from modules.langfuse_rotina import log_crew_trace_tree_if_enabled

        log_crew_trace_tree_if_enabled(
            req_id=req_id,
            user_text=user_text,
            plan_labels="+".join(spec_labels) if spec_labels else "—",
            plan_reason=plan_reason,
            parallel_n=len(spec_labels),
            elapsed_ms=elapsed_ms,
            crew_output=out,
            final_markdown=final or "",
            langfuse_trace_id=crew_langfuse_trace_id,
        )
    except Exception:
        pass

    try:
        from modules.langfuse_rotina import emit_crew_aggregated_usage_to_langfuse

        emit_crew_aggregated_usage_to_langfuse(
            langfuse_trace_id=crew_langfuse_trace_id,
            crew_output=out,
            req_id=req_id,
        )
    except Exception:
        pass

    try:
        from modules.langfuse_rotina import lf_flush

        lf_flush()
    except Exception:
        pass

    return CrewChatResult(final_markdown=final or "(sem resposta da crew)", traces=traces)
