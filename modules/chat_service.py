"""Serviços do chat Rotina: plano JSON, SQL de âmbito família, respostas determinísticas, status."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import duckdb

from core.database import validate_mutation_sql, _mensagem_csv_aberto_simples



_SQL_ALIAS_SKIP = frozenset(
    {
        "inner",
        "left",
        "right",
        "full",
        "cross",
        "join",
        "where",
        "group",
        "order",
        "limit",
        "as",
        "on",
        "using",
    }
)


def apply_parent_sql_scope(sql: str, id_aluno: int) -> str:
    """
    Restringe consultas do perfil Família às linhas do filho (subconsultas em diario / info).
    Preserva o filtro interno `WHERE id_aluno = …` do subselect (não duplica substituição).
    """
    aid = int(id_aluno)
    sub_d = f"(SELECT * FROM diario_estruturado WHERE id_aluno = {aid})"
    sub_i = f"(SELECT * FROM info_alunos WHERE id_aluno = {aid})"
    out = sql.strip().rstrip(";")

    def scope_table(s: str, table: str, subq: str, default_alias: str) -> str:
        # FROM tabela WHERE … (exceto quando já é WHERE id_aluno = …)
        s = re.sub(
            rf"\bFROM\s+{table}\s+WHERE\b(?!\s+id_aluno\s*=)",
            rf"FROM {subq} AS {default_alias} WHERE",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            rf"\bJOIN\s+{table}\s+WHERE\b(?!\s+id_aluno\s*=)",
            rf"JOIN {subq} AS {default_alias} WHERE",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            rf"\bFROM\s+{table}\s+AS\s+(\w+)\b",
            rf"FROM {subq} AS \1",
            s,
            flags=re.IGNORECASE,
        )
        s = re.sub(
            rf"\bJOIN\s+{table}\s+AS\s+(\w+)\b",
            rf"JOIN {subq} AS \1",
            s,
            flags=re.IGNORECASE,
        )
        # FROM tabela alias — alias não pode ser WHERE
        def _repl_alias(m: re.Match[str]) -> str:
            op, word = m.group(1), m.group(2)
            if word.lower() in _SQL_ALIAS_SKIP:
                return m.group(0)
            return f"{op} {subq} AS {word}"

        s = re.sub(
            rf"\b(FROM|JOIN)\s+{table}\s+(?!WHERE\b)(\w+)\b",
            _repl_alias,
            s,
            flags=re.IGNORECASE,
        )
        # FROM tabela antes de ORDER / GROUP / LIMIT / fim
        s = re.sub(
            rf"\bFROM\s+{table}\b(?=\s+(?:ORDER|GROUP|LIMIT|$))",
            rf"FROM {subq} AS {default_alias}",
            s,
            flags=re.IGNORECASE,
        )
        return s

    out = scope_table(out, "diario_estruturado", sub_d, "diario_estruturado")
    out = scope_table(out, "info_alunos", sub_i, "info_alunos")
    return out


def augment_question_for_parent_rag(question: str, id_aluno: int, nome_aluno: str) -> str:
    """Reforça o embedding com o aluno vinculado (PDFs continuam institucionais)."""
    q = (question or "").strip()
    nome = (nome_aluno or "").strip() or f"id_aluno={id_aluno}"
    return (
        f"{q}\n\n[Escopo responsável: trate dados de rotina/diário apenas no contexto do aluno "
        f"{nome} (id_aluno={id_aluno}). Documentos gerais da escola podem ser citados quando forem normativos.]"
    )
INSTITUTIONAL_SCOPE_RE = re.compile(
    r"(nome da escola|nome da instituição|como se chama a escola|qual [ée] o nome|"
    r"identidade da escola|cnpj da escola|endere[çc]o da escola|miss[ãa]o|vis[ãa]o|"
    r"quem somos|institui[çc][ãa]o|secretaria|diretoria|coordena[çc][ãa]o geral)",
    re.IGNORECASE,
)


def is_rag_identity_scope_question(message: str) -> bool:
    """Nome da escola, missão, endereço institucional, PPP/regimento como identidade — não nutrição nem protocolo de febre."""
    um = message.strip()
    ul = um.lower()
    if INSTITUTIONAL_SCOPE_RE.search(um):
        return True
    return (
        "nome" in ul
        and "escola" in ul
        and "aluno" not in ul
        and "criança" not in ul
        and "crianca" not in ul
    )


# Perguntas sobre cardápio / refeições / café da manhã → priorizar PDFs de planejamento nutricional no RAG
# (evita que o 1.º trecho na sidebar seja outro PDF que só menciona “alimentação” em geral).
NUTRITION_MEALS_SCOPE_RE = re.compile(
    r"(caf[ée]\s*(da\s*)?manh[ãa]|"
    r"lanche(\s+(da\s*)?(manh[ãa]|tarde))?|"
    r"almo[çc]o|jantar|merenda|card[áa]pio|cardapio|"
    r"refei[çc][õo]es?|refeicao|refeicoes|"
    r"nutri[çc][ãa]o|nutricional|nutricao|"
    r"alimenta[çc][ãa]o|alimentacao|"
    r"planejamento\s+nutricional|menu\s+escolar|sobremesa|"
    r"merendeira|hora\s+do\s+lanche)",
    re.IGNORECASE,
)


def is_rag_nutrition_meals_scope_question(message: str) -> bool:
    """
    Cardápio, horários de refeição, café da manhã na escola, etc.
    Restringe a busca ao PDF de planejamento nutricional (como o escopo de identidade usa regimento/PPP).
    """
    um = (message or "").strip()
    if not um:
        return False
    if NUTRITION_MEALS_SCOPE_RE.search(um):
        return True
    ul = um.lower()
    if "café" in ul or "cafe" in ul:
        if any(x in ul for x in ("manhã", "manha", "escola", "criança", "crianca", "turma")):
            return True
    return False



def _plan_mutacao_targets_csv_tables(plan: dict[str, Any]) -> bool:
    mut = plan.get("mutacao")
    if isinstance(mut, str) and mut.strip():
        if re.search(r"\b(info_alunos|diario_estruturado)\b", mut, re.IGNORECASE):
            return True
    sql = plan.get("sql")
    if isinstance(sql, str) and sql.strip() and validate_mutation_sql(sql.strip()):
        return True
    return False


# Verbos de exclusão (infinitivo e imperativo comum em PT-BR) para detetar pedidos ao planeador/chat.
_DELETE_VERB_GROUP = (
    r"apagar|apague|apaga|remover|remova|eliminar|elimine|excluir|exclua|deletar|delete"
)


def _user_natural_language_cadastro_mutation_intent(user_message: str) -> bool:
    """
    Pedidos de gravar/apagar/alterar aluno ou linha de diário — só DuckDB/CSV;
    evita que o planeador peça RAG por associação errada a PDFs.
    """
    um = (user_message or "").strip()
    if not um:
        return False
    ul = um.lower()
    if "nome da escola" in ul and "aluno" not in ul:
        return False
    patterns = (
        r"\b(?:cadastrar|registrar|adicionar|incluir|criar|inserir|salvar|gravar)\b.{0,72}\b(?:aluno|alunos)\b",
        rf"\b(?:{_DELETE_VERB_GROUP})\b.{{0,72}}\b(?:aluno|alunos|cadastro)\b",
        rf"\b(?:{_DELETE_VERB_GROUP})\b.{{0,40}}\b(?:do\s+cadastro|o\s+cadastro)\b",
        # "apagar leonardo, miguel e gustavo" — lista de nomes sem a palavra "aluno"
        rf"\b(?:{_DELETE_VERB_GROUP})\b.{{0,200}}(?:,|;|\be\b|\bou\b)",
        r"\bnov[oa]\s+aluno\b",
        r"\b(?:atualizar|alterar|modificar)\b.{0,72}\b(?:cadastro|aluno|alunos)\b",
        r"\b(?:inserir|gravar|registrar)\b.{0,72}\b(?:di[aá]rio|diario)\b",
        r"\bmuta[cç][aã]o\b.{0,40}\b(?:aluno|cadastro|csv|info_alunos|di[aá]rio)\b",
    )
    for pat in patterns:
        if re.search(pat, ul, re.IGNORECASE | re.DOTALL):
            return True
    return False


def _should_force_structured_sources_no_rag(plan: dict[str, Any], user_message: str) -> bool:
    if _plan_mutacao_targets_csv_tables(plan):
        return True
    return _user_natural_language_cadastro_mutation_intent(user_message)


_CONTINUATION_PRONOUN = re.compile(
    r"(?is)\b("
    r"ele|ela|"
    r"esse\s+aluno|essa\s+aluna|"
    r"o\s+mesmo|a\s+mesma|"
    r"mesm[oa]\s+(?:aluno|aluna|criança|crianca|menin[oa])"
    r")\b"
)


_NAME_STOP_TOKENS = frozenset(
    {
        "tem",
        "têm",
        "tém",
        "é",
        "e",
        "alergia",
        "alergias",
        "intolerância",
        "intolerancia",
        "turma",
        "turmas",
        "nome",
        "cadastro",
        "está",
        "esta",
        "com",
        "sem",
        "qual",
        "quais",
        "sobre",
        "segundo",
    }
)


def _trim_aluno_name_tokens(raw: str) -> str:
    """Corta em 'tem alergia', 'esta', etc., para não poluir o ILIKE."""
    parts = [p for p in raw.replace("?", " ").split() if p]
    out: list[str] = []
    for p in parts:
        wd = p.strip(".,;:!?").lower()
        if wd in _NAME_STOP_TOKENS:
            break
        out.append(p.strip(".,;:!?"))
    return " ".join(out).strip()


def _extract_aluno_names_from_chat_blob(blob: str) -> list[str]:
    """Extrai fragmentos de nome citados como «aluno X Y» ou variantes (ordem cronológica aproximada)."""
    found: list[str] = []
    for m in re.finditer(
        r"(?is)\b(?:aluno|aluna)\s+([^\n?!]{3,72})",
        blob,
    ):
        name = _trim_aluno_name_tokens(_strip_dates_from_student_name_fragment(m.group(1)))
        if len(name) >= 5 and name.lower() not in ("rotina viva", "nome da escola"):
            found.append(name.title() if name.islower() else name)
    out: list[str] = []
    seen: set[str] = set()
    for n in found:
        key = n.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def augment_cadastro_question_with_history(
    user_text: str,
    history: list[dict[str, str]],
    *,
    parent_scope: tuple[int, str] | None = None,
) -> str:
    """
    Reforça a pergunta enviada ao planeador com o nome do aluno quando for continuação
    (ele/ela, qual turma…) e o nome só apareceu antes no chat — evita `sql: null` e perda de contexto.
    """
    ut = (user_text or "").strip()
    if not ut:
        return ut
    low = ut.lower()
    if re.search(r"(?is)\baluno\s+\S+\s+\S+", ut):
        return ut
    if "«" in ut and "»" in ut:
        return ut
    blob = "\n".join(
        (m.get("content") or "")
        for m in history[-12:]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    )
    short_follow = len(ut) < 140
    turma_alerg = ("turma" in low or "alerg" in low or "alérg" in low or "alegic" in low)
    diario_hint = bool(
        re.search(
            r"(?i)\b(sono|refei|almo[cç]|lanche|café|cafe|jantar|comeu|dormiu|evacua|"
            r"banheiro|trocas|medicament|recado|atividade|intera[cç]|di[aá]rio|"
            r"rotina|hoje|ontem|semana)\b",
            ut,
        )
    )
    continuation = bool(_CONTINUATION_PRONOUN.search(ut)) or (
        short_follow and turma_alerg and re.search(r"(?i)\bqual\b", low)
    )
    continuation = continuation or (short_follow and diario_hint)
    if not continuation:
        return ut
    names = _extract_aluno_names_from_chat_blob(blob)
    if names:
        chosen = names[-1]
        return (
            f"{ut}\n\n[Contexto da conversa: continuação sobre o/a aluno/a «{chosen}». "
            "Use esse nome em `nome ILIKE` em `JOIN info_alunos` com `diario_estruturado`, "
            "ou só em `info_alunos` se for cadastro.]"
        )
    if parent_scope is not None:
        _aid, nome = parent_scope
        nome = (nome or "").strip()
        if nome and nome not in ("—", "?", ""):
            return (
                f"{ut}\n\n[Contexto da conversa: continuação sobre o aluno vinculado «{nome}» "
                f"(id_aluno={int(_aid)}). Use `d.id_aluno = {int(_aid)}` ou `nome ILIKE` com JOIN.]"
            )
    return ut


_DIARIO_READ_KEYWORDS = re.compile(
    r"(?i)\b(refei|c[eê]fe|café|almo[cç]|lanche|jantar|"
    r"sono|dormiu|evacua|banheiro|trocas|medicament|recado|atividade|intera[cç]|"
    r"di[aá]rio(\s+estruturado)?|rotina|comeu|anotou|registou|ontem|hoje|semana|"
    r"últim|ultim|m[eê]s|periodo|período|filho|filha|"
    r"como\s+foi\s+o\s+dia|foi\s+o\s+dia|\bdia\s+(?:de|do|da)\b|\bo\s+dia\b)\b",
)


def _sql_week_anchor_filter_by_name_tokens(tokens: list[str]) -> str:
    """
    Últimos ~25 dias de diário **a partir da data mais recente desse aluno na base**,
    não a partir do relógio do servidor (evita zero linhas quando o CSV é de outro mês).
    """
    esc_ax = []
    for t in tokens[:4]:
        safe = t.replace("'", "''")
        esc_ax.append(f"ax.nome ILIKE '%{safe}%'")
    inner = " AND ".join(esc_ax)
    return (
        "AND try_cast(d.data AS DATE) >= ("
        "SELECT coalesce(max(try_cast(dx.data AS DATE)), CURRENT_DATE) - INTERVAL 25 DAY "
        "FROM diario_estruturado AS dx JOIN info_alunos AS ax ON dx.id_aluno = ax.id_aluno "
        f"WHERE {inner})"
    )


def _sql_week_anchor_filter_by_id_aluno(aid: int) -> str:
    return (
        "AND try_cast(d.data AS DATE) >= ("
        "SELECT coalesce(max(try_cast(dx.data AS DATE)), CURRENT_DATE) - INTERVAL 25 DAY "
        f"FROM diario_estruturado AS dx WHERE dx.id_aluno = {int(aid)})"
    )


def _diario_question_wants_week_range(um: str) -> bool:
    """Perguntas tipo 'como foi a semana' → vários dias, não só o último registo."""
    low = um.lower()
    return bool(
        re.search(
            r"(?i)\b(semana|últim\w*\s+dias|ultim\w*\s+dias|últim\w*\s+semana|"
            r"ultim\w*\s+semana|nesta\s+semana|esta\s+semana|nos\s+últimos|"
            r"nos\s+ultimos|7\s+dias|sete\s+dias)\b",
            um,
        )
        or ("semana" in low and re.search(r"(?i)\b(como|foi|resumo|rotina)\b", um))
    )


def _strip_dates_from_student_name_fragment(s: str) -> str:
    """
    Remove datas coladas ao nome (ex.: «Ana Souza 2026-04-15» ou «Ana Souza 15/04/2026»),
    para não gerar `ILIKE '%2026%'` nem fragmentos inválidos.
    """
    t = (s or "").strip()
    if not t:
        return t
    t = re.sub(r"(?i)\s+(?:no\s+dia|na\s+data|em|dia)\s+\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\s*$", "", t)
    t = re.sub(r"(?i)\s+(?:no\s+dia|na\s+data|em)\s+\d{4}-\d{2}-\d{2}\s*$", "", t)
    t = re.sub(r"\s+\d{4}-\d{2}-\d{2}\s*$", "", t)
    t = re.sub(r"\s+\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\s*$", "", t)
    return t.strip()


def _parse_diary_date_filter_sql(um: str) -> str | None:
    """
    Devolve cláusula SQL `AND ...` para um dia específico.
    Datas `DD/MM/AAAA` ou `DD-MM-AAAA` tratadas como **dia/mês/ano (PT-BR)**.
    Também reconhece `AAAA-MM-DD` na pergunta.
    """
    m_iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", um)
    if m_iso:
        y, mo, d = m_iso.group(1), m_iso.group(2), m_iso.group(3)
        lit = f"{y}-{mo}-{d}"
        return f"AND d.data = '{lit}'"
    m_br = re.search(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b", um)
    if m_br:
        day, month, year = int(m_br.group(1)), int(m_br.group(2)), int(m_br.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            lit = f"{year:04d}-{month:02d}-{day:02d}"
            return f"AND d.data = '{lit}'"
    return None


def _extract_student_name_for_infer_sql(um: str) -> str | None:
    """Nome do aluno a partir da pergunta (incl. bloco «…» do augment)."""
    for pat in (
        r'["""]([^"""]{3,50})["""]',
        r"[“”]([^“”]{3,50})[“”]",
        r"\bmeu\s+filho\s+['\"]?([A-Za-zÀ-ÿ][^\n'\"?!]{2,45})",
        r"\bminha\s+filha\s+['\"]?([A-Za-zÀ-ÿ][^\n'\"?!]{2,45})",
    ):
        mq = re.search(pat, um, re.IGNORECASE)
        if mq:
            cand = _trim_aluno_name_tokens(_strip_dates_from_student_name_fragment(mq.group(1).strip()))
            if len(cand) >= 4:
                return cand
    m_ctx = re.search(r"«([^»]{3,80})»", um)
    if m_ctx:
        raw = m_ctx.group(1).strip()
        if re.match(r"(?i)^id_aluno\s*=", raw):
            return None
        name = _trim_aluno_name_tokens(_strip_dates_from_student_name_fragment(raw))
        if len(name) >= 4:
            return name
    name: str | None = None
    ma = re.search(
        r"(?is)\b(?:aluno|aluna)\s+(.+?)(?:\?|$|\n)",
        um,
    )
    if ma:
        name = _trim_aluno_name_tokens(_strip_dates_from_student_name_fragment(ma.group(1)))
    if not name:
        parts_caps = re.findall(
            r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)+)\b",
            um,
        )
        block = {
            "Qual Turma",
            "Rotina Viva",
            "Ensino Fundamental",
            "Educação Infantil",
        }
        for cand in reversed(parts_caps):
            if cand in block or len(cand) < 6:
                continue
            if re.search(r"\d{4}", cand):
                continue
            name = cand
            break
    if not name:
        return None
    name = _trim_aluno_name_tokens(_strip_dates_from_student_name_fragment(name))
    if len(name) < 4:
        return None
    return name


def infer_structured_select_sql(user_message: str) -> str | None:
    """
    SELECT de recurso quando o planeador devolve `sql` vazio: `info_alunos` e/ou
    `diario_estruturado` + JOIN, conforme palavras-chave da pergunta.
    """
    um = (user_message or "").strip()
    if not um:
        return None
    wants_diario = bool(_DIARIO_READ_KEYWORDS.search(um))
    wants_cadastro = bool(
        re.search(
            r"(?i)\b(turma|alerg|alérg|alegic|cadastro|aluno|aluna|contato|telefone|id_aluno)\b",
            um,
        )
    )
    if not wants_diario and not wants_cadastro:
        return None
    id_m = re.search(
        r"(?i)\bid_aluno\s*=\s*(\d+)\b",
        um,
    ) or re.search(r"(?i)\bid_aluno\D{0,12}(\d+)\b", um)
    name = _extract_student_name_for_infer_sql(um)
    diary_cols = (
        "d.id_registro, d.id_aluno, d.data, d.cafe_manha, d.almoco, d.lanche_tarde, d.jantar_extra, "
        "d.qualidade_sono, d.atividade_dia, d.interacao_social, d.recado_professora, "
        "d.evacuacao, d.trocas_banheiro, d.medicamentos, a.nome"
    )
    date_clause = _parse_diary_date_filter_sql(um)
    week_range = _diario_question_wants_week_range(um)
    # Perguntas de enturmação (cadastro) não devem ser desviadas para SELECT do diário só porque
    # a frase contém palavras do léxico de rotina (ex.: "rotina", "atividade" em outros contextos).
    wants_pure_turma_lookup = bool(
        re.search(r"(?i)\bqual\s+turma\b", um)
        or re.search(r"(?i)\bem\s+qual\s+turma\b", um)
        or re.search(r"(?i)\bem\s+que\s+turma\b", um)
    )
    if wants_diario and wants_cadastro and wants_pure_turma_lookup and name and not date_clause and not week_range:
        wants_diario = False
    if week_range:
        diary_limit = 120
        order_diary = "d.data ASC, d.id_registro ASC"
    else:
        order_diary = "d.data ASC, d.id_registro ASC"
        diary_limit = 50 if date_clause else 120

    if wants_diario:
        if id_m:
            aid = int(id_m.group(1))
            parts = [
                f"SELECT {diary_cols} FROM diario_estruturado d",
                "JOIN info_alunos a ON d.id_aluno = a.id_aluno",
                f"WHERE d.id_aluno = {aid}",
            ]
            if date_clause:
                parts.append(date_clause.strip())
            elif week_range:
                parts.append(_sql_week_anchor_filter_by_id_aluno(aid))
            parts.append(f"ORDER BY {order_diary} LIMIT {diary_limit}")
            return " ".join(parts)
        if name:
            tokens = [t for t in re.split(r"\s+", name.strip()) if len(t) >= 2]
            if tokens:
                esc = []
                for t in tokens[:4]:
                    safe = t.replace("'", "''")
                    esc.append(f"a.nome ILIKE '%{safe}%'")
                where = " AND ".join(esc)
                parts = [
                    f"SELECT {diary_cols} FROM diario_estruturado d",
                    "JOIN info_alunos a ON d.id_aluno = a.id_aluno",
                    f"WHERE {where}",
                ]
                if date_clause:
                    parts.append(date_clause)
                elif week_range:
                    parts.append(_sql_week_anchor_filter_by_name_tokens(tokens))
                parts.append(f"ORDER BY {order_diary} LIMIT {diary_limit}")
                return " ".join(parts)
        if not wants_cadastro:
            return None
    if not name:
        return None
    tokens = [t for t in re.split(r"\s+", name.strip()) if len(t) >= 2]
    if not tokens:
        return None
    esc = []
    for t in tokens[:4]:
        safe = t.replace("'", "''")
        esc.append(f"nome ILIKE '%{safe}%'")
    where = " AND ".join(esc)
    return (
        "SELECT id_aluno, nome, turma, alergias, contato_pais FROM info_alunos "
        f"WHERE {where} ORDER BY id_aluno LIMIT 25"
    )


def infer_info_alunos_select_sql(user_message: str) -> str | None:
    """Compatível com chamadas antigas — delega em `infer_structured_select_sql`."""
    return infer_structured_select_sql(user_message)


def apply_infer_sql_to_plan(plan: dict[str, Any] | None, planning_user_text: str) -> dict[str, Any]:
    """
    Força o SELECT inferido quando a pergunta é claramente sobre aluno + cadastro/diário.
    O planeador LLM por vezes gera SQL que não devolve linhas ou ignora o JOIN correto.
    """
    p = dict(plan) if isinstance(plan, dict) else {}
    mut = p.get("mutacao")
    if isinstance(mut, str) and mut.strip():
        return p
    inf = infer_structured_select_sql(planning_user_text)
    if not inf:
        return p
    p["sql"] = inf
    raw_f = p.get("fontes") or []
    fontes: list[str] = []
    if isinstance(raw_f, str):
        fontes = [raw_f.lower().strip()] if raw_f.lower().strip() in ("rag", "sql") else []
    elif isinstance(raw_f, list):
        for x in raw_f:
            if isinstance(x, str) and x.lower().strip() in ("rag", "sql"):
                fontes.append(x.lower().strip())
    fontes = [f for f in fontes if f != "rag"]
    if "sql" not in fontes:
        fontes.append("sql")
    if not fontes:
        fontes = ["sql"]
    p["fontes"] = fontes
    return p


def normalize_plan(plan: dict[str, Any] | None, user_message: str) -> dict[str, Any]:
    """
    Corrige planos inconsistentes do modelo pequeno (ex.: fontes só \"sql\" com sql null).
    Garante RAG para perguntas institucionais (nome da escola, PPP, regimento, etc.).
    """
    p = dict(plan) if isinstance(plan, dict) else {}
    raw = p.get("fontes")
    fontes: list[str] = []
    if isinstance(raw, str):
        s = raw.lower().strip()
        if s in ("rag", "sql"):
            fontes = [s]
    elif isinstance(raw, list):
        for x in raw:
            if isinstance(x, str) and x.lower().strip() in ("rag", "sql"):
                fontes.append(x.lower().strip())
    if not fontes:
        fontes = ["rag"]

    seen: set[str] = set()
    fontes = [f for f in fontes if not (f in seen or seen.add(f))]

    um = user_message.strip()
    ul = um.lower()

    sql_val = p.get("sql")
    sql_str = sql_val.strip() if isinstance(sql_val, str) else ""
    has_sql = bool(sql_str)

    inferred = infer_structured_select_sql(um)
    if inferred and not has_sql:
        p["sql"] = inferred
        sql_str = inferred
        has_sql = True
        if "sql" not in fontes:
            fontes.append("sql")

    if "sql" in fontes and not has_sql:
        fontes = [f for f in fontes if f != "sql"]
        p["sql"] = None
        if "rag" not in fontes:
            fontes.append("rag")
    if INSTITUTIONAL_SCOPE_RE.search(um) or (
        "nome" in ul
        and "escola" in ul
        and "aluno" not in ul
        and "criança" not in ul
        and "crianca" not in ul
    ):
        if "rag" not in fontes:
            fontes.append("rag")
        if "sql" in fontes and not has_sql:
            fontes = [f for f in fontes if f != "sql"]

    if not fontes:
        fontes = ["rag"]

    if _should_force_structured_sources_no_rag(p, um):
        fontes = [f for f in fontes if f != "rag"]
        if "sql" not in fontes:
            fontes.append("sql")
        if not fontes:
            fontes = ["sql"]

    p["fontes"] = fontes
    if "sql" not in fontes:
        p["sql"] = None
    mv = p.get("mutacao")
    if mv is not None and not isinstance(mv, str):
        p["mutacao"] = None
    elif isinstance(mv, str) and not mv.strip():
        p["mutacao"] = None
    return p


def _first_re_group(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1)


def parse_gestao_delete_aluno_nome_fragmento(user_message: str) -> str | None:
    """
    Detecta pedidos do tipo «apague o aluno [NOME]» / «remover aluno X» e devolve o fragmento de nome.
    Devolve None se não for um comando directo de exclusão por nome.
    """
    s = (user_message or "").strip()
    if not s:
        return None
    m = re.search(
        r"(?is)\b(?:apague|apagar|remova|remover|elimine|eliminar|exclua|excluir|deletar|delete)\b\s*"
        r"(?:o\s+|a\s+)?(?:(?:aluno|aluna|alunos|alunas)\s+)?['\"]?(.+?)['\"]?\s*$",
        s,
    )
    if not m:
        return None
    frag = (m.group(1) or "").strip()
    for sfx in (
        " do cadastro",
        " da escola",
        " permanente",
        " permanentemente",
        " por favor",
        " pf",
        " pf.",
    ):
        if frag.lower().endswith(sfx):
            frag = frag[: -len(sfx)].strip()
    frag = frag.strip(' \t."\'`')
    if len(frag) < 2:
        return None
    return frag


def try_gestao_delete_by_name_intent(
    user_message: str,
    data_dir: Path,
    *,
    session_role: str | None,
    allow_delete_mutations: bool,
) -> tuple[bool, bool, str]:
    """
    Se for perfil Gestão e mensagem do tipo «apague o aluno [NOME]», aplica `delete_by_name` nos CSV.

    Retorno: ``(handled, success, message)``.
    - ``handled=False`` → seguir o fluxo normal (planejador / SQL).
    - ``handled=True`` → já respondemos ao pedido; ``success`` e ``message`` descrevem o resultado.
    """
    if (session_role or "").strip().lower() != "gestao" or not allow_delete_mutations:
        return False, False, ""
    frag = parse_gestao_delete_aluno_nome_fragmento(user_message)
    if frag is None:
        return False, False, ""
    from core.database import delete_by_name

    msg, ok = delete_by_name(frag, data_dir=data_dir)
    return True, ok, msg


def _user_requests_student_delete(user_message: str) -> bool:
    """Pedido explícito de remover aluno do cadastro (verbo de exclusão antes do objeto)."""
    um = (user_message or "").strip().lower()
    if not um:
        return False
    if re.search(
        rf"\b(não|nao)\b.{{0,20}}\b(?:{_DELETE_VERB_GROUP})\b",
        um,
        re.DOTALL,
    ):
        return False
    if re.search(
        rf"\bcomo\b.{{0,48}}\b(?:{_DELETE_VERB_GROUP})\b",
        um,
        re.DOTALL,
    ):
        return False
    if re.search(rf"\bsem\b.{{0,16}}\b(?:{_DELETE_VERB_GROUP})\b", um):
        return False
    if re.search(rf"\b(?:evitar|impedir)\b.{{0,24}}\b(?:{_DELETE_VERB_GROUP})\b", um):
        return False
    if re.search(
        rf"\b(?:{_DELETE_VERB_GROUP})\b.{{0,200}}(?:,|;|\be\b|\bou\b)",
        um,
        re.DOTALL,
    ):
        return True
    # Só verbo → objeto (ex.: "apague o aluno"). Evita "novo aluno ... sem remover X" (falso positivo).
    return bool(
        re.search(
            rf"\b(?:{_DELETE_VERB_GROUP})\b.{{0,88}}\b(?:aluno|alunos|cadastro)\b",
            um,
            re.DOTALL,
        )
    )


def _reply_delete_not_persisted_no_mutation(*, perfil_educador: bool) -> str:
    """Quando o utilizador pediu apagar mas não há DELETE aplicável — evita confirmação falsa (stream)."""
    if perfil_educador:
        return (
            "Não posso **apagar** alunos no cadastro: o perfil **Educador** só permite **ler** e **gravar** "
            "(INSERT/UPDATE). **Só a Gestão** pode remover registos (DELETE). O CSV **não foi alterado**."
        )
    return (
        "**Nada foi removido do CSV.** O plano não trouxe uma mutação `DELETE` a aplicar, por isso o cadastro "
        "mantém-se igual. Se quiser eliminar um aluno, tente reformular o pedido ou confirme que está no perfil **Gestão**."
    )


def build_mutation_direct_reply(
    *,
    mut_sql: str,
    ok: bool,
    result_message: str,
    duplicate_warn: str,
    duck_block: str,
) -> str:
    """
    Resposta determinística para mutações (evita recusa contraditória do LLM).
    """
    if not ok:
        if duplicate_warn:
            return (
                "Não gravei a alteração porque detectei dados duplicados no cadastro.\n\n"
                f"{duplicate_warn}\n\n"
                "Nada foi alterado no CSV."
            )
        msg = result_message.strip() or (
            "A alteração não foi gravada. Feche o CSV se estiver aberto e tente novamente."
        )
        if msg == _mensagem_csv_aberto_simples():
            return msg
        lowm = (mut_sql or "").lower()
        if re.search(r"\bdelete\s+from\s+info_alunos\b", lowm):
            msg += "\n\nO cadastro em CSV **não foi alterado** por este pedido."
        return msg

    low = (mut_sql or "").lower()
    count = _first_re_group(r"CONTAGEM_OFICIAL_ALUNOS\s*=\s*(\d+)", duck_block) or "?"

    # Não usar apenas `"info_alunos" in low"`: INSERT no diário costuma trazer subselect a info_alunos.
    ins_diario = bool(re.search(r"\binsert\s+into\s+diario_estruturado\b", low))
    upd_diario = bool(re.search(r"\bupdate\s+diario_estruturado\b", low))
    del_diario = bool(re.search(r"\bdelete\s+from\s+diario_estruturado\b", low))
    ins_info = bool(re.search(r"\binsert\s+into\s+info_alunos\b", low))
    del_info = bool(re.search(r"\bdelete\s+from\s+info_alunos\b", low))
    upd_info = bool(re.search(r"\bupdate\s+info_alunos\b", low))

    if ins_diario:
        return (
            "Registo **gravado** no diário (`diario_estruturado.csv`).\n\n"
            "Confira na verificação abaixo se `id_registro`, `id_aluno`, `data` e os campos pedidos estão corretos."
        )
    if upd_diario:
        return (
            "Linha do diário **atualizada** em `diario_estruturado.csv`.\n\n"
            "Confira na verificação abaixo."
        )
    if del_diario:
        return (
            "Linha do diário **removida** de `diario_estruturado.csv`.\n\n"
            "Confira na verificação abaixo."
        )

    if ins_info:
        aid = _first_re_group(r"CONFIRME_id_aluno\s*=\s*(\d+)", duck_block) or "?"
        return (
            f"Cadastro realizado com sucesso. O novo aluno foi gravado com `id_aluno={aid}`.\n\n"
            f"Total atual de alunos no cadastro: {count}."
        )
    if del_info:
        return (
            "Remoção **gravada** no cadastro (CSV).\n\n"
            f"Total atual de alunos no cadastro (após verificação): **{count}**.\n\n"
            "_Confirme em `info_alunos.csv` se precisar; o número acima vem do estado lido após a mutação._"
        )
    if upd_info:
        return (
            "Cadastro de aluno **atualizado** em `info_alunos.csv`.\n\n"
            f"Total atual de alunos no cadastro (referência): {count}."
        )
    mentions_info = bool(re.search(r"\binfo_alunos\b", low))
    mentions_diario = bool(re.search(r"\bdiario_estruturado\b", low))
    return (
        "Alteração aplicada e CSV atualizado com sucesso.\n\n"
        f"Total atual de alunos no cadastro: {count}."
        if mentions_info and not mentions_diario
        else "Alteração aplicada e CSVs atualizados com sucesso."
    )


def apply_user_data_source_mode(plan: dict[str, Any], mode: str) -> dict[str, Any]:
    """Aplica escolha do sidebar por cima do plano normalizado (automático / só SQL / só RAG)."""
    if mode == "auto":
        return plan
    p = dict(plan)
    if mode == "structured":
        p["fontes"] = ["sql"]
        sql = p.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            p["sql"] = None
    elif mode == "documents":
        p["fontes"] = ["rag"]
        p["sql"] = None
        # Não limpar "mutacao": gravar cadastro/diário nos CSV não passa pelos PDFs;
        # o utilizador pode estar em "só documentos" e ainda pedir uma linha no diário.
    return p


def _student_label_for_chat(conn: duckdb.DuckDBPyConnection | None, aid: int) -> str:
    if conn is None:
        return f"id_aluno={aid}"
    try:
        rec = conn.execute(
            "SELECT nome, turma FROM info_alunos WHERE id_aluno = ? LIMIT 1",
            [aid],
        ).fetchone()
    except Exception:
        rec = None
    if not rec:
        return f"id_aluno={aid}"
    nome = str(rec[0] or "").strip() or f"id_aluno={aid}"
    turma = str(rec[1] or "").strip()
    return f"{nome} ({turma})" if turma else nome


def _processing_status_sql_line(user_text: str, sql: str | None) -> str:
    """Linha de status quando o plano usa dados estruturados: começa com CSV."""
    blob = f"{user_text} {sql or ''}".lower()
    if "alerg" in blob:
        return "CSV — verificando alergias e cadastro de alunos…"
    if "turma" in blob or ("infantil" in blob and "qual" in blob):
        return "CSV — consultando turmas e cadastro de alunos…"
    if any(
        w in blob
        for w in (
            "diário",
            "diario",
            "refei",
            "sono",
            "lanche",
            "evacu",
            "medic",
            "recado",
        )
    ):
        return "CSV — consultando diário estruturado…"
    return "CSV — consultando tabelas de cadastro e diário…"


def _processing_status_rag_line(user_text: str) -> str:
    """Linha de status quando o plano usa documentos: começa com PDF."""
    ul = user_text.lower()
    if "regimento" in ul or "norma" in ul or "regra" in ul:
        return "PDF — regimento e normas da escola…"
    if "ppp" in ul or "pedagóg" in ul or "pedagog" in ul:
        return "PDF — PPP e documentos pedagógicos…"
    if any(w in ul for w in ("cardápio", "cardapio", "nutri", "aliment")):
        return "PDF — planejamento nutricional e cardápio…"
    if "segurança" in ul or "seguranca" in ul or "saúde" in ul or "saude" in ul:
        return "PDF — guia de saúde e segurança…"
    if "nome" in ul and "escola" in ul:
        return "PDF — identidade institucional nos documentos…"
    return "PDF — documentos institucionais…"
