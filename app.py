"""
Rotina Viva — assistente para rotinas em escolas infantis (AI Factory / PUCPR).
Streamlit + DuckDB (CSVs) + ChromaDB (RAG) + txtai (segmentação) + Ollama (LLM/embeddings).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Generator, Iterable

import chromadb
import duckdb
import httpx
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


DATA_DIR = Path(os.getenv("ROTINA_DATA_DIR", "data")).resolve()
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma")).resolve()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# LLM no chat: ollama (local) ou openai (API compatível com OpenAI — OpenAI, Groq, OpenRouter, etc.)
ROTINA_CHAT_PROVIDER = os.getenv("ROTINA_CHAT_PROVIDER", "ollama").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()

# Timeouts: no Ollama em CPU o servidor pode demorar minutos até enviar o 1º byte — read=None desliga limite entre chunks/cabeçalho.
def _llm_plan_timeout() -> float:
    sec = _env_int("LLM_PLAN_TIMEOUT", 600)
    return float(max(30, sec))


def _ollama_stream_timeout() -> httpx.Timeout:
    raw = os.getenv("LLM_STREAM_READ_TIMEOUT", "0").strip().lower()
    if raw in ("", "0", "none", "unlimited"):
        return httpx.Timeout(connect=120.0, read=None, write=120.0, pool=120.0)
    try:
        rsec = float(raw)
    except ValueError:
        return httpx.Timeout(connect=120.0, read=None, write=120.0, pool=120.0)
    return httpx.Timeout(connect=120.0, read=rsec, write=120.0, pool=120.0)


def _use_openai_chat() -> bool:
    return ROTINA_CHAT_PROVIDER == "openai" and bool(OPENAI_API_KEY)

# Modo leve (padrão 1): poucos chunks, sem txtai na indexação — ideal para testar fluxo em PC fraco.
LIGHT_MODE = _env_bool("ROTINA_LIGHT_MODE", "1")
CHUNK_CHAR_SIZE = _env_int("ROTINA_CHUNK_CHARS", 2000 if LIGHT_MODE else 1000)
CHUNK_CHAR_OVERLAP = _env_int("ROTINA_CHUNK_OVERLAP", 150 if LIGHT_MODE else 120)
MAX_CHUNKS_PER_PDF = _env_int("ROTINA_MAX_CHUNKS_PER_PDF", 5 if LIGHT_MODE else 40)
MAX_CHUNKS_TOTAL = _env_int("ROTINA_MAX_CHUNKS_TOTAL", 20 if LIGHT_MODE else 300)
CHROMA_ADD_BATCH = _env_int("ROTINA_CHROMA_ADD_BATCH", 1 if LIGHT_MODE else 4)
RAG_TOP_K = _env_int("ROTINA_RAG_TOP_K", 3 if LIGHT_MODE else 6)
USE_TXTAI_CHUNKING = _env_bool("ROTINA_USE_TXTAI", "0") and not LIGHT_MODE

# Muda o cache do Streamlit quando você alterar limites no .env
INDEX_PROFILE = (
    f"light={int(LIGHT_MODE)}|cs={CHUNK_CHAR_SIZE}|ov={CHUNK_CHAR_OVERLAP}|"
    f"pp={MAX_CHUNKS_PER_PDF}|tot={MAX_CHUNKS_TOTAL}|bat={CHROMA_ADD_BATCH}|"
    f"txtai={int(USE_TXTAI_CHUNKING)}"
)

PDF_NAMES = (
    "regimento_interno_escola.pdf",
    "planejamento_nutricional_mensal.pdf",
    "guia_procedimentos_saude_seguranca.pdf",
    "PPP_DED_IBC.pdf",
)

CHROMA_COLLECTION = "rotina_viva_docs"

SYSTEM_PERSONA = """Você é o assistente "Rotina Viva", da escola infantil.
- Tom empático, claro e respeitoso com pais, mães, responsáveis e professoras.
- Use apenas as informações fornecidas nos blocos de contexto (dados tabulares e trechos de documentos).
- Se trechos trouxerem nome, denominação ou título com o nome da escola (ex.: linha começando em "Escola", ou "Título: ... Escola ..."), cite isso na resposta. Não diga que o nome "não consta" se ele aparecer literalmente no contexto.
- Se algo realmente não estiver no contexto, diga com honestidade e sugira falar com a coordenação.
- Nunca invente nomes de crianças, datas ou ocorrências que não apareçam no contexto.
- Responda em português do Brasil, de forma objetiva e acolhedora."""

SYSTEM_GROUNDING = """Leia o contexto abaixo antes de responder.
- Priorize fatos que estejam escritos nos trechos ou na tabela.
- Só diga que uma informação não aparece se, depois de verificar o contexto, ela de fato não estiver lá.
- Para perguntas sobre identidade da escola, procure linhas como nome fantasia, cabeçalho, "Escola ..." ou campo "Título:" nos documentos."""

_ASKS_SCHOOL_NAME = re.compile(
    r"(qual\s+[ée]\s+o\s+nome\s+da\s+escola|nome\s+da\s+escola|como\s+se\s+chama\s+a\s+escola)",
    re.IGNORECASE,
)


def extract_school_name_hints_from_rag(rag_block: str) -> list[str]:
    """Extrai possíveis nomes da escola do texto RAG para reforçar modelos pequenos."""
    if not rag_block or rag_block.startswith("(Nenhum") or rag_block.startswith("(sem trechos"):
        return []
    skip = ("uniforme", "logotipo", "entrada", "saída", "saida", "tolerância", "tolerancia", "uso do")
    hints: list[str] = []
    for line in rag_block.splitlines():
        s = line.strip()
        if not s.lower().startswith("escola"):
            continue
        sl = s.lower()
        if any(x in sl for x in skip):
            continue
        if 8 <= len(s) <= 90:
            hints.append(s)
    for m in re.finditer(
        r"Escola\s+Infantil\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]{2,45}?(?=\s+Vigência|\s+Horários|,|\.)",
        rag_block,
        re.IGNORECASE,
    ):
        hints.append(m.group(0).strip())
    for m in re.finditer(r"Título:\s*([^\n]{8,110})", rag_block, re.IGNORECASE):
        t = m.group(1).strip()
        if "escola" in t.lower():
            hints.append(t)
    out: list[str] = []
    seen: set[str] = set()
    for h in hints:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out[:8]


def school_name_reinforcement(user_message: str, rag_block: str) -> str | None:
    if not _ASKS_SCHOOL_NAME.search(user_message):
        return None
    hints = extract_school_name_hints_from_rag(rag_block)
    if not hints:
        return None
    joined = " | ".join(hints)
    return (
        f'A pergunta é sobre o nome da escola. Nos trechos já constam estas denominações (use na resposta, citando o texto): "{joined}". '
        "Responda de forma direta com o(s) nome(s) encontrado(s). Não diga que a informação não existe nos documentos, pois ela está listada acima a partir dos trechos."
    )

SCHEMA_FOR_LLM = """
Tabelas DuckDB (somente leitura; use SELECT):

1) info_alunos
   Colunas: id_aluno (INTEGER), nome (TEXT), turma (TEXT), alergias (TEXT), contato_pais (TEXT)

2) diario_estruturado
   Colunas: id_registro (INTEGER), id_aluno (INTEGER), data (DATE ou TEXT),
   cafe_manha, almoco, lanche_tarde, jantar_extra (TEXT),
   trocas_banheiro (INTEGER), evacuacao (TEXT), medicamentos (TEXT),
   hora_sono_inicio, hora_sono_fim (TEXT), qualidade_sono (TEXT),
   atividade_dia (TEXT), interacao_social (TEXT), recado_professora (TEXT)

Para juntar aluno e diário: JOIN diario_estruturado d ON d.id_aluno = a.id_aluno JOIN info_alunos a ...
"""


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def _resolve_info_alunos_csv(base: Path) -> Path:
    for name in ("info_alunos.csv", "info_alunos_v2.csv"):
        p = base / name
        if p.exists():
            return p
    return base / "info_alunos.csv"


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


def format_sql_rows(rows: list[Any], columns: list[str], max_rows: int = 80) -> str:
    if not rows:
        return "(nenhuma linha retornada)"
    show = rows[:max_rows]
    lines = [" | ".join(columns)]
    for r in show:
        vals = [str(r[i]) for i in range(len(columns))]
        lines.append(" | ".join(vals))
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} linhas omitidas)")
    return "\n".join(lines)


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)


def flatten_segments(seg: Any) -> list[str]:
    """Normaliza saída do txtai Segmentation para lista de strings."""
    if isinstance(seg, str):
        return [seg.strip()] if seg.strip() else []
    if isinstance(seg, (list, tuple)):
        out: list[str] = []
        for item in seg:
            out.extend(flatten_segments(item))
        return out
    return []


def chunk_text_by_chars(text: str, size: int, overlap: int, max_chunks: int) -> list[str]:
    """Chunking fixo por caracteres — leve na CPU; `max_chunks` evita indexação infinita."""
    text = text.strip()
    if not text or max_chunks <= 0:
        return []
    size = max(200, size)
    overlap = max(0, min(overlap, size // 2))
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n and len(out) < max_chunks:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if len(piece) >= 40:
            out.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return out


def chunk_pdf_for_index(text: str, per_pdf_cap: int) -> list[str]:
    """Escolhe estratégia de chunking conforme modo leve / txtai."""
    if not text.strip():
        return []
    if LIGHT_MODE or not USE_TXTAI_CHUNKING:
        return chunk_text_by_chars(text, CHUNK_CHAR_SIZE, CHUNK_CHAR_OVERLAP, per_pdf_cap)
    from txtai.pipeline import Segmentation

    segment = Segmentation(paragraphs=True, minlength=200, cleantext=True)
    raw = segment(text)
    chunks = [c for c in flatten_segments(raw) if len(c) >= 60]
    return chunks[:per_pdf_cap]


def ollama_embedding_function():
    try:
        from chromadb.utils.embedding_functions.ollama_embedding_function import (
            OllamaEmbeddingFunction,
        )
    except ImportError:
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    return OllamaEmbeddingFunction(url=OLLAMA_HOST, model_name=OLLAMA_EMBED_MODEL)


def add_with_retry(
    collection: chromadb.Collection,
    documents: list[str],
    ids: list[str],
    metadatas: list[dict[str, Any]],
    batch_size: int | None = None,
) -> None:
    """
    Indexa lotes no Chroma com retry para tolerar timeout do Ollama embeddings.
    Em caso extremo, degrada para 1 documento por chamada.
    """
    if not documents:
        return

    batch_size = max(1, batch_size or CHROMA_ADD_BATCH)
    for i in range(0, len(documents), batch_size):
        d = documents[i : i + batch_size]
        ix = ids[i : i + batch_size]
        md = metadatas[i : i + batch_size]

        ok = False
        for attempt in range(4):
            try:
                collection.add(documents=d, ids=ix, metadatas=md)
                ok = True
                break
            except Exception:
                # Backoff progressivo: 1s, 2s, 4s, 8s
                time.sleep(2**attempt)

        if ok:
            continue

        # Fallback: adiciona item a item para não perder indexação inteira.
        for j in range(len(d)):
            single_ok = False
            for attempt in range(3):
                try:
                    collection.add(documents=[d[j]], ids=[ix[j]], metadatas=[md[j]])
                    single_ok = True
                    break
                except Exception:
                    time.sleep(1 + attempt)
            if not single_ok:
                raise RuntimeError(
                    f"Falha ao indexar chunk {ix[j]} após múltiplas tentativas."
                )


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------


@st.cache_resource
def get_duckdb_connection(data_dir_str: str) -> duckdb.DuckDBPyConnection:
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
        "CREATE OR REPLACE TABLE info_alunos AS SELECT * FROM read_csv_auto(?);",
        [str(info_csv)],
    )
    conn.execute(
        "CREATE OR REPLACE TABLE diario_estruturado AS SELECT * FROM read_csv_auto(?);",
        [str(diario_csv)],
    )
    return conn


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


# ---------------------------------------------------------------------------
# ChromaDB (persistente) + indexação com chunks txtai
# ---------------------------------------------------------------------------


@st.cache_resource
def get_chroma_collection(
    persist_dir_str: str,
    data_dir_str: str,
    ollama_url: str,
    embed_model: str,
    index_profile: str,
) -> chromadb.Collection:
    _ = index_profile  # só invalida o cache do Streamlit quando o perfil muda
    persist_dir = Path(persist_dir_str)
    data_dir = Path(data_dir_str)
    persist_dir.mkdir(parents=True, exist_ok=True)

    emb = ollama_embedding_function()
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=emb,
        metadata={"description": "Rotina Viva — documentos institucionais"},
    )

    if collection.count() == 0:
        docs: list[str] = []
        ids: list[str] = []
        metas: list[dict[str, Any]] = []
        total_used = 0
        for pdf_name in PDF_NAMES:
            if total_used >= MAX_CHUNKS_TOTAL:
                break
            pdf_path = data_dir / pdf_name
            if not pdf_path.exists():
                continue
            full_text = extract_pdf_text(pdf_path)
            cap = min(MAX_CHUNKS_PER_PDF, MAX_CHUNKS_TOTAL - total_used)
            chunks = chunk_pdf_for_index(full_text, cap)
            for i, ch in enumerate(chunks):
                docs.append(ch)
                ids.append(f"{pdf_name}::{i}")
                metas.append({"source": pdf_name, "chunk": str(i)})
            total_used += len(chunks)
        if docs:
            add_with_retry(collection, docs, ids, metas, batch_size=CHROMA_ADD_BATCH)
    return collection


def retrieve_rag_context(collection: chromadb.Collection, question: str, k: int | None = None) -> str:
    n = collection.count()
    if n == 0:
        return "(Nenhum documento indexado. Coloque os PDFs em ROTINA_DATA_DIR e reinicie o app.)"
    top = k if k is not None else RAG_TOP_K
    res = collection.query(query_texts=[question], n_results=min(top, max(1, n)))
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    parts: list[str] = []
    for doc, meta in zip(docs, metas):
        src = (meta or {}).get("source", "?")
        parts.append(f"[Fonte: {src}]\n{doc}")
    return "\n\n---\n\n".join(parts) if parts else "(sem trechos relevantes)"


# ---------------------------------------------------------------------------
# Ollama — planejamento (JSON) e resposta em streaming
# ---------------------------------------------------------------------------

_INSTITUTIONAL_Q = re.compile(
    r"(nome da escola|nome da instituição|como se chama a escola|qual [ée] o nome|"
    r"identidade da escola|cnpj da escola|endere[çc]o da escola|miss[ãa]o|vis[ãa]o|"
    r"quem somos|institui[çc][ãa]o|secretaria|diretoria|coordena[çc][ãa]o geral)",
    re.IGNORECASE,
)


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

    sql_val = p.get("sql")
    sql_str = sql_val.strip() if isinstance(sql_val, str) else ""
    has_sql = bool(sql_str)

    if "sql" in fontes and not has_sql:
        fontes = [f for f in fontes if f != "sql"]
        p["sql"] = None
        if "rag" not in fontes:
            fontes.append("rag")

    um = user_message.strip()
    ul = um.lower()
    if _INSTITUTIONAL_Q.search(um) or (
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

    p["fontes"] = fontes
    if "sql" not in fontes:
        p["sql"] = None
    return p


def _routing_planner_user_content(user_message: str) -> str:
    return f"""Classifique a pergunta sobre uma escola infantil.

{SCHEMA_FOR_LLM}

Regras (siga com cuidado):
- "rag": documentos oficiais — regimento, normas, PPP, proposta pedagógica, segurança/saúde institucional, cardápio/nutrição em documento, horários gerais da escola, **nome da escola / identidade institucional / endereço / missão** quando estiver em texto oficial.
- "sql": **somente** quando a pergunta pede dados do **cadastro ou diário** (criança específica, turma, alergias cadastradas, refeições do dia no diário, sono, evacuação, medicamentos do dia, recado da professora, listagens/contagens a partir das tabelas CSV).
- **Não** use "sql" para "qual é o nome da escola?" — isso é "rag" (documentos).
- Use ambos ["rag","sql"] só se a pergunta claramente precisar de documento oficial **e** de linhas do diário/cadastro.

Responda somente JSON válido:
{{"fontes": ["rag"] ou ["sql"] ou ["rag","sql"] ou ["sql","rag"], "sql": null ou uma string com UMA consulta SELECT DuckDB}}

Se "sql" estiver em fontes, "sql" deve ser a string SELECT. Se não souber a consulta, **não** inclua "sql" em fontes — use só "rag".

Pergunta:
{user_message}
"""


def ollama_plan_sources(user_message: str) -> dict[str, Any]:
    planner = _routing_planner_user_content(user_message)
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "Responda apenas com JSON válido, sem markdown."},
            {"role": "user", "content": planner},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    try:
        r = httpx.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=_llm_plan_timeout(),
        )
        r.raise_for_status()
        data = r.json()
        raw = (data.get("message") or {}).get("content") or "{}"
        return json.loads(raw)
    except Exception:
        return {"fontes": ["rag"], "sql": None}


def ollama_chat_stream(
    user_message: str,
    duck_block: str,
    rag_block: str,
    history: Iterable[dict[str, str]],
) -> Generator[str, None, None]:
    ctx = f"""## Dados tabulares (consultas internas)
{duck_block}

## Trechos de documentos institucionais
{rag_block}
"""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {"role": "system", "content": SYSTEM_GROUNDING},
        {"role": "system", "content": ctx},
    ]
    sn = school_name_reinforcement(user_message, rag_block)
    if sn:
        messages.append({"role": "system", "content": sn})
    for m in history:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.4},
    }
    try:
        with httpx.Client(timeout=_ollama_stream_timeout()) as client:
            with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as resp:
                if resp.status_code == 404:
                    detail = resp.read().decode(errors="replace").strip()
                    yield (
                        f"**Ollama (404):** o modelo `{OLLAMA_CHAT_MODEL}` ainda não está disponível "
                        f"no servidor em `{OLLAMA_HOST}`. Normal na primeira subida: aguarde o "
                        f"`docker compose` terminar de baixar o modelo, ou rode no container Ollama: "
                        f"`ollama pull {OLLAMA_CHAT_MODEL}`.\n\n"
                    )
                    if detail:
                        yield f"_Detalhe:_ {detail[:1200]}"
                    return
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("done"):
                        break
                    msg = obj.get("message") or {}
                    piece = msg.get("content") or ""
                    if piece:
                        yield piece
    except httpx.ReadTimeout:
        yield (
            "**Timeout no Ollama:** a geração demorou demais (comum em CPU fraca). "
            "Confirme `LLM_STREAM_READ_TIMEOUT=0` (padrão: sem limite de leitura) ou use `ROTINA_CHAT_PROVIDER=openai` "
            "com API externa (veja `.env`)."
        )
    except httpx.TimeoutException as e:
        yield f"**Timeout de rede (Ollama):** `{e}` — confira se `{OLLAMA_HOST}` responde."


def _openai_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }


def openai_plan_sources(user_message: str) -> dict[str, Any]:
    planner = _routing_planner_user_content(user_message)
    messages = [
        {"role": "system", "content": "Responda apenas com JSON válido, sem markdown."},
        {"role": "user", "content": planner},
    ]
    url = f"{OPENAI_BASE_URL}/chat/completions"
    body: dict[str, Any] = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    try:
        r = httpx.post(url, headers=_openai_headers(), json=body, timeout=120.0)
        if r.status_code == 400:
            body.pop("response_format", None)
            r = httpx.post(url, headers=_openai_headers(), json=body, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content") or "{}"
        return json.loads(raw)
    except Exception:
        return {"fontes": ["rag"], "sql": None}


def openai_chat_stream(
    user_message: str,
    duck_block: str,
    rag_block: str,
    history: Iterable[dict[str, str]],
) -> Generator[str, None, None]:
    ctx = f"""## Dados tabulares (consultas internas)
{duck_block}

## Trechos de documentos institucionais
{rag_block}
"""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PERSONA},
        {"role": "system", "content": SYSTEM_GROUNDING},
        {"role": "system", "content": ctx},
    ]
    sn = school_name_reinforcement(user_message, rag_block)
    if sn:
        messages.append({"role": "system", "content": sn})
    for m in history:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    url = f"{OPENAI_BASE_URL}/chat/completions"
    body = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.4,
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=60.0, read=None, write=120.0, pool=120.0)) as client:
            with client.stream("POST", url, headers=_openai_headers(), json=body) as resp:
                if resp.status_code == 401:
                    yield "**API:** chave inválida ou ausente (`OPENAI_API_KEY`)."
                    return
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content") or ""
                        if piece:
                            yield piece
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.text[:800]
        except Exception:
            pass
        yield f"**Erro HTTP da API:** {e.response.status_code}. {detail}"
    except httpx.TimeoutException as e:
        yield f"**Timeout na API:** `{e}`"


def llm_plan_sources(user_message: str) -> dict[str, Any]:
    if _use_openai_chat():
        return openai_plan_sources(user_message)
    return ollama_plan_sources(user_message)


def llm_chat_stream(
    user_message: str,
    duck_block: str,
    rag_block: str,
    history: Iterable[dict[str, str]],
) -> Generator[str, None, None]:
    if _use_openai_chat():
        yield from openai_chat_stream(user_message, duck_block, rag_block, history)
        return
    if ROTINA_CHAT_PROVIDER == "openai" and not OPENAI_API_KEY:
        yield (
            "**Configuração:** `ROTINA_CHAT_PROVIDER=openai` exige `OPENAI_API_KEY` "
            "(e opcionalmente `OPENAI_CHAT_MODEL`, `OPENAI_BASE_URL`)."
        )
        return
    yield from ollama_chat_stream(user_message, duck_block, rag_block, history)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])


def main() -> None:
    st.set_page_config(page_title="Rotina Viva", layout="wide", initial_sidebar_state="expanded")
    init_session_state()

    st.title("Rotina Viva")
    st.caption(
        "Assistente inteligente — PUCPR · AI Factory · Ollama + DuckDB + ChromaDB"
        + (" · modo leve (poucos chunks)" if LIGHT_MODE else "")
    )

    with st.sidebar:
        st.subheader("Ambiente")
        st.markdown(
            f"- **Dados:** `{DATA_DIR}`\n"
            f"- **ChromaDB:** `{CHROMA_DIR}`\n"
            f"- **Ollama:** `{OLLAMA_HOST}`\n"
            f"- **LLM:** `{OLLAMA_CHAT_MODEL}` · **Embeddings:** `{OLLAMA_EMBED_MODEL}`\n"
            f"- **Indexação:** modo leve=`{LIGHT_MODE}` · até `{MAX_CHUNKS_TOTAL}` chunks no total "
            f"(`{MAX_CHUNKS_PER_PDF}`/PDF) · lote Chroma=`{CHROMA_ADD_BATCH}`"
        )
        if st.button("Limpar conversa"):
            st.session_state.messages = []
            st.rerun()

    try:
        conn = get_duckdb_connection(str(DATA_DIR))
    except Exception as e:
        st.error(f"Falha ao carregar DuckDB: {e}")
        conn = None

    collection = None
    if conn is not None:
        try:
            collection = get_chroma_collection(
                str(CHROMA_DIR),
                str(DATA_DIR),
                OLLAMA_HOST,
                OLLAMA_EMBED_MODEL,
                INDEX_PROFILE,
            )
            if collection.count() == 0:
                st.warning(
                    "Índice Chroma vazio: adicione os PDFs em `ROTINA_DATA_DIR` e reinicie para indexar."
                )
        except Exception as e:
            st.error(f"Falha ao preparar ChromaDB/Ollama embeddings: {e}")

    if prompt := st.chat_input("Pergunte sobre rotinas, alunos ou documentos da escola…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user":
        return

    user_text = st.session_state.messages[-1]["content"]

    if conn is None or collection is None:
        err = "Serviços de dados não disponíveis. Verifique os CSVs e a conexão com o Ollama."
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
        return

    history_for_model = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        plan = normalize_plan(llm_plan_sources(user_text), user_text)
        fontes = plan.get("fontes") or ["rag"]
        if isinstance(fontes, str):
            fontes = [fontes]

        duck_block = "(nenhuma consulta SQL executada)"
        if "sql" in fontes:
            sql = plan.get("sql")
            if isinstance(sql, str) and sql.strip():
                duck_block, ok = run_safe_select(conn, sql)
                if not ok:
                    duck_block = f"{duck_block}\n(Tente reformular com nome do aluno ou data, se aplicável.)"
            else:
                duck_block = "Nenhuma consulta SQL válida foi gerada para esta pergunta."

        rag_block = "(busca em documentos não solicitada)"
        if "rag" in fontes:
            rag_block = retrieve_rag_context(collection, user_text, k=RAG_TOP_K)

        def _gen() -> Generator[str, None, None]:
            yield from llm_chat_stream(user_text, duck_block, rag_block, history_for_model)

        full = st.write_stream(_gen) or ""

        with st.expander("Detalhes (Etapa 1)", expanded=False):
            st.json({"plano": plan, "fontes": fontes})
            st.text_area("Contexto SQL (trecho)", duck_block[:6000], height=160)
            st.text_area("Contexto RAG (trecho)", rag_block[:6000], height=160)

    st.session_state.messages.append({"role": "assistant", "content": full})


if __name__ == "__main__":
    main()
