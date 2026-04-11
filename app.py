"""
Rotina Viva — assistente para rotinas em escolas infantis (AI Factory / PUCPR).
Streamlit + DuckDB (CSVs) + ChromaDB (RAG) + txtai (segmentação) + LLM/embeddings (API ou Ollama local).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
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
from txtai.pipeline import Segmentation

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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DATA_DIR = Path(os.getenv("ROTINA_DATA_DIR", "data")).resolve()
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma")).resolve()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# LLM no chat: ollama (local) ou openai / openrouter (API compatível com OpenAI).
ROTINA_CHAT_PROVIDER = os.getenv("ROTINA_CHAT_PROVIDER", "ollama").strip().lower()
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("OPENROUTER_API_KEY", "").strip()
)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()

# Embeddings no Chroma: ollama (local) ou openrouter / openai (API compatível com OpenAI).
ROTINA_EMBED_PROVIDER = os.getenv("ROTINA_EMBED_PROVIDER", "ollama").strip().lower()
OPENAI_EMBED_BASE_URL = os.getenv("OPENAI_EMBED_BASE_URL", "").strip().rstrip("/")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "").strip()
if ROTINA_EMBED_PROVIDER in ("openrouter", "openai"):
    if not OPENAI_EMBED_BASE_URL:
        OPENAI_EMBED_BASE_URL = OPENAI_BASE_URL
    if not OPENAI_EMBED_MODEL:
        OPENAI_EMBED_MODEL = (
            "openai/text-embedding-3-small"
            if ROTINA_EMBED_PROVIDER == "openrouter"
            else "text-embedding-3-small"
        )

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


def _api_stream_httpx_timeout() -> httpx.Timeout:
    """
    Limite de espera entre chunks no SSE (OpenRouter/OpenAI).
    read=None permite conexão aberta sem dados — a UI parece travar para sempre.
    """
    raw = os.getenv("ROTINA_API_STREAM_READ_TIMEOUT", "").strip().lower()
    if raw in ("none", "unlimited"):
        return httpx.Timeout(connect=90.0, read=None, write=120.0, pool=120.0)
    if not raw or raw == "0":
        read_sec = 150.0
    else:
        try:
            read_sec = max(30.0, float(raw))
        except ValueError:
            read_sec = 150.0
    return httpx.Timeout(connect=90.0, read=read_sec, write=120.0, pool=120.0)


def _use_openai_compatible_chat() -> bool:
    return ROTINA_CHAT_PROVIDER in ("openai", "openrouter") and bool(OPENAI_API_KEY)


# Limites de chunking / RAG (ajustáveis por env; defaults para indexação “completa”).
CHUNK_CHAR_SIZE = _env_int("ROTINA_CHUNK_CHARS", 1000)
CHUNK_CHAR_OVERLAP = _env_int("ROTINA_CHUNK_OVERLAP", 120)
MAX_CHUNKS_PER_PDF = _env_int("ROTINA_MAX_CHUNKS_PER_PDF", 40)
MAX_CHUNKS_TOTAL = _env_int("ROTINA_MAX_CHUNKS_TOTAL", 300)
CHROMA_ADD_BATCH = _env_int("ROTINA_CHROMA_ADD_BATCH", 4)
RAG_TOP_K = _env_int("ROTINA_RAG_TOP_K", 6)
# >0: corta trechos cuja distância ao embedding da pergunta excede (melhor_distância + gap).
# Evita preencher K com PDFs pouco relacionados (ex.: cardápio em pergunta sobre nome da escola).
ROTINA_RAG_DISTANCE_GAP = _env_float("ROTINA_RAG_DISTANCE_GAP", 0.28)
# 0 = só ordem por embedding. >0 mistura com sobreposição de termos (ajuda tabelas/PDF “achatados”).
ROTINA_RAG_LEXICAL_WEIGHT = _env_float("ROTINA_RAG_LEXICAL_WEIGHT", 0.38)
# Fração mínima de termos da pergunta que precisam aparecer no trecho para manter o candidato
# mesmo se a distância vetorial estiver fora do gap (evita perder tabelas com números/rótulos).
ROTINA_RAG_LEXICAL_FLOOR = _env_float("ROTINA_RAG_LEXICAL_FLOOR", 0.14)
# Na indexação, prefixa trechos que parecem tabela para o embedding captar melhor “quadro/tabulação”.
ROTINA_RAG_TABLE_EMBED_PREFIX = _env_bool("ROTINA_RAG_TABLE_EMBED_PREFIX", "1")

# Limites de taxa para API gratuita (OpenRouter / OpenAI): embeddings e HTTP.
ROTINA_EMBED_API_BATCH_SIZE = _env_int("ROTINA_EMBED_API_BATCH_SIZE", 1)
ROTINA_API_EMBED_MIN_INTERVAL_SEC = _env_float(
    "ROTINA_API_EMBED_MIN_INTERVAL_SEC",
    2.0 if ROTINA_EMBED_PROVIDER in ("openrouter", "openai") else 0.0,
)
ROTINA_API_PAUSE_BETWEEN_PDF_SEC = _env_float(
    "ROTINA_API_PAUSE_BETWEEN_PDF_SEC",
    5.0 if ROTINA_EMBED_PROVIDER in ("openrouter", "openai") else 0.0,
)
ROTINA_API_PLAN_TO_CHAT_DELAY_SEC = _env_float(
    "ROTINA_API_PLAN_TO_CHAT_DELAY_SEC",
    1.5 if ROTINA_CHAT_PROVIDER in ("openrouter", "openai") else 0.0,
)
ROTINA_API_HTTP_MAX_RETRIES = _env_int("ROTINA_API_HTTP_MAX_RETRIES", 10)
ROTINA_API_ADD_MAX_RETRIES = _env_int("ROTINA_API_ADD_MAX_RETRIES", 12)
ROTINA_API_EMBED_MAX_RETRIES = _env_int("ROTINA_API_EMBED_MAX_RETRIES", 12)
# Teto de duração do streaming do chat (evita “rodando eternamente”).
ROTINA_STREAM_MAX_SECONDS = _env_float("ROTINA_STREAM_MAX_SECONDS", 600.0)
# Temperatura do chat: mais baixa quando há linhas SQL (reduz invenção em tabelas).
ROTINA_CHAT_TEMPERATURE = _env_float("ROTINA_CHAT_TEMPERATURE", 0.35)
ROTINA_CHAT_TEMP_WITH_SQL = _env_float("ROTINA_CHAT_TEMP_WITH_SQL", 0.12)


def effective_chroma_add_batch() -> int:
    if ROTINA_EMBED_PROVIDER in ("openrouter", "openai"):
        return max(1, ROTINA_EMBED_API_BATCH_SIZE)
    return CHROMA_ADD_BATCH


_EMBED_INDEX_TOKEN = (
    f"eprov=ollama|oh={OLLAMA_HOST}|em={OLLAMA_EMBED_MODEL}"
    if ROTINA_EMBED_PROVIDER == "ollama"
    else f"eprov={ROTINA_EMBED_PROVIDER}|eu={OPENAI_EMBED_BASE_URL}|em={OPENAI_EMBED_MODEL}"
)

# Muda o cache do Streamlit quando você alterar limites ou embeddings no .env
INDEX_PROFILE = (
    f"cs={CHUNK_CHAR_SIZE}|ov={CHUNK_CHAR_OVERLAP}|"
    f"pp={MAX_CHUNKS_PER_PDF}|tot={MAX_CHUNKS_TOTAL}|bat={CHROMA_ADD_BATCH}|"
    f"eab={effective_chroma_add_batch()}|emb_iv={ROTINA_API_EMBED_MIN_INTERVAL_SEC}|"
    f"pdfp={ROTINA_API_PAUSE_BETWEEN_PDF_SEC}|rdg={ROTINA_RAG_DISTANCE_GAP}|"
    f"rlw={ROTINA_RAG_LEXICAL_WEIGHT}|rlf={ROTINA_RAG_LEXICAL_FLOOR}|"
    f"tpre={int(ROTINA_RAG_TABLE_EMBED_PREFIX)}|seg=txtai|{_EMBED_INDEX_TOKEN}"
)


def _chroma_collection_name() -> str:
    """Nome estável por backend de embedding (não misturar vetores de modelos diferentes)."""
    if ROTINA_EMBED_PROVIDER == "ollama":
        return "rotina_viva_docs"
    key = f"{ROTINA_EMBED_PROVIDER}\0{OPENAI_EMBED_BASE_URL}\0{OPENAI_EMBED_MODEL}"
    return f"rv_docs_{hashlib.sha256(key.encode()).hexdigest()[:14]}"


CHROMA_COLLECTION = _chroma_collection_name()

PDF_NAMES = (
    "regimento_interno_escola.pdf",
    "planejamento_nutricional_mensal.pdf",
    "guia_procedimentos_saude_seguranca.pdf",
    "PPP_DED_IBC.pdf",
)

# RAG: perguntas de identidade institucional buscam só nestes PDFs (evita saúde/cardápio por menção genérica a “escola”).
RAG_IDENTITY_SOURCES: tuple[str, ...] = (
    "regimento_interno_escola.pdf",
    "PPP_DED_IBC.pdf",
)

_INSTITUTIONAL_Q = re.compile(
    r"(nome da escola|nome da instituição|como se chama a escola|qual [ée] o nome|"
    r"identidade da escola|cnpj da escola|endere[çc]o da escola|miss[ãa]o|vis[ãa]o|"
    r"quem somos|institui[çc][ãa]o|secretaria|diretoria|coordena[çc][ãa]o geral)",
    re.IGNORECASE,
)


def is_rag_identity_scope_question(message: str) -> bool:
    """Nome da escola, missão, endereço institucional, PPP/regimento como identidade — não nutrição nem protocolo de febre."""
    um = message.strip()
    ul = um.lower()
    if _INSTITUTIONAL_Q.search(um):
        return True
    return (
        "nome" in ul
        and "escola" in ul
        and "aluno" not in ul
        and "criança" not in ul
        and "crianca" not in ul
    )


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

SYSTEM_SQL_STRICT = """Dados tabulares (bloco "Dados tabulares" acima):
- A tabela é o resultado **literal** de uma consulta ao banco. Trate cada célula como dado real já filtrado.
- **Não invente** linhas, colunas, nomes de crianças, datas, refeições, medicamentos ou números que **não apareçam** nessa tabela.
- Se a tabela estiver vazia ou disser "(nenhuma linha retornada)", diga isso claramente — não preencha com suposições.
- Para contar, listar ou comparar, use **apenas** o que está nas linhas mostradas (e o número da coluna "linha" se existir).
- Se a pergunta pedir algo que a tabela não contém (coluna ausente), diga que o resultado atual não traz esse campo.
- Se várias linhas tiverem o mesmo nome e turmas diferentes, isso vem do cadastro (homônimos ou duplicidade): cite `id_aluno` de cada linha e não assuma um único aluno sem explicar."""

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
   **Alergias / restrições alimentares cadastradas** do aluno estão na coluna **`alergias`** (ex.: "Glúten", "Lactose", "Amendoim", "Nenhuma").
   Perguntas do tipo “tem alergia?”, “a que é alérgico?”, “quem tem alergia a leite?” → **sempre consulte `info_alunos`** com SELECT em `nome` e `alergias` (é dado de **cadastro**, não de PDF).
   Acentos: se o usuário escrever sem acento (ex. gluten), use `alergias ILIKE '%glúten%' OR alergias ILIKE '%gluten%'`
   ou busque por substantivo: `ILIKE '%lactose%'`, `ILIKE '%amendoim%'`.
   **Importante:** cada linha já liga **um cadastro** (`id_aluno`) ao **nome** e à **turma** na mesma linha.
   O mesmo texto em `nome` pode aparecer em **várias linhas** (homônimos ou erro de cadastro): aí existem várias turmas
   para “o mesmo nome”. Nesse caso liste **todas** as linhas com `id_aluno, nome, turma` e explique a ambiguidade.
   Não confunda com “lista de turmas da escola”: isso seria só `SELECT DISTINCT turma`, que **não** responde
   “qual a turma do fulano?” — para isso filtre pelo nome (ou id).

   Exemplos úteis:
   - Turma de um aluno pelo nome: `SELECT nome, turma FROM info_alunos WHERE nome ILIKE '%Rafael%Souza%'`
     (use partes do nome que o usuário disse; ILIKE ignora maiúsculas.)
   - Um aluno por id: `SELECT nome, turma FROM info_alunos WHERE id_aluno = 1`
   - Listar alunos de uma turma: `SELECT nome, turma FROM info_alunos WHERE turma = 'Infantil 3'`
   - Alergias de um aluno pelo nome: `SELECT nome, turma, alergias FROM info_alunos WHERE nome ILIKE '%Beatriz%Santos%'`
   - Alunos com certo tipo de alergia: `SELECT nome, turma, alergias FROM info_alunos WHERE alergias ILIKE '%lactose%'`
   - Listar quem não é “Nenhuma”: `SELECT nome, turma, alergias FROM info_alunos WHERE alergias NOT ILIKE '%nenhuma%'`

2) diario_estruturado
   Colunas: id_registro (INTEGER), id_aluno (INTEGER), data (DATE ou TEXT),
   cafe_manha, almoco, lanche_tarde, jantar_extra (TEXT),
   trocas_banheiro (INTEGER), evacuacao (TEXT), medicamentos (TEXT),
   hora_sono_inicio, hora_sono_fim (TEXT), qualidade_sono (TEXT),
   atividade_dia (TEXT), interacao_social (TEXT), recado_professora (TEXT)

Para juntar aluno e diário (ex.: refeições de um aluno por nome):
   `FROM diario_estruturado d JOIN info_alunos a ON d.id_aluno = a.id_aluno WHERE a.nome ILIKE '%...%'`
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


def _sql_cell_text(v: Any) -> str:
    return str(v).replace("|", "·").replace("\n", " ").strip()


def format_sql_rows(rows: list[Any], columns: list[str], max_rows: int = 80) -> str:
    if not rows:
        return (
            "(nenhuma linha retornada)\n"
            "→ Não invente dados de aluno/diário; informe que a consulta não retornou linhas."
        )
    show = rows[:max_rows]
    head_cells = ["linha"] + [_sql_cell_text(c) for c in columns]
    lines = [
        "=== Resultado SQL (literal) — use só estas linhas/células; não crie outras ===",
        f"Total de linhas retornadas: {len(rows)} (tabela abaixo: até {len(show)}).",
        "",
        "| " + " | ".join(head_cells) + " |",
        "| " + " | ".join(["---"] * len(head_cells)) + " |",
    ]
    for i, r in enumerate(show, start=1):
        cells = [str(i)] + [_sql_cell_text(r[j]) for j in range(len(columns))]
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > max_rows:
        lines.append(
            f"\n({len(rows) - max_rows} linhas omitidas — não invente o conteúdo delas.)"
        )
    return "\n".join(lines)


def duck_block_has_tabular_rows(duck_block: str) -> bool:
    """True se há pelo menos uma linha de dados da tabela SQL formatada."""
    s = (duck_block or "").strip()
    if not s:
        return False
    skip = (
        "(nenhuma consulta SQL executada)",
        "Nenhuma consulta SQL válida",
        "(nenhuma linha retornada)",
        "Consulta SQL rejeitada",
        "Erro ao executar SQL",
    )
    if any(x in s for x in skip):
        return False
    return bool(re.search(r"^\|\s*\d+\s*\|", s, re.MULTILINE))


def chat_stream_temperature(duck_block: str) -> float:
    return (
        ROTINA_CHAT_TEMP_WITH_SQL
        if duck_block_has_tabular_rows(duck_block)
        else ROTINA_CHAT_TEMPERATURE
    )


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
    """Chunking fixo por caracteres — reserva se a segmentação txtai não produzir trechos."""
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
    """Segmenta com txtai (parágrafos); só cai no chunking por caracteres se não houver trechos."""
    if not text.strip():
        return []
    segment = Segmentation(paragraphs=True, minlength=200, cleantext=True)
    raw = segment(text)
    chunks = [c for c in flatten_segments(raw) if len(c) >= 60]
    chunks = chunks[:per_pdf_cap]
    if chunks:
        return chunks
    return chunk_text_by_chars(text, CHUNK_CHAR_SIZE, CHUNK_CHAR_OVERLAP, per_pdf_cap)


_RAG_STOPWORDS = frozenset(
    {
        "que",
        "qual",
        "quais",
        "quando",
        "como",
        "onde",
        "sobre",
        "esse",
        "essa",
        "este",
        "esta",
        "isso",
        "aquilo",
        "para",
        "com",
        "sem",
        "por",
        "dos",
        "das",
        "pelo",
        "pela",
        "uma",
        "uns",
        "num",
        "numa",
        "the",
        "and",
        "from",
        "what",
        "which",
        "when",
        "how",
        "são",
        "ser",
        "tem",
        "foi",
        "está",
        "estao",
        "pode",
        "deve",
        "diz",
        "dizer",
        "nos",
        "nas",
        "pelos",
        "pelas",
    }
)


def _rag_tokenize(text: str) -> set[str]:
    """Tokens para cruzar pergunta × trecho (complementa embedding em tabelas e PDF denso)."""
    if not text:
        return set()
    low = text.lower()
    words = re.findall(r"\w+", low, flags=re.UNICODE)
    out: set[str] = set()
    for w in words:
        if len(w) >= 3 and w not in _RAG_STOPWORDS:
            out.add(w)
    for m in re.finditer(r"\d{2,}", text):
        out.add(m.group(0))
    return out


def _rag_lexical_recall(question: str, doc: str) -> float:
    qt = _rag_tokenize(question)
    if not qt:
        return 0.0
    dt = _rag_tokenize(doc)
    if not dt:
        return 0.0
    return len(qt & dt) / len(qt)


def _pdf_chunk_looks_tabular(text: str) -> bool:
    """Heurística para texto extraído de PDF que parece linhas/colunas (tabela ou quadro)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    tab_lines = sum(1 for ln in lines if "\t" in ln)
    multi_gap = sum(1 for ln in lines if len(re.split(r"\s{2,}", ln.strip())) >= 3)
    many_short = sum(1 for ln in lines if 4 <= len(ln) <= 120 and ln.count(" ") <= 8)
    return tab_lines >= 1 or multi_gap >= 3 or (multi_gap >= 2 and many_short >= 4)


def _rag_text_for_embedding_index(chunk: str) -> str:
    """Texto gravado no Chroma (e embedado). Prefixo leve em trechos tabulares."""
    if not ROTINA_RAG_TABLE_EMBED_PREFIX or not _pdf_chunk_looks_tabular(chunk):
        return chunk
    return (
        "Quadro ou tabela do documento (rótulos e valores em linhas). Conteúdo:\n" + chunk
    )


def _is_rate_limit_error(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    try:
        import openai

        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError) and getattr(exc, "status_code", None) == 429:
            return True
    except ImportError:
        pass
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "too many requests" in s


def _httpx_retry_after_seconds(resp: httpx.Response, attempt: int) -> float:
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return min(300.0, float(ra))
        except ValueError:
            pass
    return min(120.0, 6.0 * (2**attempt) + random.uniform(0, 2))


class ThrottledOpenAIEmbeddingFunction:
    """
    Encapsula OpenAIEmbeddingFunction: uma sequência de chamadas espaçadas,
    1 texto por request à API e retries com backoff em 429.
    """

    def __init__(self, inner: Any, min_interval_sec: float) -> None:
        self._inner = inner
        self._min_interval_sec = max(0.0, min_interval_sec)
        self._last_mono: float | None = None

    def _pace(self) -> None:
        if self._min_interval_sec <= 0:
            return
        now = time.monotonic()
        if self._last_mono is not None:
            gap = self._min_interval_sec - (now - self._last_mono)
            if gap > 0:
                time.sleep(gap)
        self._last_mono = time.monotonic()

    def __call__(self, input: list[str]) -> list[Any]:
        if not input:
            return []
        out: list[Any] = []
        for text in input:
            self._pace()
            last_err: BaseException | None = None
            for attempt in range(ROTINA_API_EMBED_MAX_RETRIES):
                try:
                    vecs = self._inner([text])
                    out.extend(vecs)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if not _is_rate_limit_error(e):
                        raise
                    delay = min(180.0, 10.0 * (2**attempt) + random.uniform(0, 2))
                    time.sleep(delay)
            if last_err is not None:
                raise last_err
        return out

    def embed_query(self, input: list[str]) -> list[Any]:
        """Chroma ≥0.5 usa `embed_query` em `collection.query`; sem isso quebra o RAG."""
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "throttled_openai"

    def default_space(self) -> Any:
        return self._inner.default_space()

    def supported_spaces(self) -> list[Any]:
        return self._inner.supported_spaces()


def build_chroma_embedding_function():
    if ROTINA_EMBED_PROVIDER == "ollama":
        try:
            from chromadb.utils.embedding_functions.ollama_embedding_function import (
                OllamaEmbeddingFunction,
            )
        except ImportError:
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

        return OllamaEmbeddingFunction(url=OLLAMA_HOST, model_name=OLLAMA_EMBED_MODEL)

    if ROTINA_EMBED_PROVIDER in ("openrouter", "openai"):
        if not OPENAI_API_KEY:
            raise ValueError(
                "ROTINA_EMBED_PROVIDER=openrouter ou openai exige OPENAI_API_KEY ou OPENROUTER_API_KEY."
            )
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        headers: dict[str, str] = {}
        referer = (
            os.getenv("OPENAI_HTTP_REFERER", "").strip()
            or os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        )
        if referer:
            headers["HTTP-Referer"] = referer
        title = (
            os.getenv("OPENAI_APP_TITLE", "").strip()
            or os.getenv("OPENROUTER_APP_TITLE", "").strip()
        )
        if title:
            headers["X-Title"] = title

        dim_raw = os.getenv("OPENAI_EMBED_DIMENSIONS", "").strip()
        dimensions: int | None = int(dim_raw) if dim_raw.isdigit() else None

        kwargs: dict[str, Any] = {
            "api_key": OPENAI_API_KEY,
            "model_name": OPENAI_EMBED_MODEL,
            "api_base": OPENAI_EMBED_BASE_URL,
            "default_headers": headers if headers else None,
        }
        if dimensions is not None and "text-embedding-3" in OPENAI_EMBED_MODEL:
            kwargs["dimensions"] = dimensions

        inner = OpenAIEmbeddingFunction(**kwargs)
        if ROTINA_API_EMBED_MIN_INTERVAL_SEC > 0:
            return ThrottledOpenAIEmbeddingFunction(inner, ROTINA_API_EMBED_MIN_INTERVAL_SEC)
        return inner

    raise ValueError(
        f"ROTINA_EMBED_PROVIDER inválido: {ROTINA_EMBED_PROVIDER!r} "
        "(use ollama, openrouter ou openai)."
    )


def add_with_retry(
    collection: chromadb.Collection,
    documents: list[str],
    ids: list[str],
    metadatas: list[dict[str, Any]],
    batch_size: int | None = None,
) -> None:
    """
    Indexa lotes no Chroma com retry (timeout / 429 na API de embeddings).
    Em caso extremo, degrada para 1 documento por chamada.
    """
    if not documents:
        return

    batch_size = max(1, batch_size or effective_chroma_add_batch())
    max_attempts = (
        ROTINA_API_ADD_MAX_RETRIES
        if ROTINA_EMBED_PROVIDER in ("openrouter", "openai")
        else 4
    )
    for i in range(0, len(documents), batch_size):
        d = documents[i : i + batch_size]
        ix = ids[i : i + batch_size]
        md = metadatas[i : i + batch_size]

        ok = False
        for attempt in range(max_attempts):
            try:
                collection.add(documents=d, ids=ix, metadatas=md)
                ok = True
                break
            except Exception as e:
                if _is_rate_limit_error(e):
                    time.sleep(min(120.0, 8.0 * (2**attempt) + random.uniform(0, 2)))
                else:
                    time.sleep(min(32.0, 2**attempt))

        if ok:
            continue

        # Fallback: adiciona item a item para não perder indexação inteira.
        for j in range(len(d)):
            single_ok = False
            single_max = max(5, max_attempts)
            for attempt in range(single_max):
                try:
                    collection.add(documents=[d[j]], ids=[ix[j]], metadatas=[md[j]])
                    single_ok = True
                    break
                except Exception as e:
                    if _is_rate_limit_error(e):
                        time.sleep(min(120.0, 6.0 * (2**attempt) + random.uniform(0, 1)))
                    else:
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
    index_profile: str,
) -> chromadb.Collection:
    _ = index_profile  # só invalida o cache do Streamlit quando o perfil muda
    persist_dir = Path(persist_dir_str)
    data_dir = Path(data_dir_str)
    persist_dir.mkdir(parents=True, exist_ok=True)

    emb = build_chroma_embedding_function()
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=emb,
        metadata={"description": "Rotina Viva — documentos institucionais"},
    )

    if collection.count() == 0:
        total_used = 0
        batch = effective_chroma_add_batch()
        for pdf_name in PDF_NAMES:
            if total_used >= MAX_CHUNKS_TOTAL:
                break
            pdf_path = data_dir / pdf_name
            if not pdf_path.exists():
                continue
            full_text = extract_pdf_text(pdf_path)
            cap = min(MAX_CHUNKS_PER_PDF, MAX_CHUNKS_TOTAL - total_used)
            chunks = chunk_pdf_for_index(full_text, cap)
            if not chunks:
                continue
            docs_pdf: list[str] = []
            ids_pdf: list[str] = []
            metas_pdf: list[dict[str, Any]] = []
            for i, ch in enumerate(chunks):
                docs_pdf.append(_rag_text_for_embedding_index(ch))
                ids_pdf.append(f"{pdf_name}::{i}")
                metas_pdf.append({"source": pdf_name, "chunk": str(i)})
            add_with_retry(collection, docs_pdf, ids_pdf, metas_pdf, batch_size=batch)
            total_used += len(chunks)
            if ROTINA_API_PAUSE_BETWEEN_PDF_SEC > 0:
                time.sleep(ROTINA_API_PAUSE_BETWEEN_PDF_SEC)
    return collection


def retrieve_rag_context(collection: chromadb.Collection, question: str, k: int | None = None) -> str:
    n = collection.count()
    if n == 0:
        return "(Nenhum documento indexado. Coloque os PDFs em ROTINA_DATA_DIR e reinicie o app.)"
    top = k if k is not None else RAG_TOP_K
    top = max(1, top)
    if ROTINA_RAG_DISTANCE_GAP > 0:
        fetch_n = min(n, max(top + 4, top * 3, 10))
    else:
        fetch_n = min(top, n)
    if ROTINA_RAG_LEXICAL_WEIGHT > 0:
        fetch_n = min(n, max(fetch_n, top * 4, 16))

    where_identity: dict[str, Any] | None = None
    if is_rag_identity_scope_question(question):
        where_identity = {"source": {"$in": list(RAG_IDENTITY_SOURCES)}}

    def _do_query(
        w: dict[str, Any] | None,
    ) -> Any:
        kw: dict[str, Any] = {
            "query_texts": [question],
            "n_results": max(1, fetch_n),
            "include": ["documents", "metadatas", "distances"],
        }
        if w is not None:
            kw["where"] = w
        return collection.query(**kw)

    res = _do_query(where_identity)
    if where_identity is not None:
        if not (res.get("documents") or [[]])[0]:
            res = _do_query(None)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    parts: list[str] = []

    if not docs:
        return "(sem trechos relevantes)"

    w_lex = ROTINA_RAG_LEXICAL_WEIGHT
    if w_lex > 0 and len(docs) == len(dists) and dists:
        lexs = [_rag_lexical_recall(question, d or "") for d in docs]
        d_best = min(float(x) for x in dists)
        inv = [1.0 / (1e-8 + float(d)) for d in dists]
        inv_m = max(inv) if inv else 1.0
        combined = [(1 - w_lex) * (inv[i] / inv_m) + w_lex * lexs[i] for i in range(len(docs))]
        order = sorted(range(len(docs)), key=lambda i: combined[i], reverse=True)
        picked: set[int] = set()
        for i in order:
            if len(parts) >= top:
                break
            dist_f = float(dists[i])
            if ROTINA_RAG_DISTANCE_GAP > 0:
                ok = (dist_f - d_best <= ROTINA_RAG_DISTANCE_GAP) or (
                    lexs[i] >= ROTINA_RAG_LEXICAL_FLOOR
                )
            else:
                ok = True
            if ok:
                picked.add(i)
                src = (metas[i] or {}).get("source", "?")
                parts.append(f"[Fonte: {src}]\n{docs[i]}")
        if len(parts) < top:
            for i in order:
                if len(parts) >= top:
                    break
                if i in picked:
                    continue
                picked.add(i)
                src = (metas[i] or {}).get("source", "?")
                parts.append(f"[Fonte: {src}]\n{docs[i]}")
    elif ROTINA_RAG_DISTANCE_GAP > 0 and dists and len(dists) == len(docs):
        base = float(dists[0])
        for doc, dist, meta in zip(docs, dists, metas):
            if float(dist) - base > ROTINA_RAG_DISTANCE_GAP:
                break
            src = (meta or {}).get("source", "?")
            parts.append(f"[Fonte: {src}]\n{doc}")
            if len(parts) >= top:
                break
    else:
        for doc, meta in zip(docs[:top], metas[:top]):
            src = (meta or {}).get("source", "?")
            parts.append(f"[Fonte: {src}]\n{doc}")
    return "\n\n---\n\n".join(parts) if parts else "(sem trechos relevantes)"


# ---------------------------------------------------------------------------
# Ollama — planejamento (JSON) e resposta em streaming
# ---------------------------------------------------------------------------


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


def _format_history_for_planner(
    history: Iterable[dict[str, str]], max_messages: int = 8
) -> str:
    """Trechos recentes do chat para o planejador resolver 'ele/ela', 'meu filho', etc."""
    rows = [
        m
        for m in history
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]
    if not rows:
        return ""
    tail = rows[-max_messages:]
    lines: list[str] = []
    for m in tail:
        label = "usuário" if m["role"] == "user" else "assistente"
        text = (m["content"] or "").strip()
        if len(text) > 1200:
            text = text[:1200] + "…"
        lines.append(f"[{label}]: {text}")
    return (
        "Conversa recente (use para resolver pronomes e continuações — ex.: de quem é “ele/ela”, "
        "qual nome citado antes):\n"
        + "\n".join(lines)
        + "\n\n"
    )


def _routing_planner_user_content(
    user_message: str,
    force: str | None = None,
    history: Iterable[dict[str, str]] | None = None,
) -> str:
    core = f"""Classifique a pergunta sobre uma escola infantil.

{SCHEMA_FOR_LLM}

Regras (siga com cuidado):
- "rag": documentos oficiais — regimento, normas, PPP, proposta pedagógica, segurança/saúde **institucional** (protocolos gerais da escola), cardápio/nutrição em documento, horários gerais da escola, **nome da escola / identidade institucional / endereço / missão** quando estiver em texto oficial.
- "sql": **cadastro ou diário** — turma de aluno, **alergias cadastradas por aluno** (`info_alunos.alergias`: não use só RAG por ser “saúde”), criança específica, refeições do dia no diário, sono, evacuação, medicamentos **anotados no diário**, recado da professora, listagens/contagens nas tabelas CSV.
- **Continuação de conversa:** se a pergunta atual usar só pronomes (“ele”, “ela”, “meu filho”) ou não repetir o nome, use a **conversa recente** para saber **qual criança** e monte o SQL com `WHERE nome ILIKE '%...%'` em `info_alunos` (ou `id_aluno` se tiver sido citado). Não deixe de filtrar pelo aluno quando a pergunta for claramente sobre a mesma criança do turno anterior.
- **Não** use "sql" para "qual é o nome da escola?" — isso é "rag" (documentos).
- Use ambos ["rag","sql"] só se a pergunta claramente precisar de documento oficial **e** de linhas do diário/cadastro.
- Tabelas e quadros em PDF costumam estar em "rag": peça termos que apareçam no documento (títulos de coluna, faixas etárias, nomes de refeição) — não use "sql" só porque a resposta parece uma tabela.
- Quando houver "sql" com SELECT válido, a resposta final ao usuário deve **copiar fielmente** as células retornadas — sem inventar linhas ou valores.

Responda somente JSON válido:
{{"fontes": ["rag"] ou ["sql"] ou ["rag","sql"] ou ["sql","rag"], "sql": null ou uma string com UMA consulta SELECT DuckDB}}

Se "sql" estiver em fontes, "sql" deve ser a string SELECT. Se não souber a consulta, **não** inclua "sql" em fontes — use só "rag".
"""
    if force == "sql_only":
        core += """
MODO **somente dados estruturados** (o usuário escolheu consultar só o DuckDB / tabelas CSV):
- Retorne {"fontes": ["sql"], "sql": "SELECT ..."} usando apenas tabelas e colunas do esquema acima.
- Perguntas do tipo “qual é a turma do [nome]?” → **obrigatoriamente** filtre `info_alunos` por `nome` (ex.: `WHERE nome ILIKE '%primeiro%ultimo%'`).
  **Não** use só `SELECT DISTINCT turma` ou listar turmas sem JOIN/WHERE no nome — isso não identifica o aluno.
- Perguntas sobre **alergia / intolerância / restrição alimentar de um aluno** (ou “quem tem alergia a X”) → `SELECT nome, turma, alergias FROM info_alunos` com `WHERE` em `nome` e/ou `alergias` (coluna **`alergias`**). Não devolva `sql: null` só porque a palavra parece “saúde”.
- Se a pergunta **não puder** ser respondida com essas tabelas (ex.: regulamento, PPP, texto de PDF), retorne {"fontes": [], "sql": null}. **Não** use "rag".
"""
    elif force == "rag_only":
        core += """
MODO **somente documentos** (o usuário escolheu consultar só o ChromaDB / PDFs indexados):
- Retorne sempre {"fontes": ["rag"], "sql": null}.
"""
    hist = _format_history_for_planner(history or [])
    return core + "\n" + hist + f"Pergunta atual:\n{user_message}\n"


def ollama_plan_sources(
    user_message: str,
    force: str | None = None,
    history: Iterable[dict[str, str]] | None = None,
) -> dict[str, Any]:
    planner = _routing_planner_user_content(user_message, force=force, history=history)
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
    ]
    if duck_block_has_tabular_rows(duck_block):
        messages.append({"role": "system", "content": SYSTEM_SQL_STRICT})
    messages.append({"role": "system", "content": ctx})
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
        "options": {"temperature": chat_stream_temperature(duck_block)},
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
            "Confirme `LLM_STREAM_READ_TIMEOUT=0` (padrão: sem limite de leitura) ou use `ROTINA_CHAT_PROVIDER=openrouter` "
            "/ `openai` (veja `.env`)."
        )
    except httpx.TimeoutException as e:
        yield f"**Timeout de rede (Ollama):** `{e}` — confira se `{OLLAMA_HOST}` responde."


def _format_api_error_body(text: str, max_len: int = 1500) -> str:
    """Extrai mensagem legível de JSON de erro (OpenRouter / OpenAI)."""
    text = (text or "").strip()[:max_len]
    if not text:
        return "(corpo vazio)"
    try:
        obj = json.loads(text)
        err = obj.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or err)[:max_len]
        if isinstance(err, str):
            return err[:max_len]
    except json.JSONDecodeError:
        pass
    return text


def _openai_headers() -> dict[str, str]:
    h: dict[str, str] = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    referer = (
        os.getenv("OPENAI_HTTP_REFERER", "").strip()
        or os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    )
    if referer:
        h["HTTP-Referer"] = referer
    title = (
        os.getenv("OPENAI_APP_TITLE", "").strip()
        or os.getenv("OPENROUTER_APP_TITLE", "").strip()
    )
    if title:
        h["X-Title"] = title
    return h


def openai_plan_sources(
    user_message: str,
    force: str | None = None,
    history: Iterable[dict[str, str]] | None = None,
) -> dict[str, Any]:
    planner = _routing_planner_user_content(user_message, force=force, history=history)
    messages = [
        {"role": "system", "content": "Responda apenas com JSON válido, sem markdown."},
        {"role": "user", "content": planner},
    ]
    url = f"{OPENAI_BASE_URL}/chat/completions"
    base: dict[str, Any] = {
        "model": OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": 0.1,
    }
    try:
        r: httpx.Response | None = None
        use_json_format = True
        for attempt in range(ROTINA_API_HTTP_MAX_RETRIES):
            work = {
                **base,
                **(
                    {"response_format": {"type": "json_object"}}
                    if use_json_format
                    else {}
                ),
            }
            r = httpx.post(url, headers=_openai_headers(), json=work, timeout=120.0)
            if r.status_code == 429 and attempt < ROTINA_API_HTTP_MAX_RETRIES - 1:
                time.sleep(_httpx_retry_after_seconds(r, attempt))
                continue
            if r.status_code == 400 and use_json_format:
                use_json_format = False
                continue
            break
        if r is None or r.status_code == 429:
            return {"fontes": ["rag"], "sql": None}
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
    ]
    if duck_block_has_tabular_rows(duck_block):
        messages.append({"role": "system", "content": SYSTEM_SQL_STRICT})
    messages.append({"role": "system", "content": ctx})
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
        "temperature": chat_stream_temperature(duck_block),
    }
    try:
        attempt = 0
        while attempt < ROTINA_API_HTTP_MAX_RETRIES:
            with httpx.Client(timeout=_api_stream_httpx_timeout()) as client:
                with client.stream("POST", url, headers=_openai_headers(), json=body) as resp:
                    if resp.status_code == 401:
                        yield "**API:** chave inválida ou ausente (`OPENAI_API_KEY` / `OPENROUTER_API_KEY`)."
                        return
                    if resp.status_code == 429:
                        resp.read()
                        attempt += 1
                        if attempt >= ROTINA_API_HTTP_MAX_RETRIES:
                            yield (
                                "**429 — limite de requisições da API.** "
                                "Aumente `ROTINA_API_EMBED_MIN_INTERVAL_SEC` / `ROTINA_API_PAUSE_BETWEEN_PDF_SEC` "
                                "no `.env` ou aguarde alguns minutos."
                            )
                            return
                        time.sleep(_httpx_retry_after_seconds(resp, attempt - 1))
                        continue
                    if resp.status_code == 400:
                        detail = _format_api_error_body(resp.read().decode(errors="replace"))
                        yield (
                            f"**Erro 400 da API** (pedido recusado). Detalhe: {detail}\n\n"
                            f"**Comum na OpenRouter:** `OPENAI_CHAT_MODEL` deve ser o **slug** do modelo "
                            f"(ex.: `meta-llama/llama-3.3-70b-instruct:free`), não o título da página do site.\n"
                            f"Modelo atual no `.env`: `{OPENAI_CHAT_MODEL}`"
                        )
                        return
                    resp.raise_for_status()
                    stream_deadline = time.monotonic() + max(60.0, ROTINA_STREAM_MAX_SECONDS)
                    finish_seen = False
                    try:
                        for line in resp.iter_lines():
                            if time.monotonic() > stream_deadline:
                                yield (
                                    "\n\n_(Tempo máximo de streaming atingido — "
                                    "aumente `ROTINA_STREAM_MAX_SECONDS` no `.env` se precisar de respostas mais longas.)_"
                                )
                                break
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
                            if obj.get("done") is True:
                                finish_seen = True
                            for choice in obj.get("choices") or []:
                                delta = choice.get("delta") or {}
                                piece = delta.get("content") or ""
                                if piece:
                                    yield piece
                                if choice.get("finish_reason"):
                                    finish_seen = True
                            if finish_seen:
                                break
                    except httpx.ReadTimeout:
                        yield (
                            "\n\n_(A API ficou muito tempo sem enviar dados — streaming encerrado. "
                            "Aumente `ROTINA_API_STREAM_READ_TIMEOUT` no `.env` se o modelo for lento.)_"
                        )
            return
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = _format_api_error_body(e.response.text)
        except Exception:
            pass
        hint = ""
        if e.response.status_code == 429:
            hint = (
                "\n\n**429:** reduza ritmo no `.env` (`ROTINA_API_EMBED_MIN_INTERVAL_SEC`, "
                "`ROTINA_API_PAUSE_BETWEEN_PDF_SEC`, `ROTINA_API_PLAN_TO_CHAT_DELAY_SEC`)."
            )
        yield f"**Erro HTTP da API:** {e.response.status_code}. {detail}{hint}"
    except httpx.TimeoutException as e:
        yield f"**Timeout na API:** `{e}`"


def llm_plan_sources(
    user_message: str,
    force: str | None = None,
    history: Iterable[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if _use_openai_compatible_chat():
        return openai_plan_sources(user_message, force=force, history=history)
    return ollama_plan_sources(user_message, force=force, history=history)


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
    return p


def llm_chat_stream(
    user_message: str,
    duck_block: str,
    rag_block: str,
    history: Iterable[dict[str, str]],
) -> Generator[str, None, None]:
    if _use_openai_compatible_chat():
        yield from openai_chat_stream(user_message, duck_block, rag_block, history)
        return
    if ROTINA_CHAT_PROVIDER in ("openai", "openrouter") and not OPENAI_API_KEY:
        yield (
            "**Configuração:** use `OPENAI_API_KEY` ou `OPENROUTER_API_KEY`, além de "
            "`OPENAI_BASE_URL` e `OPENAI_CHAT_MODEL` (para OpenRouter, veja `.env`)."
        )
        return
    yield from ollama_chat_stream(user_message, duck_block, rag_block, history)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("data_source_mode", "auto")


def main() -> None:
    st.set_page_config(page_title="Rotina Viva", layout="wide", initial_sidebar_state="expanded")
    init_session_state()

    st.title("Rotina Viva")
    st.caption("Assistente inteligente — PUCPR · AI Factory")

    with st.sidebar:
        st.subheader("Fonte da resposta")
        _mode_choices: tuple[tuple[str, str], ...] = (
            ("auto", "Automático (a IA escolhe SQL e/ou documentos)"),
            ("structured", "Só dados estruturados (DuckDB — cadastro e diário)"),
            ("documents", "Só documentos (ChromaDB — PDFs indexados)"),
        )
        _vals = [m[0] for m in _mode_choices]
        _labels = {m[0]: m[1] for m in _mode_choices}
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
        st.caption(
            "**Automático:** combina tabelas CSV e PDFs quando fizer sentido. "
            "**DuckDB:** alunos, turmas, diário, refeições cadastradas, etc. "
            "**Documentos:** regimento, PPP, normas e textos dos PDFs."
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

    if prompt := st.chat_input("Pergunte sobre rotinas, alunos ou documentos da escola…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.messages or st.session_state.messages[-1]["role"] != "user":
        return

    user_text = st.session_state.messages[-1]["content"]
    mode_ds = st.session_state.data_source_mode

    if conn is None:
        err = "DuckDB indisponível. Verifique os CSVs em `ROTINA_DATA_DIR`."
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
        return

    if mode_ds in ("auto", "documents") and collection is None:
        err = (
            "ChromaDB não está disponível ou o índice falhou. "
            "Para perguntas em documentos use a opção só PDFs após corrigir o ambiente, "
            "ou escolha só DuckDB se a pergunta for sobre cadastro/diário."
        )
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
        return

    history_for_model = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        plan_force: str | None = None
        if mode_ds == "structured":
            plan_force = "sql_only"
        elif mode_ds == "documents":
            plan_force = "rag_only"
        plan = normalize_plan(
            llm_plan_sources(user_text, force=plan_force, history=history_for_model),
            user_text,
        )
        plan = apply_user_data_source_mode(plan, mode_ds)
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
        if "rag" in fontes and collection is not None:
            rag_block = retrieve_rag_context(collection, user_text, k=RAG_TOP_K)

        if _use_openai_compatible_chat() and ROTINA_API_PLAN_TO_CHAT_DELAY_SEC > 0:
            time.sleep(ROTINA_API_PLAN_TO_CHAT_DELAY_SEC)

        def _gen() -> Generator[str, None, None]:
            yield from llm_chat_stream(user_text, duck_block, rag_block, history_for_model)

        full = st.write_stream(_gen) or ""

    st.session_state.messages.append({"role": "assistant", "content": full})


if __name__ == "__main__":
    main()
