"""ChromaDB RAG: indexação de PDFs e recuperação contextual (Rotina Viva)."""

from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

from modules.chat_service import (
    is_rag_identity_scope_question,
    is_rag_nutrition_meals_scope_question,
)


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


CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "data/vector_db")).resolve()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Provedor de chat (só para atraso plano→chat no UI; o motor está em `modules.ai_engine`).
ROTINA_CHAT_PROVIDER = os.getenv("ROTINA_CHAT_PROVIDER", "ollama").strip().lower()
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("OPENROUTER_API_KEY", "").strip()
)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

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

# Limites de chunking / RAG (ajustáveis por env). Chunking principal: RecursiveCharacterTextSplitter 1200/200.
RAG_RECURSIVE_CHUNK_SIZE = _env_int("ROTINA_RAG_RECURSIVE_CHUNK", 1200)
RAG_RECURSIVE_CHUNK_OVERLAP = _env_int("ROTINA_RAG_RECURSIVE_OVERLAP", 200)
CHUNK_CHAR_SIZE = _env_int("ROTINA_CHUNK_CHARS", 1200)
CHUNK_CHAR_OVERLAP = _env_int("ROTINA_CHUNK_OVERLAP", 200)
MAX_CHUNKS_PER_PDF = _env_int("ROTINA_MAX_CHUNKS_PER_PDF", 100)
MAX_CHUNKS_TOTAL = _env_int("ROTINA_MAX_CHUNKS_TOTAL", 500)
CHROMA_ADD_BATCH = _env_int("ROTINA_CHROMA_ADD_BATCH", 4)
RAG_TOP_K = _env_int("ROTINA_RAG_TOP_K", 3)
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

# Incrementar quando a extração ou o marcador entre páginas mudar (força reindexação).
RAG_PDF_EXTRACT_VERSION = 3

# PPP: sumário até ~pág. 28 — não indexar (equivalente a PyPDFLoader.load()[28:]).
PPP_PEDAGOGICO_PDF = "ppp_projeto_político_pedagógico.pdf"
PPP_SKIP_FIRST_PAGES = 28

# Linhas tipo sumário: título + pontinhos + número (remissão de página).
_RAG_TOC_DOT_LEADER_LINE = re.compile(r"\.{3,}\s*\d{1,4}\s*$")
# Entre páginas no texto extraído — usado como separador preferido no chunking recursivo.
RAG_PDF_PAGE_SPLIT_MARKER = "\n\n<<<PAGE_SPLIT>>>\n\n"

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
ROTINA_API_ADD_MAX_RETRIES = _env_int("ROTINA_API_ADD_MAX_RETRIES", 12)
ROTINA_API_EMBED_MAX_RETRIES = _env_int("ROTINA_API_EMBED_MAX_RETRIES", 12)



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
    f"rcs={RAG_RECURSIVE_CHUNK_SIZE}|rco={RAG_RECURSIVE_CHUNK_OVERLAP}|"
    f"cs={CHUNK_CHAR_SIZE}|ov={CHUNK_CHAR_OVERLAP}|"
    f"pp={MAX_CHUNKS_PER_PDF}|tot={MAX_CHUNKS_TOTAL}|bat={CHROMA_ADD_BATCH}|"
    f"eab={effective_chroma_add_batch()}|emb_iv={ROTINA_API_EMBED_MIN_INTERVAL_SEC}|"
    f"pdfp={ROTINA_API_PAUSE_BETWEEN_PDF_SEC}|rdg={ROTINA_RAG_DISTANCE_GAP}|"
    f"rlw={ROTINA_RAG_LEXICAL_WEIGHT}|rlf={ROTINA_RAG_LEXICAL_FLOOR}|"
    f"tpre={int(ROTINA_RAG_TABLE_EMBED_PREFIX)}|seg=recursive|"
    f"pexv={RAG_PDF_EXTRACT_VERSION}|{_EMBED_INDEX_TOKEN}"
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
    "planejamento_nutricional_semanal.pdf",
    "guia_procedimentos_saude_seguranca.pdf",
    "PPP_DED_IBC.pdf",
    "ppp_projeto_político_pedagógico.pdf",
)

# RAG: perguntas de identidade institucional buscam só nestes PDFs (evita saúde/cardápio por menção genérica a “escola”).
RAG_IDENTITY_SOURCES: tuple[str, ...] = (
    "regimento_interno_escola.pdf",
    "PPP_DED_IBC.pdf",
    "ppp_projeto_político_pedagógico.pdf",
)

# RAG: café da manhã, lanche, cardápio, refeições — apenas o planejamento nutricional semanal.
RAG_NUTRITION_SOURCES: tuple[str, ...] = (
    "planejamento_nutricional_semanal.pdf",
)


def _rag_pdf_manifest_fingerprint(data_dir: Path) -> str:
    """Muda com a lista `PDF_NAMES` ou com mtimes / presença dos ficheiros em `data_dir`."""
    parts: list[str] = []
    for name in PDF_NAMES:
        p = data_dir / name
        if p.is_file():
            parts.append(f"{name}\t{p.stat().st_mtime_ns}")
        else:
            parts.append(f"{name}\tMISSING")
    raw = (
        "\n".join(parts)
        + "\n|pdf_names="
        + "|".join(PDF_NAMES)
        + f"\n|pexv={RAG_PDF_EXTRACT_VERSION}|ppp_skip={PPP_SKIP_FIRST_PAGES}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _rag_fingerprint_path(persist_dir: Path) -> Path:
    return persist_dir / ".rotina_rag_index_fingerprint"


def _read_rag_stored_fingerprint(persist_dir: Path) -> str | None:
    path = _rag_fingerprint_path(persist_dir)
    try:
        t = path.read_text(encoding="utf-8").strip()
        return t or None
    except OSError:
        return None


def _write_rag_stored_fingerprint(persist_dir: Path, value: str) -> None:
    try:
        _rag_fingerprint_path(persist_dir).write_text(value, encoding="utf-8")
    except OSError:
        pass


def _chromadb_client_persist(client: Any) -> None:
    """Algumas versões do Chroma expõem `persist()` no cliente persistente."""
    persist_fn = getattr(client, "persist", None)
    if callable(persist_fn):
        try:
            persist_fn()
        except Exception:
            pass


def reset_rotina_chroma_persist(persist_dir: Path | None = None) -> None:
    """Apaga o diretório do ChromaDB (reindexação completa no próximo `get_chroma_collection`)."""
    p = (persist_dir or CHROMA_DIR).resolve()
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def _drop_rotina_chroma_collection(client: Any, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass



# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _line_is_toc_dot_leader(s: str) -> bool:
    """True se a linha parece remissão de sumário (pontos + número final)."""
    t = s.strip()
    if len(t) < 6 or t.startswith("==="):
        return False
    return bool(_RAG_TOC_DOT_LEADER_LINE.search(t))


def strip_toc_dot_leader_lines(text: str) -> str:
    """Remove linhas com padrão 'título .... 29' (resíduos de sumário / rodapé)."""
    if not text.strip():
        return text
    out_lines: list[str] = []
    for line in text.splitlines():
        if _line_is_toc_dot_leader(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def extract_pdf_text(path: Path, *, skip_first_pages: int = 0) -> str:
    """
    Extrai texto página a página com cabeçalho `=== PDF página N ===` e marcador
    `RAG_PDF_PAGE_SPLIT_MARKER` entre páginas. `skip_first_pages` ignora as N
    primeiras páginas do ficheiro (índices 0..N-1), equivalente a
    `PyPDFLoader.load()[N:]`.
    """
    reader = PdfReader(str(path))
    skip = max(0, int(skip_first_pages))
    page_list = reader.pages[skip:] if skip else reader.pages
    parts: list[str] = []
    for j, page in enumerate(page_list):
        physical_page = skip + j + 1
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        body = strip_toc_dot_leader_lines(t.strip())
        if body:
            parts.append(f"=== PDF página {physical_page} ===\n{body}")
    return RAG_PDF_PAGE_SPLIT_MARKER.join(parts)


def chunk_text_by_chars(text: str, size: int, overlap: int, max_chunks: int) -> list[str]:
    """Chunking fixo por caracteres — reserva se o splitter recursivo não produzir trechos."""
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
    """
    Fragmentação com RecursiveCharacterTextSplitter (economia de tokens vs. texto inteiro).
    Tamanho e overlap por defeito 1200 / 200 (`ROTINA_RAG_RECURSIVE_*`).
    """
    if not text.strip() or per_pdf_cap <= 0:
        return []
    size = max(200, int(RAG_RECURSIVE_CHUNK_SIZE))
    overlap = max(0, min(int(RAG_RECURSIVE_CHUNK_OVERLAP), size // 2))
    chunks: list[str] = []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[
                RAG_PDF_PAGE_SPLIT_MARKER,
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        raw_chunks = splitter.split_text(text)
        chunks: list[str] = []
        for c in raw_chunks:
            cleaned = strip_toc_dot_leader_lines(c.strip())
            if len(cleaned) >= 40:
                chunks.append(cleaned)
    except Exception:
        chunks = []
    if not chunks:
        raw_fb = chunk_text_by_chars(text, size, overlap, per_pdf_cap)
        out_fb: list[str] = []
        for c in raw_fb:
            cleaned = strip_toc_dot_leader_lines(c.strip())
            if len(cleaned) >= 40:
                out_fb.append(cleaned)
        return out_fb
    return chunks[:per_pdf_cap]


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


def _purge_chroma_test_del_under_data_dir(data_dir: Path) -> None:
    """Remove `data/_chroma_test_del/` antes da indexação (evita resíduos de testes)."""
    p = (data_dir.resolve() / "_chroma_test_del")
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


def ingest_rotina_pdf_documents(
    collection: chromadb.Collection, data_dir: Path
) -> int:
    """Lê `PDF_NAMES`, segmenta, gera embeddings e grava na coleção `CHROMA_COLLECTION`."""
    _purge_chroma_test_del_under_data_dir(data_dir)
    total_used = 0
    batch = effective_chroma_add_batch()
    for pdf_name in PDF_NAMES:
        if total_used >= MAX_CHUNKS_TOTAL:
            break
        pdf_path = data_dir / pdf_name
        if not pdf_path.exists():
            continue
        skip_pages = PPP_SKIP_FIRST_PAGES if pdf_name == PPP_PEDAGOGICO_PDF else 0
        full_text = extract_pdf_text(pdf_path, skip_first_pages=skip_pages)
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
    return total_used


def rag_will_run_full_document_ingest(persist_dir: Path, data_dir: Path) -> bool:
    """
    True se na próxima chamada a `get_chroma_collection` for necessário ingerir PDFs
    (pasta vazia, coleção inexistente ou manifest de PDFs diferente do gravado).
    """
    persist_dir = persist_dir.resolve()
    data_dir = data_dir.resolve()
    fp_now = _rag_pdf_manifest_fingerprint(data_dir)
    if _read_rag_stored_fingerprint(persist_dir) != fp_now:
        return True
    if not persist_dir.is_dir():
        return True
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        col = client.get_collection(CHROMA_COLLECTION)
        return int(col.count()) == 0
    except Exception:
        return True


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

    fp_now = _rag_pdf_manifest_fingerprint(data_dir)
    fp_stored = _read_rag_stored_fingerprint(persist_dir)

    emb = build_chroma_embedding_function()
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=emb,
        metadata={"description": "Rotina Viva — documentos institucionais"},
    )

    need_reingest = fp_stored != fp_now or collection.count() == 0
    if need_reingest and collection.count() > 0:
        _drop_rotina_chroma_collection(client, CHROMA_COLLECTION)
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=emb,
            metadata={"description": "Rotina Viva — documentos institucionais"},
        )

    if collection.count() == 0:
        ingest_rotina_pdf_documents(collection, data_dir)
        _write_rag_stored_fingerprint(persist_dir, fp_now)
        _chromadb_client_persist(client)
    elif fp_stored is None and collection.count() > 0:
        _write_rag_stored_fingerprint(persist_dir, fp_now)

    return collection


def retrieve_rag_context_and_chunks(
    collection: chromadb.Collection, question: str, k: int | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """
    Retorna o bloco de texto para o LLM e a lista dos trechos (chunks) efetivamente
    escolhidos após busca + reranking — mesma ordem do contexto enviado ao modelo.
    """
    n = collection.count()
    if n == 0:
        return (
            "(Nenhum documento indexado. Coloque os PDFs em ROTINA_DATA_DIR e reinicie o app.)",
            [],
        )
    top = k if k is not None else RAG_TOP_K
    top = max(1, top)
    if ROTINA_RAG_DISTANCE_GAP > 0:
        fetch_n = min(n, max(top + 4, top * 3, 10))
    else:
        fetch_n = min(top, n)
    if ROTINA_RAG_LEXICAL_WEIGHT > 0:
        fetch_n = min(n, max(fetch_n, top * 4, 16))

    id_q = is_rag_identity_scope_question(question)
    nut_q = is_rag_nutrition_meals_scope_question(question)
    where_scope: dict[str, Any] | None = None
    # Uma pergunta “pura” por tipo: filtro no Chroma. Pergunta mista (ex.: nome + cardápio) busca em todos.
    if id_q and not nut_q:
        where_scope = {"source": {"$in": list(RAG_IDENTITY_SOURCES)}}
    elif nut_q and not id_q:
        where_scope = {"source": {"$in": list(RAG_NUTRITION_SOURCES)}}

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

    res = _do_query(where_scope)
    if where_scope is not None:
        if not (res.get("documents") or [[]])[0]:
            res = _do_query(None)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    selected: list[tuple[str, str, float | None, dict[str, Any]]] = []

    if not docs:
        return "(sem trechos relevantes)", []

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
            if len(selected) >= top:
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
                selected.append((src, docs[i] or "", float(dists[i]), metas[i] or {}))
        if len(selected) < top:
            for i in order:
                if len(selected) >= top:
                    break
                if i in picked:
                    continue
                picked.add(i)
                src = (metas[i] or {}).get("source", "?")
                selected.append((src, docs[i] or "", float(dists[i]), metas[i] or {}))
    elif ROTINA_RAG_DISTANCE_GAP > 0 and dists and len(dists) == len(docs):
        base = float(dists[0])
        for doc, dist, meta in zip(docs, dists, metas):
            if float(dist) - base > ROTINA_RAG_DISTANCE_GAP:
                break
            src = (meta or {}).get("source", "?")
            selected.append((src, doc or "", float(dist), meta or {}))
            if len(selected) >= top:
                break
    else:
        for i, (doc, meta) in enumerate(zip(docs[:top], metas[:top])):
            dist_v: float | None = float(dists[i]) if i < len(dists) else None
            src = (meta or {}).get("source", "?")
            selected.append((src, doc or "", dist_v, meta or {}))

    if not selected:
        return "(sem trechos relevantes)", []

    # Pergunta mista (identidade + refeições): trechos do planejamento nutricional primeiro no contexto e na sidebar.
    if id_q and nut_q:
        pref = frozenset(RAG_NUTRITION_SOURCES)
        nut_first = [t for t in selected if t[0] in pref]
        nut_rest = [t for t in selected if t[0] not in pref]
        if nut_first:
            selected = nut_first + nut_rest

    parts_fmt = [f"[Fonte: {s}]\n{t}" for s, t, _, __ in selected]
    rag_block = "\n\n---\n\n".join(parts_fmt)
    chunks_ui: list[dict[str, Any]] = []
    for src, text, dist, meta in selected:
        chunks_ui.append(
            {
                "source": src,
                "chunk": str(meta.get("chunk", "")),
                "text": text,
                "distance": dist,
            }
        )
    return rag_block, chunks_ui


def retrieve_rag_context(collection: chromadb.Collection, question: str, k: int | None = None) -> str:
    block, _ = retrieve_rag_context_and_chunks(collection, question, k=k)
    return block
