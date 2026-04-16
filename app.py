"""
Rotina Viva — assistente para rotinas em escolas infantis (AI Factory / PUCPR).
Streamlit + DuckDB (CSVs) + ChromaDB (RAG) + txtai (segmentação) + LLM/embeddings (API ou Ollama local).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from typing import Any, Generator, Iterable

import altair as alt
import chromadb
import duckdb
import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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

# Transcrição de voz (Whisper HTTP). Docker Compose define URL do serviço `whisper` (sem chave).
# API OpenAI: use OPENAI_TRANSCRIBE_BASE_URL=https://api.openai.com/v1 e OPENAI_TRANSCRIBE_API_KEY (ou só chave no .env com URL da OpenAI).
OPENAI_TRANSCRIBE_BASE_URL = os.getenv(
    "OPENAI_TRANSCRIBE_BASE_URL", "https://api.openai.com/v1"
).strip().rstrip("/")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1").strip()
_stt_key_raw = os.getenv("OPENAI_TRANSCRIBE_API_KEY", "").strip()
if _stt_key_raw:
    OPENAI_TRANSCRIBE_API_KEY = _stt_key_raw
elif "api.openai.com" in OPENAI_TRANSCRIBE_BASE_URL:
    OPENAI_TRANSCRIBE_API_KEY = OPENAI_API_KEY
else:
    OPENAI_TRANSCRIBE_API_KEY = ""

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
# Teto de caracteres na resposta do assistente (0 = sem limite). Evita cópia infinita de tabelas.
ROTINA_CHAT_MAX_OUTPUT_CHARS = _env_int("ROTINA_CHAT_MAX_OUTPUT_CHARS", 4500)
# Temperatura do chat: mais baixa quando há linhas SQL (reduz invenção em tabelas).
ROTINA_CHAT_TEMPERATURE = _env_float("ROTINA_CHAT_TEMPERATURE", 0.35)
ROTINA_CHAT_TEMP_WITH_SQL = _env_float("ROTINA_CHAT_TEMP_WITH_SQL", 0.12)

# Relatório sono/alimentação: teto de minutos (ex.: 60) e faixas **dentro** de 0…teto.
# Classificação: [0, L1) pouco, [L1, L2) normal, [L2, teto] bastante (minutos > teto viram teto).
# Exemplo com padrão L1=20, L2=40 e teto 60: ~10 min → pouco, ~35 min → normal, 60 min → bastante.
if os.getenv("ROTINA_SONO_MAX_MIN", "").strip():
    ROTINA_SONO_MAX_MIN = max(15.0, min(180.0, _env_float("ROTINA_SONO_MAX_MIN", 60.0)))
else:
    ROTINA_SONO_MAX_MIN = max(15.0, min(180.0, _env_float("ROTINA_SONO_REFERENCIA_MIN", 60.0)))
ROTINA_SONO_FAIXA_LIMITE_1 = max(
    1.0, min(ROTINA_SONO_MAX_MIN - 2.0, _env_float("ROTINA_SONO_FAIXA_LIMITE_1", 20.0))
)
ROTINA_SONO_FAIXA_LIMITE_2 = max(
    ROTINA_SONO_FAIXA_LIMITE_1 + 1.0,
    min(ROTINA_SONO_MAX_MIN - 1.0, _env_float("ROTINA_SONO_FAIXA_LIMITE_2", 40.0)),
)

# Valores canônicos para `qualidade_sono` no diário (CSV + relatório).
SONO_QUAL_BASTANTE = "Dormiu bastante"
SONO_QUAL_POUCO = "Dormiu pouco"
SONO_QUAL_PADRAO = (
    f"Dormiu normal ({int(round(ROTINA_SONO_FAIXA_LIMITE_1))}–"
    f"{int(round(ROTINA_SONO_FAIXA_LIMITE_2))} min)"
)


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
- Para perguntas sobre identidade da escola, procure linhas como nome fantasia, cabeçalho, "Escola ..." ou campo "Título:" nos documentos.
- Responda ao que a **pergunta atual** pede. Não acrescente observações sobre nomes ou assuntos que só surgiram em **mensagens anteriores** do chat: o bloco de contexto desta rodada costuma estar filtrado à pergunta de agora, e a ausência de um nome nesse bloco **não** autoriza dizer "não há informações sobre [fulano]" se o utilizador **não perguntou** por essa pessoa nesta mensagem."""

SYSTEM_SQL_STRICT = """Dados tabulares (bloco "Dados tabulares" acima):
- A tabela é o resultado **literal** de uma consulta ao banco. Trate cada célula como dado real já filtrado.
- **Não invente** linhas, colunas, nomes de crianças, datas, refeições, medicamentos ou números que **não apareçam** nessa tabela.
- Se a tabela estiver vazia ou disser "(nenhuma linha retornada)", diga isso claramente — não preencha com suposições.
- Para contar, listar ou comparar, use **apenas** o que está nas linhas mostradas (e o número da coluna "linha" se existir).
- Se a pergunta pedir algo que a tabela não contém (coluna ausente), diga que o resultado atual não traz esse campo.
- Se várias linhas tiverem o mesmo nome e turmas diferentes, isso vem do cadastro (homônimos ou duplicidade): cite `id_aluno` de cada linha e não assuma um único aluno sem explicar.
- Esta tabela reflete a **pergunta atual**; não conclua pela omissão de nomes aqui que "não há dados" sobre alguém que o utilizador **não citou** nesta pergunta.
- **Resposta ao utilizador (obrigatório):** não transcreva a tabela inteira nem liste todos os alunos linha a linha — o utilizador já vê os dados na aplicação. Limite-se a **resumir** (ex.: total, ids relevantes, sim/não) em **poucas frases**; no máximo **3 exemplos** de linha se for indispensável."""

SYSTEM_MUTATION_APPLIED = """Operação nos dados (instalação autorizada — perfil Gestão):
- Nesta mesma resposta, o sistema **já executou** o pedido de INSERT/UPDATE/DELETE nos CSV locais quando a secção de verificação pós-mutação aparecer nos dados tabulares acima.
- **Ordem temporal:** essa tabela é o estado **depois** da alteração nesta mensagem — **não** é uma fotografia de “antes” nem um relatório de duplicatas pré-existentes.
- Se o utilizador pediu **incluir** um aluno (ou linha de diário) e a verificação mostra **uma** linha com nome/turma/dados pedidos, interprete como **sucesso do cadastro** (confirme id e dados). **Não** diga que “já existia” ou que “não é necessário acrescentar” só porque essa linha aparece — ela pode ser **precisamente** o registo que acabou de ser inserido.
- Só fale em duplicata ou “já cadastrado” se o utilizador pediu **apenas** verificação sem inserir, ou se a verificação mostrar **duas ou mais** linhas inequivocamente conflituosas para o mesmo pedido (explique com ids).
- **DELETE:** a amostra tabular mostra só **alguns** registos recentes (últimos ids). Depois de remover um aluno, o nome **não** deve aparecer — isso **confirma** a remoção, não significa “nunca existiu para apagar”. **Não** diga que “não há aluno com esse nome para apagar” só porque percorreu mentalmente a lista mostrada: essa lista é **parcial**. Confirme o DELETE em **uma frase** sem reescrever a tabela.
- **Total de alunos:** se existir a linha `CONTAGEM_OFICIAL_ALUNOS=N` no contexto, use **só esse N** ao mencionar quantos alunos há no cadastro — **não** invente, não subtraia 1, nem conte linhas da tabela markdown.
- **id_aluno após inserir:** se existir `CONFIRME_id_aluno=N`, use **exatamente esse N** ao citar o id do aluno recém-cadastrado — **não** diga N+1, N−1, nem um id “à frente”; ignore suposições e use só essa linha.
- Responda de forma **curta**: **não** copie todas as linhas da verificação para a resposta (máx. 1–2 linhas ou só id/nome).
- **Não** recuse o pedido nem envie à secretaria quando a operação já foi aplicada.
- Se os dados tabulares mostrarem erro de SQL na verificação, explique esse erro."""

SYSTEM_MUTATION_FAILED = """Falha ao gravar alteração nos CSV desta instalação (Rotina Viva):
- Nos dados tabulares acima há uma secção **"A alteração aos dados NÃO foi gravada"** com o motivo real (permissão, ficheiro em uso, etc.).
- Responda de forma **prática**: o utilizador deve **fechar** `info_alunos.csv` e/ou `diario_estruturado.csv` se estiverem abertos no **Excel** ou noutro editor (no Windows isto bloqueia a escrita), e voltar a pedir a alteração.
- **Não** diga que não tem acesso a "sistemas de gestão" ou que só pode "fornecer informações" — nesta app a alteração é feita aqui; o problema é **técnico local** (não foi possível escrever no disco).
- Seja breve e acertivo: uma frase sobre fechar o ficheiro + repetir o pedido."""

SYSTEM_DUPLICATE_CADASTRO = """Duplicados no cadastro (info_alunos):
- Se existir secção **"Aviso de duplicado"** nos dados tabulares, o servidor detetou **nome ou contacto** já presentes no CSV **antes de gravar**.
- Regra desta instalação: em caso de duplicado, a mutação é **bloqueada** e nada é persistido no CSV.
- Explique com clareza: ajuste os dados ou confirme outro identificador/contato para gravar.
- Não omita o aviso; seja breve."""

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
   **qualidade_sono:** padronize com um destes textos (teto e faixas: `ROTINA_SONO_MAX_MIN`, `ROTINA_SONO_FAIXA_LIMITE_1`, `ROTINA_SONO_FAIXA_LIMITE_2` no `.env`):
   `Dormiu bastante`, `Dormiu pouco`, e `Dormiu normal (L1–L2 min)` conforme configurado.
   Textos antigos (ex.: “acordou agitado”) ainda carregam: o relatório normaliza para essas três categorias.

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


def validate_mutation_sql(sql: str) -> bool:
    """Permite uma instrução INSERT/UPDATE/DELETE nas tabelas CSV (perfil gestão)."""
    if not sql or not isinstance(sql, str):
        return False
    t = sql.strip().rstrip(";")
    if ";" in t:
        return False
    low = t.lower()
    if not any(low.startswith(p) for p in ("insert", "update", "delete")):
        return False
    banned = re.compile(
        r"\b(drop|alter|create|attach|detach|pragma|copy|call|truncate|replace|into\s+sqlite|from\s+sqlite)\b",
        re.IGNORECASE,
    )
    if banned.search(low):
        return False
    if "diario_estruturado" not in low and "info_alunos" not in low:
        return False
    return True


def _ensure_utf8_bom_csv(path: Path) -> None:
    """Excel no Windows abre CSV sem BOM como ANSI — o BOM marca UTF-8 (acentos corretos)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return
    if raw.startswith(b"\xef\xbb\xbf"):
        return
    try:
        path.write_bytes(b"\xef\xbb\xbf" + raw)
    except OSError:
        pass


def persist_duckdb_tables_to_csv(conn: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    """Grava tabelas em memória de volta aos CSVs (após mutação)."""
    info_csv = _resolve_info_alunos_csv(data_dir)
    diario_csv = data_dir / "diario_estruturado.csv"
    ip = str(info_csv.resolve()).replace("\\", "/")
    dp = str(diario_csv.resolve()).replace("\\", "/")
    conn.execute(f"COPY (SELECT * FROM info_alunos) TO '{ip}' (HEADER, DELIMITER ',')")
    conn.execute(f"COPY (SELECT * FROM diario_estruturado) TO '{dp}' (HEADER, DELIMITER ',')")
    _ensure_utf8_bom_csv(info_csv)
    _ensure_utf8_bom_csv(diario_csv)


def _extract_first_values_tuple(sql: str) -> str | None:
    """Conteúdo interno do primeiro VALUES (...) num INSERT (uma linha)."""
    m = re.search(r"VALUES\s*\(", sql, re.IGNORECASE)
    if not m:
        return None
    i = m.end()
    depth = 1
    while i < len(sql) and depth:
        c = sql[i]
        if c == "'":
            i += 1
            while i < len(sql):
                if sql[i] == "'":
                    if i + 1 < len(sql) and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return sql[m.end() : i]
        i += 1
    return None


def _split_sql_value_list(inner: str) -> list[str]:
    """Divide valores de um VALUES (…) sem nested calls (caso típico do cadastro)."""
    parts: list[str] = []
    cur: list[str] = []
    i = 0
    in_q = False
    depth = 0
    while i < len(inner):
        c = inner[i]
        if in_q:
            if c == "'":
                if i + 1 < len(inner) and inner[i + 1] == "'":
                    cur.append("''")
                    i += 2
                    continue
                in_q = False
                cur.append("'")
                i += 1
                continue
            cur.append(c)
            i += 1
            continue
        if c == "'":
            in_q = True
            cur.append(c)
            i += 1
            continue
        if c == "(":
            depth += 1
            cur.append(c)
            i += 1
            continue
        if c == ")":
            if depth > 0:
                depth -= 1
            cur.append(c)
            i += 1
            continue
        if c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        if c.isspace():
            i += 1
            continue
        cur.append(c)
        i += 1
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _sql_unquote_string_token(raw: str) -> str | None:
    s = raw.strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        return s[1:-1].replace("''", "'")
    return None


def _parse_info_alunos_insert(sql: str) -> dict[str, str] | None:
    if not re.search(r"INSERT\s+INTO\s+info_alunos\b", sql, re.IGNORECASE):
        return None
    inner = _extract_first_values_tuple(sql)
    if not inner:
        return None
    parts = _split_sql_value_list(inner)
    mcols = re.search(
        r"INSERT\s+INTO\s+info_alunos\s*\(\s*([^)]+?)\s*\)\s*VALUES",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if mcols:
        cols = [c.strip().lower() for c in mcols.group(1).split(",")]
    else:
        cols = ["id_aluno", "nome", "turma", "alergias", "contato_pais"]
    out: dict[str, str] = {}
    for i, col in enumerate(cols):
        if i >= len(parts):
            break
        raw = parts[i].strip()
        uq = _sql_unquote_string_token(raw)
        out[col] = uq if uq is not None else raw
    # Fallback robusto: quando o 1º valor é subconsulta complexa, garanta os 4 campos textuais.
    # Padrão esperado em info_alunos: (id_expr, 'nome', 'turma', 'alergias', 'contato_pais')
    if "nome" not in out or "contato_pais" not in out:
        lits = re.findall(r"'((?:[^']|'')*)'", inner)
        if len(lits) >= 4:
            out["nome"] = lits[-4].replace("''", "'")
            out["turma"] = lits[-3].replace("''", "'")
            out["alergias"] = lits[-2].replace("''", "'")
            out["contato_pais"] = lits[-1].replace("''", "'")
    return out or None


def _parse_info_alunos_update(sql: str) -> dict[str, str] | None:
    if not re.search(r"UPDATE\s+info_alunos\b", sql, re.IGNORECASE):
        return None
    out: dict[str, str] = {}
    for field in ("nome", "turma", "alergias", "contato_pais"):
        pat = rf"\b{field}\s*=\s*('(?:[^']|'')*')"
        m = re.search(pat, sql, re.IGNORECASE)
        if m:
            lit = m.group(1)
            out[field] = lit[1:-1].replace("''", "'")
    mw = re.search(r"WHERE\s+id_aluno\s*=\s*(\d+)", sql, re.IGNORECASE)
    if mw:
        out["_where_id_aluno"] = mw.group(1)
    return out if out else None


def _duplicate_info_alunos_warning_before_write(
    conn: duckdb.DuckDBPyConnection,
    fields: dict[str, str],
    exclude_id_aluno: int | None,
) -> str | None:
    """
    Verifica se nome ou contacto já existem no cadastro (antes de INSERT/UPDATE persistir).
    """
    lines: list[str] = []
    nome = (fields.get("nome") or "").strip()
    contato = (fields.get("contato_pais") or "").strip()
    excl = exclude_id_aluno

    if nome:
        q = (
            "SELECT id_aluno, nome, turma FROM info_alunos "
            "WHERE TRIM(COALESCE(nome,'')) ILIKE TRIM(?)"
        )
        params: list[Any] = [nome]
        if excl is not None:
            q += " AND CAST(id_aluno AS INTEGER) != ?"
            params.append(excl)
        q += " LIMIT 8"
        try:
            rows = conn.execute(q, params).fetchall()
        except Exception:
            rows = []
        if rows:
            bits = [f"id_aluno={r[0]} ({r[1]} | {r[2]})" for r in rows]
            lines.append(
                f"Nome «{nome}» já consta no cadastro: " + "; ".join(bits)
            )

    trivial_ct = ("", "nenhum", "nenhuma", "-", "—", "na", "n/a")
    if contato and contato.lower() not in trivial_ct:
        q = (
            "SELECT id_aluno, nome, contato_pais FROM info_alunos "
            "WHERE TRIM(COALESCE(contato_pais,'')) ILIKE TRIM(?)"
        )
        params = [contato]
        if excl is not None:
            q += " AND CAST(id_aluno AS INTEGER) != ?"
            params.append(excl)
        q += " LIMIT 8"
        try:
            rows = conn.execute(q, params).fetchall()
        except Exception:
            rows = []
        if rows:
            bits = [f"id_aluno={r[0]} ({r[1]})" for r in rows]
            lines.append(
                f"Contacto «{contato}» já consta no cadastro: " + "; ".join(bits)
            )

    if not lines:
        return None
    return (
        "**Atenção**: " + " ".join(lines) + " "
        "A alteração foi bloqueada para evitar duplicidade; confirme com o utilizador se era intencional."
    )


def _precheck_duplicate_info_alunos(
    conn: duckdb.DuckDBPyConnection, sql: str
) -> str | None:
    low = sql.strip().lower()
    if re.search(r"insert\s+into\s+info_alunos\b", low):
        parsed = _parse_info_alunos_insert(sql)
        if parsed:
            return _duplicate_info_alunos_warning_before_write(conn, parsed, None)
    if re.search(r"update\s+info_alunos\b", low):
        parsed = _parse_info_alunos_update(sql)
        if parsed:
            wid = parsed.pop("_where_id_aluno", None)
            excl = int(wid) if wid is not None else None
            if parsed:
                return _duplicate_info_alunos_warning_before_write(conn, parsed, excl)
    return None


def _format_mutation_persist_error(e: BaseException) -> str:
    """Mensagem legível; destaca bloqueio por Excel/CSV aberto no Windows."""
    raw = str(e)
    el = raw.lower()
    if isinstance(e, PermissionError) or "permission denied" in el or "errno 13" in el:
        return (
            "Não foi possível **gravar** os CSV: permissão negada ou ficheiro bloqueado. "
            "Se `info_alunos.csv` ou `diario_estruturado.csv` estiver aberto no **Excel** (ou outro programa), "
            "**feche o ficheiro** e tente novamente. "
            f"(Detalhe técnico: {raw})"
        )
    if (
        "being used by another process" in el
        or "cannot access the file" in el
        or "another program" in el
    ):
        return (
            "Não foi possível gravar os CSV: o ficheiro está **em uso** por outro programa. "
            "Feche o Excel ou o editor que tiver o CSV aberto e tente de novo. "
            f"(Detalhe: {raw})"
        )
    return f"Erro ao aplicar alteração: {raw}"


def run_mutation_and_persist(
    conn: duckdb.DuckDBPyConnection, sql: str, data_dir: Path
) -> tuple[str, bool, str | None]:
    if not validate_mutation_sql(sql):
        return "Instrução de alteração rejeitada (tabelas ou tipo não permitidos).", False, None
    dup_warn: str | None = None
    try:
        dup_warn = _precheck_duplicate_info_alunos(conn, sql)
    except Exception:
        dup_warn = None
    if dup_warn:
        return (
            "Alteração **não gravada**: o texto/nome/contacto informado já existe no cadastro.\n\n"
            + dup_warn
            + "\n\nNada foi alterado no CSV. Ajuste os dados e tente novamente.",
            False,
            dup_warn,
        )
    try:
        conn.execute(sql)
        persist_duckdb_tables_to_csv(conn, data_dir)
        msg = "Alteração aplicada e CSVs atualizados."
        return msg, True, None
    except Exception as e:
        return _format_mutation_persist_error(e), False, None


def _verification_selects_after_mutation(mut_sql: str) -> list[str]:
    """SELECTs só de leitura para o LLM ver o estado pós-mutação (evita resposta genérica de recusa)."""
    low = (mut_sql or "").lower()
    out: list[str] = []
    if "info_alunos" in low:
        # Amostra pequena: reduz tentação de o modelo copiar a lista inteira para a resposta.
        # O total exato vem de `CONTAGEM_OFICIAL_ALUNOS` em `post_mutation_verification_block`, não da tabela formatada.
        out.append(
            "SELECT id_aluno, nome, turma, alergias, contato_pais FROM info_alunos ORDER BY id_aluno DESC LIMIT 12"
        )
    if "diario_estruturado" in low:
        out.append(
            "SELECT id_registro, id_aluno, data, cafe_manha, almoco, recado_professora "
            "FROM diario_estruturado ORDER BY id_registro DESC LIMIT 12"
        )
    if not out:
        out.append(
            "SELECT 'info_alunos' AS tabela, COUNT(*)::BIGINT AS n FROM info_alunos "
            "UNION ALL SELECT 'diario_estruturado', COUNT(*)::BIGINT FROM diario_estruturado"
        )
    return out


def _official_new_aluno_id_block(conn: duckdb.DuckDBPyConnection) -> str | None:
    """
    Após INSERT em info_alunos (ids monótonos), o maior id_aluno é o registo criado.
    Texto plano para o LLM não confundir com células da tabela markdown nem inventar id+1.
    """
    try:
        r = conn.execute(
            "SELECT id_aluno, nome, turma, alergias, contato_pais "
            "FROM info_alunos ORDER BY id_aluno DESC LIMIT 1"
        ).fetchone()
        if not r:
            return None
        aid = int(r[0])
        return (
            "=== Aluno com maior id_aluno no cadastro (em INSERT recente, é o registo novo) ===\n"
            f"CONFIRME_id_aluno={aid}\n"
            f"nome={r[1]}\n"
            f"turma={r[2]}\n"
            f"alergias={r[3]}\n"
            f"contato_pais={r[4]}\n"
            "Ao confirmar o cadastro, repita **só** o inteiro em CONFIRME_id_aluno (não use id+1, id−1 nem outro número)."
        )
    except Exception:
        return None


def _official_info_alunos_count_block(conn: duckdb.DuckDBPyConnection) -> str | None:
    """Uma linha inequívoca com COUNT(*) real (evita o modelo inventar ou ler mal tabelas markdown)."""
    try:
        r = conn.execute(
            "SELECT COUNT(*) FROM info_alunos WHERE TRIM(COALESCE(nome, '')) <> ''"
        ).fetchone()
        if not r:
            return None
        n = int(r[0])
        return (
            "=== TOTAL NO CADASTRO (apenas alunos com nome preenchido) ===\n"
            f"CONTAGEM_OFICIAL_ALUNOS={n}\n"
            "Ao dizer quantos alunos existem, use **exatamente** o inteiro acima (não some linhas da amostra nem estime)."
        )
    except Exception:
        return None


def post_mutation_verification_block(
    conn: duckdb.DuckDBPyConnection, mut_sql: str
) -> str:
    """Texto tabular para anexar ao contexto do chat após mutação bem-sucedida."""
    mut_low = (mut_sql or "").lower()
    note = (
        "Estas linhas reflectem o cadastro/diário **depois** da mutação aplicada **nesta mensagem**. "
        "Se o pedido foi inserir um registo, a linha com esses dados é em geral o **novo** registo, não prova de que já existia antes."
    )
    if "insert" in mut_low:
        note += (
            " (Pedido inclui INSERT: confirme o sucesso; para **id_aluno** use só a linha `CONFIRME_id_aluno=...` "
            "se existir, não interprete como duplicata por defeito.)"
        )
    if "delete" in mut_low:
        note += (
            " **DELETE:** a lista abaixo é só uma **amostra** (últimos ids); após apagar, o nome pode **não** "
            "aparecer — isso indica sucesso. Para o total de alunos use só **CONTAGEM_OFICIAL_ALUNOS** no fim; **não** reproduza a tabela na resposta."
        )
    parts: list[str] = [
        "=== Verificação pós-mutação (estado atual dos CSV após esta operação) ===",
        note,
    ]
    for sel in _verification_selects_after_mutation(mut_sql):
        block, ok = run_safe_select(conn, sel)
        parts.append(block)
        if not ok:
            break
    if "insert" in mut_low and "info_alunos" in mut_low:
        _nid = _official_new_aluno_id_block(conn)
        if _nid:
            parts.append(_nid)
    if "info_alunos" in mut_low:
        _cnt = _official_info_alunos_count_block(conn)
        if _cnt:
            parts.append(_cnt)
    return "\n\n".join(parts)


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


def _duckdb_csv_reload_token(data_dir: Path) -> str:
    """Muda quando os CSVs são alterados, para o cache do DuckDB recarregar."""
    info_csv = _resolve_info_alunos_csv(data_dir)
    diario_csv = data_dir / "diario_estruturado.csv"
    parts: list[str] = []
    for p in (info_csv, diario_csv):
        try:
            parts.append(str(p.stat().st_mtime_ns))
        except OSError:
            parts.append("0")
    return "|".join(parts)


@st.cache_resource
def get_duckdb_connection(data_dir_str: str, _csv_token: str) -> duckdb.DuckDBPyConnection:
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


_MEAL_SCORE_MAP: dict[str, int] = {
    "comeu bem": 2,
    "comeu pouco": 1,
    "recusou": 0,
}


def _meal_score_cell(val: Any) -> int:
    k = str(val or "").strip().lower()
    return _MEAL_SCORE_MAP.get(k, 1)


def _parse_clock_to_minutes(s: Any) -> int | None:
    raw = str(s or "").strip()
    if not raw:
        return None
    raw = raw.replace(".", ":")
    parts = raw.split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0].strip())
        m = int(parts[1].strip()[:2])
        return max(0, min(23, h)) * 60 + max(0, min(59, m))
    except ValueError:
        return None


def _sleep_minutes_between(start: Any, end: Any) -> int | None:
    """
    Duração em minutos entre dois horários do mesmo dia (relógio 24 h).

    Usa o **menor** arco entre os dois instantes no círculo de 24 h, para que
    `inicio=13:00` e `fim=12:00` (colunas trocadas vs. 12:00→13:00) resulte em
    **60 min**, e não ~23 h (bug que marcava todos como “Dormiu bastante”).
    Ignora durações irreais para soneca escolar (> 4 h).
    """
    a = _parse_clock_to_minutes(start)
    b = _parse_clock_to_minutes(end)
    if a is None or b is None:
        return None
    d = (b - a) % (24 * 60)
    if d == 0:
        return None
    d_short = min(d, 24 * 60 - d)
    if d_short <= 0:
        return None
    if d_short > 240:
        return None
    return int(d_short)


def _classify_sono_min_for_report(sono_min: float) -> str:
    """
    Classifica minutos de sono para o relatório: valores são limitados a `ROTINA_SONO_MAX_MIN`.
    Faixas: [0, L1) pouco, [L1, L2) normal, [L2, teto] bastante (`ROTINA_SONO_FAIXA_LIMITE_*`).
    """
    cap = float(ROTINA_SONO_MAX_MIN)
    l1 = float(ROTINA_SONO_FAIXA_LIMITE_1)
    l2 = float(ROTINA_SONO_FAIXA_LIMITE_2)
    m = max(0.0, min(float(sono_min), cap))
    if m < l1:
        return SONO_QUAL_POUCO
    if m < l2:
        return SONO_QUAL_PADRAO
    return SONO_QUAL_BASTANTE


def _normalize_qualidade_sono_val(raw: str, sono_min: Any) -> str:
    """
    Reduz `qualidade_sono` do CSV a uma das três categorias canônicas.
    Prioridade: texto reconhecível → inferência por minutos (horários início/fim) → "—".
    """
    t = (raw or "").strip().lower()
    if t in ("—", "-", "", "nan", "none"):
        t = ""

    exact = {
        SONO_QUAL_BASTANTE.lower(): SONO_QUAL_BASTANTE,
        SONO_QUAL_POUCO.lower(): SONO_QUAL_POUCO,
        SONO_QUAL_PADRAO.lower(): SONO_QUAL_PADRAO,
    }
    if t in exact:
        return exact[t]
    if t.startswith("sono no padrão") or t.startswith("sono no padrao"):
        return SONO_QUAL_PADRAO

    # Variações comuns / legado (ordem: sinais de sono ruim curto, depois pouco, bastante, padrão).
    if any(
        k in t
        for k in (
            "acordou agit",
            "muito agit",
            "sono agit",
            "interrup",
            "chorou muito",
            "acordou cedo",
            "não dormiu",
            "nao dormiu",
            "quase não dorm",
        )
    ):
        return SONO_QUAL_POUCO
    if any(
        k in t
        for k in (
            "dormiu pouco",
            "pouco sono",
            "sono curto",
            "descanso curto",
            "dormiu mal",
        )
    ):
        return SONO_QUAL_POUCO
    if any(
        k in t
        for k in (
            "dormiu bastante",
            "bastante sono",
            "sono longo",
            "dormiu bem",
            "dormiu tranquilo",
            "sono tranquilo",
            "bom sono",
            "descanso bom",
            "dormiu o dia",
        )
    ):
        return SONO_QUAL_BASTANTE
    if any(
        k in t
        for k in (
            "dormiu normal",
            "no padrão",
            "no padrao",
            "padrão da",
            "padrao da",
            "média da semana",
            "media da semana",
            "sono normal",
            "sono moderado",
            "sono regular",
            "sono leve",
            "padrão (~60",
            "padrao (~60",
        )
    ):
        return SONO_QUAL_PADRAO

    sm: float | None
    try:
        sm = float(sono_min) if sono_min is not None and pd.notna(sono_min) else None
    except (TypeError, ValueError):
        sm = None
    if sm is not None and sm > 0:
        if sm <= 240:
            return _classify_sono_min_for_report(sm)
        return "—"
    return "—"


def build_sleep_meal_report_dataframe(
    conn: duckdb.DuckDBPyConnection,
    student_name: str,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    str | None,
    str,
    str | None,
    tuple[str, str] | None,
]:
    """
    Relatório só com CSV no DuckDB. Filtra pelo nome do aluno (ILIKE), depois:
    - considera a **janela de 7 dias corridos** terminando na **data mais recente** do diário
      desse(s) aluno(s); se nessa semana não houver **nenhum** sono válido, usa a semana que termina
      na **última data com sono válido** (horários preenchidos e duração maior que zero);
    - agrega por **dia** (média de ingestão e de minutos de sono se houver mais de uma linha no mesmo dia);
    - preserva textos de **cafe_manha**, **almoco**, **lanche_tarde** por dia (moda) para gráfico de barras.

    Retorno: (df resumo diário, df refeições textuais por dia, erro|None, rótulo alunos,
              aviso_semana|None, (início_iso, fim_iso)|None).
    """
    name = (student_name or "").strip()
    if not name:
        return None, None, "Informe o nome do aluno.", "", None, None
    try:
        id_rows = conn.execute(
            "SELECT id_aluno, nome FROM info_alunos WHERE nome ILIKE ? ORDER BY nome",
            [f"%{name}%"],
        ).fetchall()
    except Exception as e:
        return None, None, str(e), "", None, None
    if not id_rows:
        return (
            None,
            None,
            f'Nenhum aluno encontrado com nome parecido com “{name}”.',
            "",
            None,
            None,
        )
    ids = [int(r[0]) for r in id_rows]
    resolved = ", ".join(sorted({str(r[1]) for r in id_rows}))
    ph = ",".join(["?"] * len(ids))
    try:
        cur = conn.execute(
            f"""
            SELECT cafe_manha, almoco, lanche_tarde, jantar_extra,
                   hora_sono_inicio, hora_sono_fim, qualidade_sono, data, id_aluno
            FROM diario_estruturado
            WHERE id_aluno IN ({ph})
            """,
            ids,
        )
        df = cur.fetchdf()
    except Exception as e:
        return None, None, str(e), resolved, None, None
    if df is None or df.empty:
        return (
            None,
            None,
            f"Sem registros de diário para: {resolved}.",
            resolved,
            None,
            None,
        )

    if "data" not in df.columns:
        return None, None, "Coluna `data` ausente no diário.", resolved, None, None

    meal_cols = ["cafe_manha", "almoco", "lanche_tarde", "jantar_extra"]
    meal_txt_cols = ["cafe_manha", "almoco", "lanche_tarde"]
    for c in meal_cols:
        if c not in df.columns:
            return None, None, f"Coluna ausente: {c}", resolved, None, None
    for c in meal_txt_cols:
        df[f"{c}_txt"] = df[c].fillna("—").astype(str).str.strip()
    for c in meal_cols:
        df[c] = df[c].map(_meal_score_cell)
    df["ingestao"] = df[meal_cols].sum(axis=1).astype(int)

    df["sono_min"] = df.apply(
        lambda r: _sleep_minutes_between(
            r.get("hora_sono_inicio"), r.get("hora_sono_fim")
        ),
        axis=1,
    )
    df["qualidade_sono"] = df["qualidade_sono"].fillna("—").astype(str).str.strip()

    df["data_dt"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data_dt"])
    if df.empty:
        return (
            None,
            None,
            "Nenhuma data válida na coluna `data` do diário.",
            resolved,
            None,
            None,
        )

    d_latest = df["data_dt"].dt.normalize().max()

    def _slice_week(df_in: pd.DataFrame, end_day: pd.Timestamp) -> pd.DataFrame:
        start_day = end_day - pd.Timedelta(days=6)
        out = df_in[
            (df_in["data_dt"].dt.normalize() >= start_day)
            & (df_in["data_dt"].dt.normalize() <= end_day)
        ].copy()
        out = out.dropna(subset=["sono_min"])
        return out[out["sono_min"] > 0]

    d_max_day = d_latest
    d_min_day = d_max_day - pd.Timedelta(days=6)
    df_w = _slice_week(df, d_max_day)
    aviso_janela: str | None = None
    if df_w.empty:
        ok = df.dropna(subset=["sono_min"])
        ok = ok[ok["sono_min"] > 0]
        if ok.empty:
            return (
                None,
                None,
                "Não há registros com **sono válido** (horários de início/fim preenchidos e duração "
                f"maior que zero) para: {resolved}. Revise os horários de início e fim do descanso nos registros.",
                resolved,
                None,
                (str(d_min_day.date()), str(d_max_day.date())),
            )
        d_anchor = ok["data_dt"].dt.normalize().max()
        df_w = _slice_week(df, d_anchor)
        d_max_day = d_anchor
        d_min_day = d_max_day - pd.Timedelta(days=6)
        if df_w.empty:
            return (
                None,
                None,
                "Não há registros com sono válido na **semana** usada para o relatório "
                f"({d_min_day.date()} a {d_max_day.date()}).",
                resolved,
                None,
                (str(d_min_day.date()), str(d_max_day.date())),
            )
        if d_anchor < d_latest:
            aviso_janela = (
                f"A data mais recente no diário (**{d_latest.date()}**) não tinha sono válido "
                f"(horários vazios, ilegíveis ou com duração zero). O relatório usa a semana que termina em "
                f"**{d_anchor.date()}**, última data com intervalo de sono válido."
            )

    df_w["dia"] = df_w["data_dt"].dt.normalize()

    _cap = float(ROTINA_SONO_MAX_MIN)
    df_w["sono_min"] = pd.to_numeric(df_w["sono_min"], errors="coerce").clip(lower=0, upper=_cap)

    df_w["qualidade_sono"] = [
        _normalize_qualidade_sono_val(str(q), sm)
        for q, sm in zip(df_w["qualidade_sono"], df_w["sono_min"])
    ]

    def _qual_pick(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iloc[0]) if len(m) > 0 else str(s.iloc[0])

    def _txt_pick(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iloc[0]) if len(m) > 0 else str(s.iloc[0])

    daily = df_w.groupby("dia", as_index=False).agg(
        ingestao=("ingestao", "mean"),
        sono_min=("sono_min", "mean"),
        qualidade_sono=("qualidade_sono", _qual_pick),
        cafe_manha=("cafe_manha_txt", _txt_pick),
        almoco=("almoco_txt", _txt_pick),
        lanche_tarde=("lanche_tarde_txt", _txt_pick),
    )
    daily["ingestao"] = daily["ingestao"].round(2)
    daily["sono_min"] = daily["sono_min"].round(1)
    daily_meals = daily[["dia", "cafe_manha", "almoco", "lanche_tarde"]].copy()
    daily = daily.drop(columns=["cafe_manha", "almoco", "lanche_tarde"])

    n_dias = len(daily)
    periodo = (str(d_min_day.date()), str(d_max_day.date()))

    if n_dias < 2:
        err = (
            "Para comparar **uma semana**, são necessários **pelo menos 2 dias distintos** "
            f"com sono e refeições registrados entre **{d_min_day.date()}** e **{d_max_day.date()}**. "
            f"No momento há apenas **1** dia com dado válido para: {resolved}. "
            "Inclua registros em mais dias desse período nos dados de rotina."
        )
        if aviso_janela:
            err = aviso_janela + "\n\n" + err
        return (None, None, err, resolved, None, periodo)

    aviso: str | None = None
    if n_dias < 7:
        aviso = (
            f"Na janela de **7 dias** ({d_min_day.date()} a {d_max_day.date()}) há apenas **{n_dias}** "
            "dia(s) com registro. Complete os dias faltantes nos dados de rotina "
            "para uma análise semanal completa."
        )
    if aviso_janela:
        aviso = aviso_janela + ("\n\n" + aviso if aviso else "")

    return daily, daily_meals, None, resolved, aviso, periodo


def _sleep_line_chart_altair(daily: pd.DataFrame) -> alt.Chart:
    """Tendência de horas + linha no teto do relatório; pontos coloridos pela classificação (CSV / faixas)."""
    cap_m = float(ROTINA_SONO_MAX_MIN)
    cap_h = cap_m / 60.0
    d = daily.sort_values("dia").copy()
    d["data_show"] = d["dia"].map(lambda x: pd.Timestamp(x).strftime("%d/%m"))
    lk = _daily_sleep_lookup_for_ref_table(daily)
    m = d.merge(lk[["data_show", "vs_escola"]], on="data_show", how="left")
    line_df = pd.DataFrame(
        {
            "Data": pd.to_datetime(m["dia"], errors="coerce").dt.normalize(),
            "Horas de sono": (pd.to_numeric(m["sono_min"], errors="coerce") / 60.0).round(2),
            "Classificação": m["vs_escola"].fillna("—").astype(str).str.strip(),
        }
    )
    _cmap = {
        SONO_QUAL_POUCO: "#c0392b",
        SONO_QUAL_PADRAO: "#2980b9",
        SONO_QUAL_BASTANTE: "#1a9850",
        "—": "#aeb6bf",
    }
    _order = [SONO_QUAL_POUCO, SONO_QUAL_PADRAO, SONO_QUAL_BASTANTE, "—"]
    _present = [c for c in _order if c in set(line_df["Classificação"])]
    for c in sorted(line_df["Classificação"].unique()):
        if c not in _present:
            _present.append(c)
    _colors = [_cmap.get(c, "#7f8c8d") for c in _present]

    base = alt.Chart(line_df).encode(
        x=alt.X(
            "Data:T",
            title="Data",
            axis=alt.Axis(format="%d/%m", labelAngle=0),
        ),
        y=alt.Y("Horas de sono:Q", title="Horas de sono"),
    )
    line = base.mark_line(strokeWidth=2, color="#34495e", interpolate="monotone")
    pts = base.mark_point(filled=True, size=95, stroke="white", strokeWidth=1).encode(
        color=alt.Color(
            "Classificação:N",
            title="Classificação",
            scale=alt.Scale(domain=_present, range=_colors),
            legend=alt.Legend(orient="bottom", direction="horizontal", labelLimit=0),
        ),
        tooltip=[
            alt.Tooltip("Data:T", title="Data", format="%d/%m/%Y"),
            alt.Tooltip("Horas de sono", title="Horas", format=".2f"),
            alt.Tooltip("Classificação", title="Classificação"),
        ],
    )
    rule_df = pd.DataFrame({"hora_teto": [cap_h]})
    rule = (
        alt.Chart(rule_df)
        .mark_rule(
            color="#e67e22",
            strokeDash=[6, 4],
            strokeWidth=2,
        )
        .encode(
            y=alt.Y("hora_teto:Q", title="Horas de sono"),
            tooltip=alt.value(f"Teto do relatório ({cap_m:.0f} min)"),
        )
    )
    return (line + pts + rule).properties(height=280).interactive()


def _vs_escola_from_csv_or_minutos(qualidade: Any, sono_min: Any) -> str:
    """
    Texto da coluna “Vs. referência escola”: prioriza `qualidade_sono` já normalizada
    no agregado diário (veio do CSV). Só usa minutos se o CSV não tiver categoria.
    """
    q = str(qualidade if qualidade is not None else "").strip()
    if q and q not in ("—", "-", "nan", "none", "NaN"):
        return q
    try:
        sm = float(sono_min)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(sm) or sm <= 0:
        return "—"
    return _classify_sono_min_for_report(sm)


def _daily_sleep_lookup_for_ref_table(daily: pd.DataFrame | None) -> pd.DataFrame:
    """
    Uma linha por dia do relatório, mesma chave `data_show` (dd/mm) usada no gráfico de refeições.

    - **min_diario:** média dos minutos já limitada ao teto (`ROTINA_SONO_MAX_MIN`).
    - **vs_escola:** **`qualidade_sono` do CSV** (normalizada por dia); se vazio, classifica pelos minutos (mesmo teto).
    """
    empty = pd.DataFrame(columns=["data_show", "min_diario", "vs_escola"])
    if daily is None or daily.empty or "sono_min" not in daily.columns:
        return empty
    d = daily.sort_values("dia").copy()
    d["data_show"] = d["dia"].map(lambda x: pd.Timestamp(x).strftime("%d/%m"))
    sm = pd.to_numeric(d["sono_min"], errors="coerce")
    d["min_diario"] = sm.round().astype("Int64")
    quals = d["qualidade_sono"] if "qualidade_sono" in d.columns else pd.Series([""] * len(d))
    d["vs_escola"] = [
        _vs_escola_from_csv_or_minutos(q, smv)
        for q, smv in zip(quals, sm.astype(float))
    ]
    out = d[["data_show", "min_diario", "vs_escola"]].drop_duplicates(subset=["data_show"])
    return out


def _sleep_reference_table_ui(daily: pd.DataFrame | None) -> pd.DataFrame:
    """Uma linha por dia: minutos no diário + classificação (mesma lógica do gráfico de sono)."""
    lk = _daily_sleep_lookup_for_ref_table(daily)
    if lk.empty:
        return pd.DataFrame(columns=["Dia", "Min. sono (diário)", "Classificação"])
    return pd.DataFrame(
        {
            "Dia": lk["data_show"],
            "Min. sono (diário)": lk["min_diario"].map(
                lambda x: f"{int(x)} min" if pd.notna(x) else "—"
            ),
            "Classificação": lk["vs_escola"].astype(str),
        }
    )


def _norm_intake_status_for_chart(raw: str) -> str:
    """Reduz textos livres a poucas categorias para cores consistentes na legenda."""
    t = str(raw or "").strip().lower()
    if "recusou" in t:
        return "Recusou"
    if "comeu bem" in t or "comeu tudo" in t:
        return "Comeu bem"
    if "comeu pouco" in t:
        return "Comeu pouco"
    if t in ("—", "-", "", "nan", "none"):
        return "Sem registro"
    return "Outro"


def _meal_intake_stacked_bar_altair(
    daily_meals: pd.DataFrame,
) -> tuple[alt.Chart, pd.DataFrame]:
    """
    Barras empilhadas (café → almoço → lanche, de baixo para cima), cor = classificação da ingestão.
    Retorna o gráfico e a tabela **só de refeições** (Dia, Refeição, texto). A tabela de sono fica à parte.
    """
    meal_pt = {
        "cafe_manha": "Café da manhã",
        "almoco": "Almoço",
        "lanche_tarde": "Lanche",
    }
    meal_ord = {"cafe_manha": 0, "almoco": 1, "lanche_tarde": 2}
    dm = daily_meals.sort_values("dia")
    if dm.empty:
        empty = pd.DataFrame()
        return alt.Chart(empty).mark_bar(), empty

    day_order = sorted(dm["dia"].unique(), key=lambda x: pd.Timestamp(x))
    sort_labels = [pd.Timestamp(d).strftime("%d/%m") for d in day_order]

    rows: list[dict[str, str | float | int]] = []
    for _, r in dm.iterrows():
        d_ts = pd.Timestamp(r["dia"])
        d_show = d_ts.strftime("%d/%m")
        for col, pt in meal_pt.items():
            registro = str(r[col]).strip() or "—"
            status = _norm_intake_status_for_chart(registro)
            rows.append(
                {
                    "data_show": d_show,
                    "fatia": 1.0,
                    "ordem": meal_ord[col],
                    "refeicao": pt,
                    "status": status,
                    "registro": registro,
                }
            )
    bar_df = pd.DataFrame(rows)

    domain_all = ["Comeu bem", "Comeu pouco", "Recusou", "Sem registro", "Outro"]
    range_all = ["#1a9850", "#f4a020", "#c0392b", "#aeb6bf", "#5d6d7e"]
    present = [s for s in domain_all if s in set(bar_df["status"])]
    colors = [range_all[domain_all.index(s)] for s in present]

    chart = (
        alt.Chart(bar_df)
        .mark_bar(cornerRadiusEnd=2, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "data_show:N",
                title="Dia",
                sort=sort_labels,
                axis=alt.Axis(labelAngle=0, labelOverlap=False),
            ),
            y=alt.Y(
                "fatia:Q",
                stack="zero",
                title=None,
                axis=None,
            ),
            color=alt.Color(
                "status:N",
                title="Ingestão (legenda)",
                scale=alt.Scale(domain=present, range=colors),
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    columns=3,
                    labelLimit=0,
                    labelFontSize=12,
                    titleFontSize=12,
                    padding=12,
                    symbolSize=80,
                ),
            ),
            order=alt.Order("ordem:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("data_show", title="Dia"),
                alt.Tooltip("refeicao", title="Refeição"),
                alt.Tooltip("registro", title="Texto registrado"),
                alt.Tooltip("status", title="Classificação"),
            ],
        )
        .properties(height=260)
    )

    ref_meals = bar_df.rename(
        columns={
            "data_show": "Dia",
            "refeicao": "Refeição",
            "registro": "Texto registrado",
        }
    )[["Dia", "Refeição", "Texto registrado"]]
    return chart, ref_meals


def sleep_meal_report_summary_md(
    df: pd.DataFrame,
    alunos_label: str,
    janela_ini: str | None,
    janela_fim: str | None,
) -> str:
    """Texto curto com leitura dos números (minimalista)."""
    n = len(df)
    mi = float(df["ingestao"].mean())
    ms = float(df["sono_min"].mean())
    mx_ing = float(df["ingestao"].max()) if n else 0.0
    mn_ing = float(df["ingestao"].min()) if n else 0.0

    parts = [
        "**Base de informação:** dados oficiais de cadastro e de rotina diária da instituição, "
        "tratados de forma confidencial e protegidos pelas normas aplicáveis, inclusive quanto a "
        "direitos autorais e privacidade.",
        f"**Aluno(s) considerado(s):** {alunos_label}.",
    ]
    if janela_ini and janela_fim:
        parts.append(
            f"**Janela semanal:** 7 dias corridos de **{janela_ini}** a **{janela_fim}** "
            f"(terminando na data mais recente do diário). **{n}** dia(s) com registro válido nesse intervalo."
        )
    else:
        parts.append(f"**{n}** dia(s) com refeições e sono registrados.")
    cap_m = int(round(ROTINA_SONO_MAX_MIN))
    l1 = int(round(ROTINA_SONO_FAIXA_LIMITE_1))
    l2 = int(round(ROTINA_SONO_FAIXA_LIMITE_2))
    parts.append(
        f"Ingestão combinada (quatro momentos, 0–8): média **{mi:.1f}** "
        f"(mín. {mn_ing:.1f}, máx. {mx_ing:.1f}). "
        f"Sono (horários do CSV, **no máximo {cap_m} min** por registro): média **{ms:.0f} min** por dia. "
        f"**Classificação:** **pouco** abaixo de {l1} min, **normal** entre {l1} e {l2} min, **bastante** a partir de {l2} min "
        f"(até {cap_m} min). **Qualidade do sono** segue três categorias canônicas alinhadas a essas faixas."
    )
    try:
        _dfq = df[df["qualidade_sono"].astype(str).str.strip() != "—"]
        if _dfq.empty:
            by_q = pd.Series(dtype=float)
        else:
            by_q = _dfq.groupby("qualidade_sono", dropna=False)["ingestao"].mean().sort_values(
                ascending=False
            )
        if len(by_q) > 1:
            top = by_q.index[0]
            parts.append(
                f"Entre essas **categorias de sono**, a maior média de ingestão aparece em “{top}”. "
                "Isso descreve o conjunto de dados, não causa médica ou pedagógica."
            )
        elif len(by_q) == 1:
            parts.append(
                "Todos os dias deste recorte compartilham a mesma categoria de sono após padronização; "
                "compare outras semanas ou turmas para ver tendências."
            )
    except Exception:
        pass
    return "\n\n".join(parts)


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

    sono_chart = _sleep_line_chart_altair(df)
    bar_chart, ref_meals = _meal_intake_stacked_bar_altair(daily_meals)
    tbl_sono = _sleep_reference_table_ui(df)

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
    mv = p.get("mutacao")
    if mv is not None and not isinstance(mv, str):
        p["mutacao"] = None
    elif isinstance(mv, str) and not mv.strip():
        p["mutacao"] = None
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
    extra_suffix: str = "",
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
    if extra_suffix.strip():
        core += "\n" + extra_suffix.strip() + "\n"
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
    extra_planner_suffix: str = "",
) -> dict[str, Any]:
    planner = _routing_planner_user_content(
        user_message,
        force=force,
        history=history,
        extra_suffix=extra_planner_suffix,
    )
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
    extra_system: str | None = None,
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
    if extra_system and extra_system.strip():
        messages.append({"role": "system", "content": extra_system.strip()})
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
    extra_planner_suffix: str = "",
) -> dict[str, Any]:
    planner = _routing_planner_user_content(
        user_message,
        force=force,
        history=history,
        extra_suffix=extra_planner_suffix,
    )
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
    extra_system: str | None = None,
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
    if extra_system and extra_system.strip():
        messages.append({"role": "system", "content": extra_system.strip()})
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
    extra_planner_suffix: str = "",
) -> dict[str, Any]:
    if _use_openai_compatible_chat():
        return openai_plan_sources(
            user_message,
            force=force,
            history=history,
            extra_planner_suffix=extra_planner_suffix,
        )
    return ollama_plan_sources(
        user_message,
        force=force,
        history=history,
        extra_planner_suffix=extra_planner_suffix,
    )


def _cap_chat_stream(
    gen: Generator[str, None, None], max_chars: int
) -> Generator[str, None, None]:
    """Corta o streaming quando excede o teto (evita cópia interminável de tabelas)."""
    if max_chars <= 0:
        yield from gen
        return
    n = 0
    for piece in gen:
        if not piece:
            continue
        remain = max_chars - n
        if remain <= 0:
            yield (
                "\n\n_(Resposta truncada: limite `ROTINA_CHAT_MAX_OUTPUT_CHARS` no `.env`.)_"
            )
            break
        if len(piece) <= remain:
            yield piece
            n += len(piece)
        else:
            yield piece[:remain]
            yield (
                "\n\n_(Resposta truncada: limite `ROTINA_CHAT_MAX_OUTPUT_CHARS` no `.env`.)_"
            )
            break


def _first_re_group(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1)


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
        return result_message.strip() or (
            "A alteração não foi gravada. Feche o CSV se estiver aberto e tente novamente."
        )

    low = (mut_sql or "").lower()
    count = _first_re_group(r"CONTAGEM_OFICIAL_ALUNOS\s*=\s*(\d+)", duck_block) or "?"
    if "insert" in low and "info_alunos" in low:
        aid = _first_re_group(r"CONFIRME_id_aluno\s*=\s*(\d+)", duck_block) or "?"
        return (
            f"Cadastro realizado com sucesso. O novo aluno foi gravado com `id_aluno={aid}`.\n\n"
            f"Total atual de alunos no cadastro: {count}."
        )
    if "delete" in low and "info_alunos" in low:
        return (
            "Remoção realizada com sucesso no cadastro.\n\n"
            f"Total atual de alunos no cadastro: {count}."
        )
    return (
        "Alteração aplicada e CSV atualizado com sucesso.\n\n"
        f"Total atual de alunos no cadastro: {count}."
        if "info_alunos" in low
        else "Alteração aplicada e CSV atualizado com sucesso."
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
        p["mutacao"] = None
    return p


def llm_chat_stream(
    user_message: str,
    duck_block: str,
    rag_block: str,
    history: Iterable[dict[str, str]],
    extra_system: str | None = None,
) -> Generator[str, None, None]:
    if _use_openai_compatible_chat():
        yield from _cap_chat_stream(
            openai_chat_stream(
                user_message,
                duck_block,
                rag_block,
                history,
                extra_system=extra_system,
            ),
            ROTINA_CHAT_MAX_OUTPUT_CHARS,
        )
        return
    if ROTINA_CHAT_PROVIDER in ("openai", "openrouter") and not OPENAI_API_KEY:
        yield (
            "**Configuração:** use `OPENAI_API_KEY` ou `OPENROUTER_API_KEY`, além de "
            "`OPENAI_BASE_URL` e `OPENAI_CHAT_MODEL` (para OpenRouter, veja `.env`)."
        )
        return
    yield from _cap_chat_stream(
        ollama_chat_stream(
            user_message,
            duck_block,
            rag_block,
            history,
            extra_system=extra_system,
        ),
        ROTINA_CHAT_MAX_OUTPUT_CHARS,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def _whisper_upload_name_and_mime(audio_bytes: bytes, reported_name: str) -> tuple[str, str]:
    """
    O Streamlit costuma gravar WebM no navegador, mas o nome do ficheiro pode ser .wav.
    Ajusta extensão e MIME com base nos magic bytes para o ffmpeg do Whisper aceitar o ficheiro.
    """
    raw = (reported_name or "").strip() or "gravacao"
    base = raw.rsplit(".", 1)[0] if "." in raw else raw
    b = audio_bytes
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        return f"{base}.wav", "audio/wav"
    if len(b) >= 4 and b[0] == 0x1A and b[1:4] == b"\x45\xdf\xa3":
        return f"{base}.webm", "audio/webm"
    if len(b) >= 4 and b[:4] == b"OggS":
        return f"{base}.ogg", "audio/ogg"
    if len(b) >= 3 and b[:3] == b"ID3":
        return f"{base}.mp3", "audio/mpeg"
    if len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
        return f"{base}.mp3", "audio/mpeg"
    low = raw.lower()
    if low.endswith(".webm"):
        return raw, "audio/webm"
    if low.endswith(".ogg"):
        return raw, "audio/ogg"
    if low.endswith(".mp3"):
        return raw, "audio/mpeg"
    # Gravação típica do browser (Chrome/Edge): WebM, por vezes com nome enganador.
    return f"{base}.webm", "audio/webm"


def _rotina_st_audio_format(audio_bytes: bytes) -> str:
    """MIME/formato para `st.audio` a partir dos bytes (mesma heurística que o Whisper)."""
    _, mime = _whisper_upload_name_and_mime(audio_bytes, "gravacao.wav")
    return {
        "audio/wav": "audio/wav",
        "audio/webm": "audio/webm",
        "audio/ogg": "audio/ogg",
        "audio/mpeg": "audio/mp3",
    }.get(mime, "audio/wav")


def _transcribe_whisper_http(audio_bytes: bytes, filename: str) -> tuple[str | None, str | None]:
    """Whisper via API compatível com OpenAI (multipart). Chave opcional. Retorna (texto, erro)."""
    if not audio_bytes:
        return None, "Nenhum dado de áudio para enviar ao Whisper."
    base = (OPENAI_TRANSCRIBE_BASE_URL or "").strip().rstrip("/")
    if not base:
        return None, "URL de transcrição não configurada (`OPENAI_TRANSCRIBE_BASE_URL`)."
    key = (OPENAI_TRANSCRIBE_API_KEY or "").strip()
    upload_name, mime = _whisper_upload_name_and_mime(audio_bytes, filename)
    url = f"{base}/audio/transcriptions"
    try:
        headers: dict[str, str] = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = httpx.post(
            url,
            headers=headers,
            files={"file": (upload_name, audio_bytes, mime)},
            data={
                "model": OPENAI_TRANSCRIBE_MODEL or "whisper-1",
                "language": "pt",
            },
            timeout=httpx.Timeout(180.0, connect=30.0),
        )
        if r.status_code >= 400:
            detail = (r.text or "").strip().replace("\n", " ")
            if len(detail) > 400:
                detail = detail[:400] + "…"
            return None, f"Whisper respondeu {r.status_code}: {detail or 'sem detalhe'}"
        try:
            out = r.json()
        except json.JSONDecodeError:
            raw = (r.text or "").strip()
            if not raw:
                return None, "__EMPTY_TRANSCRIPT__"
            return raw, None
        if not isinstance(out, dict):
            return None, f"Whisper devolveu JSON inesperado: {type(out).__name__}"
        t = (out.get("text") or "").strip()
        if not t:
            # Código interno: UI trata com aviso curto no chat e reabre o microfone.
            return None, "__EMPTY_TRANSCRIPT__"
        return t, None
    except httpx.ConnectError:
        return None, (
            f"Não foi possível ligar ao Whisper em `{base}`. "
            "Confirme `docker compose up`, espere o modelo carregar (`docker logs rotina-whisper`) "
            "e, se corre o Streamlit **fora** do Docker, use no `.env`: "
            "`OPENAI_TRANSCRIBE_BASE_URL=http://127.0.0.1:9000/v1`."
        )
    except httpx.TimeoutException:
        return None, (
            "Tempo esgotado ao transcrever. O primeiro uso do Whisper pode demorar; tente de novo."
        )
    except Exception as e:
        return None, f"Erro ao chamar Whisper: {type(e).__name__}: {e!s}"[:400]


def _transcribe_google_sr(audio_bytes: bytes) -> str | None:
    """Fallback: Google Web Speech API (requer rede; áudio em WAV)."""
    try:
        import speech_recognition as sr  # type: ignore[import-untyped]
    except ImportError:
        return None
    r = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = r.record(source)
        return (r.recognize_google(audio, language="pt-BR") or "").strip() or None
    except Exception:
        return None


def transcribe_voice_bytes(audio_bytes: bytes, filename: str) -> tuple[str | None, str | None]:
    """
    Retorna (texto, erro). Tenta Whisper (HTTP) e depois reconhecimento Google.
    """
    if not audio_bytes or len(audio_bytes) < 80:
        return (
            None,
            "Gravação muito curta ou vazia. Grave novamente por alguns segundos.",
        )
    txt, werr = _transcribe_whisper_http(audio_bytes, filename)
    if txt:
        return txt, None
    gtxt = _transcribe_google_sr(audio_bytes)
    if gtxt:
        return gtxt, None
    if werr == "__EMPTY_TRANSCRIPT__":
        return None, "__EMPTY_TRANSCRIPT__"
    if werr:
        return None, werr
    return (
        None,
        "Não foi possível transcrever o áudio (sem resposta útil do Whisper nem do fallback Google). "
        "Confirme o serviço **Whisper** (`docker compose up`, `docker logs rotina-whisper`). "
        "Se corre o Streamlit **no PC** (fora do Docker), no `.env` use "
        "`OPENAI_TRANSCRIBE_BASE_URL=http://127.0.0.1:9000/v1` (não `http://whisper:...`). "
        "Alternativa: API OpenAI com `OPENAI_TRANSCRIBE_BASE_URL` e `OPENAI_TRANSCRIBE_API_KEY`.",
    )


ROTINA_CHAT_QUERY_PARAM = "rotina_chat"
ROTINA_CHAT_SESSION_SUBDIR = ".rotina_chat"
ROTINA_BROWSER_SESSION_QUERY_PARAM = "rotina_session"
ROTINA_BROWSER_SESSION_SUBDIR = ".rotina_browser_sessions"
ROTINA_DIRECT_CHAT_FILE = "chat_familia_educadores.json"


def _query_param_first(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        return str(v[0]) if v else None
    return str(v)


def _is_safe_chat_session_id(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _chat_session_dir() -> Path:
    d = DATA_DIR / ROTINA_CHAT_SESSION_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _browser_session_dir() -> Path:
    d = DATA_DIR / ROTINA_BROWSER_SESSION_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _issue_browser_session_token() -> str:
    return str(uuid.uuid4())


def _save_browser_session_token(token: str, username: str) -> None:
    p = _browser_session_dir() / f"{token}.json"
    try:
        p.write_text(
            json.dumps({"username": username.strip(), "v": 1}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _delete_browser_session_file(token: str) -> None:
    if not _is_safe_chat_session_id(token):
        return
    try:
        (_browser_session_dir() / f"{token}.json").unlink(missing_ok=True)
    except OSError:
        pass


def try_restore_rotina_browser_session() -> bool:
    """
    Após F5 o session_state do Streamlit reinicia; restaura login se a URL tiver
    `?rotina_session=<uuid>` e existir o ficheiro em disco (token opaco).
    """
    if st.session_state.get("rotina_authenticated"):
        return True
    raw = _query_param_first(st.query_params.get(ROTINA_BROWSER_SESSION_QUERY_PARAM))
    if not raw or not _is_safe_chat_session_id(raw):
        return False
    path = _browser_session_dir() / f"{raw}.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    username = str(data.get("username") or "").strip()
    if not username:
        return False
    users = load_rotina_users()
    rec = users.get(username) if isinstance(users, dict) else None
    if not isinstance(rec, dict):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    role = str(rec.get("role", "")).strip().lower()
    if role not in ("gestao", "educador", "familia"):
        return False
    st.session_state.rotina_authenticated = True
    st.session_state.rotina_role = role
    st.session_state.rotina_user_label = str(rec.get("display_name") or username).strip()
    if role == "familia":
        try:
            st.session_state.rotina_parent_id_aluno = int(rec.get("id_aluno"))
        except (TypeError, ValueError):
            st.session_state.rotina_authenticated = False
            return False
    else:
        st.session_state.rotina_parent_id_aluno = None
    st.session_state.setdefault("rotina_sidebar_screen", "assistant")
    st.session_state.setdefault("rotina_direct_chat_student", None)
    return True


def _clear_browser_session_query_param() -> None:
    if ROTINA_BROWSER_SESSION_QUERY_PARAM not in st.query_params:
        return
    try:
        del st.query_params[ROTINA_BROWSER_SESSION_QUERY_PARAM]
    except Exception:
        pass


def _direct_chat_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / ROTINA_DIRECT_CHAT_FILE


def _legacy_direct_chat_path() -> Path:
    return (DATA_DIR / ".familia_educador_chat") / "mensagens.json"


def _normalize_direct_chat_sender_role(raw: str) -> str | None:
    """`familia` ↔ `educador` (gestão conta como lado escola)."""
    r = (raw or "").strip().lower()
    if r == "familia":
        return "familia"
    if r in ("educador", "gestao"):
        return "educador"
    return None


def _direct_chat_viewer_side(session_role: str) -> str | None:
    sr = (session_role or "").strip().lower()
    if sr == "familia":
        return "familia"
    if sr in ("educador", "gestao"):
        return "educador"
    return None


def _load_direct_chat_store() -> dict[str, list[dict[str, str]]]:
    p = _direct_chat_path()
    if not p.is_file():
        lp = _legacy_direct_chat_path()
        if lp.is_file():
            p = lp
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    clean: dict[str, list[dict[str, str]]] = {}
    for key, msgs in raw.items():
        if not isinstance(key, str) or not isinstance(msgs, list):
            continue
        parsed: list[dict[str, str]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            txt = str(m.get("content") or "").strip()
            sender = str(m.get("sender") or "").strip()
            role = _normalize_direct_chat_sender_role(str(m.get("sender_role") or ""))
            if not txt or role is None:
                continue
            parsed.append({"content": txt, "sender": sender, "sender_role": role})
        clean[key] = parsed[-300:]
    return clean


def _persist_direct_chat_store(store: dict[str, list[dict[str, str]]]) -> None:
    try:
        _direct_chat_path().write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return


def _ensure_direct_chat_store_file() -> None:
    p = _direct_chat_path()
    if p.is_file():
        return
    _persist_direct_chat_store({})


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


def _coerce_stored_chat_messages_list(data: list[Any]) -> list[dict[str, str]] | None:
    out: list[dict[str, str]] = []
    for m in data:
        if not isinstance(m, dict):
            return None
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            return None
        out.append({"role": str(role), "content": content})
    return out


ROTINA_REPORT_PHASES = frozenset({"idle", "ask_name", "result"})


def _report_blob_for_disk() -> dict[str, Any] | None:
    """Estado mínimo do relatório sono/refeições para restaurar após F5 (gráficos refeitos no DuckDB)."""
    phase = st.session_state.get("sleep_rep_phase", "idle")
    if phase not in ROTINA_REPORT_PHASES:
        phase = "idle"
    if phase == "idle":
        return None
    blob: dict[str, Any] = {
        "phase": phase,
        "query_name": str(st.session_state.get("sleep_rep_query_name") or ""),
        "resolved_label": str(st.session_state.get("sleep_rep_resolved_label") or ""),
    }
    if phase == "ask_name":
        blob["nome_field"] = str(st.session_state.get("sleep_rep_nome_field") or "")
    return blob


def _apply_report_blob_from_disk(blob: Any) -> None:
    if not isinstance(blob, dict):
        return
    phase = blob.get("phase")
    if phase not in ROTINA_REPORT_PHASES:
        return
    qn = str(blob.get("query_name") or "").strip()
    if phase == "result" and not qn:
        st.session_state.sleep_rep_phase = "idle"
        return
    if phase == "idle":
        st.session_state.sleep_rep_phase = "idle"
        return
    st.session_state.sleep_rep_phase = phase
    st.session_state.sleep_rep_query_name = str(blob.get("query_name") or "")
    st.session_state.sleep_rep_resolved_label = str(blob.get("resolved_label") or "")
    if phase == "ask_name" and "nome_field" in blob:
        st.session_state.sleep_rep_nome_field = str(blob.get("nome_field") or "")


def _parse_session_file_payload(data: Any) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Ficheiro legado: lista só de mensagens. Novo: `{\"messages\": [...], \"report\": {...}}`."""
    if isinstance(data, list):
        parsed = _coerce_stored_chat_messages_list(data)
        return (parsed or [], None)
    if isinstance(data, dict) and "messages" in data:
        raw_m = data["messages"]
        if not isinstance(raw_m, list):
            return ([], None)
        parsed = _coerce_stored_chat_messages_list(raw_m)
        if parsed is None:
            return ([], None)
        rep = data.get("report")
        if isinstance(rep, dict):
            return (parsed, rep)
        return (parsed, None)
    return ([], None)


def _rotina_session_serial_current() -> str:
    payload = {
        "messages": st.session_state.get("messages") or [],
        "report": _report_blob_for_disk(),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def ensure_rotina_chat_session_id() -> str:
    """Garante `?rotina_chat=<uuid>` na URL; o mesmo id reaparece após F5 e liga ao ficheiro em disco."""
    raw = _query_param_first(st.query_params.get(ROTINA_CHAT_QUERY_PARAM))
    if raw and _is_safe_chat_session_id(raw):
        return raw
    st.query_params[ROTINA_CHAT_QUERY_PARAM] = str(uuid.uuid4())
    st.rerun()


def sync_rotina_chat_from_disk(chat_id: str) -> None:
    """Carrega mensagens e estado do relatório quando a sessão Streamlit é nova ou o id da URL mudou."""
    if st.session_state.get("_chat_disk_synced_for") == chat_id:
        return
    path = _chat_session_dir() / f"{chat_id}.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            msgs, rep = _parse_session_file_payload(data)
            st.session_state.messages = msgs
            if rep is not None:
                _apply_report_blob_from_disk(rep)
        except (OSError, json.JSONDecodeError):
            pass
    st.session_state._chat_disk_synced_for = chat_id
    st.session_state._rotina_session_serial = _rotina_session_serial_current()


def persist_rotina_chat_to_disk(chat_id: str) -> None:
    """Grava conversa + relatório em `ROTINA_DATA_DIR/.rotina_chat/<uuid>.json`."""
    msgs: list[dict[str, str]] = st.session_state.get("messages") or []
    rep = _report_blob_for_disk()
    payload = {"messages": msgs, "report": rep}
    serial = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if st.session_state.get("_rotina_session_serial") == serial:
        return
    st.session_state._rotina_session_serial = serial
    path = _chat_session_dir() / f"{chat_id}.json"
    if not msgs and rep is None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    try:
        path.write_text(serial, encoding="utf-8")
    except OSError:
        pass


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


def _rotina_chat_footer_css() -> None:
    """Espaço no fundo do conteúdo + estilo da barra fixa (microfone + chat)."""
    st.markdown(
        """
<style>
section.main div.block-container {
    padding-bottom: 5.85rem !important;
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
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.09) !important;
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


def _rotina_pin_chat_footer_row() -> None:
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


def _render_rag_sidebar_body(body: Any) -> None:
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
            ch = _chunks[0]
            if _lq:
                st.caption(f"Pergunta: {_lq[:180]}{'…' if len(_lq) > 180 else ''}")
            src = str(ch.get("source", "?"))
            ck = ch.get("chunk")
            dist = ch.get("distance")
            meta_bits: list[str] = []
            if ck not in (None, "", "?"):
                meta_bits.append(f"índice {ck}")
            if isinstance(dist, (int, float)):
                meta_bits.append(f"distância {float(dist):.4f}")
            with st.expander(src, expanded=False):
                if meta_bits:
                    st.caption(" · ".join(meta_bits))
                txt = (ch.get("text") or "").strip()
                if len(txt) > 4000:
                    txt = txt[:4000] + "…"
                st.text(txt)


ROTINA_USERS_FILENAME = "rotina_users.json"


def load_rotina_users() -> dict[str, Any]:
    p = DATA_DIR / ROTINA_USERS_FILENAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _planner_suffix_gestao() -> str:
    return (
        'Inclua no JSON o campo opcional "mutacao": null ou UMA string SQL com INSERT, UPDATE ou DELETE '
        "apenas nas tabelas `info_alunos` e `diario_estruturado` (colunas do esquema acima). "
        'Use "mutacao" só se o utilizador pedir para criar, alterar ou apagar registos; caso contrário "mutacao": null. '
        "**Novos alunos (`INSERT` em `info_alunos`):** defina `id_aluno` como o próximo id livre — "
        "use `(SELECT COALESCE(MAX(id_aluno), 0) + 1 FROM info_alunos)` como primeiro valor em `VALUES` "
        "(não invente um número fixo nem reutilize ids existentes). "
        "**Novas linhas de diário (`INSERT` em `diario_estruturado`):** id_registro com "
        "`(SELECT COALESCE(MAX(id_registro), 0) + 1 FROM diario_estruturado)` da mesma forma. "
        "Após mutação bem-sucedida o servidor corre SELECTs de verificação (estado **final** dos CSV); "
        "na resposta, confirme o sucesso do pedido — não trate a linha inserida como duplicata pré-existente. "
        "Antes de gravar, o servidor pode avisar se **nome** ou **contacto** já existiam noutra linha — repita esse aviso ao utilizador. "
        'Pode omitir "sql" no JSON ou devolver só um SELECT complementar. '
        'Formato: {"fontes": [...], "sql": null ou "SELECT ...", "mutacao": null ou "DELETE ..."}.'
    )


def _planner_suffix_educador_readonly() -> str:
    return (
        'RBAC — Perfil Educador (só leitura nos CSV): não inclua alterações. Use sempre `"mutacao": null`. '
        "Apenas SELECT em `info_alunos` e `diario_estruturado`. "
        'Formato: {"fontes": [...], "sql": null ou "SELECT ...", "mutacao": null}.'
    )


def _planner_suffix_familia(id_aluno: int, nome: str) -> str:
    return (
        f"RBAC — Perfil Família (só leitura): o responsável vê apenas o aluno **{nome}** (id_aluno={id_aluno}). "
        'Não inclua "mutacao". Todas as consultas SQL devem restringir-se a esse aluno.'
    )


def _chat_system_familia(id_aluno: int, nome: str) -> str:
    return (
        f"O utilizador é um responsável (perfil leitura). Para dados de cadastro ou diário, aborde apenas o aluno "
        f"**{nome}** (id_aluno={id_aluno}). Não revele dados de outras crianças."
    )


def render_login() -> None:
    _pad_l, _center, _pad_r = st.columns([1, 2, 1])
    with _center:
        _il, _inner, _ir = st.columns([1, 2, 1])
        with _inner:
            _logo_path = DATA_DIR / "logo_rotina_viva.png"
            if _logo_path.is_file():
                st.image(str(_logo_path), use_container_width=True)
            else:
                st.title("Rotina Viva")
            with st.form("rotina_login_form"):
                username = st.text_input("Usuário")
                password = st.text_input("Senha", type="password")
                submitted = st.form_submit_button("Entrar", use_container_width=True)
            if submitted:
                users = load_rotina_users()
                key = (username or "").strip()
                rec = users.get(key) if isinstance(users, dict) else None
                if isinstance(rec, dict) and rec.get("password") == password:
                    role = str(rec.get("role", "")).strip().lower()
                    if role not in ("gestao", "educador", "familia"):
                        st.error(
                            "Perfil inválido: use `gestao`, `educador` ou `familia` no ficheiro de utilizadores."
                        )
                        return
                    st.session_state.rotina_authenticated = True
                    st.session_state.rotina_role = role
                    st.session_state.rotina_user_label = str(
                        rec.get("display_name") or key
                    ).strip()
                    if role == "familia":
                        try:
                            st.session_state.rotina_parent_id_aluno = int(
                                rec.get("id_aluno")
                            )
                        except (TypeError, ValueError):
                            st.error(
                                "Para o perfil Família é obrigatório um campo numérico `id_aluno`."
                            )
                            return
                    else:
                        st.session_state.rotina_parent_id_aluno = None
                    st.session_state.rotina_sidebar_screen = "assistant"
                    st.session_state.rotina_direct_chat_student = None
                    st.session_state.messages = []
                    st.session_state.pop("_rotina_session_serial", None)
                    st.session_state.pop("_chat_disk_synced_for", None)
                    _tok = _issue_browser_session_token()
                    _save_browser_session_token(_tok, key)
                    st.query_params[ROTINA_BROWSER_SESSION_QUERY_PARAM] = _tok
                    st.query_params[ROTINA_CHAT_QUERY_PARAM] = str(uuid.uuid4())
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos.")


def render_auth_sidebar() -> None:
    label = st.session_state.get("rotina_user_label") or "—"
    role = st.session_state.get("rotina_role") or ""
    role_lc = str(role).strip().lower()
    st.markdown(f"**{label}**")
    if role_lc == "gestao":
        st.caption("Perfil: Gestão")
    elif role_lc == "educador":
        st.caption("Perfil: Educador (leitura)")
    elif role_lc == "familia":
        st.caption("Perfil: Família")
    _left, _right = st.columns([1, 1], gap="small")
    with _left:
        if st.button("IA", key="rotina_sidebar_assistente_btn", help="Abrir Assistente IA"):
            st.session_state.rotina_sidebar_screen = "assistant"
            st.rerun()
    with _right:
        _direct_btn_label = (
            "Chat direto escola"
            if role_lc == "familia"
            else "Chat direto família"
        )
        _direct_btn_help = (
            "Abrir mensagens com a escola (educadores / gestão)."
            if role_lc == "familia"
            else "Abrir mensagens com as famílias (por aluno)."
        )
        if st.button(
            _direct_btn_label,
            key="rotina_sidebar_direct_chat_btn",
            help=_direct_btn_help,
        ):
            st.session_state.rotina_sidebar_screen = "direct_chat"
            st.rerun()
    if st.button("Sair", key="rotina_logout_btn"):
        _ltok = _query_param_first(
            st.query_params.get(ROTINA_BROWSER_SESSION_QUERY_PARAM)
        )
        if _ltok:
            _delete_browser_session_file(_ltok)
        _clear_browser_session_query_param()
        st.session_state.rotina_authenticated = False
        st.session_state.rotina_role = None
        st.session_state.rotina_user_label = ""
        st.session_state.rotina_parent_id_aluno = None
        st.session_state.rotina_sidebar_screen = "assistant"
        st.session_state.rotina_direct_chat_student = None
        st.session_state.messages = []
        st.session_state.pop("_rotina_session_serial", None)
        st.session_state.pop("_chat_disk_synced_for", None)
        st.query_params[ROTINA_CHAT_QUERY_PARAM] = str(uuid.uuid4())
        st.rerun()
    st.divider()


def _render_chat_sidebar_internals() -> Any:
    """Controlos de fonte + limpar conversa + placeholder dos trechos RAG."""
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


def render_rotina_chat(
    chat_id: str,
    conn: duckdb.DuckDBPyConnection | None,
    collection: Any,
    rag_sidebar_body: Any,
    *,
    read_only_db: bool,
    allow_mutations: bool,
    parent_scope: tuple[int, str] | None,
    planner_extra: str,
    chat_extra_system: str | None,
    report_parent_lock: tuple[int, str] | None,
) -> None:
    """Área principal: logo, relatório, chat (texto + áudio), processamento SQL/RAG/mutação."""
    _rep_phase = st.session_state.get("sleep_rep_phase", "idle")
    _exp_relatorio = _rep_phase != "idle"
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
            st.markdown(msg["content"])

    _rotina_voice_spinner_slot = st.empty()

    _msgs = st.session_state.messages
    _last_is_user = bool(_msgs and _msgs[-1]["role"] == "user")

    _gutter_l, _center_wrap, _gutter_r = st.columns([1, 2.2, 1], gap="small")
    _voice_blob = None
    with _center_wrap:
        _icol, _vcol = st.columns([5, 1], gap="small")
        with _icol:
            prompt = st.chat_input("Pergunte sobre rotinas, alunos ou documentos da escola…")
        with _vcol:
            _voice_preview = st.session_state.get("rotina_voice_preview_bytes")
            if _voice_preview is not None:
                st.audio(_voice_preview, format=_rotina_st_audio_format(_voice_preview))
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

    _rotina_pin_chat_footer_row()

    if _voice_blob is not None:
        _raw = _voice_blob.getvalue()
        if _raw:
            _vh = hashlib.sha256(_raw).hexdigest()
            if st.session_state.get("rotina_voice_hash") != _vh:
                _vname = getattr(_voice_blob, "name", None) or "gravacao.wav"
                with _rotina_voice_spinner_slot.container():
                    with st.spinner("Processando áudio…"):
                        _vtxt, _ver = transcribe_voice_bytes(_raw, _vname)
                if _vtxt:
                    st.session_state.rotina_voice_preview_bytes = _raw
                    st.session_state.rotina_voice_hash = _vh
                    st.session_state.messages.append({"role": "user", "content": _vtxt})
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        persist_rotina_chat_to_disk(chat_id)
        st.rerun()

    if _last_is_user:
        user_text = _msgs[-1]["content"]
        mode_ds = st.session_state.data_source_mode

        if conn is None:
            err = "DuckDB indisponível. Verifique os CSVs em `ROTINA_DATA_DIR`."
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        elif mode_ds in ("auto", "documents") and collection is None:
            err = (
                "ChromaDB não está disponível ou o índice falhou. "
                "Para perguntas em documentos use a opção só PDFs após corrigir o ambiente, "
                "ou escolha só DuckDB se a pergunta for sobre cadastro/diário."
            )
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            history_for_model = _msgs[:-1]

            with st.chat_message("assistant"):
                plan_force: str | None = None
                if mode_ds == "structured":
                    plan_force = "sql_only"
                elif mode_ds == "documents":
                    plan_force = "rag_only"

                progress_ui = st.empty()
                with progress_ui.container():
                    with st.status("Processando sua pergunta…", expanded=True) as proc:
                        proc.write("Analisando a pergunta e planejando consultas…")
                        plan = normalize_plan(
                            llm_plan_sources(
                                user_text,
                                force=plan_force,
                                history=history_for_model,
                                extra_planner_suffix=planner_extra,
                            ),
                            user_text,
                        )
                        plan = apply_user_data_source_mode(plan, mode_ds)
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
                        if (
                            isinstance(mut_sql, str)
                            and mut_sql.strip()
                            and _conn is not None
                        ):
                            mutation_attempted = True
                            proc.write("Aplicando alteração nos dados (CSV)…")
                            _mmsg, mok, _dup_w = run_mutation_and_persist(
                                _conn, mut_sql.strip(), DATA_DIR
                            )
                            mutation_result_msg = _mmsg
                            proc.write(_mmsg)
                            if _dup_w:
                                mutation_duplicate_warn = _dup_w
                            if mok:
                                mutation_ok = True
                                mut_sql_done = mut_sql.strip()
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
                                collection, rag_question, k=RAG_TOP_K
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
                            _use_openai_compatible_chat()
                            and ROTINA_API_PLAN_TO_CHAT_DELAY_SEC > 0
                        ):
                            time.sleep(ROTINA_API_PLAN_TO_CHAT_DELAY_SEC)

                _extra_chat = (chat_extra_system or "").strip()
                if mutation_ok:
                    _extra_chat = (
                        (_extra_chat + "\n\n") if _extra_chat else ""
                    ) + SYSTEM_MUTATION_APPLIED
                    if mutation_duplicate_warn:
                        _extra_chat += "\n\n" + SYSTEM_DUPLICATE_CADASTRO
                elif mutation_fail_detail:
                    _extra_chat = (
                        (_extra_chat + "\n\n") if _extra_chat else ""
                    ) + (
                        SYSTEM_DUPLICATE_CADASTRO
                        if mutation_duplicate_warn
                        else SYSTEM_MUTATION_FAILED
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
                    _render_rag_sidebar_body(rag_sidebar_body)
                    return

                def _gen() -> Generator[str, None, None]:
                    yield from llm_chat_stream(
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

            st.session_state.messages.append({"role": "assistant", "content": full})

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
    _render_rag_sidebar_body(rag_sidebar_body)


def _render_gestao_ou_educador(
    *,
    allow_mutations: bool,
    read_only_db: bool,
    planner_extra: str,
) -> None:
    _chat_id = ensure_rotina_chat_session_id()
    sync_rotina_chat_from_disk(_chat_id)
    _rotina_chat_footer_css()
    _screen = st.session_state.get("rotina_sidebar_screen", "assistant")
    with st.sidebar:
        render_auth_sidebar()
        rag_sidebar_body = (
            _render_chat_sidebar_internals()
            if _screen == "assistant"
            else st.empty()
        )

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

    st.subheader("Alunos cadastrados")
    if conn is not None:
        try:
            _df_alunos = conn.execute(
                "SELECT id_aluno, nome, turma, alergias FROM info_alunos ORDER BY nome"
            ).fetchdf()
            st.dataframe(_df_alunos, use_container_width=True, hide_index=True)
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
        parent_scope=None,
        planner_extra=planner_extra,
        chat_extra_system=None,
        report_parent_lock=None,
    )


def render_gestao() -> None:
    _render_gestao_ou_educador(
        allow_mutations=True,
        read_only_db=False,
        planner_extra=_planner_suffix_gestao(),
    )


def render_educador() -> None:
    _render_gestao_ou_educador(
        allow_mutations=False,
        read_only_db=True,
        planner_extra=_planner_suffix_educador_readonly(),
    )


def render_familia() -> None:
    _chat_id = ensure_rotina_chat_session_id()
    sync_rotina_chat_from_disk(_chat_id)
    _rotina_chat_footer_css()
    _screen = st.session_state.get("rotina_sidebar_screen", "assistant")
    with st.sidebar:
        render_auth_sidebar()
        rag_sidebar_body = (
            _render_chat_sidebar_internals()
            if _screen == "assistant"
            else st.empty()
        )

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
        parent_scope=_ps,
        planner_extra=_planner_suffix_familia(aid, nome_filho)
        if isinstance(aid, int)
        else "",
        chat_extra_system=_chat_system_familia(aid, nome_filho)
        if isinstance(aid, int)
        else None,
        report_parent_lock=_ps,
    )


def main() -> None:
    st.set_page_config(page_title="Rotina Viva", layout="wide", initial_sidebar_state="expanded")
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
        st.error("Sessão inválida. Entre novamente.")
        render_login()


if __name__ == "__main__":
    main()
