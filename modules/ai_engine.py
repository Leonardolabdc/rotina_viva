"""
Motor de IA: Ollama, OpenAI/OpenRouter, instruções de sistema e transcrição (Whisper).
Chaves e URLs via os.getenv, alinhado ao comportamento anterior em app.py.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Generator, Iterable
from urllib.parse import urlparse, urlunparse

import httpx
from dotenv import load_dotenv

load_dotenv()


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


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")

ROTINA_CHAT_PROVIDER = os.getenv("ROTINA_CHAT_PROVIDER", "ollama").strip().lower()
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY", "").strip()
    or os.getenv("OPENROUTER_API_KEY", "").strip()
)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()


def _normalize_transcribe_base_url(raw: str) -> str:
    u = (raw or "").strip().rstrip("/")
    if not u:
        return u
    try:
        p = urlparse(u)
    except ValueError:
        return u
    if (p.hostname or "").lower() != "whisper":
        return u
    if Path("/.dockerenv").is_file():
        return u
    port = p.port or 9000
    return urlunparse(
        (
            p.scheme or "http",
            f"127.0.0.1:{port}",
            p.path or "",
            "",
            p.query,
            p.fragment,
        )
    ).rstrip("/")


OPENAI_TRANSCRIBE_BASE_URL = _normalize_transcribe_base_url(
    os.getenv("OPENAI_TRANSCRIBE_BASE_URL", "https://api.openai.com/v1")
)
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1").strip()
_stt_key_raw = os.getenv("OPENAI_TRANSCRIBE_API_KEY", "").strip()
if _stt_key_raw:
    OPENAI_TRANSCRIBE_API_KEY = _stt_key_raw
elif "api.openai.com" in OPENAI_TRANSCRIBE_BASE_URL:
    OPENAI_TRANSCRIBE_API_KEY = OPENAI_API_KEY
else:
    OPENAI_TRANSCRIBE_API_KEY = ""

ROTINA_API_HTTP_MAX_RETRIES = _env_int("ROTINA_API_HTTP_MAX_RETRIES", 10)
ROTINA_STREAM_MAX_SECONDS = _env_float("ROTINA_STREAM_MAX_SECONDS", 600.0)
ROTINA_CHAT_MAX_OUTPUT_CHARS = _env_int("ROTINA_CHAT_MAX_OUTPUT_CHARS", 4500)
ROTINA_CHAT_TEMPERATURE = _env_float("ROTINA_CHAT_TEMPERATURE", 0.35)
ROTINA_CHAT_TEMP_WITH_SQL = _env_float("ROTINA_CHAT_TEMP_WITH_SQL", 0.12)


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


def use_openai_compatible_chat() -> bool:
    return ROTINA_CHAT_PROVIDER in ("openai", "openrouter") and bool(OPENAI_API_KEY)


# --- Instruções de sistema (expostas por funções para app.py ficar legível) ---


def system_persona() -> str:
    return """Você é o assistente "Rotina Viva", da escola infantil.
- Tom empático, claro e respeitoso com pais, mães, responsáveis e professoras.
- Use apenas as informações fornecidas nos blocos de contexto (dados tabulares e trechos de documentos).
- Se trechos trouxerem nome, denominação ou título com o nome da escola (ex.: linha começando em "Escola", ou "Título: ... Escola ..."), cite isso na resposta. Não diga que o nome "não consta" se ele aparecer literalmente no contexto.
- Se algo realmente não estiver no contexto, diga com honestidade e sugira falar com a coordenação.
- Nunca invente nomes de crianças, datas ou ocorrências que não apareçam no contexto.
- Responda em português do Brasil, de forma objetiva e acolhedora."""


def system_grounding() -> str:
    return """Leia o contexto abaixo antes de responder.
- Priorize fatos que estejam escritos nos trechos ou na tabela.
- Só diga que uma informação não aparece se, depois de verificar o contexto, ela de fato não estiver lá.
- Para perguntas sobre identidade da escola, procure linhas como nome fantasia, cabeçalho, "Escola ..." ou campo "Título:" nos documentos.
- Responda ao que a **pergunta atual** pede. Não acrescente observações sobre nomes ou assuntos que só surgiram em **mensagens anteriores** do chat: o bloco de contexto desta rodada costuma estar filtrado à pergunta de agora, e a ausência de um nome nesse bloco **não** autoriza dizer "não há informações sobre [fulano]" se o utilizador **não perguntou** por essa pessoa nesta mensagem."""


def system_sql_strict() -> str:
    return """Dados tabulares (bloco "Dados tabulares" acima):
- A tabela é o resultado **literal** de uma consulta ao banco. Trate cada célula como dado real já filtrado.
- **Não invente** linhas, colunas, nomes de crianças, datas, refeições, medicamentos ou números que **não apareçam** nessa tabela.
- Se a tabela estiver vazia ou disser "(nenhuma linha retornada)", diga isso claramente — não preencha com suposições.
- Para contar, listar ou comparar, use **apenas** o que está nas linhas mostradas (e o número da coluna "linha" se existir).
- Se a pergunta pedir algo que a tabela não contém (coluna ausente), diga que o resultado atual não traz esse campo.
- Se várias linhas tiverem o mesmo nome e turmas diferentes, isso vem do cadastro (homônimos ou duplicidade): cite `id_aluno` de cada linha e não assuma um único aluno sem explicar.
- Esta tabela reflete a **pergunta atual**; não conclua pela omissão de nomes aqui que "não há dados" sobre alguém que o utilizador **não citou** nesta pergunta.
- **Resposta ao utilizador (obrigatório):** não transcreva a tabela inteira nem liste todos os alunos linha a linha — o utilizador já vê os dados na aplicação. Limite-se a **resumir** (ex.: total, ids relevantes, sim/não) em **poucas frases**; no máximo **3 exemplos** de linha se for indispensável."""


def system_mutation_applied() -> str:
    return """Operação nos dados (instalação autorizada — perfil Gestão):
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


def system_mutation_failed() -> str:
    return """Falha ao gravar alteração nos CSV desta instalação (Rotina Viva):
- Nos dados tabulares acima há uma secção **"A alteração aos dados NÃO foi gravada"** com o motivo real (permissão, ficheiro em uso, etc.).
- Responda de forma **prática**: o utilizador deve **fechar** `info_alunos.csv` e/ou `diario_estruturado.csv` se estiverem abertos no **Excel** ou noutro editor (no Windows isto bloqueia a escrita), e voltar a pedir a alteração.
- **Não** diga que não tem acesso a "sistemas de gestão" ou que só pode "fornecer informações" — nesta app a alteração é feita aqui; o problema é **técnico local** (não foi possível escrever no disco).
- Seja breve e acertivo: uma frase sobre fechar o ficheiro + repetir o pedido."""


def system_duplicate_cadastro() -> str:
    return """Duplicados no cadastro (info_alunos):
- Se existir secção **"Aviso de duplicado"** nos dados tabulares, o servidor detetou **nome ou contacto** já presentes no CSV **antes de gravar**.
- Regra desta instalação: em caso de duplicado, a mutação é **bloqueada** e nada é persistido no CSV.
- Explique com clareza: ajuste os dados ou confirme outro identificador/contato para gravar.
- Não omita o aviso; seja breve."""


def schema_duckdb_for_llm() -> str:
    return """
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
   - **INSERT de novo aluno (campo "mutacao" do planejador):** liste sempre as colunas e **inclua id_aluno em primeiro**, por exemplo:
     `INSERT INTO info_alunos (id_aluno, nome, turma, alergias, contato_pais) VALUES ((SELECT COALESCE(MAX(id_aluno), 0) + 1 FROM info_alunos), 'Maria Silva', 'Infantil 1', 'Nenhuma', '41999999999')`.
     Sem id_aluno na lista de colunas a gravação é rejeitada.

2) diario_estruturado
   **Diário ≠ novo aluno:** pedidos do tipo “registar / anotar / lançar no **diário**”, “refeição do dia”, “sono hoje”
   → **INSERT ou UPDATE em `diario_estruturado`** com o `id_aluno` **já existente** (use `SELECT id_aluno FROM info_alunos WHERE nome ILIKE …` só para obter o id).
   **Não** faça `INSERT INTO info_alunos` para anotar refeições ou rotina diária — isso é cadastro de criança nova.
   **Data:** se o utilizador **não** indicar dia, use **`CURRENT_DATE`** (ou `CAST(CURRENT_DATE AS VARCHAR)`) na coluna `data`. Formato preferido na mutação: `'YYYY-MM-DD'` (ex.: `'2026-04-17'`).
   **Horas de sono:** se não forem mencionadas, use **`''`** (string vazia) em `hora_sono_inicio` e `hora_sono_fim` — não invente horários.
   Colunas: id_registro (INTEGER), id_aluno (INTEGER), data (TEXT — use `'AAAA-MM-DD'` ou expressão de data),
   cafe_manha, almoco, lanche_tarde, jantar_extra (TEXT),
   trocas_banheiro (INTEGER), evacuacao (TEXT), medicamentos (TEXT),
   hora_sono_inicio, hora_sono_fim (TEXT — vazio `''` se não houver horário de sono),
   qualidade_sono (TEXT),
   atividade_dia (TEXT), interacao_social (TEXT), recado_professora (TEXT)
   **qualidade_sono:** use **exatamente** um dos três textos canónicos já usados no CSV (nunca “Dormiu bem” solto):
   `Dormiu bastante`, `Dormiu pouco`, e o terceiro no formato **`Dormiu normal (…min)`** com os mesmos números que aparecem nas linhas recentes desse aluno (faixas vêm do `.env` do servidor).
   Se o utilizador disser “dormiu bem / dormiu muito” → **`Dormiu bastante`**. “Dormiu mal / pouco” → **`Dormiu pouco`**. “Dormiu normal” → copie o literal **`Dormiu normal (…)`** de `SELECT qualidade_sono FROM diario_estruturado WHERE id_aluno = … ORDER BY id_registro DESC LIMIT 1` se existir; senão use o padrão descrito nas instruções de sistema.
   **Mapeamento — não confunda colunas (erro comum da IA):**
   - **cafe_manha, almoco, lanche_tarde, jantar_extra:** só estes valores canónicos (como no CSV): **`Comeu bem`**, **`Comeu pouco`**, **`Recusou`**. **Proibido** gravar só `Bem`, `Mal`, `OK`.
     Se o utilizador disser “comeu bem / alimentou-se bem” **sem** dizer qual refeição: preencha **só `almoco`** com `Comeu bem` e deixe **`cafe_manha`, `lanche_tarde`, `jantar_extra`** como **`''`**. Só preencha várias colunas se ele citar explicitamente café, almoço, lanche ou jantar.
   - **atividade_dia:** atividade pedagógica do dia (ex.: “Brincadeiras livres”, “Pintura”, “Musicalização”). **Não** coloque aqui “ativo”, “quieto”, “interagiu” — isso não é nome de atividade.
   - **interacao_social:** como foi com colegas / disposição social (ex.: “Interagiu bem com colegas”, “Preferiu brincar sozinho”, “Muito ativo nas atividades”, “Participou bem das atividades”). Frases tipo **“ficou ativo”** vão **aqui**, não em `atividade_dia`.
   - **recado_professora:** observação livre da educadora (opcional); pode resumir o dia numa frase curta se couber.
   - **trocas_banheiro:** número inteiro (ex.: 2) ou **0** se não mencionado; **evacuacao:** ex.: `Normal`, `Pastoso`, `Diarreia` ou **`''`** se não disseram.
   - **medicamentos:** texto ou **`''`**.
   Antes de montar o INSERT, pode usar um SELECT rápido (`SELECT cafe_manha, qualidade_sono FROM diario_estruturado WHERE id_aluno = … LIMIT 3`) para copiar o **estilo** dos valores já gravados.

Para juntar aluno e diário (ex.: refeições de um aluno por nome):
   `FROM diario_estruturado d JOIN info_alunos a ON d.id_aluno = a.id_aluno WHERE a.nome ILIKE '%...%'`
"""


_ASKS_SCHOOL_NAME = re.compile(
    r"(qual\s+[ée]\s+o\s+nome\s+da\s+escola|nome\s+da\s+escola|como\s+se\s+chama\s+a\s+escola)",
    re.IGNORECASE,
)


def extract_school_name_hints_from_rag(rag_block: str) -> list[str]:
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


def duck_block_has_tabular_rows(duck_block: str) -> bool:
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


def _format_history_for_planner(
    history: Iterable[dict[str, str]], max_messages: int = 8
) -> str:
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


def build_routing_planner_prompt(
    user_message: str,
    force: str | None = None,
    history: Iterable[dict[str, str]] | None = None,
    extra_suffix: str = "",
) -> str:
    sch = schema_duckdb_for_llm()
    core = f"""Classifique a pergunta sobre uma escola infantil.

{sch}

Regras (siga com cuidado):
- "rag": documentos oficiais — regimento, normas, PPP, proposta pedagógica, segurança/saúde **institucional** (protocolos gerais da escola), cardápio/nutrição em documento, horários gerais da escola, **nome da escola / identidade institucional / endereço / missão** quando estiver em texto oficial.
- "sql": **cadastro ou diário** — turma de aluno, **alergias cadastradas por aluno** (`info_alunos.alergias`: não use só RAG por ser “saúde”), criança específica, refeições do dia no diário, sono, evacuação, medicamentos **anotados no diário**, recado da professora, listagens/contagens nas tabelas CSV.
- **Continuação de conversa:** se a pergunta atual usar só pronomes (“ele”, “ela”, “meu filho”) ou não repetir o nome, use a **conversa recente** para saber **qual criança** e monte o SQL com `WHERE nome ILIKE '%...%'` em `info_alunos` (ou `id_aluno` se tiver sido citado). Não deixe de filtrar pelo aluno quando a pergunta for claramente sobre a mesma criança do turno anterior.
- **Não** use "sql" para "qual é o nome da escola?" — isso é "rag" (documentos).
- Use ambos ["rag","sql"] só se a pergunta claramente precisar de documento oficial **e** de linhas do diário/cadastro.
- **Criar, apagar ou alterar aluno / cadastro (`info_alunos`) ou linha de diário (`diario_estruturado`):** use **apenas** `["sql"]` em "fontes" — **não** inclua "rag" (PDFs não são a fonte de mutações nos CSV).
- Tabelas e quadros em PDF costumam estar em "rag": peça termos que apareçam no documento (títulos de coluna, faixas etárias, nomes de refeição) — não use "sql" só porque a resposta parece uma tabela.
- Quando houver "sql" com SELECT válido, a resposta final ao usuário deve **copiar fielmente** as células retornadas — sem inventar linhas ou valores.

Responda somente JSON válido com **três chaves** (use `null` quando não aplicável):
{{"fontes": ["rag"] ou ["sql"] ou ["rag","sql"] ou ["sql","rag"], "sql": null ou uma string com UMA consulta SELECT DuckDB, "mutacao": null ou UMA string SQL (INSERT/UPDATE/DELETE) apenas nas tabelas `info_alunos` e `diario_estruturado`}}

- **"mutacao"** só preencha quando o utilizador pedir **criar, alterar ou apagar** dados nessas tabelas; caso contrário `"mutacao": null`. Regras de **quem pode DELETE** ou só INSERT/UPDATE vêm das instruções de perfil/RBAC abaixo (se existirem) — respeite-as sempre.
- Se "sql" estiver em fontes e precisar de dados tabulares, "sql" deve ser a string SELECT. Se não souber a consulta, **não** inclua "sql" em fontes — use só "rag".
"""
    if force == "sql_only":
        core += """
MODO **somente dados estruturados** (sidebar: só DuckDB / CSV — sem PDFs nesta rodada):
- **Se o utilizador pedir gravar dados** (cadastrar aluno, inserir/atualizar diário, alterar turma, etc.): `"fontes": ["sql"]` e preencha **"mutacao"** com **um** INSERT ou UPDATE conforme o esquema. Acompanhe com um `"sql"` SELECT útil quando fizer falta; `"sql"` pode ser `null` só se a mutação bastar. **Não** devolva `"mutacao": null` só porque este modo chama-se “só dados” — esse modo significa sem RAG, **não** proíbe escrita.
- **Se for só consulta** (listar, contar, turma de X, alergias…): `{"fontes": ["sql"], "sql": "SELECT ...", "mutacao": null}`.
- Perfis / DELETE: siga o bloco **RBAC** colocado **depois** deste modo (última palavra sobre o que pode ir em `"mutacao"`).
- Perguntas do tipo “qual é a turma do [nome]?” → **obrigatoriamente** filtre `info_alunos` por `nome` (ex.: `WHERE nome ILIKE '%primeiro%ultimo%'`).
  **Não** use só `SELECT DISTINCT turma` ou listar turmas sem JOIN/WHERE no nome — isso não identifica o aluno.
- Perguntas sobre **alergia / intolerância / restrição alimentar de um aluno** (ou “quem tem alergia a X”) → `SELECT nome, turma, alergias FROM info_alunos` com `WHERE` em `nome` e/ou `alergias` (coluna **`alergias`**). Não devolva `sql: null` só porque a palavra parece “saúde”.
- Se a pergunta **não puder** ser respondida com essas tabelas (ex.: regulamento, PPP, texto de PDF), retorne {"fontes": [], "sql": null, "mutacao": null}. **Não** use "rag".
"""
    elif force == "rag_only":
        core += """
MODO **somente documentos** (sidebar: prioridade ao ChromaDB / PDFs indexados):
- Para **perguntas só de leitura** em documentos: {"fontes": ["rag"], "sql": null, "mutacao": null}.
- Se o utilizador pedir **criar, alterar ou apagar** dados em `info_alunos` ou `diario_estruturado`,
  isso **não** vem de PDF: use {"fontes": ["sql"], "sql": null ou um SELECT útil, "mutacao": "INSERT/UPDATE/DELETE…"}
  conforme o esquema CSV e o RBAC abaixo. **Não** devolva "mutacao": null só por estar neste modo.
"""
    if extra_suffix.strip():
        core += (
            "\n---\n**RBAC / instruções de perfil (têm prioridade sobre tudo o que precede):**\n"
            + extra_suffix.strip()
            + "\n"
        )
    hist = _format_history_for_planner(history or [])
    return core + "\n" + hist + f"Pergunta atual:\n{user_message}\n"


def _httpx_retry_after_seconds(resp: httpx.Response, attempt: int) -> float:
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return min(300.0, float(ra))
        except ValueError:
            pass
    return min(120.0, 6.0 * (2**attempt) + random.uniform(0, 2))


def _format_api_error_body(text: str, max_len: int = 1500) -> str:
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


def ollama_plan_sources(
    user_message: str,
    force: str | None = None,
    history: Iterable[dict[str, str]] | None = None,
    extra_planner_suffix: str = "",
) -> dict[str, Any]:
    planner = build_routing_planner_prompt(
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
        {"role": "system", "content": system_persona()},
        {"role": "system", "content": system_grounding()},
    ]
    if extra_system and extra_system.strip():
        messages.append({"role": "system", "content": extra_system.strip()})
    if duck_block_has_tabular_rows(duck_block):
        messages.append({"role": "system", "content": system_sql_strict()})
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


def openai_plan_sources(
    user_message: str,
    force: str | None = None,
    history: Iterable[dict[str, str]] | None = None,
    extra_planner_suffix: str = "",
) -> dict[str, Any]:
    planner = build_routing_planner_prompt(
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
        {"role": "system", "content": system_persona()},
        {"role": "system", "content": system_grounding()},
    ]
    if extra_system and extra_system.strip():
        messages.append({"role": "system", "content": extra_system.strip()})
    if duck_block_has_tabular_rows(duck_block):
        messages.append({"role": "system", "content": system_sql_strict()})
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
    if use_openai_compatible_chat():
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


def llm_chat_stream(
    user_message: str,
    duck_block: str,
    rag_block: str,
    history: Iterable[dict[str, str]],
    extra_system: str | None = None,
) -> Generator[str, None, None]:
    if use_openai_compatible_chat():
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


def whisper_upload_name_and_mime(audio_bytes: bytes, reported_name: str) -> tuple[str, str]:
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
    return f"{base}.webm", "audio/webm"


def rotina_st_audio_format(audio_bytes: bytes) -> str:
    """MIME/formato para `st.audio` a partir dos bytes (mesma heurística que o Whisper)."""
    _, mime = whisper_upload_name_and_mime(audio_bytes, "gravacao.wav")
    return {
        "audio/wav": "audio/wav",
        "audio/webm": "audio/webm",
        "audio/ogg": "audio/ogg",
        "audio/mpeg": "audio/mp3",
    }.get(mime, "audio/wav")


def _transcribe_whisper_http(audio_bytes: bytes, filename: str) -> tuple[str | None, str | None]:
    if not audio_bytes:
        return None, "Nenhum dado de áudio para enviar ao Whisper."
    base = (OPENAI_TRANSCRIBE_BASE_URL or "").strip().rstrip("/")
    if not base:
        return None, "URL de transcrição não configurada (`OPENAI_TRANSCRIBE_BASE_URL`)."
    key = (OPENAI_TRANSCRIBE_API_KEY or "").strip()
    upload_name, mime = whisper_upload_name_and_mime(audio_bytes, filename)
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


def processar_resposta_chat_stream(
    user_message: str,
    duck_block: str,
    rag_block: str,
    history: Iterable[dict[str, str]],
    extra_system: str | None = None,
) -> Generator[str, None, None]:
    """Alias legível: streaming da resposta do assistente (Ollama ou API OpenAI-compatível)."""
    yield from llm_chat_stream(user_message, duck_block, rag_block, history, extra_system)
