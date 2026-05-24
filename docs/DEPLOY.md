# Deploy no Streamlit Community Cloud (Fase 0 — com agentes CrewAI)

Guia para publicar o **Rotina Viva** em [share.streamlit.io](https://share.streamlit.io) com chat, RAG e **multi-agente (CrewAI)** activo.

## O que já está preparado no repositório

| Ficheiro | Função |
|----------|--------|
| `requirements.txt` | Aponta para `requirements-streamlit-cloud.txt` (prod + CrewAI) |
| `packages.toml` | Python 3.12 |
| `.streamlit/config.toml` | Tema e opções do servidor |
| `.streamlit/secrets.toml.example` | Modelo de secrets para colar no painel |
| `src/core/cloud_bootstrap.py` | Secrets → env, utilizadores demo, defaults CrewAI |
| `scripts/build_rag_index.py` | Pré-indexar PDFs antes do deploy |

## Passo a passo (≈30 min)

### 1. Pré-indexar RAG (recomendado)

No PC, com `.env` e `OPENROUTER_API_KEY` válida:

```bash
python scripts/build_rag_index.py
git add -f data/vector_db/
git commit -m "chore: índice RAG para Streamlit Cloud"
```

Sem isto, o **primeiro login** indexa os PDFs na cloud (lento + tokens).

Windows:

```powershell
.\scripts\prepare-streamlit-cloud.ps1
```

### 2. Push para o GitHub

```bash
git push origin main
```

### 3. Criar a app no Streamlit Cloud

1. Aceda a [share.streamlit.io](https://share.streamlit.io) e ligue o GitHub.
2. **New app** → repositório `Leonardolabdc/Rotina-Viva`.
3. **Main file path:** `app.py`
4. **Branch:** `main`
5. **Advanced settings → Python:** 3.12 (lido de `packages.toml`).

### 4. Secrets (Settings → Secrets)

Copie de [`.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example) e substitua a chave OpenRouter:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_CHAT_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENAI_EMBED_MODEL = "openai/text-embedding-3-small"
ROTINA_CHAT_PROVIDER = "openrouter"
ROTINA_EMBED_PROVIDER = "openrouter"
OPENROUTER_HTTP_REFERER = "https://github.com/Leonardolabdc/Rotina-Viva"
OPENROUTER_APP_TITLE = "Rotina Viva"
ROTINA_ENABLE_CREWAI = "true"
CREWAI_DISABLE_TELEMETRY = "true"
ROTINA_ENABLE_ML_LAB = "false"
ROTINA_LANGFUSE_ENABLED = "false"
```

### 5. Deploy e teste

1. Aguarde o build (CrewAI + Chroma demoram mais que prod slim).
2. Login: `gestao.demo` / `demo123`
3. Na **sidebar**, active **Modo CrewAI (multi-agente)**.
4. Envie uma pergunta sobre regimento ou rotina.

## Utilizadores demo

| Utilizador | Senha | Perfil |
|------------|-------|--------|
| `gestao.demo` | `demo123` | Gestão |
| `professor.demo` | `demo123` | Educador |
| `pai.demo` | `demo123` | Família |

Criados automaticamente pelo `cloud_bootstrap` se `data/rotina_users.json` não existir.

## Agentes CrewAI

- Dependências incluídas em `requirements-streamlit-cloud.txt`.
- `ROTINA_ENABLE_CREWAI=true` nos secrets (default no bootstrap).
- Toggle na UI: sidebar → **Modo CrewAI**.
- Requer `ROTINA_CHAT_PROVIDER=openrouter` (ou openai) + chave API.

## Limitações do tier gratuito

| Item | Nota |
|------|------|
| **RAM** | CrewAI + Chroma + LiteLLM — se a app reiniciar por OOM, reduza chunks RAG ou desligue ML lab (`ROTINA_ENABLE_ML_LAB=false`, já default). |
| **Disco** | Efémero — CSV editados e Chroma podem resetar após redeploy/reboot. |
| **Whisper / voz** | Não há container `whisper` no Cloud. Configure `OPENAI_TRANSCRIBE_BASE_URL` para Oracle VM ou omita (chat por texto funciona). |
| **Cold start** | 30–90 s após inactividade. |

## Voz (opcional — Oracle VM)

Quando tiver Whisper na VM:

```toml
OPENAI_TRANSCRIBE_BASE_URL = "https://SEU-IP-OU-DOMINIO/v1"
OPENAI_TRANSCRIBE_MODEL = "whisper-1"
```

## Desenvolvimento local vs Cloud

| Ambiente | Requirements |
|----------|----------------|
| **Streamlit Cloud** | `requirements.txt` → `requirements-streamlit-cloud.txt` |
| **Docker dev** | `requirements-dev.txt` (inclui FLAML, DeepEval) |
| **Local sem Docker** | `pip install -r requirements-dev.txt` |

## Checklist antes de apresentar na faculdade

- [ ] Secrets OpenRouter configurados no painel
- [ ] Login demo funciona
- [ ] Chat + RAG responde sobre PDFs
- [ ] Toggle CrewAI activo e resposta multi-agente
- [ ] URL pública copiada para o relatório/README

## Alternativas

Ver secção Docker/VPS no topo deste ficheiro e [ARQUITETURA.md](ARQUITETURA.md).
