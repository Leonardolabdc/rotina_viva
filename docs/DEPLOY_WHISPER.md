# Voz (STT) — Streamlit Cloud e produção

## Opção A — OpenRouter Whisper (**recomendado, sem VM**)

Usa a **mesma** `OPENROUTER_API_KEY` do chat. O servidor Streamlit envia áudio em base64 para:

`POST https://openrouter.ai/api/v1/audio/transcriptions`

Custo aproximado: **~$0,006/min** (`openai/whisper-1`). Ver [OpenRouter STT](https://openrouter.ai/docs/guides/overview/multimodal/stt).

### Secrets (Streamlit Cloud)

Se já tens `OPENROUTER_API_KEY`, o `cloud_bootstrap` preenche STT automaticamente. Para explicitar:

```toml
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENAI_TRANSCRIBE_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_TRANSCRIBE_MODEL = "openai/whisper-1"
```

**Reboot app** → login educador/gestão → grave áudio no chat.

Legenda esperada: *Voz: **OpenRouter Whisper STT***.

### Local (.env)

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_TRANSCRIBE_BASE_URL=https://openrouter.ai/api/v1
OPENAI_TRANSCRIBE_MODEL=openai/whisper-1
```

### Docker local

Se `.env` apontar para `http://whisper:9000/v1`, o Docker continua a usar Whisper local. Para testar OpenRouter no PC, sobrescreve com a URL OpenRouter acima.

---

## Opção B — Whisper self-hosted (VM / Docker)

Para **grátis** com servidor próprio (Oracle, Render, PC + túnel):

```
Browser → Streamlit Cloud → HTTP multipart → Whisper :9000
```

Ver secção **VM** abaixo.

---

## Fluxo técnico

```
Browser (st.audio_input) → bytes → Python (Streamlit) → STT API
```

**Sem CORS no browser** — a chamada HTTP parte do **servidor** Streamlit.

| Backend | URL | Protocolo |
|---------|-----|-----------|
| **OpenRouter** | `https://openrouter.ai/api/v1` | JSON + base64 |
| Whisper Docker | `http://host:9000/v1` | multipart (OpenAI-compatível) |
| OpenAI API | `https://api.openai.com/v1` | multipart |

Código: `src/modules/transcribe_service.py` → `ai_engine.transcribe_voice_bytes()`.

---

## Arquitetura (Fase 0 → Fase 1)

```
Fase 0   Streamlit Cloud ──► OpenRouter STT  (ou Whisper VM)
Fase 1   FastAPI / Next.js ──► mesma URL / proxy
```

---

## VM Whisper (Opção B — Oracle ou outro VPS)

### 1. VM Ubuntu + Docker

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER
git clone https://github.com/Leonardolabdc/rotina_viva.git
cd rotina_viva
docker compose -f docker-compose.whisper.yml up -d
docker compose -f docker-compose.whisper.yml logs -f whisper
```

### 2. Firewall — porta **9000** (ingress TCP 0.0.0.0/0 na demo)

### 3. Teste do PC

```powershell
python scripts\test_whisper_url.py http://SEU-IP:9000/v1
```

### 4. Secrets

```toml
OPENAI_TRANSCRIBE_BASE_URL = "http://SEU-IP:9000/v1"
OPENAI_TRANSCRIBE_MODEL = "whisper-1"
```

---

## Resolução de problemas

| Sintoma | Solução |
|---------|---------|
| "OpenRouter STT: defina OPENROUTER_API_KEY" | Adicionar chave nos Secrets |
| HTTP 402 OpenRouter | Créditos insuficientes na conta OpenRouter |
| "Transcrição não configurada" | Definir `OPENAI_TRANSCRIBE_BASE_URL` |
| Funciona local, falha Cloud | URL `127.0.0.1` ou `whisper:` inválida na cloud |
| Transcrição vazia | Falar mais alto; gravar ≥ 2 s |

---

## Segurança

- **OpenRouter:** chave só nos Secrets; áudio enviado ao provedor (política OpenRouter/OpenAI).
- **VM Whisper:** restringir firewall ou Bearer token em produção.

Ver [DEPLOY.md](DEPLOY.md).
