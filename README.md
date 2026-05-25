# 🤖 Rotina Viva 🌿

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/Leonardolabdc/Rotina-Viva)

> **Status do Projeto:** 🛠️ Em desenvolvimento (Etapa 1 - CBL)

O **Rotina Viva** é um assistente inteligente projetado para ser a ponte digital entre a escola e os pais. Ele utiliza Inteligência Artificial para transformar a montanha de dados burocráticos de uma escola infantil em informações úteis e acessíveis.

Ideal para **escolas infantis**, **educadores** e **famílias** que precisam de **comunicação clara, registro ágil da rotina escolar e transparência** no acompanhamento das crianças.

## 📸 Demo

[![Assista ao vídeo de apresentação](assets/demo_rotina_viva.gif)](https://youtu.be/3QAkjsPqgK4)

*Clique no GIF acima para ver a apresentação completa com áudio no YouTube.*

🔗 [Acesse a aplicação](http://localhost:8501) (após subir o ambiente — veja [Como Executar o Projeto](#-como-executar-o-projeto))

---

## Metodologia CBL (Challenge-Based Learning)

Este projeto foi estruturado seguindo os pilares do aprendizado baseado em desafios:

### Grande Ideia

**Comunicação escolar e acompanhamento do desenvolvimento na educação infantil.** A base do projeto é fortalecer o vínculo entre a instituição de ensino e os responsáveis, garantindo que o desenvolvimento da criança seja acompanhado de perto e com clareza.

### Pergunta Essencial

> Como a IA pode otimizar o registro da rotina escolar e melhorar a transparência para os pais, garantindo que os educadores tenham mais tempo de qualidade para se dedicar ao desenvolvimento dos alunos?

### O Desafio

**Desafio: Desenvolver o Rotina Viva**, um assistente inteligente robusto que automatiza o registro diário de alimentação, sono e higiene, além de atuar como consultor pedagógico instantâneo para sanar dúvidas sobre o regimento e diretrizes da escola. O sistema elimina gargalos de comunicação manual ao permitir que pais e educadores interajam de forma natural e acessível, garantindo agilidade no preenchimento de dados e humanizando o acompanhamento do desenvolvimento infantil.

---

##   Justificativa Pessoal


> A partir de conversas com minha esposa, observamos as limitações das agendas de papel tradicionais e a excessiva carga de trabalho manual imposta aos educadores. Acredito que uma agenda virtual, orientada por uma Inteligência Artificial bem estruturada e robusta, pode devolver o tempo para o que realmente importa: o cuidado e a educação das crianças, além de elevar significativamente a transparência das informações para os pais.

---

## 🚀 Instalação

### Pré-requisitos

* [Docker](https://www.docker.com/) instalado.
* [Git](https://git-scm.com/) instalado.
* Conta e chave no [OpenRouter](https://openrouter.ai/) (para chat e embeddings).

### Passo a passo (Docker — recomendado)

1. **Clone o repositório:**

```bash
git clone https://github.com/Leonardolabdc/Rotina-Viva.git
cd Rotina-Viva
```

2. **Configure o `.env`** (copie de `.env.example` e preencha a chave — veja [Configuração](#%EF%B8%8F-configuração)).

3. **Suba o ambiente com Docker:**

```bash
docker compose up --build -d
```

Isso sobe dois serviços:
- **`rotina-viva`** — app Streamlit na porta **8501**
- **`rotina-whisper`** — transcrição de voz (API compatível com OpenAI na porta **9000**)

4. **Aguarde o Whisper carregar** (primeira execução pode demorar alguns minutos):

```bash
docker logs rotina-whisper
```

5. **Acesse** [http://localhost:8501](http://localhost:8501).

### Instalação local (sem Docker)

```bash
git clone https://github.com/Leonardolabdc/Rotina-Viva.git
cd Rotina-Viva

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements-dev.txt
cp .env.example .env            # Linux/macOS — no Windows: copy .env.example .env
```

No `.env` local, se quiser voz, use `OPENAI_TRANSCRIBE_BASE_URL=http://127.0.0.1:9000/v1` com o Whisper a correr (via Docker ou serviço local).

---

## ⚙️ Configuração

Crie um arquivo `.env` na raiz do projeto:

```env
ROTINA_CHAT_PROVIDER=openrouter
ROTINA_EMBED_PROVIDER=openrouter

OPENROUTER_API_KEY=sk-or-v1-EXEMPLO_SUBSTITUA_PELA_SUA_CHAVE
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_CHAT_MODEL=meta-llama/llama-3.3-70b-instruct
OPENAI_EMBED_MODEL=openai/text-embedding-3-small

OPENROUTER_HTTP_REFERER=https://github.com/Leonardolabdc/Rotina-Viva
OPENROUTER_APP_TITLE=Rotina Viva

# Transcrição de voz (dentro do Docker Compose)
OPENAI_TRANSCRIBE_BASE_URL=http://whisper:9000/v1
OPENAI_TRANSCRIBE_MODEL=whisper-1
```

Para login de demonstração, copie o ficheiro de utilizadores:

```bash
copy data\rotina_users.example.json data\rotina_users.json   # Windows
# cp data/rotina_users.example.json data/rotina_users.json   # Linux/macOS
```

---

## 📖 Como Executar o Projeto

### Com Docker (forma principal)

Na raiz do projeto, com o `.env` já configurado:

```bash
docker compose up --build -d
```

Comandos úteis:

```bash
docker compose ps              # ver se os contentores estão a correr
docker compose logs rotina-viva   # logs da app
docker compose down            # parar tudo
```

Abra [http://localhost:8501](http://localhost:8501) no navegador.

### Sem Docker (desenvolvimento)

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

### Utilizadores de demonstração

Após copiar `data/rotina_users.example.json` para `data/rotina_users.json`, use:

| Utilizador | Senha | Perfil |
|------------|-------|--------|
| `gestao.demo` | `demo123` | Gestão (CRUD nos dados) |
| `professor.demo` | `demo123` | Educador |
| `pai.demo` | `demo123` | Família (consulta + RAG) |

### O que fazer na app

1. **Faça login** com um dos perfis acima.
2. **Chat** — pergunte sobre regimento, alunos, rotina, saúde ou documentos da escola (RAG em PDFs).
3. **Educador / Gestão** — registe sono, refeições e higiene via texto ou voz.
4. **Família** — consulte informações do aluno vinculado ao perfil.
5. **Sidebar** — opções como CrewAI (multi-agente) e ML de emoções, conforme o perfil.

### Deploy na nuvem (Streamlit Community Cloud)

Para entregar o trabalho com URL pública **e agentes CrewAI**, siga [docs/DEPLOY.md](docs/DEPLOY.md):

1. Push do repositório para o GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → `app.py`.
3. Cole os **Secrets** de [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example).
4. (Recomendado) Pré-indexe o RAG: `python scripts/build_rag_index.py` e `git add -f data/vector_db/`.

---

* **Linguagem:** Python 3.12
* **Interface:** Streamlit
* **Orquestração de IA:** OpenRouter (CrewAI / LiteLLM)
* **Banco de Dados Vetorial:** ChromaDB (busca semântica e embeddings em PDFs)
* **Banco de Dados Relacional:** DuckDB (análise de dados de rotina)
* **Infraestrutura:** Docker & Docker Compose
* **Versionamento:** Git & GitHub

### Modelos de IA em uso

| Função | Modelo |
|--------|--------|
| Chat (OpenRouter) | `meta-llama/llama-3.3-70b-instruct` |
| Embeddings RAG (OpenRouter) | `openai/text-embedding-3-small` |
| Transcrição de voz (Docker) | `whisper-1` (faster-whisper **base**, PT) |

---

## Arquitetura e Fluxo de Dados

### Diagrama de arquitetura

```mermaid
flowchart TB
    subgraph usuarios["Utilizadores (RBAC)"]
        G["Gestão<br/>CRUD completo"]
        E["Educador<br/>registro + chat"]
        F["Família<br/>consulta + RAG"]
    end

    subgraph cliente["Cliente"]
        NAV["Navegador<br/>localhost:8501"]
    end

    subgraph docker["Docker Compose"]
        APP["rotina-viva<br/>Streamlit · porta 8501"]
        WH["rotina-whisper<br/>faster-whisper PT · porta 9000"]
    end

    subgraph app["Camadas da aplicação (src/)"]
        UI["ui/<br/>components · styles"]
        CORE["core/<br/>auth · database · security"]
        MOD["modules/<br/>chat · ai_engine · RAG · CrewAI · ML"]
    end

    subgraph dados["Persistência (data/)"]
        DUCK["DuckDB<br/>CSVs de rotina"]
        CHROMA["ChromaDB<br/>vector_db/"]
        DOCS["PDFs + JSON<br/>regimento · usuários"]
    end

    subgraph apis["Serviços externos"]
        OR["OpenRouter<br/>chat + embeddings"]
        LF["Langfuse<br/>observabilidade (opcional)"]
    end

    G & E & F --> NAV
    NAV --> APP
    APP --> UI
    UI --> CORE
    UI --> MOD
    CORE --> MOD
    MOD --> DUCK
    MOD --> CHROMA
    MOD --> DOCS
    CORE --> DOCS
    E -->|"áudio (voz)"| MOD
    MOD -->|"transcrição"| WH
    MOD --> OR
    MOD -.-> LF
    CHROMA --> DOCS
```

| Camada | Responsabilidade |
|--------|------------------|
| **Interface** | `app.py` orquestra login e telas por perfil (`gestao`, `educador`, `familia`). |
| **Core** | Autenticação (`auth_manager`), SQL validado e DuckDB (`database`), guardrails e PII (`security`). |
| **Módulos** | Chat com plano SQL/RAG (`chat_service`, `ai_engine`), índice vetorial (`rag_index`), multi-agente (`rotina_crew`), emoções ML local (`ml_emotion_chat`), voz (`transcribe_service`). |
| **Dados** | Rotina estruturada em CSV via DuckDB; documentos da escola em PDF indexados no ChromaDB. |
| **Infra** | Dois contentores: app Streamlit e API Whisper compatível com OpenAI. |

### Fluxo de uma mensagem no chat

```mermaid
sequenceDiagram
    actor U as Utilizador
    participant UI as Streamlit UI
    participant SEC as Segurança
    participant PL as Plano SQL/RAG/ML
    participant DB as DuckDB
    participant RAG as ChromaDB
    participant LLM as OpenRouter
    participant CR as CrewAI (opcional)

    U->>UI: texto ou voz
    opt voz
        UI->>UI: Whisper (Docker)
    end
    UI->>SEC: validação + anonimização PII
    SEC->>PL: intenção e âmbito (RBAC)
    PL->>DB: SELECT / mutações (educador/gestão)
    PL->>RAG: busca semântica (PDFs)
    alt CrewAI ativo
        PL->>CR: contexto agregado
        CR->>LLM: especialistas + redação
    else Chat direto
        PL->>LLM: prompt com grounding
    end
    LLM-->>UI: resposta em PT-BR
    UI-->>U: markdown + gráficos (Altair)
```

### Fluxo macro (visão geral)

![Fluxo Macro do Rotina Viva](assets/rotina_viva_fluxo_macro.png)

**Destaque Técnico:**
O Rotina Viva utiliza uma arquitetura baseada em perfis de acesso (RBAC - Role-Based Access Control). Educadores possuem privilégios de escrita (CRUD) via voz ou texto para alimentar o DuckDB, enquanto os pais possuem acesso restrito apenas para consulta e interação via RAG no ChromaDB. Toda comunicação é mediada por uma camada de segurança que inclui Guardrails e Anonimização de dados sensíveis (PII).

Mais detalhes: [docs/ARQUITETURA.md](docs/ARQUITETURA.md) · [docs/CBL.md](docs/CBL.md) · [docs/SYSTEM_PROMPT.md](docs/SYSTEM_PROMPT.md)

O assistente usa prompts de persona, grounding e SQL (tom empático, respostas só com base no contexto RAG + DuckDB). Texto completo em [`docs/SYSTEM_PROMPT.md`](docs/SYSTEM_PROMPT.md).

---

## Estrutura do Projeto

O projeto segue uma arquitetura modular para garantir escalabilidade e facilitar a manutenção, separando a interface da lógica de negócio e persistência:

```
Rotina-Viva/
├── app.py                 # Ponto de entrada
├── README.md
├── requirements.txt
├── .env                   # Segredos (não commitado)
├── .env.example           # Template de segredos
├── .gitignore
├── LICENSE
│
├── src/                   # Código-fonte
│   ├── __init__.py
│   ├── core/              # Auth, DuckDB, segurança
│   ├── modules/           # IA, RAG, CrewAI, ML emoções
│   └── ui/                # Interface Streamlit
│
├── tests/                 # Testes automatizados
│   └── fixtures/          # golden_dataset.json
│
├── data/                  # CSVs, PDFs, persistência
├── docs/                  # Documentação extra
├── assets/                # Demo e diagramas
├── scripts/               # Utilitários (testes Docker)
└── docker/                # API Whisper
```

**Detalhamento dos módulos:**

`app.py`: Ponto de entrada (orquestrador) da aplicação.

`src/core/`: Módulos fundamentais do sistema.

`auth_manager.py`: Gerenciamento de sessões, login e níveis de acesso (RBAC).

`database.py`: Camada de persistência e integração com DuckDB e sistemas de arquivos.

`src/modules/`: Lógica de negócio e inteligência.

`ai_engine.py`: Motor de IA (OpenRouter/OpenAI/Ollama) e processamento de áudio (Whisper).

`services.py`: Regras de negócio para Chat Direto e Diário de Classe.

`src/ui/`: Interface do usuário e componentes.

`components.py`: Componentes visuais reutilizáveis e telas por perfil de usuário.

`styles.py`: Definições de CSS customizado e identidade visual futurista.

`data/`: Repositório de arquivos de entrada e persistência local (CSVs, JSONs).

`Dockerfile` & `docker-compose.yml`: Configuração de infraestrutura para ambiente containerizado e replicável.

---

## Testes

### Testes de segurança (Docker)

Os scripts `tests/test_security_rotina.py` e `scripts/demo_security_before_after.py` **não correm no PC anfitrião** — entram na imagem e executam-se **dentro** do contentor `rotina-viva`.

Depois de alterar o código ou o `Dockerfile`, reconstrua:

```powershell
docker compose build rotina-viva
docker compose up -d
```

**PowerShell (Windows), na raiz do projeto:**

```powershell
.\scripts\run-security-tests-docker.ps1
```

**Ou manualmente:**

```powershell
docker compose exec rotina-viva python tests/test_security_rotina.py
docker compose exec rotina-viva python scripts/demo_security_before_after.py --write /data/SEGURANCA_ANTES_DEPOIS.md
```

O relatório antes/depois fica em `data/SEGURANCA_ANTES_DEPOIS.md` (pasta `data` montada no contentor como `/data`).

> **Nota:** `docker compose exec` é diferente de `docker run` — o contentor `rotina-viva` tem de estar **a correr** (`docker compose ps`).

### Avaliação e testes locais

```powershell
# Avaliação DeepEval (com .env configurado)
python tests/test_rotina_viva.py

# Heurísticas ML de emoção
python tests/test_ml_emotion_chat.py
```

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Leonardo**
- GitHub: [@Leonardolabdc](https://github.com/Leonardolabdc)
