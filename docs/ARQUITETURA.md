# Arquitetura e fluxo de dados

![Fluxo macro do Rotina Viva](../assets/rotina_viva_fluxo_macro.png)

## Destaque técnico

O Rotina Viva utiliza uma arquitetura baseada em perfis de acesso (RBAC). Educadores possuem privilégios de escrita (CRUD) via voz ou texto para alimentar o DuckDB, enquanto os pais possuem acesso restrito apenas para consulta e interação via RAG no ChromaDB. Toda comunicação é mediada por uma camada de segurança que inclui guardrails e anonimização de dados sensíveis (PII).

## Estrutura de pastas

| Pasta / ficheiro | Função |
|------------------|--------|
| `app.py` | Ponto de entrada Streamlit |
| `src/core/` | Autenticação, base de dados DuckDB, segurança |
| `src/modules/` | Motor de IA, RAG, CrewAI, serviços de chat e ML de emoções |
| `src/ui/` | Componentes e estilos da interface |
| `data/` | CSVs, PDFs indexados, persistência local |
| `docker/` | Serviço Whisper (transcrição) |
| `scripts/` | Utilitários (testes de segurança no Docker) |
| `tests/` | Avaliação DeepEval e testes unitários |
| `docs/` | Documentação técnica e CBL |

## System prompts (resumo)

Persona do assistente: tom empático, uso exclusivo do contexto fornecido (dados tabulares e trechos RAG), sem inventar nomes ou ocorrências, respostas em português do Brasil.

Regras de grounding e SQL: priorizar factos do contexto; para dados tabulares, não inventar linhas; resumir em poucas frases sem transcrever tabelas inteiras.

Definições completas em `src/modules/ai_engine.py` (`SYSTEM_PERSONA`, `SYSTEM_GROUNDING`, `SYSTEM_SQL_STRICT`).

## Segurança

Relatório detalhado: [RELATORIO_SEGURANCA_LLM.md](RELATORIO_SEGURANCA_LLM.md).

Testes no Docker:

```powershell
.\scripts\run-security-tests-docker.ps1
```
