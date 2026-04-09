# 🌿 Rotina Viva - Assistente Inteligente de Gestão Escolar

O **Rotina Viva** é um sistema inteligente desenvolvido para transformar a comunicação entre escolas infantis e pais. O projeto automatiza o registro de rotinas diárias (alimentação, sono, higiene) e oferece um canal de consulta instantâneo baseado em documentos oficiais da instituição e dados históricos dos alunos.

Este projeto faz parte da disciplina **AI Factory: Building Intelligent Systems** da PUCPR.

---

## 🚀 Diferenciais do Projeto
- **Interface Natural:** Professores podem registrar eventos via chat de forma rápida.
- **Memória Semântica (RAG):** Consulta automática ao Regimento Interno, Cardápio e Guias de Saúde.
- **Análise de Dados Estruturados:** Monitoramento de padrões de saúde e comportamento via DuckDB.
- **Alta Performance:** Implementado com arquitetura de API para garantir baixa latência mesmo em hardware limitado.

---

## 🛠️ Stack Tecnológica
- **Linguagem:** Python 3.10+
- **Interface:** [Streamlit](https://streamlit.io/)
- **Cérebro (LLM):** OpenAI API / Groq Cloud (Llama 3 / GPT-4o)
- **Banco de Dados Estruturados:** [DuckDB](https://duckdb.org/)
- **Banco Vetorial (RAG):** [ChromaDB](https://www.trychroma.com/)
- **Infraestrutura:** [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)

---

## 📁 Estrutura de Arquivos
```text
Rotina-Viva/
├── data/                       # Documentos e CSVs base
│   ├── diario_estruturado.csv
│   ├── info_alunos.csv
│   ├── guia_saude_seguranca.pdf
│   ├── planejamento_nutricional_mensal.pdf
│   └── regimento_interno_escola.pdf
├── .env                        # Chaves de API (não versionado)
├── .gitignore                  # Proteção de dados sensíveis
├── app.py                      # Aplicação principal Streamlit
├── Dockerfile                  # Receita da imagem Docker
├── docker-compose.yml          # Orquestração do container
└── requirements.txt            # Dependências do projeto