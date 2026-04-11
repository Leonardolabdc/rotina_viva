# Rotina Viva 🌿

> **Status do Projeto:** 🛠️ Em desenvolvimento (Etapa 1 - CBL)

O **Rotina Viva** é um assistente inteligente projetado para ser a ponte digital entre a escola e os pais. Ele utiliza Inteligência Artificial para transformar a montanha de dados burocráticos de uma escola infantil em informações úteis e acessíveis.

---

## 💡 Metodologia CBL (Challenge-Based Learning)

Este projeto foi estruturado seguindo os pilares do aprendizado baseado em desafios:

### 🌟 Grande Ideia
**Comunicação escolar e acompanhamento do desenvolvimento na educação infantil.** A base do projeto é fortalecer o vínculo entre a instituição de ensino e os responsáveis, garantindo que o desenvolvimento da criança seja acompanhado de perto e com clareza.

### ❓ Pergunta Essencial
> Como a IA pode otimizar o registro da rotina escolar e melhorar a transparência para os pais, garantindo que os educadores tenham mais tempo de qualidade para se dedicar ao desenvolvimento dos alunos?

### 🎯 O Desafio
**Desafio: Desenvolver o Rotina Viva**, um assistente inteligente robusto que automatiza o registro diário de alimentação, sono e higiene, além de atuar como consultor pedagógico instantâneo para sanar dúvidas sobre o regimento e diretrizes da escola. O sistema elimina gargalos de comunicação manual ao permitir que pais e educadores interajam de forma natural e acessível, garantindo agilidade no preenchimento de dados e humanizando o acompanhamento do desenvolvimento infantil.

---

## 📝 Justificativa Pessoal


> A partir de conversas com minha esposa, observamos as limitações das agendas de papel tradicionais e a excessiva carga de trabalho manual imposta aos educadores. Acredito que uma agenda virtual, orientada por uma Inteligência Artificial bem estruturada e robusta, pode devolver o tempo para o que realmente importa: o cuidado e a educação das crianças, além de elevar significativamente a transparência das informações para os pais.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.12
* **Interface:** Streamlit
* **Orquestração de IA:** OpenRouter (Modelos meta-llama/llama-3.3-70b-instruct, openai/text-embedding-3-small)
* **Banco de Dados Vetorial:** ChromaDB (para busca semântica e embeddings em PDFs)
* **Banco de Dados Relacional:** DuckDB (para análise de dados de rotina)
* **Infraestrutura:** Docker & Docker Compose
* **Versionamento:** Git & GitHub

---

## 📦 Estrutura do Projeto

* `data/`: Contém os arquivos de entrada (ex: `diario_estruturado.csv`).
* `scripts/`: Scripts auxiliares de automação e limpeza de dados.
* `Dockerfile` & `docker-compose.yml`: Configurações para ambiente isolado e replicável.
* `app.py`: Script principal para processamento e análise.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
* [Docker](https://www.docker.com/) instalado.
* [Git](https://git-scm.com/) instalado.

### Passo a Passo
1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Leonardolabdc/Rotina-Viva.git](https://github.com/Leonardolabdc/Rotina-Viva.git)
    cd Rotina-Viva
    ```
2.  **Suba o ambiente com Docker:**
    ```bash
    docker-compose up --build
    ```

