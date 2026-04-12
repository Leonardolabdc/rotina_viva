# Rotina Viva 🌿

> **Status do Projeto:** 🛠️ Em desenvolvimento (Etapa 1 - CBL)

O **Rotina Viva** é um assistente inteligente projetado para ser a ponte digital entre a escola e os pais. Ele utiliza Inteligência Artificial para transformar a montanha de dados burocráticos de uma escola infantil em informações úteis e acessíveis.

---

##   Metodologia CBL (Challenge-Based Learning)

Este projeto foi estruturado seguindo os pilares do aprendizado baseado em desafios:

###   Grande Ideia
**Comunicação escolar e acompanhamento do desenvolvimento na educação infantil.** A base do projeto é fortalecer o vínculo entre a instituição de ensino e os responsáveis, garantindo que o desenvolvimento da criança seja acompanhado de perto e com clareza.

###   Pergunta Essencial
> Como a IA pode otimizar o registro da rotina escolar e melhorar a transparência para os pais, garantindo que os educadores tenham mais tempo de qualidade para se dedicar ao desenvolvimento dos alunos?

###   O Desafio
**Desafio: Desenvolver o Rotina Viva**, um assistente inteligente robusto que automatiza o registro diário de alimentação, sono e higiene, além de atuar como consultor pedagógico instantâneo para sanar dúvidas sobre o regimento e diretrizes da escola. O sistema elimina gargalos de comunicação manual ao permitir que pais e educadores interajam de forma natural e acessível, garantindo agilidade no preenchimento de dados e humanizando o acompanhamento do desenvolvimento infantil.

---

##   Justificativa Pessoal


> A partir de conversas com minha esposa, observamos as limitações das agendas de papel tradicionais e a excessiva carga de trabalho manual imposta aos educadores. Acredito que uma agenda virtual, orientada por uma Inteligência Artificial bem estruturada e robusta, pode devolver o tempo para o que realmente importa: o cuidado e a educação das crianças, além de elevar significativamente a transparência das informações para os pais.

---

##   Tecnologias Utilizadas

* **Linguagem:** Python 3.12
* **Interface:** Streamlit
* **Orquestração de IA:** OpenRouter (Modelos meta-llama/llama-3.3-70b-instruct, openai/text-embedding-3-small)
* **Banco de Dados Vetorial:** ChromaDB (para busca semântica e embeddings em PDFs)
* **Banco de Dados Relacional:** DuckDB (para análise de dados de rotina)
* **Infraestrutura:** Docker & Docker Compose
* **Versionamento:** Git & GitHub

---

##   System Prompt

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
- Esta tabela reflete a **pergunta atual**; não conclua pela omissão de nomes aqui que "não há dados" sobre alguém que o utilizador **não citou** nesta pergunta."""

---

##   Estrutura do Projeto

* `data/`: Contém os arquivos de entrada (ex: `diario_estruturado.csv`).
* `scripts/`: Scripts auxiliares de automação e limpeza de dados.
* `Dockerfile` & `docker-compose.yml`: Configurações para ambiente isolado e replicável.
* `app.py`: Script principal para processamento e análise.

---

##   Como Executar o Projeto

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

