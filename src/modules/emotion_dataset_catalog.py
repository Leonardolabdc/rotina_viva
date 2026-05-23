"""
Metadados científicos do dataset dair-ai/emotion (Hugging Face).
Fonte: https://huggingface.co/datasets/dair-ai/emotion
"""

from __future__ import annotations

from typing import Any

# Identificação no Hub
EMOTION_DATASET_ID = "dair-ai/emotion"
EMOTION_CONFIG = "split"  # train / validation / test (20k no total)

# Nomes por id (ordem do paper / card do dataset)
EMOTION_LABEL_ID_TO_NAME: dict[int, str] = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

EMOTION_LABEL_NAME_TO_ID: dict[str, int] = {v: k for k, v in EMOTION_LABEL_ID_TO_NAME.items()}

# Descrição de cada coluna (schema oficial)
COLUMN_DESCRIPTIONS: dict[str, str] = {
    "text": (
        "Mensagem em inglês (estilo Twitter / texto curto) que expressa um estado emocional. "
        "É a única feature bruta para modelos tabulares clássicos após vetorização (ex.: TF-IDF)."
    ),
    "label": (
        "Classe alvo inteira 0–5: sadness, joy, love, anger, fear, surprise. "
        "Não use como feature no treino — apenas como y."
    ),
}

# Colunas que normalmente NÃO devem entrar como features numéricas/tabular direto
# (neste dataset o schema é mínimo; a lista orienta extensões futuras)
SUGGESTED_COLUMNS_TO_EXCLUDE_FROM_FEATURES: list[str] = []

# Notas para preparação — o dataset já vem “limpo” em colunas
DATA_PREPARATION_NOTES: list[str] = [
    "O split `split` já traz train / validation / test separados — use validação oficial "
    "para comparar com literatura, ou faça subamostragem só para protótipos rápidos.",
    "Não remova `text` nem `label` para o fluxo supervisionado padrão.",
    "Se no futuro juntar metadados (ids, timestamps), colunas de **leakage** (ex.: id do "
    "autor correlacionado ao alvo) devem ser excluídas da matriz X.",
    "Textos duplicados com labels diferentes são ruído real; para EDA pode contar duplicados, "
    "mas não os apague automaticamente sem política explícita.",
    "O domínio é inglês informal; métricas no teste não generalizam para português sem adaptação.",
]


def emotion_dataset_card_markdown() -> str:
    """Texto explicativo para a UI (Markdown)."""
    lines = [
        f"**Dataset:** `{EMOTION_DATASET_ID}` (config `{EMOTION_CONFIG}`)",
        "",
        "### Colunas",
    ]
    for col, desc in COLUMN_DESCRIPTIONS.items():
        lines.append(f"- **`{col}`:** {desc}")
    lines.extend(
        [
            "",
            "### Mapeamento `label` → emoção",
        ]
    )
    for i, name in sorted(EMOTION_LABEL_ID_TO_NAME.items()):
        lines.append(f"- `{i}` → **{name}**")
    lines.append("")
    lines.append("### Sugestão de exclusão de colunas (features)")
    if not SUGGESTED_COLUMNS_TO_EXCLUDE_FROM_FEATURES:
        lines.append(
            "_Nenhuma coluna extra no schema base — não há o que remover além do que você "
            "mesmo acrescentar em merges. Mantenha `label` só como alvo._"
        )
    else:
        for c in SUGGESTED_COLUMNS_TO_EXCLUDE_FROM_FEATURES:
            lines.append(f"- `{c}`")
    lines.append("")
    lines.append("### Notas de preparação")
    for n in DATA_PREPARATION_NOTES:
        lines.append(f"- {n}")
    return "\n".join(lines)


def catalog_dict() -> dict[str, Any]:
    """Estrutura serializável (ex.: inspecionar no Streamlit)."""
    return {
        "dataset_id": EMOTION_DATASET_ID,
        "config": EMOTION_CONFIG,
        "column_descriptions": COLUMN_DESCRIPTIONS,
        "suggested_drop_from_features": list(SUGGESTED_COLUMNS_TO_EXCLUDE_FROM_FEATURES),
        "label_map": {str(k): v for k, v in EMOTION_LABEL_ID_TO_NAME.items()},
        "preparation_notes": list(DATA_PREPARATION_NOTES),
    }
