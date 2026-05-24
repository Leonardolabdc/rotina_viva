# Modelo ML de emoções (produção — Fase 0)

| Ficheiro | Descrição |
|----------|-----------|
| `emotion_flaml_bundle.pkl` | Pipeline TF-IDF + classificador (FLAML), dataset [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| `manifest.json` | Metadados, métricas e SHA256 do artefacto |

**Uso na app:** inferência no chat (`/emotion`, IA preditiva) e ferramenta CrewAI. Treino FLAML permanece no laboratório local (`ROTINA_ENABLE_ML_LAB`).

**Nota de escala:** em produção futura, este `.pkl` deve sair do repo público (storage privado, URL secreta ou formato sem pickle).
