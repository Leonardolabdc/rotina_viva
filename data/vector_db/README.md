# Índice vetorial Chroma (RAG)

Gerado por `scripts/build_rag_index.py`. Para deploy no Streamlit Cloud, construa localmente e faça commit:

```bash
python scripts/build_rag_index.py
git add -f data/vector_db/
```

Sem esta pasta no repositório, a app indexa os PDFs no primeiro login na cloud.
