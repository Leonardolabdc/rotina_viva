#!/usr/bin/env bash
set -euo pipefail

ollama serve &
PID=$!

echo "Aguardando Ollama..."
for _ in $(seq 1 120); do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "Baixando modelos (primeira execução pode demorar)..."
ollama pull llama3.2:3b
ollama pull nomic-embed-text

echo "Ollama pronto."
wait "${PID}"
