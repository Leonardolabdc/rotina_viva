# Streamlit Cloud — prepare deploy (Windows)
# Na raiz: .\scripts\prepare-streamlit-cloud.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "== Rotina Viva — preparar Streamlit Cloud ==" -ForegroundColor Cyan

if (-not (Test-Path "data\rotina_users.json")) {
    Copy-Item "data\rotina_users.example.json" "data\rotina_users.json"
    Write-Host "Criado data\rotina_users.json (demo)" -ForegroundColor Green
}

if (-not (Test-Path ".env")) {
    Write-Host "AVISO: .env ausente — configure chaves antes de build_rag_index.py" -ForegroundColor Yellow
} else {
    Write-Host "A construir índice RAG (opcional mas recomendado)..." -ForegroundColor Cyan
    python scripts/build_rag_index.py
    if ($LASTEXITCODE -eq 0 -and (Test-Path "data\vector_db")) {
        Write-Host "Índice em data\vector_db — considere: git add -f data\vector_db\" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Próximos passos:" -ForegroundColor Cyan
Write-Host "  1. Push para GitHub (branch main ou deploy)"
Write-Host "  2. https://share.streamlit.io → New app → app.py"
Write-Host "  3. Secrets: copie de .streamlit\secrets.toml.example"
Write-Host "  4. Login demo: gestao.demo / demo123 — active CrewAI na sidebar"
