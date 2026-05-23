# Testes de seguranca dentro do contentor rotina-viva (Docker Compose).
# Uso (na raiz do projeto): .\scripts\run-security-tests-docker.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

Write-Host "A verificar contentor rotina-viva..."
$running = docker compose ps --services --filter "status=running" 2>$null
if ($running -notmatch "rotina-viva") {
    Write-Host "Contentor parado - a subir docker compose up -d ..."
    docker compose up -d rotina-viva
    Start-Sleep -Seconds 3
}

Write-Host 'A correr testes de seguranca (deps ja vêm do requirements.txt na imagem)...'

Write-Host "`n=== tests/test_security_rotina.py ===" -ForegroundColor Cyan
docker compose exec -T rotina-viva python tests/test_security_rotina.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Relatorio antes/depois em data/SEGURANCA_ANTES_DEPOIS.md ===" -ForegroundColor Cyan
docker compose exec -T rotina-viva python scripts/demo_security_before_after.py --write /data/SEGURANCA_ANTES_DEPOIS.md
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nConcluido. Abra: data\SEGURANCA_ANTES_DEPOIS.md" -ForegroundColor Green
