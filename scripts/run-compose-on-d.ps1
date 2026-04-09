#Requires -Version 5.1
<#
.SYNOPSIS
  Reduz uso do disco C: ao rodar o Rotina Viva com Docker Compose.

  - TEMP/TMP e cache de pip da sessão vão para D:\Dev\RotinaViva\.cache (no disco D:).
  - Modelos Ollama ficam em .\docker-data\ollama (já configurado no docker-compose.yml).

  IMPORTANTE: imagens e camadas Docker ainda ficam onde o Docker Desktop estiver configurado.
  Se o C: estiver cheio, mova os dados do Docker para D: (passos abaixo).
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $ProjectRoot "docker-compose.yml"))) {
    $ProjectRoot = "D:\Dev\RotinaViva"
}

$CacheRoot = Join-Path $ProjectRoot ".cache"
$TmpDir = Join-Path $CacheRoot "tmp"
$PipCache = Join-Path $CacheRoot "pip"
$OllamaHostDir = Join-Path $ProjectRoot "docker-data\ollama"

foreach ($d in @($CacheRoot, $TmpDir, $PipCache, $OllamaHostDir)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

$env:TEMP = $TmpDir
$env:TMP = $TmpDir
$env:PIP_CACHE_DIR = $PipCache
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

Write-Host ""
Write-Host "=== Rotina Viva - pastas no disco do projeto (drive D) ===" -ForegroundColor Cyan
Write-Host " TEMP/TMP:     $TmpDir"
Write-Host " PIP_CACHE:   $PipCache"
Write-Host " Ollama data: $OllamaHostDir"
Write-Host ""

Write-Host 'Se o build ainda lotar o disco do sistema, mova o Docker Desktop para o drive D.' -ForegroundColor Yellow
Write-Host '  1) Feche containers (Ctrl+C) e Docker Desktop - Quit'
Write-Host '  2) Docker Desktop - Settings - Resources - Advanced'
Write-Host '  3) Disk image location - pasta em D: (ex.: D:\DockerDesktopData)'
Write-Host '  4) Apply and Restart'
Write-Host '  5) Opcional: docker system prune -a (remove imagens nao usadas)'
Write-Host ""

Set-Location $ProjectRoot
docker compose up --build
