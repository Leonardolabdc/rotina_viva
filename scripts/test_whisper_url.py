#!/usr/bin/env python3
"""Testa reachability do serviço Whisper (OpenAI-compatível) a partir do PC ou CI."""

from __future__ import annotations

import os
import sys

import httpx


def main() -> int:
    base = (
        (sys.argv[1] if len(sys.argv) > 1 else "")
        or os.getenv("ROTINA_TRANSCRIBE_SERVICE_URL", "")
        or os.getenv("OPENAI_TRANSCRIBE_BASE_URL", "")
    ).strip().rstrip("/")
    if not base:
        print("Uso: python scripts/test_whisper_url.py http://IP:9000/v1")
        print("  ou defina OPENAI_TRANSCRIBE_BASE_URL / ROTINA_TRANSCRIBE_SERVICE_URL")
        return 1
    docs = f"{base.rstrip('/')}/../docs".replace("/v1/../docs", "/docs")
    if not docs.endswith("/docs"):
        docs = base.replace("/v1", "") + "/docs" if base.endswith("/v1") else f"{base}/docs"
    # hwdsl2 expõe /docs na raiz do servidor (porta 9000)
    root_docs = base.split("/v1")[0] + "/docs"
    print(f"Base STT: {base}")
    for label, url in [("docs", root_docs), ("base", base)]:
        try:
            r = httpx.get(url, timeout=httpx.Timeout(15.0, connect=10.0))
            print(f"  [{label}] {url} → HTTP {r.status_code}")
            if r.status_code < 400:
                print("OK — serviço acessível.")
                return 0
        except httpx.ConnectError as e:
            print(f"  [{label}] FALHA ligação: {e}")
        except httpx.TimeoutException:
            print(f"  [{label}] timeout")
    print(
        "Não foi possível contactar o Whisper. Verifique firewall (porta 9000), "
        "docker compose -f docker-compose.whisper.yml ps, e logs."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
