"""
Langfuse opcional para a LLM Streamlit + CrewAI/LiteLLM.

Usa **Langfuse Python SDK 2.x** (`langfuse>=2.57,<3`): o LiteLLM integra-se com `Langfuse.trace()` /
`.generation()`, API removida na série 3.x (causa erros em silêncio no callback).

Variáveis: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` e/ou `LANGFUSE_BASE_URL`.
LiteLLM tende a usar `LANGFUSE_HOST`; no arranque `ensure_litellm_langfuse_env()` sincroniza host e base URL.
Desligar: `ROTINA_LANGFUSE_ENABLED=false`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Generator

_log = logging.getLogger(__name__)

_LF_CLIENT: Any | None = None
_LANGFUSE_IMPORT_ERROR: str | None = None
_LITELLM_CALLBACKS_REGISTERED = False
_LANGFUSE_OK_LOGGED = False
_LANGFUSE_PARTIAL_WARNED = False
_LANGFUSE_LITELLM_PATCHED = False

_MAX_CHARS_PER_MESSAGE = int(os.getenv("ROTINA_LANGFUSE_MAX_CHARS_PER_MSG", "12000"))
_langfuse_logged_disabled = False


def _strip_env_quotes(raw: str | None) -> str:
    """Remove aspas envolventes comuns em `.env` (ex.: \"sk-lf-...\") que o loader pode deixar."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def langfuse_integration_enabled() -> bool:
    if os.getenv("ROTINA_LANGFUSE_ENABLED", "false").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    sk = _strip_env_quotes(os.getenv("LANGFUSE_SECRET_KEY"))
    pk = _strip_env_quotes(os.getenv("LANGFUSE_PUBLIC_KEY"))
    base = _strip_env_quotes(os.getenv("LANGFUSE_BASE_URL")) or _strip_env_quotes(
        os.getenv("LANGFUSE_HOST")
    )
    return bool(sk and pk and base)


def _apply_langfuse_litellm_sdk_patch() -> None:
    """
    LiteLLM integra Langfuse passando `sdk_integration=` ao construtor.
    Langfuse Python ≥3 removeu esse argumento → TypeError em silêncio no callback.
    """
    global _LANGFUSE_LITELLM_PATCHED
    if _LANGFUSE_LITELLM_PATCHED:
        return
    try:
        from langfuse import Langfuse as LF

        _orig = LF.__init__

        def _wrap_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.pop("sdk_integration", None)
            return _orig(self, *args, **kwargs)

        LF.__init__ = _wrap_init  # type: ignore[method-assign]
        _LANGFUSE_LITELLM_PATCHED = True
    except Exception as e:
        _log.debug("Langfuse LiteLLM sdk patch: %s", e)


def ensure_litellm_langfuse_env() -> None:
    """
    LiteLLM (callback Langfuse) lê sobretudo `LANGFUSE_HOST`; o SDK aceita `LANGFUSE_BASE_URL`.
    Sincroniza os dois para `.env` com só um dos campos preenchidos.
    """
    base = _strip_env_quotes(os.getenv("LANGFUSE_BASE_URL")).rstrip("/")
    host = _strip_env_quotes(os.getenv("LANGFUSE_HOST")).rstrip("/")
    if host and not base:
        os.environ["LANGFUSE_BASE_URL"] = host
        base = host
    if base and not host:
        os.environ["LANGFUSE_HOST"] = base


def configure_litellm_observability() -> None:
    """
    Configuração central LiteLLM para observabilidade (Etapa 2 / PUCPR):
    - `success_callback` + async para Langfuse quando integração activa;
    - `identify_installation` se existir na versão do LiteLLM (telemetria/repasse de usage em versões recentes).

    Idempotente: pode chamar-se após `crewai.llm.LLM` limpar callbacks.
    """
    global _LITELLM_CALLBACKS_REGISTERED
    try:
        import litellm
    except ImportError:
        return
    _sv = getattr(litellm, "set_verbose", None)
    if callable(_sv):
        _sv(False)
    else:
        try:
            litellm.set_verbose = False  # type: ignore[misc]
        except Exception:
            pass
    if hasattr(litellm, "identify_installation"):
        try:
            litellm.identify_installation = True  # type: ignore[misc]
        except Exception as e:
            _log.debug("litellm.identify_installation: %s", e)
    if not langfuse_integration_enabled():
        return
    try:
        import langfuse  # noqa: F401

        _apply_langfuse_litellm_sdk_patch()
        ensure_litellm_langfuse_env()
        litellm.success_callback = ["langfuse"]
        if hasattr(litellm, "_async_success_callback"):
            litellm._async_success_callback = ["langfuse"]
        litellm.logging_callback_manager.add_litellm_success_callback("langfuse")
        litellm.logging_callback_manager.add_litellm_async_success_callback("langfuse")
        _LITELLM_CALLBACKS_REGISTERED = True
    except Exception as e:
        _log.debug("configure_litellm_observability: %s", e)


def _truncate_chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    try:
        from core.security import mask_messages_for_observability

        masked = mask_messages_for_observability(messages)
    except Exception:
        masked = messages
    out: list[dict[str, str]] = []
    cap = max(500, min(_MAX_CHARS_PER_MESSAGE, 200_000))
    for m in masked:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "?")
        c = str(m.get("content") or "")
        if len(c) > cap:
            c = c[: cap - 1] + "…"
        out.append({"role": role, "content": c})
    return out


def init_langfuse_integration() -> None:
    """Chamar uma vez ao arrancar (`app.py` após load_dotenv). Idempotente."""
    global _langfuse_logged_disabled, _LANGFUSE_OK_LOGGED, _LANGFUSE_PARTIAL_WARNED

    ensure_litellm_langfuse_env()
    configure_litellm_observability()

    if not langfuse_integration_enabled():
        sk = _strip_env_quotes(os.getenv("LANGFUSE_SECRET_KEY"))
        pk = _strip_env_quotes(os.getenv("LANGFUSE_PUBLIC_KEY"))
        bu = _strip_env_quotes(os.getenv("LANGFUSE_BASE_URL")) or _strip_env_quotes(
            os.getenv("LANGFUSE_HOST")
        )
        if (sk or pk or bu) and not _LANGFUSE_PARTIAL_WARNED:
            _LANGFUSE_PARTIAL_WARNED = True
            missing = []
            if not sk:
                missing.append("LANGFUSE_SECRET_KEY")
            if not pk:
                missing.append("LANGFUSE_PUBLIC_KEY")
            if not bu:
                missing.append("LANGFUSE_BASE_URL (ou LANGFUSE_HOST)")
            _log.warning(
                "Langfuse: variáveis incompletas (faltam: %s). Nada será enviado ao Langfuse.",
                ", ".join(missing) if missing else "combinação inválida",
            )
        return
    try:
        import langfuse  # noqa: F401

        _apply_langfuse_litellm_sdk_patch()
    except ImportError as e:
        if not _langfuse_logged_disabled:
            _langfuse_logged_disabled = True
            _log.warning("Pacote `langfuse` não instalado (%s). `pip install langfuse`", e)
        return

    try:
        if _LITELLM_CALLBACKS_REGISTERED and not _LANGFUSE_OK_LOGGED:
            _LANGFUSE_OK_LOGGED = True
            host = _strip_env_quotes(os.getenv("LANGFUSE_HOST")) or _strip_env_quotes(
                os.getenv("LANGFUSE_BASE_URL")
            )
            _log.info(
                "Langfuse: activo — SDK + callbacks LiteLLM (sync/async). Host=%s",
                host or "(n/d)",
            )
            print(
                f"[rotina] Langfuse: LiteLLM callbacks registados (host={host or 'n/d'})",
                file=sys.stderr,
                flush=True,
            )
    except Exception as e:
        _log.warning("LiteLLM indisponível para callback Langfuse (%s)", e)


def refresh_litellm_langfuse_callbacks() -> None:
    """
    O `crewai.llm.LLM` no __init__ chama `set_env_callbacks()` / `set_callbacks()` e pode
    limpar ou sobrescrever `litellm.success_callback` (ex.: `LITELLM_FAILURE_CALLBACKS` sem
    `LITELLM_SUCCESS_CALLBACKS`). Voltar a registar o callback **depois** de instanciar o LLM da Crew.
    """
    configure_litellm_observability()


def _get_langfuse() -> Any | None:
    global _LF_CLIENT, _LANGFUSE_IMPORT_ERROR
    if not langfuse_integration_enabled():
        return None
    if _LF_CLIENT is not None:
        return _LF_CLIENT
    try:
        import langfuse  # noqa: F401

        _apply_langfuse_litellm_sdk_patch()
        from langfuse import Langfuse as _LF

        sk = _strip_env_quotes(os.getenv("LANGFUSE_SECRET_KEY"))
        pk = _strip_env_quotes(os.getenv("LANGFUSE_PUBLIC_KEY"))
        base = (
            _strip_env_quotes(os.getenv("LANGFUSE_BASE_URL")).rstrip("/")
            or _strip_env_quotes(os.getenv("LANGFUSE_HOST")).rstrip("/")
        )
        _LF_CLIENT = _LF(
            public_key=pk,
            secret_key=sk,
            host=base,
            debug=_strip_env_quotes(os.getenv("LANGFUSE_DEBUG")).lower() in ("1", "true"),
        )
        return _LF_CLIENT
    except Exception as e:
        _LANGFUSE_IMPORT_ERROR = str(e)
        _log.warning("Langfuse client init falhou (%s)", e)
        return None


def lf_flush() -> None:
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        lf.flush()
    except Exception as e:
        _log.debug("langfuse.flush: %s", e)


def _litellm_model_slug_for_rotina_cost() -> str:
    """Nome de modelo no formato LiteLLM (ex.: `openrouter/...`) para `completion_cost`."""
    from modules import ai_engine

    model = (ai_engine.OPENAI_CHAT_MODEL or "").strip()
    if not model:
        return "gpt-4o-mini"
    low = model.lower()
    base = (ai_engine.OPENAI_BASE_URL or "").strip().lower()
    use_openrouter = ai_engine.ROTINA_CHAT_PROVIDER == "openrouter" or "openrouter.ai" in base
    if use_openrouter and not low.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


def emit_crew_aggregated_usage_to_langfuse(
    *,
    langfuse_trace_id: str,
    crew_output: Any,
    req_id: str,
) -> None:
    """
    Fail-safe: o contador de tokens do CrewAI agrega usage por `kickoff()`; envia para o Langfuse
    como geração filha do trace (usage_details + custo estimado LiteLLM), complementando o callback LiteLLM.
    """
    if not (langfuse_trace_id and str(langfuse_trace_id).strip()) or not langfuse_integration_enabled():
        return
    lf = _get_langfuse()
    if lf is None:
        return
    u = getattr(crew_output, "token_usage", None)
    if u is None:
        return
    pt = int(getattr(u, "prompt_tokens", 0) or 0)
    ct = int(getattr(u, "completion_tokens", 0) or 0)
    tt = int(getattr(u, "total_tokens", 0) or 0)
    if pt == 0 and ct == 0 and tt == 0:
        return
    if tt <= 0:
        tt = pt + ct
    model = _litellm_model_slug_for_rotina_cost()
    cost_details: dict[str, float] | None = None
    try:
        import litellm

        c = litellm.completion_cost(
            completion_response={
                "model": model,
                "usage": {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": tt,
                },
            },
        )
        if isinstance(c, (int, float)) and c >= 0:
            cost_details = {"total": float(c)}
    except Exception as e:
        _log.debug("emit_crew_aggregated_usage_to_langfuse.completion_cost: %s", e)

    meta_usage = {
        "crew_prompt_tokens": pt,
        "crew_completion_tokens": ct,
        "crew_total_tokens": tt,
        "crew_successful_requests": int(getattr(u, "successful_requests", 0) or 0),
    }
    gen_id = f"rotina-crew-usage::{langfuse_trace_id}"
    try:
        trace = lf.trace(id=langfuse_trace_id)
        trace.update(metadata={"rotina.crew.token_usage": meta_usage})
        try:
            trace.update(
                usage={
                    "promptTokens": pt,
                    "completionTokens": ct,
                    "totalTokens": tt,
                }
            )
        except Exception:
            pass
        g = trace.generation(
            id=gen_id,
            name="CrewAI — uso LLM agregado (fail-safe)",
            model=model,
            metadata={
                "source": "rotina-viva-crew-token-counter",
                "rotina.crew.req_id": req_id,
            },
            input={
                "note": "Tokens somados pelo CrewAI após kickoff; custo estimado via LiteLLM quando aplicável.",
            },
            usage_details={"input": pt, "output": ct},
            cost_details=cost_details,
            output=f"total_tokens={tt} (prompt={pt}, completion={ct})",
        )
        g.end()
        lf_flush()
    except Exception as e:
        _log.warning("emit_crew_aggregated_usage_to_langfuse: %s", e)


def start_chat_stream_generation(
    *,
    name: str,
    model: str,
    messages: list[dict[str, str]],
    model_parameters: dict[str, Any] | None,
) -> Any | None:
    lf = _get_langfuse()
    if lf is None:
        return None
    try:
        return lf.generation(
            name=name,
            model=model,
            input=_truncate_chat_messages(messages),
            model_parameters=model_parameters or None,
            metadata={"source": "rotina-viva-streamlit"},
        )
    except Exception as e:
        _log.warning("Langfuse generation (streamlit) falhou (%s)", e)
        return None


def finish_generation_stream(gen: Any | None, output_parts: list[str]) -> None:
    if gen is None:
        return
    try:
        gen.update(output="".join(output_parts))
        gen.end()
        lf_flush()
    except Exception as e:
        _log.debug("Langfuse finish_generation_stream: %s", e)


def finish_generation_error(gen: Any | None, message: str) -> None:
    if gen is None:
        return
    try:
        gen.update(status_message=str(message)[:2000], level="ERROR")
        gen.end()
        lf_flush()
    except Exception:
        pass


def observe_openai_generation(
    *,
    name: str,
    model: str,
    messages: list[dict[str, str]],
    model_parameters: dict[str, Any] | None,
    output: Any,
    err: BaseException | None = None,
    usage_details: dict[str, int] | None = None,
    cost_details: dict[str, float] | None = None,
) -> None:
    lf = _get_langfuse()
    if lf is None:
        return
    try:
        g = lf.generation(
            name=name,
            model=model,
            input=_truncate_chat_messages(messages),
            model_parameters=model_parameters or None,
            metadata={"source": "rotina-viva"},
        )
        if err is not None:
            g.update(status_message=str(err)[:2000], level="ERROR")
        else:
            out = output if isinstance(output, str) else str(output)
            if len(out) > _MAX_CHARS_PER_MESSAGE:
                out = out[: _MAX_CHARS_PER_MESSAGE - 1] + "…"
            upd: dict[str, Any] = {"output": out}
            if usage_details:
                upd["usage_details"] = usage_details
            if cost_details:
                upd["cost_details"] = cost_details
            g.update(**upd)
        g.end()
        lf_flush()
    except Exception as e:
        _log.debug("observe_openai_generation: %s", e)


def begin_rotina_crew_langfuse_trace_if_enabled(
    *,
    langfuse_trace_id: str,
    req_id: str,
    user_text: str,
    plan_labels: str,
    plan_reason: str,
) -> None:
    """
    Abre um trace com ID fixo **antes** do `crew.kickoff()` para coincidir com
    `metadata.existing_trace_id` nas completions LiteLLM (mesma árvore no Langfuse).
    """
    if not langfuse_integration_enabled():
        return
    try:
        lf = _get_langfuse()
        if lf is None:
            return
        um = (user_text or "").strip()
        if len(um) > 8000:
            um = um[:7999] + "…"
        lf.trace(
            id=langfuse_trace_id,
            name="CrewAI · Rotina Viva",
            session_id=req_id,
            input={
                "pergunta_utilizador": um,
                "plano_especialistas": plan_labels,
            },
            metadata={
                "rotina.crew.req_id": req_id,
                "rotina.crew.reason": plan_reason,
            },
            tags=["rotina-viva", "crewai"],
        )
        lf.flush()
    except Exception as e:
        _log.warning("Langfuse pré-crew trace: %s", e)


def _crew_task_output_body(item: Any) -> str:
    if item is None:
        return ""
    r = getattr(item, "raw", None)
    if isinstance(r, str) and r.strip():
        return r.strip()
    o = getattr(item, "output", None)
    if isinstance(o, str) and o.strip():
        return o.strip()
    jd = getattr(item, "json_dict", None)
    if isinstance(jd, dict) and jd:
        try:
            import json

            return json.dumps(jd, ensure_ascii=False)
        except Exception:
            pass
    p = getattr(item, "pydantic", None)
    if p is not None:
        try:
            return str(p).strip()
        except Exception:
            pass
    return str(r or o or "").strip()


def log_crew_trace_tree_if_enabled(
    *,
    req_id: str,
    user_text: str,
    plan_labels: str,
    plan_reason: str,
    parallel_n: int,
    elapsed_ms: int,
    crew_output: Any,
    final_markdown: str,
    langfuse_trace_id: str | None = None,
) -> None:
    """
    Um trace Langfuse com spans aninhados por agente (árvore no UI).
    Complementa os eventos LiteLLM (LLM) com visão de **papel** (Receção, Dados, …).

    Use o mesmo ``langfuse_trace_id`` que no ``build_crew_chat_llm(...)`` para unir
    gerações LiteLLM e estes spans no **mesmo** trace (evita ficar só com `litellm-completion`).
    """
    if not langfuse_integration_enabled():
        return
    max_body = max(4000, min(int(os.getenv("ROTINA_LANGFUSE_CREW_BODY_MAX", "60000")), 200_000))
    try:
        from modules.rotina_crew.trace_labels import trace_agent_label_for_task_output
    except ImportError:
        _log.warning("Langfuse árvore crew: trace_labels inacessível (PYTHONPATH?).")
        return
    raw_list = getattr(crew_output, "tasks_output", None) or []
    if not raw_list:
        _log.debug("Langfuse árvore crew: tasks_output vazio (req=%s)", req_id)
    try:
        lf = _get_langfuse()
        if lf is None:
            return
        um = (user_text or "").strip()
        if len(um) > 8000:
            um = um[:7999] + "…"
        trace_kw: dict[str, Any] = {
            "name": "CrewAI · Rotina Viva",
            "session_id": req_id,
            "input": {
                "pergunta_utilizador": um,
                "plano_especialistas": plan_labels,
            },
            "metadata": {
                "rotina.crew.req_id": req_id,
                "rotina.crew.reason": plan_reason,
                "rotina.crew.parallel": parallel_n,
                "rotina.crew.ms": elapsed_ms,
            },
            "tags": ["rotina-viva", "crewai"],
        }
        if langfuse_trace_id:
            trace_kw["id"] = langfuse_trace_id
        trace = lf.trace(**trace_kw)
        for item in raw_list:
            lab = trace_agent_label_for_task_output(item)
            body = _crew_task_output_body(item)
            if len(body) > max_body:
                body = body[: max_body - 1] + "…"
            tn = getattr(item, "name", None)
            sp = trace.span(
                name=f"Agente · {lab}",
                output=body or "(sem texto)",
                metadata={"task_name": str(tn or ""), "rotina.crew.req_id": req_id},
            )
            sp.end()
        fin = (final_markdown or "").strip()
        if len(fin) > max_body:
            fin = fin[: max_body - 1] + "…"
        trace.update(output=fin or "(vazio)")
        lf.flush()
    except Exception as e:
        _log.warning("Langfuse árvore crew falhou (req=%s): %s", req_id, e)


def iterate_stream_with_langfuse(
    gen_obs: Any | None, inner: Generator[str, None, None]
) -> Generator[str, None, None]:
    parts: list[str] = []
    err: BaseException | None = None
    try:
        for piece in inner:
            parts.append(piece)
            yield piece
    except BaseException as e:
        err = e
        raise
    finally:
        if err is not None and not isinstance(err, GeneratorExit):
            finish_generation_error(gen_obs, str(err))
        else:
            finish_generation_stream(gen_obs, parts)
