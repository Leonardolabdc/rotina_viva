"""LLM para CrewAI alinhado às variáveis de ambiente do `ai_engine` (OpenAI / OpenRouter)."""

from __future__ import annotations

from typing import Any

from modules import ai_engine


def _crew_litellm_model_name() -> str:
    """
    CrewAI usa LiteLLM por baixo dos panos; nomes tipo `vendor/model` (ex.: meta-llama/…)
    exigem prefixo de provider (`openrouter/...`). Sem isso LiteLLm acusa provider desconhecido.
    """
    model = (ai_engine.OPENAI_CHAT_MODEL or "").strip()
    if not model:
        return model
    low = model.lower()
    base = (ai_engine.OPENAI_BASE_URL or "").strip().lower()
    use_openrouter = ai_engine.ROTINA_CHAT_PROVIDER == "openrouter" or "openrouter.ai" in base
    if use_openrouter and not low.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


def build_crew_chat_llm(
    *,
    langfuse_trace_id: str | None = None,
    langfuse_session_id: str | None = None,
) -> Any:
    """
    Devolve `crewai.llm.LLM` (LiteLLM) com o mesmo endpoint/chave que o chat OpenAI-compatível.
    Instanciar `ChatOpenAI` fazia o CrewAI extrair só o nome do modelo e perder o contexto
    OpenRouter (`meta-llama/...` → provider inválido).
    """
    if not ai_engine.use_openai_compatible_chat():
        raise RuntimeError(
            "CrewAI neste projecto usa LangChain OpenAI: defina `ROTINA_CHAT_PROVIDER=openai` "
            "(ou openrouter) e `OPENAI_API_KEY` / `OPENROUTER_API_KEY`."
        )
    from crewai.llm import LLM as CrewAILLM

    url = ai_engine.OPENAI_BASE_URL.rstrip("/")
    litellm_meta: dict[str, Any] = {}
    if langfuse_trace_id:
        litellm_meta["existing_trace_id"] = langfuse_trace_id
    if langfuse_session_id:
        litellm_meta["session_id"] = langfuse_session_id
    extra: dict[str, Any] = {}
    if litellm_meta:
        extra["metadata"] = litellm_meta

    llm = CrewAILLM(
        model=_crew_litellm_model_name(),
        api_key=ai_engine.OPENAI_API_KEY,
        base_url=url,
        api_base=url,
        temperature=min(0.7, max(0.0, float(ai_engine.ROTINA_CHAT_TEMPERATURE))),
        **extra,
    )
    try:
        from modules.langfuse_rotina import refresh_litellm_langfuse_callbacks

        refresh_litellm_langfuse_callbacks()
    except Exception:
        pass
    return llm
