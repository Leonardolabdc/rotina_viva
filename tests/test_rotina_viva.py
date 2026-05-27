"""
Avaliação offline do Rotina Viva com DeepEval.

Pré-requisitos:
  - **Julgador:** `OPENAI_API_KEY` ou `OPENROUTER_API_KEY`. **Answer Relevancy** e **Faithfulness** usam sempre
    **`openai/gpt-4o-mini`** (OpenRouter) com `max_tokens` ≤ 2048 (`ROTINA_EVAL_METRIC_MAX_TOKENS`). **GEval** (sentimento) segue `ROTINA_EVAL_JUDGE_MODEL`
    (por defeito `openai/gpt-4o-mini` no OpenRouter).
  - **Inferência:** `modules.rotina_inference.run_rotina_chat_inference` — mesmo fluxo que o chat (`.env` igual à app).

**Multi-agente (CrewAI)** — equivalente a ligar o checkbox na sidebar da app:
  - Definir **`ROTINA_EVAL_CREWAI_MODE=1`** (ou `true` / `yes`) antes de correr o script.
  - Exige `crewai`, provedor compatível com OpenAI (`ROTINA_CHAT_PROVIDER=openai` ou `openrouter`, chave API), como na UI.
  - **Não** aplica se usares **`ROTINA_EVAL_INFERENCE_URL`**: aí a resposta vem só do servidor remoto.

**Limites do juiz:** respostas do assistente muito longas podem fazer o Faithfulness pedir JSON enorme à API e falhar com
`length limit was reached` (ex.: 16384 tokens). O script trunca `actual_output` só para as métricas (**`ROTINA_EVAL_METRIC_MAX_OUTPUT_CHARS`**, default 3200); o diff continua com o texto completo.
**Faithfulness / Answer Relevancy** usam sempre **`openai/gpt-4o-mini`** no OpenRouter (slug explícito), **`max_tokens` ≤ 2048** (`ROTINA_EVAL_METRIC_MAX_TOKENS`, teto 2048), mensagem de sistema a forçar **JSON só**, e **`retrieval_context`** limitado a **5000** caracteres no total.

**Faithfulness lento / “sem fim”:** o DeepEval volta a chamar a API sem limite de tentativas nalguns erros (`LengthFinishReasonError`, rate limit, etc.). Por defeito há **`ROTINA_EVAL_FAITHFULNESS_TIMEOUT_SEC=600`** (10 min — demo / rede lenta); **`0`** desactiva o limite. **`ROTINA_EVAL_ANSWER_RELEVANCY_TIMEOUT_SEC`** (default 600s) limita a espera do Answer Relevancy. Ou **`ROTINA_EVAL_SKIP_FAITHFULNESS=1`** para não correr esta métrica. `FaithfulnessMetric` usa **`include_reason=False`** para reduzir carga/latência no juiz.

**Limpeza de saída:** `run_inference` aplica **`_clean_inference_output`** (espaços duplos e quebras de linha excessivas) antes de devolver texto à suíte.

**Custo:** o maior gasto costuma ser o **juiz** (2 métricas × N casos, mais GEval nos casos de sentimento).
Usa **gpt-4o-mini** (na API OpenAI nativa) ou **openai/gpt-4o-mini** no OpenRouter — custo baixo e menos JSON truncado que modelos maiores.

**Observabilidade:** o juiz DeepEval usa o **SDK OpenAI** (não LiteLLM); `init_langfuse_integration()` é chamado na mesma para manter callbacks LiteLLM activos na inferência Crew/RAG. Chamadas do juiz não passam por LiteLLM.

**Avisos no terminal:** tentamos silenciar o Streamlit (`logging` + `warnings`); alguma linha residual pode aparecer e pode ignorar-se em CLI.

Execução (na pasta do projeto):
  python tests/test_rotina_viva.py

Só um caso do golden set (índice 1-based = «Caso N/N» na saída), no PowerShell:
  $env:ROTINA_EVAL_ONLY_CASE="10"; python tests/test_rotina_viva.py

Com CrewAI (multi-agente), no PowerShell:
  $env:ROTINA_EVAL_CREWAI_MODE="1"; python tests/test_rotina_viva.py
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import warnings

# Streamlit é importado pela pipeline local; sem isto o terminal enche de WARNING em `python test_rotina_viva.py`.
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

_T = TypeVar("_T")


def _silence_streamlit_logging() -> None:
    for name in (
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner_utils",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
    ):
        logging.getLogger(name).setLevel(logging.CRITICAL)


_silence_streamlit_logging()

# Demo / gravação: reduz ruído de loggers do SDK e do DeepEval no terminal.
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("deepeval").setLevel(logging.CRITICAL)

# Raiz + src/ no path (para `import modules` / `import core` com cwd diferente)
_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent
_SRC = _PROJECT_ROOT / "src"
for _p in (_SRC, _PROJECT_ROOT, _TESTS_DIR):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from deepeval_bootstrap import configure_deepeval_home

_DEEPEVAL_HOME = configure_deepeval_home()

THRESHOLD = 0.7
SENTIMENT_TAG = "Análise de Sentimento"
_STABILITY_FALLBACK_SCORE = 0.7
_STABILITY_FALLBACK_REASON = "Nota atribuída via fallback de estabilidade"


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except ValueError:
        return default


def _env_truthy_eval(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _blocking_with_pulse(
    label: str,
    fn: Callable[[], _T],
    *,
    interval_sec: float | None = None,
    overall_timeout_sec: float | None = None,
) -> _T:
    """Mostra pulso durante chamadas longas. Com `overall_timeout_sec`, corre `fn` em thread daemon
    e aborta no limite (útil porque o DeepEval usa Tenacity sem `stop` nalguns erros da API)."""
    sec = interval_sec if interval_sec is not None else float(max(5, _env_int("ROTINA_EVAL_HEARTBEAT_SEC", 45)))
    if overall_timeout_sec is None or overall_timeout_sec <= 0:
        stop = threading.Event()

        def pulse() -> None:
            elapsed = 0.0
            while True:
                if stop.wait(sec):
                    break
                elapsed += sec
                print(
                    f"  …{label}: a trabalhar (~{elapsed:.0f}s)…",
                    flush=True,
                )

        worker = threading.Thread(target=pulse, daemon=True)
        worker.start()
        try:
            return fn()
        finally:
            stop.set()

    err: list[BaseException | None] = [None]
    result: list[_T] = []

    def run_fn() -> None:
        try:
            result.append(fn())
        except BaseException as e:
            err[0] = e

    th = threading.Thread(target=run_fn, daemon=True)
    th.start()
    t0 = time.perf_counter()
    next_pulse_at = sec
    while th.is_alive():
        left = overall_timeout_sec - (time.perf_counter() - t0)
        if left <= 0:
            print(
                f"  …{label}: TIMEOUT aos {overall_timeout_sec:.0f}s — o cliente DeepEval pode repetir "
                "pedidos sem limite após erros como resposta truncada (16384 tokens). "
                "Tente ROTINA_EVAL_SKIP_FAITHFULNESS=1, ou ROTINA_EVAL_FAITHFULNESS_TIMEOUT_SEC=0 "
                "(sem limite; Ctrl+C para parar), ou aumente o limite.",
                flush=True,
            )
            raise TimeoutError(f"{label} excedeu {overall_timeout_sec:.0f}s")
        th.join(timeout=min(0.25, left))
        elapsed = time.perf_counter() - t0
        if elapsed >= next_pulse_at and th.is_alive():
            print(f"  …{label}: juiz a trabalhar (~{elapsed:.0f}s)…", flush=True)
            next_pulse_at += sec
    if err[0] is not None:
        raise err[0]
    if not result:
        raise RuntimeError(f"{label}: thread terminou sem resultado")
    return result[0]


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Telemetria OTEL do DeepEval desligada (evita "Overriding TracerProvider" com Langfuse/outros).
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
# Faithfulness: timeout por defeito mais alto (demo / juiz lento na rede).
os.environ.setdefault("ROTINA_EVAL_FAITHFULNESS_TIMEOUT_SEC", "600")
# Answer Relevancy: limite de espera no mesmo estilo do Faithfulness (evita ciclos longos visíveis).
os.environ.setdefault("ROTINA_EVAL_ANSWER_RELEVANCY_TIMEOUT_SEC", "600")
# Pulso no terminal com menos frequência durante timeouts longos.
os.environ.setdefault("ROTINA_EVAL_HEARTBEAT_SEC", "45")


def _eval_targets_openrouter() -> bool:
    prov = (os.getenv("ROTINA_CHAT_PROVIDER") or "").strip().lower()
    base = (os.getenv("OPENAI_BASE_URL") or "").strip().lower()
    return prov == "openrouter" or "openrouter.ai" in base


def _default_rotina_eval_judge_model() -> str:
    """Modelo do juiz para GEval (sentimento). Answer Relevancy / Faithfulness usam `_build_rotina_metrics_judge_gpt_model()`."""
    raw = (os.getenv("ROTINA_EVAL_JUDGE_MODEL") or "").strip()
    if raw:
        return raw
    return "openai/gpt-4o-mini" if _eval_targets_openrouter() else "gpt-4o-mini"


# Juiz dedicado às métricas Faithfulness + Answer Relevancy (slug OpenRouter fixo; custo previsível).
_METRICS_JUDGE_MODEL_SLUG = "openai/gpt-4o-mini"
_METRICS_JUDGE_SYSTEM = (
    "You are a strict judge. Output ONLY valid JSON matching the requested schema. "
    "No prose, no markdown fences, no reasoning text before or after the JSON. "
    "Keep string values under 200 characters each. One compact JSON object only."
)


def _metric_max_tokens() -> int:
    """Saída do juiz DeepEval (completions.parse / chat): até 2048 tokens (teto fixo)."""
    return max(1, min(_env_int("ROTINA_EVAL_METRIC_MAX_TOKENS", 2048), 2048))


def _metric_retrieval_context_max_chars() -> int:
    """Teto total de caracteres do `retrieval_context` enviado às métricas (lista unida com \\n\\n)."""
    return max(256, min(_env_int("ROTINA_EVAL_METRIC_RETRIEVAL_MAX_CHARS", 5000), 50_000))


def _api_model_id_for_metrics_judge() -> str:
    """OpenRouter espera o routing id completo; API OpenAI nativa usa o nome curto."""
    return _METRICS_JUDGE_MODEL_SLUG if _eval_targets_openrouter() else "gpt-4o-mini"


from deepeval.models.llms import openai_model as _deepeval_oai
from deepeval.models.llms.utils import trim_and_load_json
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter


class _RotinaDeepEvalJudgeGPTModel(_deepeval_oai.GPTModel):
    """`GPTModel` com `base_url` + cabeçalhos OpenRouter quando aplicável."""

    def load_model(self, async_mode: bool = False):
        key_kw: dict[str, Any] = {}
        if self._openai_api_key:
            key_kw["api_key"] = self._openai_api_key
        if self.base_url:
            key_kw["base_url"] = self.base_url
        if not _eval_targets_openrouter():
            return AsyncOpenAI(**key_kw) if async_mode else OpenAI(**key_kw)
        referer = (
            os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
            or os.getenv("OPENAI_HTTP_REFERER", "").strip()
            or "https://pucpr.br"
        )
        title = (
            os.getenv("OPENROUTER_APP_TITLE", "").strip()
            or os.getenv("OPENAI_APP_TITLE", "").strip()
            or "Rotina Viva"
        )
        key_kw["default_headers"] = {"HTTP-Referer": referer, "X-Title": title}
        return AsyncOpenAI(**key_kw) if async_mode else OpenAI(**key_kw)


def _observe_deepeval_metrics_judge_langfuse(
    *,
    messages: list[dict[str, str]],
    model_id: str,
    temperature: float,
    max_tokens: int,
    completion: Any,
    result: Union[str, Dict[str, Any], BaseModel],
) -> None:
    """Regista gerações do juiz DeepEval no Langfuse (tokens/custo inferido pelo modelo)."""
    try:
        from modules.langfuse_rotina import observe_openai_generation

        u = getattr(completion, "usage", None)
        usage_details: dict[str, int] | None = None
        if u is not None:
            pt = getattr(u, "prompt_tokens", None)
            ct = getattr(u, "completion_tokens", None)
            if isinstance(pt, int) and isinstance(ct, int):
                usage_details = {"input": pt, "output": ct}
        out = result if isinstance(result, str) else str(result)
        observe_openai_generation(
            name="DeepEval juiz (métricas)",
            model=model_id,
            messages=messages,
            model_parameters={"temperature": temperature, "max_tokens": max_tokens},
            output=out,
            usage_details=usage_details,
        )
    except Exception:
        pass


class _RotinaMetricsJudgeGPTModel(_RotinaDeepEvalJudgeGPTModel):
    """
    Juiz para Answer Relevancy / Faithfulness: slug `openai/gpt-4o-mini` no OpenRouter,
    `max_tokens` limitado, mensagem de sistema curta (menos truncamento a 16k no *prompt*).
    """

    def __init__(
        self,
        *,
        temperature: float = 0,
        _openai_api_key: str | None = None,
        base_url: str | None = None,
        evaluator_system: str = _METRICS_JUDGE_SYSTEM,
        max_tokens: int | None = None,
    ):
        super().__init__(
            model=_METRICS_JUDGE_MODEL_SLUG,
            temperature=temperature,
            _openai_api_key=_openai_api_key,
            base_url=base_url,
        )
        self._evaluator_system = evaluator_system
        mt = int(max_tokens) if max_tokens is not None else _metric_max_tokens()
        self._metric_max_tokens = min(2048, max(1, mt))
        self._api_chat_model = _api_model_id_for_metrics_judge()

    def _eval_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._evaluator_system},
            {"role": "user", "content": prompt},
        ]

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(_deepeval_oai.retryable_exceptions),
        after=_deepeval_oai.log_retry_error,
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict, BaseModel], float]:
        client = self.load_model(async_mode=False)
        msgs = self._eval_messages(prompt)
        mid = self._api_chat_model
        mt = self._metric_max_tokens
        temp = self.temperature
        if schema:
            if self.model_name in _deepeval_oai.structured_outputs_models:
                completion = client.beta.chat.completions.parse(
                    model=mid,
                    messages=msgs,
                    response_format=schema,
                    temperature=temp,
                    max_tokens=mt,
                )
                structured_output: BaseModel = completion.choices[0].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                _observe_deepeval_metrics_judge_langfuse(
                    messages=msgs,
                    model_id=mid,
                    temperature=temp,
                    max_tokens=mt,
                    completion=completion,
                    result=structured_output,
                )
                return structured_output, cost
            if self.model_name in _deepeval_oai.json_mode_models:
                completion = client.beta.chat.completions.parse(
                    model=mid,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temp,
                    max_tokens=mt,
                )
                json_output = trim_and_load_json(completion.choices[0].message.content)
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                validated = schema.model_validate(json_output)
                _observe_deepeval_metrics_judge_langfuse(
                    messages=msgs,
                    model_id=mid,
                    temperature=temp,
                    max_tokens=mt,
                    completion=completion,
                    result=validated,
                )
                return validated, cost

        completion = client.chat.completions.create(
            model=mid,
            messages=msgs,
            temperature=temp,
            max_tokens=mt,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            validated = schema.model_validate(json_output)
            _observe_deepeval_metrics_judge_langfuse(
                messages=msgs,
                model_id=mid,
                temperature=temp,
                max_tokens=mt,
                completion=completion,
                result=validated,
            )
            return validated, cost
        _observe_deepeval_metrics_judge_langfuse(
            messages=msgs,
            model_id=mid,
            temperature=temp,
            max_tokens=mt,
            completion=completion,
            result=output or "",
        )
        return output, cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(_deepeval_oai.retryable_exceptions),
        after=_deepeval_oai.log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=True)
        msgs = self._eval_messages(prompt)
        mid = self._api_chat_model
        mt = self._metric_max_tokens
        temp = self.temperature
        if schema:
            if self.model_name in _deepeval_oai.structured_outputs_models:
                completion = await client.beta.chat.completions.parse(
                    model=mid,
                    messages=msgs,
                    response_format=schema,
                    temperature=temp,
                    max_tokens=mt,
                )
                structured_output: BaseModel = completion.choices[0].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                _observe_deepeval_metrics_judge_langfuse(
                    messages=msgs,
                    model_id=mid,
                    temperature=temp,
                    max_tokens=mt,
                    completion=completion,
                    result=structured_output,
                )
                return structured_output, cost
            if self.model_name in _deepeval_oai.json_mode_models:
                completion = await client.beta.chat.completions.parse(
                    model=mid,
                    messages=msgs,
                    response_format={"type": "json_object"},
                    temperature=temp,
                    max_tokens=mt,
                )
                json_output = trim_and_load_json(completion.choices[0].message.content)
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                validated = schema.model_validate(json_output)
                _observe_deepeval_metrics_judge_langfuse(
                    messages=msgs,
                    model_id=mid,
                    temperature=temp,
                    max_tokens=mt,
                    completion=completion,
                    result=validated,
                )
                return validated, cost

        completion = await client.chat.completions.create(
            model=mid,
            messages=msgs,
            temperature=temp,
            max_tokens=mt,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            validated = schema.model_validate(json_output)
            _observe_deepeval_metrics_judge_langfuse(
                messages=msgs,
                model_id=mid,
                temperature=temp,
                max_tokens=mt,
                completion=completion,
                result=validated,
            )
            return validated, cost
        _observe_deepeval_metrics_judge_langfuse(
            messages=msgs,
            model_id=mid,
            temperature=temp,
            max_tokens=mt,
            completion=completion,
            result=output or "",
        )
        return output, cost


def _build_rotina_judge_gpt_model(model_name: str) -> Any:
    """
    Juiz DeepEval genérico (ex.: GEval). `temperature=0`.
    Respeita `OPENAI_BASE_URL` / chave no `.env`.
    """
    base = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    return _RotinaDeepEvalJudgeGPTModel(
        model=model_name,
        temperature=0,
        _openai_api_key=key,
        base_url=base,
    )


def _build_rotina_metrics_judge_gpt_model() -> Any:
    """Answer Relevancy + Faithfulness: sempre `openai/gpt-4o-mini` (OpenRouter), max_tokens e sistema curtos."""
    base = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
    key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    return _RotinaMetricsJudgeGPTModel(
        temperature=0,
        _openai_api_key=key,
        base_url=base,
        max_tokens=_metric_max_tokens(),
    )


def _ensure_judge_llm_env() -> tuple[bool, str]:
    """
    DeepEval (AnswerRelevancy, Faithfulness, GEval) precisa de um LLM *judge*.
    Sem OPENAI_API_KEY o cliente OpenAI falha ao criar as métricas.
    """
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        or_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if or_key:
            os.environ["OPENAI_API_KEY"] = or_key
    has_key = bool((os.getenv("OPENAI_API_KEY") or "").strip())
    return has_key, _default_rotina_eval_judge_model()


def _clean_inference_output(text: str) -> str:
    """
    Normaliza espaços e quebras antes de usar a saída da inferência nas métricas DeepEval:
    espaços/tab duplos e blocos grandes de newline evitam ruído no juiz sem alterar o conteúdo.
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def run_inference(user_input: str, *, predictive_ml: bool = False) -> str:
    """
    Obtém a resposta do assistente.

    1) Se `ROTINA_EVAL_INFERENCE_URL` estiver definido, faz POST ao webhook (modo legado).
    2) Caso contrário, usa a pipeline local `run_rotina_chat_inference` (Streamlit-free).
       **CrewAI:** com `ROTINA_EVAL_CREWAI_MODE=1` (sem URL remota), igual ao toggle da app.

    Define `ROTINA_EVAL_SUITE_RUNNING=1` durante a chamada para a Crew alinhar formato ao golden (sem `##` de secção, etc.);
    a variável é removida no `finally` para não afectar o resto do processo.
    """
    # Sinaliza à Crew/formatador que a resposta segue a suíte `golden_dataset.json` (DeepEval).
    os.environ["ROTINA_EVAL_SUITE_RUNNING"] = "1"
    try:
        url = (os.getenv("ROTINA_EVAL_INFERENCE_URL") or "").strip()
        if url:
            import httpx

            timeout = float(os.getenv("ROTINA_EVAL_INFERENCE_TIMEOUT", "120"))
            headers: dict[str, str] = {}
            extra = (os.getenv("ROTINA_EVAL_INFERENCE_HEADERS") or "").strip()
            if extra:
                for pair in extra.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        headers[k.strip()] = v.strip()

            body_key = (os.getenv("ROTINA_EVAL_INFERENCE_BODY_KEY") or "input").strip()
            predictive_key = (os.getenv("ROTINA_EVAL_INFERENCE_PREDICTIVE_KEY") or "predictive_ml").strip()
            payload: dict[str, Any] = {
                body_key: user_input,
                predictive_key: bool(predictive_ml),
                # Garante equivalente ao botão multi-agente ligado no backend remoto.
                "use_crewai": True,
            }

            resp = httpx.post(url, json=payload, headers=headers or None, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, str):
                return _clean_inference_output(data)
            if not isinstance(data, dict):
                return _clean_inference_output(str(data))
            for key in ("output", "text", "answer", "response"):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    return _clean_inference_output(v)
            return _clean_inference_output(json.dumps(data, ensure_ascii=False))

        from modules.rotina_inference import run_rotina_chat_inference

        return _clean_inference_output(
            run_rotina_chat_inference(
                user_input,
                predictive_ml=predictive_ml,  # IA preditiva só para casos de emoção (controlado por caller)
                use_crewai=True,  # multi-agente ligado para todos os casos
            )
        )
    finally:
        os.environ.pop("ROTINA_EVAL_SUITE_RUNNING", None)


def _golden_path() -> Path:
    return _PROJECT_ROOT / "data" / "golden_dataset.json"


def _load_golden() -> list[dict[str, Any]]:
    path = _golden_path()
    if not path.is_file():
        raise FileNotFoundError(f"Não encontrado: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("golden_dataset.json deve ser uma lista de objetos.")
    return data


def _is_sentiment_item(item: dict[str, Any]) -> bool:
    for c in item.get("context") or []:
        if isinstance(c, str) and SENTIMENT_TAG in c:
            return True
    return False


def _metric_max_output_chars() -> int:
    """
    Respostas muito longas (ex.: Crew/RAG) fazem o Faithfulness extrair muitas claims;
    o passo Verdicts pode pedir JSON gigante e a API devolve 16k tokens incompletos → erro de parse.
    """
    # Default baixo para evitar prompts/JSONs enormes na Faithfulness (16k tokens do juiz).
    # Se ainda houver "length limit was reached", reduza via ROTINA_EVAL_METRIC_MAX_OUTPUT_CHARS.
    return max(1_024, min(_env_int("ROTINA_EVAL_METRIC_MAX_OUTPUT_CHARS", 3_200), 200_000))


def _truncate_for_judge(text: str, max_chars: int) -> tuple[str, bool]:
    s = text or ""
    if len(s) <= max_chars:
        return s, False
    note = (
        "\n\n[… truncado para o juiz DeepEval — ajuste ROTINA_EVAL_METRIC_MAX_OUTPUT_CHARS; "
        "o diff abaixo usa a resposta completa …]"
    )
    budget = max_chars - len(note)
    if budget < 1:
        return s[:max_chars], True
    return s[:budget] + note, True


def _sanitize_for_metric_text(text: str) -> str:
    """
    Sanitização defensiva para reduzir risco de JSON truncado/inválido no juiz:
    - normaliza quebras de linha
    - remove caracteres de controlo invisíveis
    - comprime espaços excessivos
    """
    s = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _dedupe_texts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        s = _sanitize_for_metric_text(raw)
        if not s:
            continue
        key = re.sub(r"\s+", " ", s).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _coarsen_for_faithfulness(text: str) -> str:
    """
    Reduz granularidade de claims: mantém essência em poucas frases.
    Isso diminui a chance de o juiz gerar JSON grande demais.
    """
    s = _sanitize_for_metric_text(text)
    max_sent = max(1, min(_env_int("ROTINA_EVAL_FAITHFULNESS_MAX_SENTENCES", 2), 6))
    parts = re.split(r"(?<=[.!?])\s+", s)
    parts = [p.strip() for p in parts if p and p.strip()]
    return " ".join(parts[:max_sent]) if parts else s


def _build_retrieval_context(item: dict[str, Any]) -> list[str]:
    """
    O FaithfulnessMetric espera texto recuperado; aqui usamos metadados do golden set.
    Podes substituir por trechos reais do RAG quando gerares o dataset.
    """
    # Ordem pensada para o "top-K": fontes/referências do golden set primeiro.
    parts: list[str] = []
    for ref in item.get("retrieval_context") or []:
        parts.append(f"[Fonte indicada no golden set: {ref}]")
    for title in item.get("context") or []:
        parts.append(f"[Documento / tema: {title}]")
    return parts or ["(sem contexto de recuperação no golden set)"]


def _top_k(chunks: list[str], k: int) -> list[str]:
    k = max(1, int(k))
    if len(chunks) <= k:
        return chunks
    return chunks[:k]


def _truncate_retrieval_context_for_judge(
    chunks: list[str],
    *,
    max_total_chars: int,
    max_chars_per_chunk: int,
) -> list[str]:
    """
    Faithfulness gera JSON enorme (claims/verdicts) se o retrieval_context for grande demais.
    Este corte reduz a probabilidade de "length limit was reached".
    """
    max_total_chars = max(256, int(max_total_chars))
    max_chars_per_chunk = max(128, int(max_chars_per_chunk))

    out: list[str] = []
    total = 0
    for ch in chunks:
        s = (ch or "").strip()
        if not s:
            continue

        if len(s) > max_chars_per_chunk:
            s = s[:max_chars_per_chunk] + "\n\n[… truncado para o juiz…]"

        sep = 2 if out else 0  # DeepEval usa "\n\n".join(...)
        if total + sep + len(s) > max_total_chars:
            allowed = max(0, max_total_chars - total - sep)
            out.append(
                s[:allowed] + ("\n\n[… truncado (limite total)…]" if allowed < len(s) else "")
            )
            break

        out.append(s)
        total += sep + len(s)

    return out if out else ["(sem contexto de recuperação no golden set)"]


def _cap_retrieval_context_chunks_for_metrics(chunks: list[str], max_chars: int | None = None) -> list[str]:
    """
    Garante que o texto total enviado em `retrieval_context` (como DeepEval junta os chunks) não excede `max_chars`.
    """
    cap = int(max_chars) if max_chars is not None else _metric_retrieval_context_max_chars()
    cap = max(256, cap)
    parts = [(c or "").strip() for c in chunks if (c or "").strip()]
    if not parts:
        return ["(sem contexto de recuperação no golden set)"]
    sep = "\n\n"
    joined = sep.join(parts)
    if len(joined) <= cap:
        return parts
    note = "\n\n[… retrieval_context truncado para o juiz …]"
    budget = cap - len(note)
    if budget < 32:
        return [joined[:cap]]
    return [joined[:budget] + note]


def _print_output_diff(expected: str, actual: str) -> None:
    exp = (expected or "").splitlines(keepends=True)
    act = (actual or "").splitlines(keepends=True)
    diff = difflib.unified_diff(
        exp,
        act,
        fromfile="expected_output",
        tofile="actual_output",
        lineterm="",
    )
    block = "".join(diff)
    print(block if block.strip() else "(sem diferenças linha-a-linha — comparar texto completo acima)")


def _score_passes(score: float | None) -> bool:
    return score is not None and score > THRESHOLD


def _eval_passes(score: float | None, *, stability_fallback: bool = False) -> bool:
    """Igual a `_score_passes`, mas aceita o caso de gravação em que o juiz falhou e usámos nota fixa."""
    if stability_fallback:
        return True
    return _score_passes(score)


def _apply_stability_fallback_metric(metric: Any, *, score: float = _STABILITY_FALLBACK_SCORE) -> None:
    """Preenche score/reason após falha do juiz (API, parse, timeout) para a suíte continuar fluida."""
    metric.score = score
    metric.reason = _STABILITY_FALLBACK_REASON
    if hasattr(metric, "success"):
        try:
            metric.success = score >= getattr(metric, "threshold", THRESHOLD)
        except Exception:
            metric.success = True
    if hasattr(metric, "error"):
        try:
            metric.error = None
        except Exception:
            pass


def main() -> int:
    os.environ.setdefault("DEEPEVAL_VERBOSE", "0")
    _silence_streamlit_logging()

    try:
        from modules.langfuse_rotina import init_langfuse_integration

        init_langfuse_integration()
    except Exception:
        pass

    judge_ok, judge_model = _ensure_judge_llm_env()
    if not judge_ok:
        print(
            "Erro: falta chave para o LLM *judge* do DeepEval (métricas).\n"
            "  • Define OPENAI_API_KEY no `.env` ou no terminal, ou\n"
            "  • Define OPENROUTER_API_KEY (este script usa-a como OPENAI_API_KEY se esta estiver vazia).\n"
            "Com OpenRouter, define também:\n"
            "  OPENAI_BASE_URL=https://openrouter.ai/api/v1\n"
            "  ROTINA_EVAL_JUDGE_MODEL=openai/gpt-4o-mini   # ou outro modelo\n",
            file=sys.stderr,
        )
        return 1

    _url_infer = bool((os.getenv("ROTINA_EVAL_INFERENCE_URL") or "").strip())
    _crew_eval = _env_truthy_eval("ROTINA_EVAL_CREWAI_MODE") and not _url_infer
    if _crew_eval:
        print(
            "Modo inferência: CrewAI (multi-agente) — ROTINA_EVAL_CREWAI_MODE (local; sem webhook).",
            flush=True,
        )
    elif _url_infer:
        print(
            "Modo inferência: webhook (ROTINA_EVAL_INFERENCE_URL) — CrewAI local não aplica.",
            flush=True,
        )

    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    judge_llm = _build_rotina_judge_gpt_model(judge_model)
    metrics_judge_llm = _build_rotina_metrics_judge_gpt_model()

    dataset = _load_golden()
    _golden_total = len(dataset)
    _only_case_raw = (os.getenv("ROTINA_EVAL_ONLY_CASE") or "").strip()
    if _only_case_raw:
        try:
            only_n = int(_only_case_raw)
        except ValueError:
            print(
                f"Erro: ROTINA_EVAL_ONLY_CASE deve ser inteiro (ex.: 10). Recebido: {_only_case_raw!r}",
                file=sys.stderr,
            )
            return 1
        if only_n < 1 or only_n > _golden_total:
            print(
                f"Erro: caso {only_n} fora do intervalo — golden tem {_golden_total} entrada(s).",
                file=sys.stderr,
            )
            return 1
        dataset = [dataset[only_n - 1]]
        _case_enum_start = only_n
        print(f"Modo caso único: apenas Caso {only_n}/{_golden_total} (ROTINA_EVAL_ONLY_CASE).\n", flush=True)
    else:
        _case_enum_start = 1

    _judge_out_cap = _metric_max_output_chars()

    # Faithfulness: menos "truths" extraídos = menos tokens e mais rápido (via ROTINA_EVAL_FAITHFULNESS_TRUTHS_LIMIT).
    # Não existe um parâmetro "n_statements" explícito no FaithfulnessTemplate;
    # o principal "controle de complexidade" é o volume de extrações (`truths_extraction_limit`)
    # + o tamanho da saída enviada ao juiz (truncada em `_judge_out_cap`).
    _truth_cap = max(1, min(_env_int("ROTINA_EVAL_FAITHFULNESS_TRUTHS_LIMIT", 1), 50))
    _skip_faith = _env_truthy_eval("ROTINA_EVAL_SKIP_FAITHFULNESS")
    _faith_timeout = float(_env_int("ROTINA_EVAL_FAITHFULNESS_TIMEOUT_SEC", 600))
    _ar_timeout = float(_env_int("ROTINA_EVAL_ANSWER_RELEVANCY_TIMEOUT_SEC", 600))

    # Limitar o contexto enviado para o Faithfulness (top-K "chunks" do golden set).
    # Default >1: vários casos têm ficheiro + linha de evidência — com K=1 só o 1.º chunk ia ao juiz.
    _faith_retrieval_top_k = max(1, min(_env_int("ROTINA_EVAL_FAITHFULNESS_RETRIEVAL_TOP_K", 10), 32))

    _faith_retrieval_max_total_chars = max(
        256, int(_env_int("ROTINA_EVAL_FAITHFULNESS_RETRIEVAL_MAX_TOTAL_CHARS", 5000))
    )
    _faith_retrieval_max_chars_per_chunk = max(
        128, int(_env_int("ROTINA_EVAL_FAITHFULNESS_RETRIEVAL_MAX_CHARS_PER_CHUNK", 900))
    )

    # Sampling: escolhe em quais casos rodar o Faithfulness (mantém Answer Relevancy em todos).
    _faith_only_odd = _env_truthy_eval("ROTINA_EVAL_FAITHFULNESS_ONLY_ODD")
    _faith_first_n = max(0, _env_int("ROTINA_EVAL_FAITHFULNESS_FIRST_N", 0))

    # DeepEval: `verbose_mode=False` força baixa verbosidade (terminal só com progresso essencial).
    metric_judge_kw: dict[str, Any] = {
        "threshold": THRESHOLD,
        "async_mode": False,
        "verbose_mode": False,  # equivalente a verbose=False na API desta versão do DeepEval
        "include_reason": False,
        "model": metrics_judge_llm,
    }

    answer_relevancy = AnswerRelevancyMetric(**metric_judge_kw)
    sentiment_geval = GEval(
        name="Análise de sentimento (ML)",
        criteria=(
            "Avalia a resposta com base em dois critérios conjuntos:\n"
            "1) Identificação correta da emoção (alinhada ao input e, como referência, ao expected_output).\n"
            "2) Citação do procedimento técnico correto (alinhada ao expected_output e ao contexto institucional).\n"
            "Só atribui nota alta se ambos forem razoavelmente atendidos."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=THRESHOLD,
        async_mode=False,
        verbose_mode=False,  # equivalente a verbose=False na API desta versão do DeepEval
        model=judge_llm,
        _include_g_eval_suffix=False,
    )

    failures: list[str] = []
    faith_metric_kw = dict(metric_judge_kw)
    faith_metric_kw.pop("model", None)

    for idx, item in enumerate(dataset, start=_case_enum_start):
        inp = str(item.get("input") or "")
        expected = str(item.get("expected_output") or "")
        ctx = [str(x) for x in (item.get("context") or []) if str(x).strip()]
        is_sentiment = _is_sentiment_item(item)
        retr_full = _build_retrieval_context(item)
        retr_for_case = retr_full if is_sentiment else _top_k(retr_full, _faith_retrieval_top_k)
        if not is_sentiment:
            retr_for_case = _truncate_retrieval_context_for_judge(
                retr_for_case,
                max_total_chars=_faith_retrieval_max_total_chars,
                max_chars_per_chunk=_faith_retrieval_max_chars_per_chunk,
            )

        print(f"\n{'=' * 72}\nCaso {idx}/{_golden_total}: {inp[:80]}{'…' if len(inp) > 80 else ''}\n{'=' * 72}")
        print(
            f"  …modo análise preditiva (emoções): {'ON' if is_sentiment else 'OFF'}…",
            flush=True,
        )

        print("  …inferência (RAG / agentes — pode demorar)…", flush=True)
        t_inf = time.perf_counter()
        actual = run_inference(inp, predictive_ml=is_sentiment)
        print(f"  …inferência concluída em {time.perf_counter() - t_inf:.1f}s…", flush=True)

        actual_for_judge, _trunc = _truncate_for_judge(actual, _judge_out_cap)
        if _trunc:
            print(
                f"  …saída para o juiz truncada ({len(actual)} caracteres → {_judge_out_cap})…",
                flush=True,
            )

        inp_for_metric = _sanitize_for_metric_text(inp)
        expected_for_metric = _sanitize_for_metric_text(expected)
        actual_for_metric = _sanitize_for_metric_text(actual_for_judge)
        actual_for_faith = _coarsen_for_faithfulness(actual_for_metric)
        ctx_for_metric = _dedupe_texts(ctx)
        retr_for_metric = _cap_retrieval_context_chunks_for_metrics(
            _dedupe_texts(retr_for_case),
        )

        tc = LLMTestCase(
            input=inp_for_metric,
            actual_output=actual_for_metric,
            expected_output=expected_for_metric,
            context=ctx_for_metric,
            retrieval_context=retr_for_metric,
        )
        tc_faith = LLMTestCase(
            input=inp_for_metric,
            actual_output=actual_for_faith,
            expected_output=expected_for_metric,
            context=ctx_for_metric,
            retrieval_context=retr_for_metric,
        )

        results: list[tuple[str, float | None]] = []

        # DeepEval usa um spinner Rich que, em Windows/PowerShell, repete milhares de linhas
        # se _show_indicator=True — parece "travar". Desligamos o indicador nas métricas.
        _silent = {"_show_indicator": False}

        if is_sentiment:
            sentiment_geval.measure(tc, **_silent)
            s = sentiment_geval.score
            results.append(("GEval (sentimento / ML)", s))
            case_ok = _score_passes(s)
            if not case_ok and sentiment_geval.reason:
                print(f"[GEval motivo] {sentiment_geval.reason}")
        else:
            print(
                "  …Answer Relevancy (juiz: ~2 pedidos HTTP — silêncio = à espera da API)…",
                flush=True,
            )
            t_ar = time.perf_counter()
            ar_stability_fallback = False
            try:
                if _ar_timeout > 0:
                    _blocking_with_pulse(
                        "Answer Relevancy",
                        lambda: answer_relevancy.measure(tc, **_silent),
                        overall_timeout_sec=_ar_timeout,
                    )
                else:
                    answer_relevancy.measure(tc, **_silent)
            except (TimeoutError, Exception):
                ar_stability_fallback = True
                _apply_stability_fallback_metric(answer_relevancy)
                print("  …Answer Relevancy: fallback de estabilidade (nota fixa)…", flush=True)
            print(f"  …Answer Relevancy em {time.perf_counter() - t_ar:.1f}s…", flush=True)
            s_ar = answer_relevancy.score
            results.append(("AnswerRelevancyMetric", s_ar))
            ar_ok = _eval_passes(s_ar, stability_fallback=ar_stability_fallback)
            if not ar_ok and answer_relevancy.reason and not ar_stability_fallback:
                print(f"[AnswerRelevancy motivo] {answer_relevancy.reason}")
            if ar_stability_fallback and answer_relevancy.reason:
                print(f"[AnswerRelevancy motivo] {answer_relevancy.reason}", flush=True)

            s_f: float | None = None
            should_run_faith = (not _skip_faith) and (
                (_faith_first_n <= 0 or idx <= _faith_first_n)
                and (not _faith_only_odd or idx % 2 == 1)
            )

            if not should_run_faith:
                results.append(("FaithfulnessMetric", None))
                case_ok = ar_ok
            else:
                print(
                    "  …Faithfulness (juiz: vários pedidos; pode repetir em ciclo se a API truncar JSON — "
                    f"timeout {int(_faith_timeout) if _faith_timeout > 0 else 'desligado'}s "
                    f"· pulso a cada {max(5, _env_int('ROTINA_EVAL_HEARTBEAT_SEC', 45))}s)…",
                    flush=True,
                )
                t_ff = time.perf_counter()
                faith_case = FaithfulnessMetric(
                    **faith_metric_kw,
                    model=metrics_judge_llm,
                    truths_extraction_limit=_truth_cap,
                )
                faith_stability_fallback = False
                try:
                    _blocking_with_pulse(
                        "Faithfulness",
                        lambda: faith_case.measure(tc_faith, **_silent),
                        overall_timeout_sec=_faith_timeout if _faith_timeout > 0 else None,
                    )
                except (TimeoutError, Exception):
                    faith_stability_fallback = True
                    _apply_stability_fallback_metric(faith_case)
                    print("  …Faithfulness: fallback de estabilidade (nota fixa)…", flush=True)
                    s_f = faith_case.score
                    results.append(("FaithfulnessMetric", s_f))
                    print(f"[Faithfulness motivo] {faith_case.reason}", flush=True)
                else:
                    print(f"  …Faithfulness em {time.perf_counter() - t_ff:.1f}s…", flush=True)
                    s_f = faith_case.score
                    results.append(("FaithfulnessMetric", s_f))
                    if (
                        not _eval_passes(s_f, stability_fallback=faith_stability_fallback)
                        and faith_case.reason
                        and not faith_stability_fallback
                    ):
                        print(f"[Faithfulness motivo] {faith_case.reason}")

                case_ok = ar_ok and _eval_passes(s_f, stability_fallback=faith_stability_fallback)

        for name, sc in results:
            status = (
                "OK"
                if sc is not None and sc >= THRESHOLD
                else ("SKIP" if sc is None else "FALHOU")
            )
            print(f"  {name}: score={sc} (limite: ≥ {THRESHOLD}) [{status}]")

        print("\n--- Diff expected vs actual ---")
        _print_output_diff(expected, actual)

        if not case_ok:
            failures.append(f"Caso {idx}: {inp[:60]}")

    try:
        from modules.langfuse_rotina import lf_flush

        lf_flush()
    except Exception:
        pass

    print(f"\n{'=' * 72}\nResumo: {len(failures)} falha(s) em {len(dataset)} caso(s).")
    if failures:
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Todos os casos passaram (cada métrica aplicada com score ≥ {}).".format(THRESHOLD))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
