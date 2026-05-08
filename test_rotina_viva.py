"""
Avaliação offline do Rotina Viva com DeepEval.

Pré-requisitos:
  - **Julgador (métricas):** `OPENAI_API_KEY` ou `OPENROUTER_API_KEY`. Por defeito o juiz usa o modelo
    **gpt-4o-mini** (Answer Relevancy / GEval via `ROTINA_EVAL_JUDGE_MODEL`). A **Faithfulness** usa por defeito
    também **gpt-4o-mini** (`ROTINA_EVAL_FAITHFULNESS_MODEL` só se quiseres outro modelo só para esta métrica).
  - **Inferência:** `modules.rotina_inference.run_rotina_chat_inference` — mesmo fluxo que o chat (`.env` igual à app).

**Multi-agente (CrewAI)** — equivalente a ligar o checkbox na sidebar da app:
  - Definir **`ROTINA_EVAL_CREWAI_MODE=1`** (ou `true` / `yes`) antes de correr o script.
  - Exige `crewai`, provedor compatível com OpenAI (`ROTINA_CHAT_PROVIDER=openai` ou `openrouter`, chave API), como na UI.
  - **Não** aplica se usares **`ROTINA_EVAL_INFERENCE_URL`**: aí a resposta vem só do servidor remoto.

**Limites do juiz:** respostas do assistente muito longas podem fazer o Faithfulness pedir JSON enorme à API e falhar com
`length limit was reached` (ex.: 16384 tokens). O script trunca `actual_output` só para as métricas (**`ROTINA_EVAL_METRIC_MAX_OUTPUT_CHARS`**, default 12000); o diff continua com o texto completo.

**Faithfulness lento / “sem fim”:** o DeepEval volta a chamar a API sem limite de tentativas nalguns erros (`LengthFinishReasonError`, rate limit, etc.). Por defeito há **`ROTINA_EVAL_FAITHFULNESS_TIMEOUT_SEC=300`** (5 min); **`0`** desactiva o limite. Ou **`ROTINA_EVAL_SKIP_FAITHFULNESS=1`** para não correr esta métrica.

**Custo:** o maior gasto costuma ser o **juiz** (2 métricas × N casos, mais GEval nos casos de sentimento).
Reduz custo com modelo juiz barato (`gpt-4o-mini`), evitando `gpt-4o` nas métricas.

**Avisos no terminal:** tentamos silenciar o Streamlit (`logging` + `warnings`); alguma linha residual pode aparecer e pode ignorar-se em CLI.

Execução (na pasta do projeto):
  python test_rotina_viva.py

Só um caso do golden set (índice 1-based = «Caso N/N» na saída), no PowerShell:
  $env:ROTINA_EVAL_ONLY_CASE="10"; python test_rotina_viva.py

Com CrewAI (multi-agente), no PowerShell:
  $env:ROTINA_EVAL_CREWAI_MODE="1"; python test_rotina_viva.py
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
from typing import Any, TypeVar

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

# Raiz do projeto no path (para `import modules` mesmo com cwd diferente)
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

THRESHOLD = 0.7
SENTIMENT_TAG = "Análise de Sentimento"


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
    sec = interval_sec if interval_sec is not None else float(max(5, _env_int("ROTINA_EVAL_HEARTBEAT_SEC", 20)))
    if overall_timeout_sec is None or overall_timeout_sec <= 0:
        stop = threading.Event()

        def pulse() -> None:
            elapsed = 0.0
            while True:
                if stop.wait(sec):
                    break
                elapsed += sec
                print(
                    f"  …{label}: juiz ainda a trabalhar (~{elapsed:.0f}s — típico para Faithfulness)…",
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
            print(f"  …{label}: juiz ainda a trabalhar (~{elapsed:.0f}s)…", flush=True)
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
    judge_model = (os.getenv("ROTINA_EVAL_JUDGE_MODEL") or "").strip() or "gpt-4o-mini"
    return has_key, judge_model


def run_inference(user_input: str, *, predictive_ml: bool = False) -> str:
    """
    Obtém a resposta do assistente.

    1) Se `ROTINA_EVAL_INFERENCE_URL` estiver definido, faz POST ao webhook (modo legado).
    2) Caso contrário, usa a pipeline local `run_rotina_chat_inference` (Streamlit-free).
       **CrewAI:** com `ROTINA_EVAL_CREWAI_MODE=1` (sem URL remota), igual ao toggle da app.
    """
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
            return data
        if not isinstance(data, dict):
            return str(data)
        for key in ("output", "text", "answer", "response"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return json.dumps(data, ensure_ascii=False)

    from modules.rotina_inference import run_rotina_chat_inference

    return run_rotina_chat_inference(
        user_input,
        predictive_ml=predictive_ml,  # IA preditiva só para casos de emoção (controlado por caller)
        use_crewai=True,  # multi-agente ligado para todos os casos
    )


def _golden_path() -> Path:
    return Path(__file__).resolve().parent / "golden_dataset.json"


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
    # Default mais baixo para evitar prompts/JSONs enormes na Faithfulness.
    # Se ainda houver "length limit was reached", reduza via ROTINA_EVAL_METRIC_MAX_OUTPUT_CHARS.
    return max(1_024, min(_env_int("ROTINA_EVAL_METRIC_MAX_OUTPUT_CHARS", 4_000), 200_000))


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


def main() -> int:
    os.environ.setdefault("DEEPEVAL_VERBOSE", "0")
    _silence_streamlit_logging()

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
    _faith_timeout = float(_env_int("ROTINA_EVAL_FAITHFULNESS_TIMEOUT_SEC", 300))

    # Limitar o contexto enviado para o Faithfulness (top-K "chunks" do golden set).
    _faith_retrieval_top_k = max(1, min(_env_int("ROTINA_EVAL_FAITHFULNESS_RETRIEVAL_TOP_K", 1), 10))

    _faith_retrieval_max_total_chars = max(
        256, int(_env_int("ROTINA_EVAL_FAITHFULNESS_RETRIEVAL_MAX_TOTAL_CHARS", 2000))
    )
    _faith_retrieval_max_chars_per_chunk = max(
        128, int(_env_int("ROTINA_EVAL_FAITHFULNESS_RETRIEVAL_MAX_CHARS_PER_CHUNK", 900))
    )

    # Sampling: escolhe em quais casos rodar o Faithfulness (mantém Answer Relevancy em todos).
    _faith_only_odd = _env_truthy_eval("ROTINA_EVAL_FAITHFULNESS_ONLY_ODD")
    _faith_first_n = max(0, _env_int("ROTINA_EVAL_FAITHFULNESS_FIRST_N", 0))

    # Faithfulness: juiz rápido. Default ``gpt-4o-mini`` (evita timeouts vs. modelos maiores).
    # Override: ``ROTINA_EVAL_FAITHFULNESS_MODEL`` (ex.: slug OpenRouter), senão não herda ``ROTINA_EVAL_JUDGE_MODEL``.
    _faith_judge_model = (os.getenv("ROTINA_EVAL_FAITHFULNESS_MODEL") or "").strip() or "gpt-4o-mini"

    metric_kw: dict[str, Any] = {
        "threshold": THRESHOLD,
        "async_mode": False,
        "verbose_mode": False,
        "include_reason": False,
        "model": judge_model,
    }

    answer_relevancy = AnswerRelevancyMetric(**metric_kw)
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
        verbose_mode=False,
        model=judge_model,
    )

    failures: list[str] = []
    # Faithfulness usa um juiz próprio (mais rápido). Não duplicar `model`.
    faith_metric_kw = dict(metric_kw)
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
        retr_for_metric = _dedupe_texts(retr_for_case)

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
            answer_relevancy.measure(tc, **_silent)
            print(f"  …Answer Relevancy em {time.perf_counter() - t_ar:.1f}s…", flush=True)
            s_ar = answer_relevancy.score
            results.append(("AnswerRelevancyMetric", s_ar))
            if not _score_passes(s_ar) and answer_relevancy.reason:
                print(f"[AnswerRelevancy motivo] {answer_relevancy.reason}")

            s_f: float | None = None
            should_run_faith = (not _skip_faith) and (
                (_faith_first_n <= 0 or idx <= _faith_first_n)
                and (not _faith_only_odd or idx % 2 == 1)
            )

            if not should_run_faith:
                results.append(("FaithfulnessMetric", None))
                case_ok = _score_passes(s_ar)
            else:
                print(
                    "  …Faithfulness (juiz: vários pedidos; pode repetir em ciclo se a API truncar JSON — "
                    f"timeout {int(_faith_timeout) if _faith_timeout > 0 else 'desligado'}s "
                    f"· pulso a cada {max(5, _env_int('ROTINA_EVAL_HEARTBEAT_SEC', 20))}s)…",
                    flush=True,
                )
                t_ff = time.perf_counter()
                faith_case = FaithfulnessMetric(
                    **faith_metric_kw,
                    model=_faith_judge_model,
                    truths_extraction_limit=_truth_cap,
                )
                try:
                    _blocking_with_pulse(
                        "Faithfulness",
                        lambda: faith_case.measure(tc_faith, **_silent),
                        overall_timeout_sec=_faith_timeout if _faith_timeout > 0 else None,
                    )
                except TimeoutError:
                    print(
                        "  FaithfulnessMetric: sem score (timeout). Caso falha para efeito de resumo.",
                        flush=True,
                    )
                    results.append(("FaithfulnessMetric", None))
                    case_ok = False
                except Exception as faith_exc:
                    # Ex.: o modelo juiz devolve JSON truncado quando bate "length limit reached".
                    # Não queremos abortar a suíte inteira.
                    print(
                        f"  FaithfulnessMetric: erro ao avaliar ({faith_exc!s}). Marcando como falha parcial.",
                        flush=True,
                    )
                    results.append(("FaithfulnessMetric", None))
                    case_ok = False
                else:
                    print(f"  …Faithfulness em {time.perf_counter() - t_ff:.1f}s…", flush=True)
                    s_f = faith_case.score
                    results.append(("FaithfulnessMetric", s_f))
                    if not _score_passes(s_f) and faith_case.reason:
                        print(f"[Faithfulness motivo] {faith_case.reason}")

                    case_ok = _score_passes(s_ar) and _score_passes(s_f)

        for name, sc in results:
            status = "OK" if sc is not None and sc > THRESHOLD else ("SKIP" if sc is None else "FALHOU")
            print(f"  {name}: score={sc} (limite: > {THRESHOLD}) [{status}]")

        print("\n--- Diff expected vs actual ---")
        _print_output_diff(expected, actual)

        if not case_ok:
            failures.append(f"Caso {idx}: {inp[:60]}")

    print(f"\n{'=' * 72}\nResumo: {len(failures)} falha(s) em {len(dataset)} caso(s).")
    if failures:
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Todos os casos passaram (cada métrica aplicada com score > {}).".format(THRESHOLD))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
