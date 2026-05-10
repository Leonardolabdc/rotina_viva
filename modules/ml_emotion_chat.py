"""
Integração do bundle emotion (.pkl) no chat: inferência local + contexto para a LLM.
Comando: prefixo numa linha (ex. `/emotion`) seguido de uma ou mais linhas de texto.
O modelo ML foi treinado em inglês; em modo `auto`/`always` o backend pode traduzir via LLM antes da inferência.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from modules import ai_engine
from modules.ml_traditional_emotion import unpickle_bundle

_bundle_cache: dict[str, tuple[int, Any, tuple[str, ...]]] = {}

# `auto`: só chama a LLM para traduzir quando o texto parece português (ou misto).
# `always`: sempre pede tradução/reformulação em inglês antes do ML.
# `never`: nunca chama a LLM; o utilizador deve escrever em inglês para melhor resultado.
_EMOTION_TRANSLATE_MODES = frozenset({"auto", "always", "never"})

_PT_CHARS_RE = re.compile(
    r"[ãáàâäéèêëíìîïóòôöõúùûüçÃÁÀÂÄÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÇ]"
)
_PT_WORD_RE = re.compile(
    r"\b(não|nao|com|para|por|uma|uns|umas|estou|está|esta|este|estamos|muito|hoje|"
    r"minha|meu|nosso|feliz|triste|raiva|medo|tenho|tem|tinha|exausta|exausto|cansad[oa]|"
    r"radiante|conseguir|alguma|pouco|pouca|sempre|nunca|coisa|também|tambem|"
    r"quando|como|onde|porque|porquê|pq|você|vc|tb|obrigad[oa]|gostaria|"
    r"criança|escola|professor[ae]?)\b",
    re.IGNORECASE,
)

# Partir uma linha com várias frases (mesmo parágrafo) — uma predição por frase.
_EMOTION_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")

# Quando o interruptor "IA preditiva" está ligado, evitamos contaminar perguntas
# factuais (cardápio, turma, horário, protocolo etc.) com o addon de emoções.
_PREDICTIVE_FACTUAL_HINT_RE = re.compile(
    r"\b("
    r"card[áa]pio|almo[çc]o|cafe\s+da\s+manha|merenda|lanche|jantar|"
    r"turma|matr[ií]cula|alergia|protocolo|hor[áa]rio|funcionamento|"
    r"regimento|ppp|manual|guia|csv|documento|tabela|base\s+de\s+dados|"
    r"qual|quando|onde|quem|quanto|tem|qual\s+[ée]"
    r")\b",
    re.IGNORECASE,
)

_PREDICTIVE_EMOTION_HINT_RE = re.compile(
    r"\b("
    r"emo[cç][aã]o|sentimento|sentiu|sentiu|est[áa]\s+(triste|feliz|nervos[oa]|ansios[oa]|irritad[oa]|"
    r"com\s+medo|chatead[oa]|frustrad[oa]|calm[oa])|"
    r"chorou|gritou|bateu|raiva|medo|alegria|tristeza|frustra[cç][aã]o|"
    r"engajamento|exclus[aã]o"
    r")\b",
    re.IGNORECASE,
)

# Relatos de agressão física: o classificador em inglês pode devolver «sadness»; o gabarito pede raiva/frustração.
_AGGRESSION_PHYSICAL_PT_RE = re.compile(
    r"(?is)\b(bateu|bater\s+no|empurrou|empurrar|socos?|socou|agrediu|agress[aã]o)\b",
)


def expand_emotion_lines(lines: list[str], *, max_segments: int = 40) -> list[str]:
    """
    Cada linha do utilizador pode conter várias frases separadas por `.` / `!` / `?`.
    O classificador foi treinado em **frases curtas** (estilo tweet); um parágrafo inteiro
    produz predições pouco fiáveis (muitas vezes puxa para uma única classe, ex. *joy*).
    """
    out: list[str] = []
    for raw in lines:
        s = (raw or "").strip()
        if not s:
            continue
        chunks = [
            c.strip() for c in _EMOTION_SENTENCE_SPLIT.split(s) if c and c.strip()
        ]
        if len(chunks) >= 2:
            out.extend(chunks)
        else:
            out.append(s)
        if len(out) >= max_segments:
            return out[:max_segments]
    return out[:max_segments]


def _emotion_translate_mode() -> str:
    raw = (os.getenv("ROTINA_EMOTION_TRANSLATE") or "auto").strip().lower()
    return raw if raw in _EMOTION_TRANSLATE_MODES else "auto"


def _emotion_translate_fallback_enabled() -> bool:
    """Se a LLM falhar ou devolver PT, tenta traduzir com `deep-translator` (rede externa)."""
    raw = (os.getenv("ROTINA_EMOTION_TRANSLATE_FALLBACK") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _fallback_translate_lines_to_english(lines: list[str]) -> list[str] | None:
    """
    Traduz cada linha para inglês via Google (pacote `deep-translator`), sem chave API.

    Falha silenciosamente (None) se o pacote não estiver instalado, não houver rede,
    ou o serviço limitar / bloquear pedidos.
    """
    if not lines:
        return None
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return None
    translator = GoogleTranslator(source="auto", target="en")
    out: list[str] = []
    for i, ln in enumerate(lines):
        s = (ln or "").strip()
        if not s:
            return None
        try:
            out.append(translator.translate(s))
        except Exception:
            return None
        if i < len(lines) - 1:
            time.sleep(float(os.getenv("ROTINA_EMOTION_FALLBACK_DELAY_SEC", "0.35")))
    return out if len(out) == len(lines) else None


def lines_likely_non_english(lines: list[str]) -> bool:
    """Heurística barata: se parecer PT (ou não-inglês óbvio), pedimos tradução em `auto`."""
    blob = "\n".join(lines)
    if _PT_CHARS_RE.search(blob):
        return True
    if _PT_WORD_RE.search(blob):
        return True
    return False


def predictive_message_looks_emotional(user_text: str) -> bool:
    """
    Heurística para o modo "IA preditiva" contínuo:
    - Se houver sinais claros de emoção/comportamento, aplica ML.
    - Se for claramente consulta factual, não injeta addon ML.
    """
    text = (user_text or "").strip()
    if not text:
        return False
    if _PREDICTIVE_EMOTION_HINT_RE.search(text):
        return True
    if _PREDICTIVE_FACTUAL_HINT_RE.search(text):
        return False
    # fallback conservador para evitar enviesar perguntas objetivas.
    return False

# Prefixos que disparam inferência (comparação case-insensitive só no início da mensagem)
_EMOTION_TRIGGERS: tuple[str, ...] = (
    "/emotion",
    "/emocao",
    "/emoção",
    "classificar emoção:",
    "classificar emocao:",
    "predição de emoção:",
    "predicao de emocao:",
    "emotion:",
    "emotion ml:",
)

# Pedido explícito de gravar cadastro/diário na mesma mensagem (senão não escrevemos CSV no fluxo ML).
_STRUCTURED_WRITE_INTENT_RE = re.compile(
    r"(?is)"
    r"(?:^|\s)#(?:guardar|gravar|persistir|salvar)\b"
    r"|(?:"
    r"\b(?:guarda(?:r)?|grav(?:ar|e)|salva(?:r)?|regist(?:rar|ra|re)|"
    r"insere(?:ir|r)?|atualiza(?:r)?|cadastra(?:r)?|anota(?:r)?)\b)"
    r".{0,160}?"
    r"\b(?:no|na|em)\s+(?:o\s+)?(?:di[aá]rio|cadastro|\bcsv\b|di[aá]rios?|planilha)"
    r"|"
    r"\b(?:no|na|em)\s+(?:o\s+)?(?:di[aá]rio|cadastro|\bcsv\b)"
    r".{0,120}?"
    r"\b(?:guarda(?:r)?|grav(?:ar|e)|salva(?:r)?|regist(?:rar|ra|re)|insere(?:ir|r)?)\b"
)


def emotion_command_requests_structured_persist(user_message: str) -> bool:
    """True se o texto pede gravar no diário/cadastro/CSV (na mesma mensagem que o comando ML)."""
    return bool(_STRUCTURED_WRITE_INTENT_RE.search(user_message or ""))


def chat_round_suppress_csv_mutations(
    user_message: str, *, predictive_session: bool
) -> bool:
    """
    Rodada em modo ML (comando `/emotion` **ou** interruptor «IA preditiva») sem pedido explícito
    de gravar: não aplicar INSERT/UPDATE/DELETE em `info_alunos` / `diario_estruturado`.
    """
    if emotion_command_requests_structured_persist(user_message or ""):
        return False
    if parse_emotion_command(user_message or "")[0] is not None:
        return True
    if predictive_session:
        return True
    return False


def emotion_command_suppress_csv_mutations(user_message: str) -> bool:
    """Compatível com chamadas antigas (só comando, sem interruptor de sessão)."""
    return chat_round_suppress_csv_mutations(
        user_message, predictive_session=False
    )


def resolve_emotion_pkl_path(data_dir: Path) -> Path | None:
    """
    Caminho do `.pkl`: `ROTINA_EMOTION_PKL` (absoluto ou relativo a `DATA_DIR`), senão ficheiros comuns.
    """
    raw = (os.getenv("ROTINA_EMOTION_PKL") or "").strip()
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else (Path(data_dir) / p).resolve())
    candidates.extend(
        [
            Path(data_dir) / "ml_models" / "emotion_flaml_bundle.pkl",
            Path(data_dir) / "ml_models" / "emotion_flaml_tfidf_bundle.pkl",
            Path("/data/ml_models/emotion_flaml_bundle.pkl"),
        ]
    )
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    return None


def load_emotion_bundle_for_path(pkl_path: Path) -> tuple[Any | None, list[str]]:
    """
    Carrega o EmotionMLBundle com cache em memória invalidado pelo mtime do ficheiro.

    Devolve ``(bundle, avisos)``; ``avisos`` inclui avisos de versão do sklearn ao deserializar.
    """
    key = str(pkl_path.resolve())
    try:
        st = pkl_path.stat()
        mt = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except OSError:
        return None, []
    hit = _bundle_cache.get(key)
    if hit is not None and hit[0] == mt:
        if len(hit) >= 3:
            return hit[1], list(hit[2])
        return hit[1], []
    load_warnings: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b = unpickle_bundle(pkl_path.read_bytes())
        for w in caught:
            msg = str(w.message).strip()
            if msg and msg not in load_warnings:
                load_warnings.append(msg)
    except Exception:
        return None, []
    wtuple = tuple(load_warnings)
    _bundle_cache[key] = (mt, b, wtuple)
    return b, list(wtuple)


def parse_emotion_command(user_text: str) -> tuple[list[str] | None, str]:
    """
    Devolve ``(None, _)`` se não for comando ML.

    Se for comando: ``([], texto)`` = falta texto a classificar; ``([...], _)`` = linhas a inferir.
    """
    raw = (user_text or "").strip()
    if not raw:
        return None, raw
    first, *rest_lines = raw.split("\n", 1)
    first_s = first.strip()
    for tr in _EMOTION_TRIGGERS:
        m = re.match(re.escape(tr) + r"\s*(.*)", first_s, flags=re.IGNORECASE)
        if not m:
            continue
        suffix = (m.group(1) or "").strip()
        rest = (rest_lines[0] if rest_lines else "").strip()
        combined = "\n".join(x for x in (suffix, rest) if x).strip()
        lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
        return (lines, combined)
    return None, raw


def build_emotion_ml_llm_addon(
    user_text: str,
    data_dir: Path,
    *,
    predictive_session: bool = False,
) -> str:
    """
    Texto (Markdown) a acrescentar ao `extra_system` do chat para a LLM usar as predições do ML.
    String vazia se não houver comando `/emotion` (etc.) nem sessão **IA preditiva** ligada.
    """
    cmd_lines, _ = parse_emotion_command(user_text)
    if cmd_lines is not None:
        lines = cmd_lines
    elif predictive_session:
        if not predictive_message_looks_emotional(user_text):
            return ""
        raw = (user_text or "").strip()
        if not raw:
            return ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            lines = [raw]
    else:
        return ""

    if cmd_lines is not None and not lines:
        return (
            "### Modelo ML de emoções (pedido pelo utilizador)\n"
            "O prefixo de comando foi reconhecido, mas **não há frases** para classificar. "
            "Peça ao utilizador para repetir com **uma frase por linha** após `/emotion` "
            "(português ou inglês; em português a app pode traduzir automaticamente para o modelo).\n"
        )

    lines = expand_emotion_lines(lines)
    if not lines:
        return (
            "### Modelo ML de emoções (pedido pelo utilizador)\n"
            "Não restou texto válido após partir frases (verifique pontuação).\n"
        )

    pkl = resolve_emotion_pkl_path(Path(data_dir))
    if pkl is None:
        return (
            "### Modelo ML de emoções (pedido pelo utilizador)\n"
            "Não foi encontrado nenhum ficheiro `.pkl` válido. "
            f"Coloque o modelo em `{Path(data_dir) / 'ml_models' / 'emotion_flaml_bundle.pkl'}` "
            "ou defina a variável de ambiente **`ROTINA_EMOTION_PKL`** com o caminho completo ou relativo a `ROTINA_DATA_DIR`.\n\n"
            "Explica isto ao utilizador em **português** e sugere treinar/guardar o modelo no laboratório **ML clássico**.\n"
        )

    bundle, pickle_warnings = load_emotion_bundle_for_path(pkl)
    if bundle is None:
        return (
            "### Modelo ML de emoções (pedido pelo utilizador)\n"
            f"O ficheiro `{pkl}` existe mas **não foi possível carregar** o pickle (versão Python/sklearn incompatível ou ficheiro corrompido). "
            "Explica o problema em **português** de forma simples.\n"
        )

    mode = _emotion_translate_mode()
    user_lines = list(lines)
    translation_warning: str | None = None
    used_llm_translate = False
    used_google_fallback = False
    post_translate_pt_warning: str | None = None

    if mode == "never":
        lines_for_ml = user_lines
    elif mode == "always" or (mode == "auto" and lines_likely_non_english(user_lines)):
        en_try, tr_err = ai_engine.llm_translate_plain_lines_to_english(user_lines)
        used_llm_translate = True
        if en_try is not None:
            lines_for_ml = en_try
            if any(lines_likely_non_english([e]) for e in en_try):
                post_translate_pt_warning = (
                    "A tradução via LLM ainda devolveu linhas com padrões de **português**."
                )
        else:
            lines_for_ml = user_lines
            translation_warning = tr_err

        if _emotion_translate_fallback_enabled() and (
            translation_warning is not None
            or (
                en_try is not None
                and any(lines_likely_non_english([e]) for e in en_try)
            )
        ):
            g_lines = _fallback_translate_lines_to_english(user_lines)
            if g_lines is not None:
                lines_for_ml = g_lines
                used_google_fallback = True
                translation_warning = None
                post_translate_pt_warning = None
            elif translation_warning is None and post_translate_pt_warning:
                post_translate_pt_warning += (
                    " O **fallback** Google (`deep-translator`) não conseguiu traduzir "
                    "(rede, bloqueio ou pacote em falta); defina `ROTINA_EMOTION_TRANSLATE_FALLBACK=0` "
                    "se não quiser este passo."
                )
    else:
        lines_for_ml = user_lines

    try:
        ids, names = bundle.predict_labels(lines_for_ml)
    except Exception as e:
        return (
            "### Modelo ML de emoções (pedido pelo utilizador)\n"
            f"A inferência falhou: `{e!s}`\n\n"
            "Explica em **português** sem inventar resultados.\n"
        )

    df = pd.DataFrame(
        {
            "texto_utilizador": user_lines,
            "texto_inferencia_en": lines_for_ml,
            "id_classe": ids,
            "emoção": names,
        }
    )
    table = "```csv\n" + df.to_csv(index=False) + "```"

    aggression_override = ""
    user_blob = "\n".join(user_lines)
    if _AGGRESSION_PHYSICAL_PT_RE.search(user_blob):
        aggression_override = (
            "\n\n---\n**Prioridade pedagógica (sobre a etiqueta ML):** o texto descreve **agressão física** "
            "(p.ex. «bateu no colega»). Para análise e resposta final, trate como **raiva ou frustração** e "
            "**intervenção imediata e mediação de conflito conforme o Regimento Interno**, mesmo que a coluna "
            "**`emoção`** acima mostre **tristeza** ou outra classe pouco adequada ao contexto.\n"
        )

    tr_blurb = ""
    if used_google_fallback:
        tr_blurb = (
            "- **Tradução:** usou-se o **fallback** (`deep-translator` / Google), "
            "porque a LLM do chat falhou ou devolveu texto ainda em português. "
            "Os dados saem do serviço público de tradução — desative com `ROTINA_EMOTION_TRANSLATE_FALLBACK=0` "
            "se não for aceitável (privacidade / rede).\n"
        )
    elif used_llm_translate and translation_warning is None:
        tr_blurb = (
            "- **Tradução:** o backend pediu à **mesma LLM** do chat uma versão em **inglês** "
            "de cada linha antes da inferência (preserva o sentido emocional o melhor possível).\n"
        )
        if post_translate_pt_warning:
            tr_blurb += f"- **Aviso pós-tradução:** {post_translate_pt_warning}\n"
    elif translation_warning:
        tr_blurb = (
            f"- **Tradução:** não foi possível traduzir automaticamente ({translation_warning}) "
            "— a inferência usou o **texto original**; os rótulos podem ser menos fiáveis se não for inglês.\n"
        )
    elif mode == "never":
        tr_blurb = (
            "- **Tradução:** desativada (`ROTINA_EMOTION_TRANSLATE=never`); o texto foi passado **direto** ao modelo.\n"
        )
        if lines_likely_non_english(user_lines):
            tr_blurb += (
                "- **Aviso:** o texto parece **português** — sem tradução para inglês o classificador "
                "costuma devolver **rótulos errados** (ex.: tudo *joy*). Use `auto` ou `always`, ou escreva em inglês.\n"
            )
    else:
        tr_blurb = (
            "- **Tradução:** heurística `auto` considerou o texto **já utilizável em inglês** "
            "(sem chamada extra à LLM).\n"
        )

    split_blurb = (
        "- **Frases:** o texto foi **partido por `.` `!` `?` …** quando vinha tudo na mesma linha — "
        "o modelo espera **frases curtas**; um parágrafo inteiro costuma dar uma única classe errada.\n"
    )

    pickle_blurb = ""
    if pickle_warnings:
        joined = " ".join(w.replace("\n", " ")[:400] for w in pickle_warnings[:2])
        pickle_blurb = (
            f"- **Pickle / sklearn:** ao carregar o `.pkl` o Python emitiu avisos (ex.: versão do scikit-learn). "
            f"Resumo: {joined}\n"
            "  Recomenda-se **`scikit-learn` 1.8.x** igual ao ambiente de treino; reinstale dependências e reinicie a app.\n"
        )

    persist_blurb = ""
    if not emotion_command_requests_structured_persist(user_text):
        persist_blurb = (
            "\n\n---\n**Política (ML de emoções):** O utilizador **não** pediu explicitamente "
            "gravar no diário, cadastro ou CSV nesta mensagem (interruptor **IA preditiva** ou `/emotion`). "
            "**Não** digas que os dados foram guardados em ficheiros; **não** descrevas `INSERT`/`UPDATE` "
            "em `diario_estruturado` ou `info_alunos`. Limita-te à classificação e à interpretação pedagógica.\n"
        )

    return (
        "### Modelo ML de emoções (factos para a tua resposta)\n"
        "O backend executou **inferência local** com o bundle treinado (TF-IDF + classificador). "
        "Trata a tabela abaixo como **resultado do modelo**, não como opinião tua sem esse contexto.\n"
        f"{split_blurb}"
        f"{pickle_blurb}"
        "- A coluna **`texto_utilizador`** é o que a pessoa escreveu; **`texto_inferencia_en`** é o que entrou no classificador.\n"
        f"{tr_blurb}"
        "- O classificador foi treinado em **inglês** (dataset dair-ai/emotion); tradução automática reduz mas não elimina erro.\n"
        "- Responde ao utilizador em **português**: relaciona cada frase original à emoção prevista; "
        "se as etiquetas de emoção estiverem em inglês, podes **traduzir o nome** da categoria ao explicar.\n\n"
        f"**Ficheiro usado:** `{pkl}`\n\n"
        f"{table}\n"
        f"{aggression_override}"
        f"{persist_blurb}"
    )


def emotion_command_help_caption() -> str:
    return (
        "**IA preditiva** (interruptor ao lado do chat): ligado = cada mensagem é classificação ML até desligar "
        "(não precisa de `/emotion`). **Sem** pedido explícito de gravar no diário/cadastro/CSV, nada é escrito "
        "nos ficheiros; para guardar, diga p.ex. «gravar no diário» na mesma mensagem. "
        "Atalho opcional: **`/emotion`** + texto. Frases na mesma linha partidas por `.` `!` `?`; PT → EN via LLM; "
        "**fallback** Google (`deep-translator`, `ROTINA_EMOTION_TRANSLATE_FALLBACK=1`). "
        "`ROTINA_EMOTION_TRANSLATE` auto/always/never."
    )
