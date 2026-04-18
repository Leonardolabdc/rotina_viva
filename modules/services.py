"""Serviços de domínio: relatório sono/alimentação (DuckDB, pandas, Altair) e acesso ao módulo de chat."""
from __future__ import annotations

import os
import re
from typing import Any

import altair as alt
import duckdb
import pandas as pd

from modules import chat_service


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Mesmas variáveis de ambiente que o app (teto e faixas de sono no relatório).
if os.getenv("ROTINA_SONO_MAX_MIN", "").strip():
    ROTINA_SONO_MAX_MIN = max(15.0, min(180.0, _env_float("ROTINA_SONO_MAX_MIN", 60.0)))
else:
    ROTINA_SONO_MAX_MIN = max(15.0, min(180.0, _env_float("ROTINA_SONO_REFERENCIA_MIN", 60.0)))
ROTINA_SONO_FAIXA_LIMITE_1 = max(
    1.0, min(ROTINA_SONO_MAX_MIN - 2.0, _env_float("ROTINA_SONO_FAIXA_LIMITE_1", 20.0))
)
ROTINA_SONO_FAIXA_LIMITE_2 = max(
    ROTINA_SONO_FAIXA_LIMITE_1 + 1.0,
    min(ROTINA_SONO_MAX_MIN - 1.0, _env_float("ROTINA_SONO_FAIXA_LIMITE_2", 40.0)),
)

SONO_QUAL_BASTANTE = "Dormiu bastante"
SONO_QUAL_POUCO = "Dormiu pouco"
SONO_QUAL_PADRAO = (
    f"Dormiu normal ({int(round(ROTINA_SONO_FAIXA_LIMITE_1))}–"
    f"{int(round(ROTINA_SONO_FAIXA_LIMITE_2))} min)"
)

_MEAL_SCORE_MAP: dict[str, int] = {
    "comeu bem": 2,
    "comeu pouco": 1,
    "recusou": 0,
}


def _meal_score_cell(val: Any) -> int:
    k = str(val or "").strip().lower()
    return _MEAL_SCORE_MAP.get(k, 1)


def _parse_clock_to_minutes(s: Any) -> int | None:
    raw = str(s or "").strip()
    if not raw:
        return None
    raw = raw.replace(".", ":")
    parts = raw.split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0].strip())
        m = int(parts[1].strip()[:2])
        return max(0, min(23, h)) * 60 + max(0, min(59, m))
    except ValueError:
        return None


def _sleep_minutes_between(start: Any, end: Any) -> int | None:
    """
    Duração em minutos entre dois horários do mesmo dia (relógio 24 h).

    Usa o **menor** arco entre os dois instantes no círculo de 24 h, para que
    `inicio=13:00` e `fim=12:00` (colunas trocadas vs. 12:00→13:00) resulte em
    **60 min**, e não ~23 h (bug que marcava todos como “Dormiu bastante”).
    Ignora durações irreais para soneca escolar (> 4 h).
    """
    a = _parse_clock_to_minutes(start)
    b = _parse_clock_to_minutes(end)
    if a is None or b is None:
        return None
    d = (b - a) % (24 * 60)
    if d == 0:
        return None
    d_short = min(d, 24 * 60 - d)
    if d_short <= 0:
        return None
    if d_short > 240:
        return None
    return int(d_short)


def _classify_sono_min_for_report(sono_min: float) -> str:
    """
    Classifica minutos de sono para o relatório: valores são limitados a `ROTINA_SONO_MAX_MIN`.
    Faixas: [0, L1) pouco, [L1, L2) normal, [L2, teto] bastante (`ROTINA_SONO_FAIXA_LIMITE_*`).
    """
    cap = float(ROTINA_SONO_MAX_MIN)
    l1 = float(ROTINA_SONO_FAIXA_LIMITE_1)
    l2 = float(ROTINA_SONO_FAIXA_LIMITE_2)
    m = max(0.0, min(float(sono_min), cap))
    if m < l1:
        return SONO_QUAL_POUCO
    if m < l2:
        return SONO_QUAL_PADRAO
    return SONO_QUAL_BASTANTE


def _normalize_qualidade_sono_val(raw: str, sono_min: Any) -> str:
    """
    Reduz `qualidade_sono` do CSV a uma das três categorias canônicas.
    Prioridade: texto reconhecível → inferência por minutos (horários início/fim) → "—".
    """
    t = (raw or "").strip().lower()
    if t in ("—", "-", "", "nan", "none"):
        t = ""

    exact = {
        SONO_QUAL_BASTANTE.lower(): SONO_QUAL_BASTANTE,
        SONO_QUAL_POUCO.lower(): SONO_QUAL_POUCO,
        SONO_QUAL_PADRAO.lower(): SONO_QUAL_PADRAO,
    }
    if t in exact:
        return exact[t]
    if t.startswith("sono no padrão") or t.startswith("sono no padrao"):
        return SONO_QUAL_PADRAO

    # Variações comuns / legado (ordem: sinais de sono ruim curto, depois pouco, bastante, padrão).
    if any(
        k in t
        for k in (
            "acordou agit",
            "muito agit",
            "sono agit",
            "interrup",
            "chorou muito",
            "acordou cedo",
            "não dormiu",
            "nao dormiu",
            "quase não dorm",
        )
    ):
        return SONO_QUAL_POUCO
    if any(
        k in t
        for k in (
            "dormiu pouco",
            "pouco sono",
            "sono curto",
            "descanso curto",
            "dormiu mal",
        )
    ):
        return SONO_QUAL_POUCO
    if any(
        k in t
        for k in (
            "dormiu bastante",
            "bastante sono",
            "sono longo",
            "dormiu bem",
            "dormiu tranquilo",
            "sono tranquilo",
            "bom sono",
            "descanso bom",
            "dormiu o dia",
        )
    ):
        return SONO_QUAL_BASTANTE
    if any(
        k in t
        for k in (
            "dormiu normal",
            "no padrão",
            "no padrao",
            "padrão da",
            "padrao da",
            "média da semana",
            "media da semana",
            "sono normal",
            "sono moderado",
            "sono regular",
            "sono leve",
            "padrão (~60",
            "padrao (~60",
        )
    ):
        return SONO_QUAL_PADRAO

    sm: float | None
    try:
        sm = float(sono_min) if sono_min is not None and pd.notna(sono_min) else None
    except (TypeError, ValueError):
        sm = None
    if sm is not None and sm > 0:
        if sm <= 240:
            return _classify_sono_min_for_report(sm)
        return "—"
    return "—"


def _week_slice_by_dates_only(df_in: pd.DataFrame, end_day: pd.Timestamp) -> pd.DataFrame:
    """Janela de 7 dias corridos terminando em `end_day`, sem exigir sono válido."""
    start_day = end_day - pd.Timedelta(days=6)
    return df_in[
        (df_in["data_dt"].dt.normalize() >= start_day)
        & (df_in["data_dt"].dt.normalize() <= end_day)
    ].copy()


def build_sleep_meal_report_dataframe(
    conn: duckdb.DuckDBPyConnection,
    student_name: str,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    str | None,
    str,
    str | None,
    tuple[str, str] | None,
]:
    """
    Relatório só com CSV no DuckDB. Filtra pelo nome do aluno (ILIKE), depois:
    - usa **todos** os registos do diário na **janela de 7 dias corridos** que termina na **data mais recente**
      desse(s) aluno(s) (não remove dias só porque o sono não tem horários válidos — refeições e texto contam);
    - agrega por **dia** (média de ingestão e de minutos de sono quando existirem; dias sem intervalo de sono
      ficam com minutos em falta no gráfico de tendência);
    - preserva textos de **cafe_manha**, **almoco**, **lanche_tarde** por dia (moda) para gráfico de barras.

    Retorno: (df resumo diário, df refeições textuais por dia, erro|None, rótulo alunos,
              aviso_semana|None, (início_iso, fim_iso)|None).
    """
    name = (student_name or "").strip()
    if not name:
        return None, None, "Informe o nome do aluno.", "", None, None
    try:
        id_rows = conn.execute(
            "SELECT id_aluno, nome FROM info_alunos WHERE nome ILIKE ? ORDER BY nome",
            [f"%{name}%"],
        ).fetchall()
    except Exception as e:
        return None, None, str(e), "", None, None
    if not id_rows:
        return (
            None,
            None,
            f'Nenhum aluno encontrado com nome parecido com “{name}”.',
            "",
            None,
            None,
        )
    ids = [int(r[0]) for r in id_rows]
    resolved = ", ".join(sorted({str(r[1]) for r in id_rows}))
    ph = ",".join(["?"] * len(ids))
    try:
        cur = conn.execute(
            f"""
            SELECT cafe_manha, almoco, lanche_tarde, jantar_extra,
                   hora_sono_inicio, hora_sono_fim, qualidade_sono, data, id_aluno
            FROM diario_estruturado
            WHERE id_aluno IN ({ph})
            """,
            ids,
        )
        df = cur.fetchdf()
    except Exception as e:
        return None, None, str(e), resolved, None, None
    if df is None or df.empty:
        return (
            None,
            None,
            f"Sem registros de diário para: {resolved}.",
            resolved,
            None,
            None,
        )

    if "data" not in df.columns:
        return None, None, "Coluna `data` ausente no diário.", resolved, None, None

    meal_cols = ["cafe_manha", "almoco", "lanche_tarde", "jantar_extra"]
    meal_txt_cols = ["cafe_manha", "almoco", "lanche_tarde"]
    for c in meal_cols:
        if c not in df.columns:
            return None, None, f"Coluna ausente: {c}", resolved, None, None
    for c in meal_txt_cols:
        df[f"{c}_txt"] = df[c].fillna("—").astype(str).str.strip()
    for c in meal_cols:
        df[c] = df[c].map(_meal_score_cell)
    df["ingestao"] = df[meal_cols].sum(axis=1).astype(int)

    df["sono_min"] = df.apply(
        lambda r: _sleep_minutes_between(
            r.get("hora_sono_inicio"), r.get("hora_sono_fim")
        ),
        axis=1,
    )
    df["qualidade_sono"] = df["qualidade_sono"].fillna("—").astype(str).str.strip()

    df["data_dt"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data_dt"])
    if df.empty:
        return (
            None,
            None,
            "Nenhuma data válida na coluna `data` do diário.",
            resolved,
            None,
            None,
        )

    d_latest = df["data_dt"].dt.normalize().max()
    d_max_day = d_latest
    d_min_day = d_max_day - pd.Timedelta(days=6)
    # Incluir todas as linhas da semana (até 7 datas distintas): não excluir dias só com refeições.
    df_w = _week_slice_by_dates_only(df, d_max_day)
    if df_w.empty:
        return (
            None,
            None,
            f"Sem linhas de diário entre **{d_min_day.date()}** e **{d_max_day.date()}** "
            f"para: {resolved}. Confirme o nome no cadastro e as datas registadas.",
            resolved,
            None,
            (str(d_min_day.date()), str(d_max_day.date())),
        )

    df_w["dia"] = df_w["data_dt"].dt.normalize()

    _cap = float(ROTINA_SONO_MAX_MIN)
    df_w["sono_min"] = pd.to_numeric(df_w["sono_min"], errors="coerce").clip(lower=0, upper=_cap)

    df_w["qualidade_sono"] = [
        _normalize_qualidade_sono_val(str(q), sm)
        for q, sm in zip(df_w["qualidade_sono"], df_w["sono_min"])
    ]

    def _qual_pick(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iloc[0]) if len(m) > 0 else str(s.iloc[0])

    def _txt_pick(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iloc[0]) if len(m) > 0 else str(s.iloc[0])

    daily = df_w.groupby("dia", as_index=False).agg(
        ingestao=("ingestao", "mean"),
        sono_min=("sono_min", "mean"),
        qualidade_sono=("qualidade_sono", _qual_pick),
        cafe_manha=("cafe_manha_txt", _txt_pick),
        almoco=("almoco_txt", _txt_pick),
        lanche_tarde=("lanche_tarde_txt", _txt_pick),
    )
    daily["ingestao"] = daily["ingestao"].round(2)
    daily["sono_min"] = daily["sono_min"].round(1)
    daily_meals = daily[["dia", "cafe_manha", "almoco", "lanche_tarde"]].copy()
    daily = daily.drop(columns=["cafe_manha", "almoco", "lanche_tarde"])

    n_dias = len(daily)
    periodo = (str(d_min_day.date()), str(d_max_day.date()))

    if n_dias < 1:
        err = (
            f"Sem dias com dados agregados entre **{d_min_day.date()}** e **{d_max_day.date()}** "
            f"para: {resolved}."
        )
        return (None, None, err, resolved, None, periodo)

    return daily, daily_meals, None, resolved, None, periodo


def _sleep_line_chart_altair(daily: pd.DataFrame) -> alt.Chart:
    """Tendência de horas + linha no teto do relatório; pontos coloridos pela classificação (CSV / faixas)."""
    cap_m = float(ROTINA_SONO_MAX_MIN)
    cap_h = cap_m / 60.0
    d = daily.sort_values("dia").copy()
    sm_plot = pd.to_numeric(d["sono_min"], errors="coerce")
    if sm_plot.notna().sum() == 0 or float(sm_plot.fillna(0).max()) <= 0:
        msg = (
            "Sem horas de sono neste período: preencha **hora_sono_inicio** e **hora_sono_fim** no diário "
            "(intervalo válido de 1 min a 4 h) para ver a tendência de horas."
        )
        ph = pd.DataFrame({"x": [0], "y": [0], "t": [msg]})
        return (
            alt.Chart(ph)
            .mark_text(
                align="left",
                baseline="top",
                dx=8,
                dy=8,
                lineHeight=18,
                fontSize=13,
                color="#566573",
            )
            .encode(
                x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=(-0.5, 8))),
                y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=(1, 0))),
                text=alt.Text("t:N"),
            )
            .properties(height=220, width=560)
        )
    d["data_show"] = d["dia"].map(lambda x: pd.Timestamp(x).strftime("%d/%m"))
    lk = _daily_sleep_lookup_for_ref_table(daily)
    m = d.merge(lk[["data_show", "vs_escola"]], on="data_show", how="left")
    line_df = pd.DataFrame(
        {
            "Data": pd.to_datetime(m["dia"], errors="coerce").dt.normalize(),
            "Horas de sono": (pd.to_numeric(m["sono_min"], errors="coerce") / 60.0).round(2),
            "Classificação": m["vs_escola"].fillna("—").astype(str).str.strip(),
        }
    )
    _cmap = {
        SONO_QUAL_POUCO: "#c0392b",
        SONO_QUAL_PADRAO: "#2980b9",
        SONO_QUAL_BASTANTE: "#1a9850",
        "—": "#aeb6bf",
    }
    _order = [SONO_QUAL_POUCO, SONO_QUAL_PADRAO, SONO_QUAL_BASTANTE, "—"]
    _present = [c for c in _order if c in set(line_df["Classificação"])]
    for c in sorted(line_df["Classificação"].unique()):
        if c not in _present:
            _present.append(c)
    _colors = [_cmap.get(c, "#7f8c8d") for c in _present]

    base = alt.Chart(line_df).encode(
        x=alt.X(
            "Data:T",
            title="Data",
            axis=alt.Axis(format="%d/%m", labelAngle=0),
        ),
        y=alt.Y("Horas de sono:Q", title="Horas de sono"),
    )
    line = base.mark_line(strokeWidth=2, color="#34495e", interpolate="monotone")
    pts = base.mark_point(filled=True, size=95, stroke="white", strokeWidth=1).encode(
        color=alt.Color(
            "Classificação:N",
            title="Classificação",
            scale=alt.Scale(domain=_present, range=_colors),
            legend=alt.Legend(orient="bottom", direction="horizontal", labelLimit=0),
        ),
        tooltip=[
            alt.Tooltip("Data:T", title="Data", format="%d/%m/%Y"),
            alt.Tooltip("Horas de sono", title="Horas", format=".2f"),
            alt.Tooltip("Classificação", title="Classificação"),
        ],
    )
    rule_df = pd.DataFrame({"hora_teto": [cap_h]})
    rule = (
        alt.Chart(rule_df)
        .mark_rule(
            color="#e67e22",
            strokeDash=[6, 4],
            strokeWidth=2,
        )
        .encode(
            y=alt.Y("hora_teto:Q", title="Horas de sono"),
            tooltip=alt.value(f"Teto do relatório ({cap_m:.0f} min)"),
        )
    )
    return (line + pts + rule).properties(height=280).interactive()


def _vs_escola_from_csv_or_minutos(qualidade: Any, sono_min: Any) -> str:
    """
    Texto da coluna “Vs. referência escola”: prioriza `qualidade_sono` já normalizada
    no agregado diário (veio do CSV). Só usa minutos se o CSV não tiver categoria.
    """
    q = str(qualidade if qualidade is not None else "").strip()
    if q and q not in ("—", "-", "nan", "none", "NaN"):
        return q
    try:
        sm = float(sono_min)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(sm) or sm <= 0:
        return "—"
    return _classify_sono_min_for_report(sm)


def _daily_sleep_lookup_for_ref_table(daily: pd.DataFrame | None) -> pd.DataFrame:
    """
    Uma linha por dia do relatório, mesma chave `data_show` (dd/mm) usada no gráfico de refeições.

    - **min_diario:** média dos minutos já limitada ao teto (`ROTINA_SONO_MAX_MIN`).
    - **vs_escola:** **`qualidade_sono` do CSV** (normalizada por dia); se vazio, classifica pelos minutos (mesmo teto).
    """
    empty = pd.DataFrame(columns=["data_show", "min_diario", "vs_escola"])
    if daily is None or daily.empty or "sono_min" not in daily.columns:
        return empty
    d = daily.sort_values("dia").copy()
    d["data_show"] = d["dia"].map(lambda x: pd.Timestamp(x).strftime("%d/%m"))
    sm = pd.to_numeric(d["sono_min"], errors="coerce")
    d["min_diario"] = sm.round().astype("Int64")
    quals = d["qualidade_sono"] if "qualidade_sono" in d.columns else pd.Series([""] * len(d))
    d["vs_escola"] = [
        _vs_escola_from_csv_or_minutos(q, smv)
        for q, smv in zip(quals, sm.astype(float))
    ]
    out = d[["data_show", "min_diario", "vs_escola"]].drop_duplicates(subset=["data_show"])
    return out


def _sleep_reference_table_ui(daily: pd.DataFrame | None) -> pd.DataFrame:
    """Uma linha por dia: minutos no diário + classificação (mesma lógica do gráfico de sono)."""
    lk = _daily_sleep_lookup_for_ref_table(daily)
    if lk.empty:
        return pd.DataFrame(columns=["Dia", "Min. sono (diário)", "Classificação"])
    return pd.DataFrame(
        {
            "Dia": lk["data_show"],
            "Min. sono (diário)": lk["min_diario"].map(
                lambda x: f"{int(x)} min" if pd.notna(x) else "—"
            ),
            "Classificação": lk["vs_escola"].astype(str),
        }
    )


def _norm_intake_status_for_chart(raw: str) -> str:
    """Reduz textos livres a poucas categorias para cores consistentes na legenda."""
    t = str(raw or "").strip().lower()
    if "recusou" in t:
        return "Recusou"
    if "comeu bem" in t or "comeu tudo" in t:
        return "Comeu bem"
    if "comeu pouco" in t:
        return "Comeu pouco"
    if t in ("—", "-", "", "nan", "none"):
        return "Sem registro"
    return "Outro"


def _meal_intake_stacked_bar_altair(
    daily_meals: pd.DataFrame,
) -> tuple[alt.Chart, pd.DataFrame]:
    """
    Barras empilhadas (café → almoço → lanche, de baixo para cima), cor = classificação da ingestão.
    Retorna o gráfico e a tabela **só de refeições** (Dia, Refeição, texto). A tabela de sono fica à parte.
    """
    meal_pt = {
        "cafe_manha": "Café da manhã",
        "almoco": "Almoço",
        "lanche_tarde": "Lanche",
    }
    meal_ord = {"cafe_manha": 0, "almoco": 1, "lanche_tarde": 2}
    dm = daily_meals.sort_values("dia")
    if dm.empty:
        empty = pd.DataFrame()
        return alt.Chart(empty).mark_bar(), empty

    day_order = sorted(dm["dia"].unique(), key=lambda x: pd.Timestamp(x))
    sort_labels = [pd.Timestamp(d).strftime("%d/%m") for d in day_order]

    rows: list[dict[str, str | float | int]] = []
    for _, r in dm.iterrows():
        d_ts = pd.Timestamp(r["dia"])
        d_show = d_ts.strftime("%d/%m")
        for col, pt in meal_pt.items():
            registro = str(r[col]).strip() or "—"
            status = _norm_intake_status_for_chart(registro)
            rows.append(
                {
                    "data_show": d_show,
                    "fatia": 1.0,
                    "ordem": meal_ord[col],
                    "refeicao": pt,
                    "status": status,
                    "registro": registro,
                }
            )
    bar_df = pd.DataFrame(rows)

    domain_all = ["Comeu bem", "Comeu pouco", "Recusou", "Sem registro", "Outro"]
    range_all = ["#1a9850", "#f4a020", "#c0392b", "#aeb6bf", "#5d6d7e"]
    present = [s for s in domain_all if s in set(bar_df["status"])]
    colors = [range_all[domain_all.index(s)] for s in present]

    chart = (
        alt.Chart(bar_df)
        .mark_bar(cornerRadiusEnd=2, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "data_show:N",
                title="Dia",
                sort=sort_labels,
                axis=alt.Axis(labelAngle=0, labelOverlap=False),
            ),
            y=alt.Y(
                "fatia:Q",
                stack="zero",
                title=None,
                axis=None,
            ),
            color=alt.Color(
                "status:N",
                title="Ingestão (legenda)",
                scale=alt.Scale(domain=present, range=colors),
                legend=alt.Legend(
                    orient="bottom",
                    direction="horizontal",
                    columns=3,
                    labelLimit=0,
                    labelFontSize=12,
                    titleFontSize=12,
                    padding=12,
                    symbolSize=80,
                ),
            ),
            order=alt.Order("ordem:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("data_show", title="Dia"),
                alt.Tooltip("refeicao", title="Refeição"),
                alt.Tooltip("registro", title="Texto registrado"),
                alt.Tooltip("status", title="Classificação"),
            ],
        )
        .properties(height=260)
    )

    ref_meals = bar_df.rename(
        columns={
            "data_show": "Dia",
            "refeicao": "Refeição",
            "registro": "Texto registrado",
        }
    )[["Dia", "Refeição", "Texto registrado"]]
    return chart, ref_meals


def sleep_meal_report_summary_md(
    df: pd.DataFrame,
    alunos_label: str,
    janela_ini: str | None,
    janela_fim: str | None,
) -> str:
    """Texto curto com leitura dos números (minimalista)."""
    n = len(df)
    mi = float(df["ingestao"].mean())
    _ms = df["sono_min"].mean()
    if n and pd.notna(_ms):
        sono_resumo = f"média **{float(_ms):.0f} min** por dia"
    else:
        sono_resumo = (
            "**sem** média de minutos (faltam horários de início e fim de sono nos registos desta janela)"
        )
    mx_ing = float(df["ingestao"].max()) if n else 0.0
    mn_ing = float(df["ingestao"].min()) if n else 0.0

    parts = [
        "**Base de informação:** dados oficiais de cadastro e de rotina diária da instituição, "
        "tratados de forma confidencial e protegidos pelas normas aplicáveis, inclusive quanto a "
        "direitos autorais e privacidade.",
        f"**Aluno(s) considerado(s):** {alunos_label}.",
    ]
    if janela_ini and janela_fim:
        parts.append(
            f"**Janela semanal:** 7 dias corridos de **{janela_ini}** a **{janela_fim}** "
            f"(terminando na data mais recente do diário). **{n}** dia(s) com registro válido nesse intervalo."
        )
    else:
        parts.append(f"**{n}** dia(s) com dados na janela (refeições e/ou sono).")
    cap_m = int(round(ROTINA_SONO_MAX_MIN))
    l1 = int(round(ROTINA_SONO_FAIXA_LIMITE_1))
    l2 = int(round(ROTINA_SONO_FAIXA_LIMITE_2))
    parts.append(
        f"Ingestão combinada (quatro momentos, 0–8): média **{mi:.1f}** "
        f"(mín. {mn_ing:.1f}, máx. {mx_ing:.1f}). "
        f"Sono (horários do CSV, **no máximo {cap_m} min** por registro): {sono_resumo}. "
        f"**Classificação:** **pouco** abaixo de {l1} min, **normal** entre {l1} e {l2} min, **bastante** a partir de {l2} min "
        f"(até {cap_m} min). **Qualidade do sono** segue três categorias canônicas alinhadas a essas faixas."
    )
    try:
        _dfq = df[df["qualidade_sono"].astype(str).str.strip() != "—"]
        if _dfq.empty:
            by_q = pd.Series(dtype=float)
        else:
            by_q = _dfq.groupby("qualidade_sono", dropna=False)["ingestao"].mean().sort_values(
                ascending=False
            )
        if len(by_q) > 1:
            top = by_q.index[0]
            parts.append(
                f"Entre essas **categorias de sono**, a maior média de ingestão aparece em “{top}”. "
                "Isso descreve o conjunto de dados, não causa médica ou pedagógica."
            )
        elif len(by_q) == 1:
            parts.append(
                "Todos os dias deste recorte compartilham a mesma categoria de sono após padronização; "
                "compare outras semanas ou turmas para ver tendências."
            )
    except Exception:
        pass
    return "\n\n".join(parts)


# API pública (UI importa estes nomes)
sleep_line_chart_altair = _sleep_line_chart_altair
meal_intake_stacked_bar_altair = _meal_intake_stacked_bar_altair
sleep_reference_table_df = _sleep_reference_table_ui

__all__ = [
    "chat_service",
    "ROTINA_SONO_MAX_MIN",
    "ROTINA_SONO_FAIXA_LIMITE_1",
    "ROTINA_SONO_FAIXA_LIMITE_2",
    "build_sleep_meal_report_dataframe",
    "sleep_line_chart_altair",
    "meal_intake_stacked_bar_altair",
    "sleep_reference_table_df",
    "sleep_meal_report_summary_md",
]
