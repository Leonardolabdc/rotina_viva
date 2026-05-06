"""Laboratório de ML clássico (FLAML + dataset emotion) — Streamlit."""

from __future__ import annotations

import json
import pickle
from typing import Any

import pandas as pd
import streamlit as st

from core.database import DATA_DIR
from modules.emotion_dataset_catalog import catalog_dict, emotion_dataset_card_markdown
from modules.ml_traditional_emotion import (
    EmotionMLBundle,
    evaluate_on_test,
    load_emotion_split_pandas,
    make_emotion_bundle,
    pickle_bundle,
    save_bundle_pkl_to_data_dir,
    train_flaml_emotion_pipeline,
    unpickle_bundle,
)


ML_TAB_PARAMETER_GUIDE = """
### TF-IDF
- **max_features**: número máximo de tokens no vocabulário (os mais frequentes no treino). Valores maiores captam mais nuance mas aumentam RAM e tempo.
- **ngram_range (1, N)**: N=1 só palavras isoladas; N=2 inclui pares consecutivos (“not bad”); N=3 trios. N>1 ajuda em expressões curtas mas aumenta dimensão.
- **min_df**: ignora tokens que aparecem em menos de N documentos (ou fração, se <1). Aumentar remove ruído raro; demasiado alto esvazia o vocabulário.
- **max_df**: ignora tokens demasiado frequentes (ex.: “the”). Reduz se quiseres cortar palavras muito genéricas.
- **sublinear_tf**: aplica log ao TF para suavizar contagens muito altas (comum em texto).
- **Fração do train**: usa só uma amostra aleatória do split `train` do Hugging Face para protótipos rápidos; 1,0 = todas as linhas.

### Holdout (treino vs validação interna)
- **Percentagem para treino**: parte do `train` (já subamostrado) que serve para **ajustar o TF-IDF e treinar** cada candidato. O restante é **validação** onde o FLAML mede a métrica escolhida. *Não* é o split `test` oficial do dataset.

### FLAML
- **Métrica otimizada**: o AutoML **minimiza o erro** derivado desta métrica na validação interna (ex.: para acurácia, minimiza 1−acurácia).
- **time_budget**: tempo máximo (s) para procurar hiperparâmetros entre os estimadores selecionados.
- **estimator_list**: famílias de modelos candidatos (sobre a matriz esparsa TF-IDF). Lineares são o padrão forte em texto; árvores podem ser mais pesadas.
- **random seed**: reprodutibilidade do holdout estratificado e de partes aleatórias dos learners.
- **FLAML n_jobs**: paralelismo interno do FLAML (−1 = todos os núcleos).

### Progresso (trials)
- Cada **trial** é um ensaio com um estimador e hiperparâmetros. **perda_validação** é o valor que o FLAML compara entre trials (transformação da métrica). A tabela só aparece **após** o treino terminar; o Streamlit não atualiza a UI linha-a-linha durante o `fit` síncrono.
"""


def _metrics_for_display(metrics: dict[str, Any]) -> dict[str, Any]:
    """Evita `st.json` gigante ou não serializável (trunca repr de config)."""
    out = dict(metrics)
    br = out.get("best_config_repr")
    if isinstance(br, str) and len(br) > 2_500:
        out["best_config_repr"] = br[:2_500] + "… [truncado]"
    return out


def _deps_ok() -> tuple[bool, str]:
    try:
        from flaml import AutoML  # noqa: F401
        import datasets  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as e:
        return False, str(e)
    return True, ""


@st.cache_data(show_spinner="A carregar dair-ai/emotion (primeira vez pode demorar)…")
def _cached_emotion_frames() -> dict[str, object]:
    return load_emotion_split_pandas()


def render_ml_traditional_page() -> None:
    st.subheader("ML clássico — classificação de emoções (FLAML)")
    st.caption(
        "Dataset [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) · "
        "TF-IDF (hiperparâmetros manuais) + AutoML FLAML no classificador · exportação `.pkl`."
    )
    b_back, _ = st.columns([1, 4])
    with b_back:
        if st.button("← Assistente IA", key="rotina_ml_back_assistant_btn", help="Volta ao chat principal."):
            st.session_state.rotina_sidebar_screen = "assistant"
            if "rotina_assistant_view_choice" in st.session_state:
                st.session_state.rotina_assistant_view_choice = st.session_state.get(
                    "data_source_mode", "auto"
                )
            st.rerun()

    ok, err = _deps_ok()
    if not ok:
        st.error(
            "Faltam dependências para este laboratório. Instale no ambiente do projeto, por exemplo:"
        )
        st.code("pip install \"flaml[automl]\" datasets scikit-learn", language="bash")
        st.caption(f"Detalhe: {err}")
        return

    tab_data, tab_train, tab_model = st.tabs(
        ["Dados e dicionário", "Treino (FLAML)", "Pickle e novas predições"]
    )

    with tab_data:
        st.markdown(emotion_dataset_card_markdown())
        with st.expander("Catálogo (JSON para inspeção)", expanded=False):
            st.json(catalog_dict())
        try:
            dfs = _cached_emotion_frames()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Train", len(dfs["train"]))
            with c2:
                st.metric("Validation", len(dfs["validation"]))
            with c3:
                st.metric("Test", len(dfs["test"]))
            _tr = dfs["train"]
            if len(_tr) == 0:
                st.warning("O split `train` veio vazio (inesperado para este dataset).")
            else:
                st.dataframe(_tr.head(12), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Não foi possível carregar o dataset: {e}")

    with tab_train:
        st.markdown(
            "**Fluxo preparado:** a partir do split `train` do Hugging Face (com a subamostra que definir), "
            "um **holdout estratificado** separa treino vs validação conforme o **split_ratio** (barra acima). "
            "O TF-IDF ajusta-se **só no treino**; o FLAML otimiza no holdout; no fim podes avaliar no `test` oficial."
        )
        with st.expander("O que significa cada parâmetro?", expanded=False):
            st.markdown(ML_TAB_PARAMETER_GUIDE)
        dfs = None
        try:
            dfs = _cached_emotion_frames()
        except Exception as e:
            st.error(str(e))
            return

        st.divider()
        st.markdown("##### TF-IDF (pré-processamento)")
        c1, c2, c3 = st.columns(3)
        with c1:
            tfidf_max_features = st.number_input(
                "max_features",
                min_value=500,
                max_value=100_000,
                value=20_000,
                step=500,
                help="Teto de vocabulário após ordenação por frequência.",
            )
        with c2:
            tfidf_ngram_max = st.slider("ngram_range (1, N)", 1, 3, 2)
        with c3:
            tfidf_min_df = st.number_input("min_df", min_value=1, max_value=50, value=1)

        c4, c5, c6 = st.columns(3)
        with c4:
            tfidf_max_df = st.slider("max_df", 0.5, 1.0, 0.95, 0.05)
        with c5:
            tfidf_sublinear = st.checkbox("sublinear_tf (log)", value=True)
        with c6:
            sample_fraction = st.slider(
                "Fração do train (prototipagem)",
                0.05,
                1.0,
                1.0,
                0.05,
                help="< 1.0 acelera iterações; use 1.0 para resultado final.",
            )

        st.divider()
        st.markdown("##### Holdout (split_ratio, estilo FLAML)")
        train_pct = st.slider(
            "Percentagem do `train` (após subamostra) usada em **treino**",
            min_value=50,
            max_value=95,
            value=80,
            step=5,
            key="rotina_ml_train_pct_slider",
            help="O restante fica para **validação** durante o AutoML (holdout estratificado). "
            "Equivale ao `split_ratio` do FLAML: fração para validação = 1 − (treino %).",
        )
        split_ratio = (100.0 - float(train_pct)) / 100.0
        st.caption(
            f"Treino ≈ **{train_pct}%** das linhas subamostradas · validação ≈ **{100 - train_pct}%** · "
            f"`split_ratio` (FLAML) ≈ **{split_ratio:.2f}**"
        )

        st.divider()
        st.markdown("##### FLAML (AutoML)")
        _metric_labels = {
            "accuracy": "Acurácia",
            "macro_f1": "F1 macro (média por classe)",
            "micro_f1": "F1 micro (agregado por instância)",
            "log_loss": "Log loss (probabilidades; menor é melhor)",
            "roc_auc_ovr": "ROC AUC OvR multiclasse (probabilidades)",
        }
        flaml_metric = st.selectbox(
            "Métrica que o FLAML otimiza (no split de validação)",
            options=list(_metric_labels.keys()),
            index=0,
            format_func=lambda k: _metric_labels.get(str(k), str(k)),
            key="rotina_ml_flaml_metric_select",
            help="O AutoML minimiza o erro desta métrica no conjunto de **validação** definido pelo holdout acima. "
            "O relatório inclui outras métricas no mesmo conjunto `val_*`.",
        )
        time_budget = st.slider(
            "time_budget (s)",
            min_value=30,
            max_value=900,
            value=120,
            step=30,
            help="Tempo máximo de busca de hiperparâmetros do classificador.",
        )
        estimator_list = st.multiselect(
            "estimator_list (FLAML — matriz esparsa TF-IDF)",
            options=[
                "lrl1",
                "lrl2",
                "svc",
                "sgd",
                "lgbm",
                "xgboost",
                "xgb_limitdepth",
                "catboost",
                "rf",
                "extra_tree",
                "kneighbor",
            ],
            default=["lrl1", "lrl2"],
            key="rotina_ml_estimator_multiselect",
            help="**Não precisas** de árvores/boosting para um baseline sólido em texto+TF-IDF: `lrl1`/`lrl2`/`svc`/`sgd` "
            "são o habitual e muitas vezes competem bem. `lgbm`/`xgboost`/`catboost`/`rf`/`extra_tree` estão aqui "
            "para experimentares; podem ser **mais lentos** e exigir mais **RAM** com vocabulários grandes. "
            "`kneighbor` costuma ser o mais pesado em alta dimensão.",
        )
        seed = st.number_input(
            "random seed",
            value=42,
            min_value=0,
            max_value=2_000_000_000,
            key="rotina_ml_seed_input",
        )
        automl_n_jobs = st.number_input(
            "FLAML n_jobs",
            value=-1,
            min_value=-1,
            max_value=32,
            key="rotina_ml_njobs_input",
        )

        run_eval_test = st.checkbox(
            "Avaliar também no split `test` após treino",
            value=True,
            key="rotina_ml_run_test_checkbox",
        )

        train_ready = bool(estimator_list)
        if not train_ready:
            st.warning("Selecione **pelo menos um estimador** para ativar o treino (os outros separadores continuam disponíveis).")

        _n_after_sample = max(1, int(len(dfs["train"]) * float(sample_fraction)))
        _n_fit_eff = max(1, int(_n_after_sample * (float(train_pct) / 100.0)))
        if int(tfidf_min_df) > _n_fit_eff:
            st.error(
                f"`min_df` ({int(tfidf_min_df)}) é maior que o número aproximado de linhas de **treino** "
                f"após subamostra e holdout (~{_n_fit_eff}). Reduza `min_df`, aumente a fração do train "
                "ou a percentagem de treino no slider acima."
            )
            train_ready = False

        if st.button(
            "Treinar com FLAML",
            type="primary",
            key="rotina_ml_train_flaml_btn",
            disabled=not train_ready,
        ):
            st.session_state.pop("_rotina_ml_train_error", None)
            try:
                with st.spinner(
                    "A treinar (pode demorar conforme `time_budget` e estimadores)…"
                ):
                    pipe, info = train_flaml_emotion_pipeline(
                        train_df=dfs["train"],
                        text_col="text",
                        label_col="label",
                        tfidf_max_features=int(tfidf_max_features),
                        tfidf_ngram_max=int(tfidf_ngram_max),
                        tfidf_min_df=int(tfidf_min_df),
                        tfidf_max_df=float(tfidf_max_df),
                        tfidf_sublinear_tf=tfidf_sublinear,
                        sample_fraction=float(sample_fraction),
                        seed=int(seed),
                        time_budget=int(time_budget),
                        estimator_list=estimator_list,
                        automl_n_jobs=int(automl_n_jobs),
                        flaml_metric=str(flaml_metric),
                        split_ratio=float(split_ratio),
                    )
                    metrics = dict(info["metrics"])
                    hp = dict(info["hyperparams_snapshot"])
                    if run_eval_test:
                        test_m = evaluate_on_test(pipe, dfs["test"], "text", "label")
                        metrics.update(test_m)
                        hp["evaluated_on_test"] = True
                    bundle = make_emotion_bundle(
                        pipe, metrics=metrics, hyperparams_snapshot=hp
                    )
                    st.session_state["_rotina_emotion_ml_bundle"] = bundle
                    st.session_state["_rotina_ml_last_trials"] = list(info.get("trials") or [])
            except Exception as e:
                st.session_state["_rotina_ml_train_error"] = str(e)
                st.error(f"Treino falhou: {e}")
            else:
                st.session_state.pop("_rotina_ml_train_error", None)
                st.success(
                    "Treino concluído. Abra **Pickle e novas predições** para descarregar o `.pkl` e testar."
                )
                m = st.session_state.get("_rotina_emotion_ml_bundle")
                if isinstance(m, EmotionMLBundle):
                    try:
                        st.json(_metrics_for_display(m.metrics))
                    except Exception:
                        st.write(_metrics_for_display(m.metrics))

        trials_cached = st.session_state.get("_rotina_ml_last_trials") or []
        if trials_cached:
            st.divider()
            st.markdown("##### Trials do último treino (FLAML)")
            st.caption(
                "Cada linha é um ensaio de hiperparâmetros. **perda_validação** é o que o FLAML **minimiza** "
                "(transformação da métrica escolhida). A tabela só se atualiza **quando o treino termina** — "
                "o `fit` do FLAML é síncrono e o Streamlit não mostra linhas intermediárias em tempo real."
            )
            st.dataframe(
                pd.DataFrame(trials_cached),
                use_container_width=True,
                hide_index=True,
                height=min(460, 44 + 36 * len(trials_cached)),
            )

        err_prev = st.session_state.get("_rotina_ml_train_error")
        if err_prev and not st.session_state.get("_rotina_emotion_ml_bundle"):
            st.caption(f"Último erro: {err_prev}")

    with tab_model:
        st.markdown(
            "Guarde o **EmotionMLBundle** (pipeline sklearn + metadados). Na inferência, "
            "o pipeline espera **lista de textos em inglês** no mesmo estilo do dataset."
        )
        b = st.session_state.get("_rotina_emotion_ml_bundle")
        if isinstance(b, EmotionMLBundle):
            pkl_bytes = pickle_bundle(b)
            st.markdown("##### Guardar modelo em `.pkl`")
            _ml_dir = DATA_DIR / "ml_models"
            st.caption(
                f"No servidor, o ficheiro fica em **`{_ml_dir.resolve()}`** "
                "(pasta `ml_models` dentro de `ROTINA_DATA_DIR` / `data/`)."
            )
            c_save_a, c_save_b = st.columns([3, 2])
            with c_save_a:
                save_filename = st.text_input(
                    "Nome do ficheiro (sem pastas)",
                    value="emotion_flaml_bundle",
                    key="rotina_ml_save_filename_input",
                    help="Apenas letras, números, ponto, _ e -. O sufixo .pkl é aplicado automaticamente.",
                )
            with c_save_b:
                st.write("")
                if st.button(
                    "Guardar .pkl no servidor",
                    type="primary",
                    key="rotina_ml_save_server_pkl_btn",
                    help="Escreve o pickle em disco na pasta ml_models.",
                ):
                    try:
                        out_path = save_bundle_pkl_to_data_dir(
                            b, save_filename, data_dir=DATA_DIR
                        )
                        st.success(f"Guardado: `{out_path}`")
                    except OSError as e:
                        st.error(f"Não foi possível gravar o ficheiro: {e}")
                    except ValueError as e:
                        st.error(str(e))
            st.download_button(
                "Descarregar modelo (.pkl) para o teu PC",
                data=pkl_bytes,
                file_name="emotion_flaml_tfidf_bundle.pkl",
                mime="application/octet-stream",
                key="rotina_ml_download_pkl_btn",
            )
            with st.expander("Métricas e hiperparâmetros gravados no bundle"):
                try:
                    st.json(
                        {
                            "metrics": _metrics_for_display(b.metrics),
                            "hyperparams_snapshot": b.hyperparams_snapshot,
                        }
                    )
                except (TypeError, ValueError):
                    st.text(
                        json.dumps(
                            {
                                "metrics": _metrics_for_display(b.metrics),
                                "hyperparams_snapshot": b.hyperparams_snapshot,
                            },
                            default=str,
                            ensure_ascii=False,
                        )
                    )
        else:
            st.info("Ainda não há modelo na sessão. Treine no separador **Treino (FLAML)**.")

        st.divider()
        st.markdown("##### Inferência com ficheiro `.pkl`")
        st.warning(
            "**Segurança:** pickle pode executar código arbitrário. Carregue **apenas** ficheiros `.pkl` "
            "que você mesmo gerou neste laboratório (ou de fontes em que confia totalmente)."
        )
        up = st.file_uploader(
            "Carregar bundle pickle",
            type=["pkl"],
            key="rotina_ml_pkl_uploader",
        )
        default_lines = (
            "i feel wonderful today\n"
            "im so angry at this situation\n"
            "i am scared about the interview"
        )
        raw = st.text_area(
            "Novos textos (um por linha)",
            value=default_lines,
            height=140,
            key="rotina_ml_infer_textarea",
        )

        if st.button(
            "Predizer com ficheiro carregado",
            disabled=up is None,
            key="rotina_ml_predict_upload_btn",
        ):
            try:
                data = up.getvalue() if hasattr(up, "getvalue") else up.read()
                bundle2 = unpickle_bundle(data)
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                if not lines:
                    st.warning("Introduza pelo menos uma linha de texto não vazia.")
                else:
                    ids, names = bundle2.predict_labels(lines)
                    out = pd.DataFrame({"text": lines, "label_id": ids, "emotion": names})
                    st.dataframe(out, use_container_width=True, hide_index=True)
            except (TypeError, ValueError, pickle.UnpicklingError) as e:
                st.error(f"Não foi possível carregar ou usar o pickle: {e}")
            except Exception as e:
                st.error(f"Erro na predição: {e}")

        if isinstance(b, EmotionMLBundle) and st.button(
            "Predizer com modelo da sessão (sem ficheiro)",
            key="rotina_ml_predict_session_btn",
        ):
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            if not lines:
                st.warning("Introduza pelo menos uma linha de texto.")
            else:
                try:
                    ids, names = b.predict_labels(lines)
                    out = pd.DataFrame({"text": lines, "label_id": ids, "emotion": names})
                    st.dataframe(out, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
