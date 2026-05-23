"""
Treino com FLAML + TF-IDF no dataset emotion, serialização pickle e inferência.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path
import pickle
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modules.emotion_dataset_catalog import (
    EMOTION_CONFIG,
    EMOTION_DATASET_ID,
    EMOTION_LABEL_ID_TO_NAME,
)

# O FLAML usa `_runner` global em `flaml.tune.tune`; treinos em paralelo (ex.: duas abas) ou
# certas combinações n_jobs deixam `_runner` a None e falha em `stop_trial`.
_FLAML_TUNE_GLOBAL_LOCK = threading.RLock()


@dataclass
class EmotionMLBundle:
    """
    Artefacto único para gravar/lêr com pickle: pipeline + metadados para produção/auditoria.
    """

    pipeline: Pipeline
    label_id_to_name: dict[int, str]
    dataset_id: str
    dataset_config: str
    text_column: str
    label_column: str
    metrics: dict[str, Any] = field(default_factory=dict)
    hyperparams_snapshot: dict[str, Any] = field(default_factory=dict)

    def predict_labels(self, texts: list[str]) -> tuple[np.ndarray, list[str]]:
        if not texts:
            return np.array([], dtype=np.int64), []
        ids = self.pipeline.predict(texts)
        names = [self.label_id_to_name.get(int(i), str(i)) for i in ids]
        return ids, names


def load_emotion_split_pandas() -> dict[str, pd.DataFrame]:
    """Carrega os três splits oficiais (requer `datasets` e rede na primeira vez)."""
    from datasets import load_dataset

    ds = load_dataset(EMOTION_DATASET_ID, EMOTION_CONFIG)
    return {
        "train": ds["train"].to_pandas(),
        "validation": ds["validation"].to_pandas(),
        "test": ds["test"].to_pandas(),
    }


def _flaml_sklearn_estimator(automl: Any) -> Any:
    """
    O `automl.model` do FLAML é um *wrapper*; o sklearn (ou XGBoost/LGBM) treinado está em `.estimator` / `._model`.
    O `sklearn.pipeline.Pipeline` exige passos já `fit` — por isso extraímos o estimador interno e chamamos `pipe.fit(...)`.
    """
    wrapper = automl.model
    if wrapper is None:
        raise RuntimeError("FLAML não devolveu um modelo treinado (`model` é None).")
    inner = getattr(wrapper, "estimator", None)
    if inner is None:
        inner = getattr(wrapper, "_model", None)
    if inner is None:
        inner = wrapper
    return inner


def trials_from_automl_config_history(automl: Any) -> list[dict[str, Any]]:
    """Fallback quando o ficheiro de log não tem linhas (ex.: versão/caminho do FLAML)."""
    ch = getattr(automl, "config_history", None) or {}
    if not isinstance(ch, dict):
        return []
    out: list[dict[str, Any]] = []
    for k in sorted(ch.keys(), key=lambda x: (isinstance(x, int), x)):
        v = ch[k]
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        est, cfg = v[0], v[1]
        t = v[2] if len(v) > 2 else None
        cfg_s = repr(cfg)
        if len(cfg_s) > 600:
            cfg_s = cfg_s[:600] + "…"
        out.append(
            {
                "id": k,
                "estimador": str(est),
                "perda_validação": None,
                "métrica_log": None,
                "tempo_trial_s": None,
                "tempo_acumulado_s": float(t) if t is not None else None,
                "amostras": None,
                "config_resumo": cfg_s,
                "nota": "entrada do histórico FLAML (melhor modelo atualizado neste passo)",
            }
        )
    return out


def read_flaml_training_log(log_path: str) -> list[dict[str, Any]]:
    """
    Lê o ficheiro de log de trials do FLAML (`log_type='all'`).
    O formato JSON real pode ter `logged_metric` como float **ou** como dict (ex.: pred_time) —
    o `TrainingLogRecord` oficial assume float, por isso fazemos parse linha-a-linha.
    """
    if not log_path or not os.path.isfile(log_path):
        return []
    import json

    out: list[dict[str, Any]] = []
    try:
        with open(log_path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if len(data) == 1:
                    continue
                lm = data.get("logged_metric")
                if isinstance(lm, dict):
                    if not lm:
                        lm_disp = None
                    else:
                        k, v = next(iter(lm.items()))
                        lm_disp = f"{k}={v}"
                else:
                    lm_disp = lm
                cfg = data.get("config")
                try:
                    cfg_s = repr(cfg)
                except Exception:
                    cfg_s = str(cfg)
                if len(cfg_s) > 600:
                    cfg_s = cfg_s[:600] + "…"
                out.append(
                    {
                        "id": data.get("record_id"),
                        "estimador": data.get("learner"),
                        "perda_validação": data.get("validation_loss"),
                        "métrica_log": lm_disp,
                        "tempo_trial_s": data.get("trial_time"),
                        "tempo_acumulado_s": data.get("wall_clock_time"),
                        "amostras": data.get("sample_size"),
                        "config_resumo": cfg_s,
                    }
                )
    except Exception:
        return []
    return out


def _subsample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 0.9999:
        return df
    return df.sample(frac=fraction, random_state=seed).reset_index(drop=True)


def compute_split_metrics(
    pipe: Pipeline,
    texts: list[str],
    y: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    """
    Métricas de classificação multiclasse no conjunto dado (acurácia, F1, log-loss, ROC AUC OvR).
    `log_loss` / `roc_auc_ovr` exigem `predict_proba` no classificador final.
    """
    y = np.asarray(y)
    pred = pipe.predict(texts)
    out: dict[str, float] = {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        f"{prefix}_micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
    }
    try:
        proba = pipe.predict_proba(texts)
        clf = pipe.named_steps["clf"]
        classes = np.asarray(getattr(clf, "classes_", np.unique(y)))
        out[f"{prefix}_log_loss"] = float(log_loss(y, proba, labels=classes))
        if len(classes) >= 2:
            out[f"{prefix}_roc_auc_ovr"] = float(
                roc_auc_score(
                    y,
                    proba,
                    multi_class="ovr",
                    average="weighted",
                    labels=classes,
                )
            )
    except Exception:
        pass
    return out


def make_emotion_bundle(
    pipe: Pipeline,
    metrics: dict[str, Any] | None,
    hyperparams_snapshot: dict[str, Any] | None,
) -> EmotionMLBundle:
    return EmotionMLBundle(
        pipeline=pipe,
        label_id_to_name=dict(EMOTION_LABEL_ID_TO_NAME),
        dataset_id=EMOTION_DATASET_ID,
        dataset_config=EMOTION_CONFIG,
        text_column="text",
        label_column="label",
        metrics=dict(metrics or {}),
        hyperparams_snapshot=dict(hyperparams_snapshot or {}),
    )


def train_flaml_emotion_pipeline(
    *,
    train_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    tfidf_max_features: int,
    tfidf_ngram_max: int,
    tfidf_min_df: int,
    tfidf_max_df: float,
    tfidf_sublinear_tf: bool,
    sample_fraction: float,
    seed: int,
    time_budget: int,
    estimator_list: list[str],
    automl_n_jobs: int,
    flaml_metric: str = "accuracy",
    split_ratio: float = 0.2,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Vetoriza com TF-IDF (só no conjunto de treino do holdout) e deixa o FLAML otimizar o classificador.

    ``split_ratio`` segue a convenção do **FLAML / sklearn**: fração das linhas (após subamostra do
    split ``train`` do HF) reservada ao conjunto de **validação**; o restante é **treino**
    (ex.: ``0.2`` → ~20% validação, ~80% treino). O TF-IDF ajusta-se apenas no treino para evitar leakage.
    """
    from flaml import AutoML

    for col in (text_col, label_col):
        if col not in train_df.columns:
            raise ValueError(
                f"Coluna `{col}` em falta no DataFrame train. "
                f"Esperado schema com `{text_col}` e `{label_col}`."
            )

    tr = _subsample(train_df, sample_fraction, seed)
    if len(tr) == 0:
        raise ValueError(
            "O conjunto de treino ficou vazio (verifique `sample_fraction` e o tamanho do split)."
        )

    sr = float(split_ratio)
    if not 0.05 <= sr <= 0.45:
        raise ValueError(
            "split_ratio deve estar entre 0.05 e 0.45 (fração do train subamostrado para validação)."
        )
    try:
        train_fit_df, val_eval_df = train_test_split(
            tr,
            test_size=sr,
            random_state=int(seed),
            stratify=tr[label_col],
        )
    except ValueError as e:
        raise ValueError(
            "Não foi possível aplicar split_ratio com estratificação (poucas amostras por classe?). "
            "Reduza split_ratio ou aumente a subamostra do train."
        ) from e
    if len(train_fit_df) < 10 or len(val_eval_df) < 6:
        raise ValueError(
            "split_ratio deixou poucos exemplos em treino ou validação. "
            "Use um valor mais baixo ou mais linhas no train."
        )

    if int(tfidf_min_df) > len(train_fit_df):
        raise ValueError(
            f"`min_df`={tfidf_min_df} não pode exceder o número de documentos de **treino** "
            f"({len(train_fit_df)}) após subamostragem e split_ratio."
        )

    X_train = train_fit_df[text_col].astype(str).tolist()
    y_train = train_fit_df[label_col].astype(int).to_numpy()
    X_val_eval = val_eval_df[text_col].astype(str).tolist()
    y_val_eval = val_eval_df[label_col].astype(int).to_numpy()

    vec = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1, int(tfidf_ngram_max)),
        min_df=int(tfidf_min_df),
        max_df=float(tfidf_max_df),
        sublinear_tf=bool(tfidf_sublinear_tf),
    )
    try:
        X_tr_sp = vec.fit_transform(X_train)
        X_val_sp = vec.transform(X_val_eval)
    except ValueError as e:
        raise ValueError(
            "TF-IDF falhou (vocabulário vazio ou parâmetros incompatíveis). "
            "Tente reduzir `min_df`, aumentar `max_df` ou `max_features`, ou ampliar a subamostra."
        ) from e

    _metric = str(flaml_metric or "accuracy").strip().lower()
    trial_rows: list[dict[str, Any]] = []
    automl: Any = None
    _fit_failed: BaseException | None = None
    _n_jobs_attempts = (
        [int(automl_n_jobs), 1] if int(automl_n_jobs) != 1 else [int(automl_n_jobs)]
    )

    for _nj in _n_jobs_attempts:
        fd, log_path = tempfile.mkstemp(prefix="rotina_flaml_", suffix=".log")
        os.close(fd)
        try:
            import flaml.tune.tune as _fl_tune

            with _FLAML_TUNE_GLOBAL_LOCK:
                _fl_tune._runner = None
                automl = AutoML()
                automl.fit(
                    X_tr_sp,
                    y_train,
                    eval_method="holdout",
                    X_val=X_val_sp,
                    y_val=y_val_eval,
                    task="classification",
                    time_budget=int(time_budget),
                    metric=_metric,
                    estimator_list=list(estimator_list),
                    n_jobs=int(_nj),
                    seed=int(seed),
                    verbose=0,
                    log_file_name=log_path,
                    log_type="all",
                )
            try:
                trial_rows = read_flaml_training_log(log_path)
            except Exception:
                trial_rows = []
            if not trial_rows:
                trial_rows = trials_from_automl_config_history(automl)
            _fit_failed = None
            break
        except AttributeError as exc:
            _fit_failed = exc
            if (
                "stop_trial" not in str(exc)
                or int(_nj) == 1
                or int(automl_n_jobs) == 1
            ):
                break
        finally:
            try:
                import flaml.tune.tune as _fl_tune

                _fl_tune._runner = None
            except Exception:
                pass
            try:
                os.unlink(log_path)
            except OSError:
                pass

    if _fit_failed is not None:
        if "stop_trial" in str(_fit_failed):
            raise RuntimeError(
                "O FLAML entrou em estado inconsistente (erro interno `stop_trial` / `_runner`). "
                "Isto costuma acontecer ao **treinar de novo** com `n_jobs=-1` ou com **várias abas** "
                "a correr treinos em simultâneo. Tente **FLAML n_jobs = 1**, feche outras abas do Streamlit "
                "e volte a treinar; se continuar, actualize `flaml` (`pip install -U flaml`)."
            ) from _fit_failed
        raise _fit_failed

    inner = _flaml_sklearn_estimator(automl)
    pipe = Pipeline([("tfidf", vec), ("clf", inner)])
    pipe.fit(X_train, y_train)

    metrics: dict[str, Any] = {
        "flaml_optimization_metric": _metric,
        "split_ratio": sr,
        "best_estimator": str(getattr(automl, "best_estimator", "") or ""),
    }
    metrics.update(compute_split_metrics(pipe, X_val_eval, y_val_eval, prefix="val"))
    _cfg = getattr(automl, "_best_estimator_config", None)
    if _cfg is not None:
        metrics["best_config_repr"] = repr(_cfg)

    hp = {
        "tfidf_max_features": tfidf_max_features,
        "tfidf_ngram_max": tfidf_ngram_max,
        "tfidf_min_df": tfidf_min_df,
        "tfidf_max_df": tfidf_max_df,
        "tfidf_sublinear_tf": tfidf_sublinear_tf,
        "sample_fraction": sample_fraction,
        "seed": seed,
        "time_budget": time_budget,
        "estimator_list": list(estimator_list),
        "flaml_metric": _metric,
        "split_ratio": sr,
        "n_trials_logged": len(trial_rows),
    }
    return pipe, {"metrics": metrics, "hyperparams_snapshot": hp, "trials": trial_rows}


def evaluate_on_test(pipe: Pipeline, test_df: pd.DataFrame, text_col: str, label_col: str) -> dict[str, float]:
    X = test_df[text_col].astype(str).tolist()
    y = test_df[label_col].astype(int).to_numpy()
    return compute_split_metrics(pipe, X, y, prefix="test")


def safe_ml_pkl_filename(raw: str, *, default: str = "emotion_flaml_bundle") -> str:
    """Nome seguro para gravar em disco (apenas basename, caracteres limitados)."""
    import re

    s = (raw or "").strip()
    s = Path(s).name
    if s.lower().endswith(".pkl"):
        s = s[:-4]
    s = re.sub(r"[^a-zA-Z0-9._-]", "_", s).strip("._") or default
    return (s[:100] if len(s) > 100 else s) + ".pkl"


def pickle_bundle(bundle: EmotionMLBundle) -> bytes:
    buf = io.BytesIO()
    pickle.dump(bundle, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def save_bundle_pkl_to_data_dir(bundle: EmotionMLBundle, filename: str, *, data_dir: Path) -> Path:
    """
    Grava o bundle em ``<data_dir>/ml_models/<filename>.pkl``.
    ``data_dir`` deve ser o ``DATA_DIR`` resolvido da aplicação.
    """
    base_dir = (Path(data_dir).resolve() / "ml_models").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    name = safe_ml_pkl_filename(filename)
    target = (base_dir / name).resolve()
    if not target.is_relative_to(base_dir):
        raise ValueError("Caminho de saída inválido.")
    target.write_bytes(pickle_bundle(bundle))
    return target


def unpickle_bundle(data: bytes) -> EmotionMLBundle:
    if not data:
        raise ValueError("Ficheiro vazio.")
    obj = pickle.loads(data)
    if not isinstance(obj, EmotionMLBundle):
        raise TypeError(
            "O pickle não contém um EmotionMLBundle desta aplicação. "
            "Só carregue ficheiros `.pkl` gerados por este laboratório."
        )
    return obj


def make_train_val_from_train(
    train_df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quando quiser validação holdout a partir do split train (opcional)."""
    return train_test_split(
        train_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_df["label"],
    )
