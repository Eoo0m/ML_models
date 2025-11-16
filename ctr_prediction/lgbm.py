import os, sys, logging, hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ================== 설정 ==================
CHUNKS_DIR = "data"
PARQUET_FILE = "train.parquet"  # 단일 파일만 읽음

LABEL_COL = "clicked"
USE_COLS = None
SEED = 42
NTHREAD = os.cpu_count() or 8
NEG_POS_RATIO = 1

TRAIN_FRAC, VALID_FRAC, TEST_FRAC = 0.95, 0.1, 0.1  # test_frac=0이면 test skip

# ================== 로깅 ==================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ================== 유틸 ==================
def read_full_parquet(path: str, columns=None) -> pd.DataFrame:
    t = pq.read_table(path, columns=columns)
    return t.combine_chunks().to_pandas()


def hash_uniform(x) -> float:
    if not isinstance(x, (str, bytes)):
        x = str(x)
    if isinstance(x, str):
        x = x.encode("utf-8")
    h = hashlib.md5(x).hexdigest()
    v = int(h[:16], 16)
    return (v % (1 << 53)) / float(1 << 53)


def _add_cyclic_features(X: pd.DataFrame, col: str, period: int):
    if col not in X.columns:
        return X
    vals = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)
    X[f"{col}_sin"] = np.sin(2 * np.pi * vals / period).astype("float32")
    X[f"{col}_cos"] = np.cos(2 * np.pi * vals / period).astype("float32")
    return X.drop(columns=[col])


def feval_final_score(preds, dataset: lgb.Dataset):
    y_true = dataset.get_label().astype(int)
    p = 1.0 / (1.0 + np.exp(-preds))

    ap = average_precision_score(y_true, p)

    eps = 1e-12
    pprob = np.clip(p, eps, 1 - eps)

    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    w_pos, w_neg = 0.5 / max(pos, 1), 0.5 / max(neg, 1)
    w = np.where(y_true == 1, w_pos, w_neg)

    wll = -np.sum(w * (y_true * np.log(pprob) + (1 - y_true) * np.log(1 - pprob)))
    final_score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))

    return ("final_score", float(final_score), True)


# ================== 전처리 ==================
def preprocess(df: pd.DataFrame, label_col: str | None, cat_fit: dict | None = None):
    # 라벨 분리
    if label_col is not None:
        y = df[label_col].astype(np.int8)
        X = df.drop(columns=[label_col]).copy()
    else:
        y = None
        X = df.copy()

    # seq32, seq_emb float 처리
    seq_cols = [
        c for c in X.columns if c.startswith("seq32_") or c.startswith("seq_emb_")
    ]
    for c in seq_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype("float32")

    # age_group
    if "age_group" in X.columns:
        X["age_group"] = (
            pd.to_numeric(X["age_group"], errors="coerce").fillna(0).astype(np.int8)
        )

    # hour cyclic
    X = _add_cyclic_features(X, col="hour", period=24)

    # float64 → float32
    for c in X.select_dtypes(include=["float64"]).columns:
        X[c] = X[c].astype(np.float32)
    # int64 → smaller int
    for c in X.select_dtypes(include=["int64"]).columns:
        X[c] = pd.to_numeric(X[c], downcast="integer")

    # ----- 카테고리 핸들링 -----
    if cat_fit is None:
        cat_fit = {"objects": {}}
    else:
        cat_fit.setdefault("objects", {})

    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        X[c] = X[c].fillna("NA")
        if c not in cat_fit["objects"]:
            X[c] = X[c].astype("category")
            cat_fit["objects"][c] = X[c].dtype
        else:
            base = cat_fit["objects"][c]
            X[c] = pd.Categorical(X[c], categories=base.categories).add_categories(
                ["__UNK__"]
            )
            X.loc[~X[c].isin(base.categories), c] = "__UNK__"
            X[c] = X[c].astype("category")

    return X, y, cat_fit


# ================== 메인 ==================
def main():
    rng = np.random.default_rng(SEED)

    # 1) 단일 parquet 로드
    path = os.path.join(CHUNKS_DIR, PARQUET_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일 없음: {path}")

    df = read_full_parquet(path, columns=USE_COLS)
    if LABEL_COL not in df.columns:
        raise KeyError(f"라벨 '{LABEL_COL}' 없음.")

    logger.info(f"파일 로드 완료: {path} | rows={len(df):,}")

    # 2) 해시 기반 split
    keys = np.array([f"0:{i}" for i in range(len(df))], dtype=object)
    u = np.fromiter((hash_uniform(k) for k in keys), dtype=float, count=len(keys))

    m_train = u < TRAIN_FRAC
    m_valid = (u >= TRAIN_FRAC) & (u < TRAIN_FRAC + VALID_FRAC)
    m_test = u >= (TRAIN_FRAC + VALID_FRAC)

    df_train = df.loc[m_train]
    df_valid = df.loc[m_valid]
    df_test = df.loc[m_test]

    logger.info(f"Train={df_train.shape}, Valid={df_valid.shape}, Test={df_test.shape}")

    # 3) 언더샘플
    yb = df_train[LABEL_COL].astype(np.int8).values
    pos_idx = np.where(yb == 1)[0]
    neg_idx = np.where(yb == 0)[0]

    n_pos = len(pos_idx)
    n_neg_keep = min(len(neg_idx), n_pos * NEG_POS_RATIO)

    if n_pos > 0 and n_neg_keep > 0:
        keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        keep_idx = np.concatenate([pos_idx, keep_neg])
        rng.shuffle(keep_idx)
        df_train_us = df_train.iloc[keep_idx].reset_index(drop=True)
    else:
        df_train_us = df_train.reset_index(drop=True)

    logger.info(f"언더샘플 후 Train(us)={df_train_us.shape}")

    # 4) 전처리
    X_train, y_train, cat_fit = preprocess(df_train_us, LABEL_COL, cat_fit=None)
    X_valid, y_valid, _ = preprocess(df_valid, LABEL_COL, cat_fit=cat_fit)
    X_test, y_test, _ = preprocess(df_test, LABEL_COL, cat_fit=cat_fit)

    # 5) LightGBM 학습
    params = {
        "objective": "binary",
        "metric": "None",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "num_threads": NTHREAD,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 200,
        "max_depth": 10,
        "max_bin": 511,
        "extra_trees": True,
        "first_metric_only": True,
        "verbose": -1,
        "force_col_wise": True,
    }

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature="auto")
    valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature="auto")

    logger.info("LightGBM 학습 시작…")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        valid_names=["valid"],
        num_boost_round=12000,
        feval=feval_final_score,
        callbacks=[lgb.early_stopping(stopping_rounds=600), lgb.log_evaluation(100)],
    )
    logger.info(f"학습 완료! best_iter={model.best_iteration}")

    # 6) threshold 튜닝
    valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    prec, rec, th = precision_recall_curve(y_valid, valid_pred)
    th = np.append(th, 1.0)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best_idx = int(np.nanargmax(f1s))
    best_th = float(th[best_idx])

    logger.info(
        f"best_th={best_th:.4f} | F1={f1s[best_idx]:.4f}, "
        f"P={prec[best_idx]:.4f}, R={rec[best_idx]:.4f}"
    )

    # 7) optional test eval
    if TEST_FRAC > 0:
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_test, test_pred)
        ap = average_precision_score(y_test, test_pred)
        logger.info(f"Test AUC={auc:.4f}, AP={ap:.4f}")


if __name__ == "__main__":
    main()
