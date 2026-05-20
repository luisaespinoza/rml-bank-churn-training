"""
Benchmark Tiny ANN inference runtime against XGBoost for the bank churn project.

Purpose
-------
This script adds deployment-focused runtime metrics to the model comparison story:
XGBoost may win raw tabular performance, while the Tiny ANN may be preferable when
minimal model size, simple static deployment, and lightweight browser inference matter.

Default behavior
----------------
- Loads the saved Tiny ANN checkpoint from artifacts/models/tiny.pt.
- Recreates the same train/validation/test split used by the experiments.
- Trains an XGBoost baseline unless a saved XGBoost pipeline is provided.
- Benchmarks two latency views:
    1. model_only: preprocessing is done once; only model inference is timed.
    2. end_to_end_python: preprocessing + model prediction is timed.

Outputs
-------
- report/runtime_tiny_vs_xgboost.json
- report/runtime_tiny_vs_xgboost.csv

Example
-------
python benchmark_tiny_vs_xgboost_runtime.py --seed 314
python benchmark_tiny_vs_xgboost_runtime.py --seed 314 --repeats 500 --batch-sizes 1 32 300 1500
python benchmark_tiny_vs_xgboost_runtime.py --xgboost-model artifacts/baselines/xgboost_pipeline.joblib
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model import Tiny
from utils import precision_at_top_k, recall_at_top_k

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


DATA_PATH = Path("data/Bank_Churn.csv")
TINY_CHECKPOINT = Path("artifacts/models/tiny.pt")
DEFAULT_XGB_MODEL_PATH = Path("artifacts/baselines/xgboost_pipeline.joblib")
OUTPUT_PREFIX = Path("report/runtime_tiny_vs_xgboost")
TARGET = "Exited"
TOP_K = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Tiny ANN runtime against XGBoost.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Bank churn CSV path.")
    parser.add_argument("--tiny-checkpoint", type=Path, default=TINY_CHECKPOINT, help="Saved Tiny .pt checkpoint.")
    parser.add_argument("--xgboost-model", type=Path, default=None, help="Optional saved XGBoost pipeline joblib path.")
    parser.add_argument("--save-xgboost-model", type=Path, default=DEFAULT_XGB_MODEL_PATH, help="Where to save a newly trained XGBoost pipeline.")
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX, help="Output prefix without extension.")
    parser.add_argument("--seed", type=int, default=314, help="Train/test split seed.")
    parser.add_argument("--repeats", type=int, default=300, help="Timed repetitions per benchmark case.")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup repetitions before timing.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 300, 1500], help="Batch sizes to benchmark.")
    parser.add_argument("--torch-threads", type=int, default=1, help="Torch CPU threads for reproducible small-model timing.")
    parser.add_argument("--no-save-xgboost", action="store_true", help="Do not save a newly trained XGBoost pipeline.")
    return parser.parse_args()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data_split(data_path: Path, seed: int):
    df = pd.read_csv(data_path)
    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    _X_val, X_test, _y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_tiny_model(checkpoint_path: Path) -> Tiny:
    checkpoint = safe_torch_load(checkpoint_path)
    input_dim = int(checkpoint.get("input_dim", 13))
    model = Tiny(input_dim=input_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_ann_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", make_one_hot_encoder(), categorical_cols),
        ]
    )


def build_tree_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", make_one_hot_encoder(), categorical_cols),
        ]
    )


def build_xgboost_pipeline(numeric_cols: List[str], categorical_cols: List[str], seed: int) -> Pipeline:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed. Install xgboost or provide --xgboost-model.")

    return Pipeline(
        steps=[
            ("prep", build_tree_preprocessor(numeric_cols, categorical_cols)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,
                    reg_lambda=5.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def to_dense_array(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def predict_tiny_prob(model: Tiny, x_np: np.ndarray) -> np.ndarray:
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
        return torch.sigmoid(logits).squeeze(1).cpu().numpy()


def classification_metrics(model_name: str, y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float | str]:
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "recall_at_top_20_pct": float(recall_at_top_k(y_true, y_prob, k=TOP_K)),
        "precision_at_top_20_pct": float(precision_at_top_k(y_true, y_prob, k=TOP_K)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob_clipped)),
    }


def slice_batch(obj: Any, batch_size: int):
    return obj.iloc[:batch_size] if hasattr(obj, "iloc") else obj[:batch_size]


def benchmark_callable(fn: Callable[[], Any], repeats: int, warmup: int) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()

    times_ms: List[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    arr = np.asarray(times_ms, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def add_rate_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    batch_size = int(row["batch_size"])
    mean_ms = float(row["mean_ms"])
    median_ms = float(row["median_ms"])
    row["mean_ms_per_customer"] = mean_ms / batch_size
    row["median_ms_per_customer"] = median_ms / batch_size
    row["customers_per_second_mean"] = 1000.0 * batch_size / mean_ms if mean_ms > 0 else None
    return row


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))

    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = load_data_split(args.data, args.seed)
    max_batch = len(X_test)
    batch_sizes = [b for b in args.batch_sizes if 1 <= b <= max_batch]
    if not batch_sizes:
        raise ValueError(f"No valid batch sizes. Test set has {max_batch} rows.")

    tiny = load_tiny_model(args.tiny_checkpoint)
    ann_prep = build_ann_preprocessor(numeric_cols, categorical_cols)
    X_train_ann = to_dense_array(ann_prep.fit_transform(X_train))
    X_test_ann = to_dense_array(ann_prep.transform(X_test))

    if args.xgboost_model is not None and args.xgboost_model.exists():
        xgb_pipeline = joblib.load(args.xgboost_model)
        print(f"[xgboost] loaded pipeline: {args.xgboost_model}")
    else:
        xgb_pipeline = build_xgboost_pipeline(numeric_cols, categorical_cols, args.seed)
        print("[xgboost] training baseline pipeline for runtime benchmark...")
        xgb_pipeline.fit(X_train, y_train)
        if not args.no_save_xgboost and args.save_xgboost_model is not None:
            args.save_xgboost_model.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(xgb_pipeline, args.save_xgboost_model)
            print(f"[xgboost] saved pipeline: {args.save_xgboost_model}")

    xgb_prep = xgb_pipeline.named_steps["prep"]
    xgb_clf = xgb_pipeline.named_steps["clf"]
    X_test_xgb = xgb_prep.transform(X_test)

    tiny_probs = predict_tiny_prob(tiny, X_test_ann)
    xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]

    metric_rows = [
        classification_metrics("tiny_ann", y_test, tiny_probs),
        classification_metrics("xgboost", y_test, xgb_probs),
    ]

    runtime_rows: List[Dict[str, Any]] = []
    for batch_size in batch_sizes:
        X_raw_batch = slice_batch(X_test, batch_size)
        X_ann_batch = X_test_ann[:batch_size]
        X_xgb_batch = slice_batch(X_test_xgb, batch_size)

        cases = [
            (
                "tiny_ann",
                "model_only_preprocessed",
                lambda X_ann_batch=X_ann_batch: predict_tiny_prob(tiny, X_ann_batch),
            ),
            (
                "tiny_ann",
                "end_to_end_python_preprocess_plus_model",
                lambda X_raw_batch=X_raw_batch: predict_tiny_prob(
                    tiny, to_dense_array(ann_prep.transform(X_raw_batch))
                ),
            ),
            (
                "xgboost",
                "model_only_preprocessed",
                lambda X_xgb_batch=X_xgb_batch: xgb_clf.predict_proba(X_xgb_batch)[:, 1],
            ),
            (
                "xgboost",
                "end_to_end_python_preprocess_plus_model",
                lambda X_raw_batch=X_raw_batch: xgb_pipeline.predict_proba(X_raw_batch)[:, 1],
            ),
        ]

        for model_name, benchmark_type, fn in cases:
            row = benchmark_callable(fn, repeats=args.repeats, warmup=args.warmup)
            row.update(
                {
                    "model": model_name,
                    "benchmark_type": benchmark_type,
                    "batch_size": int(batch_size),
                    "repeats": int(args.repeats),
                    "warmup": int(args.warmup),
                    "seed": int(args.seed),
                }
            )
            runtime_rows.append(add_rate_fields(row))
            print(
                f"[{model_name}] {benchmark_type} batch={batch_size} "
                f"median={row['median_ms']:.4f} ms "
                f"mean/customer={row['mean_ms_per_customer']:.6f} ms"
            )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "classification_metrics": metric_rows,
        "runtime_metrics": runtime_rows,
        "notes": [
            "model_only_preprocessed times only the estimator after preprocessing has already been applied.",
            "end_to_end_python_preprocess_plus_model times Python preprocessing plus model inference.",
            "Browser runtime should be measured separately in the deployed web app; these are local Python CPU benchmarks.",
        ],
    }

    with open(args.output_prefix.with_suffix(".json"), "w") as f:
        json.dump(output, f, indent=2)

    pd.DataFrame(runtime_rows).to_csv(args.output_prefix.with_suffix(".csv"), index=False)
    pd.DataFrame(metric_rows).to_csv(args.output_prefix.with_name(args.output_prefix.name + "_classification.csv"), index=False)

    print("\nClassification metrics:")
    print(pd.DataFrame(metric_rows).to_string(index=False))

    print("\nSaved runtime artifacts:")
    print(f"- {args.output_prefix.with_suffix('.json')}")
    print(f"- {args.output_prefix.with_suffix('.csv')}")
    print(f"- {args.output_prefix.with_name(args.output_prefix.name + '_classification.csv')}")


if __name__ == "__main__":
    main()
