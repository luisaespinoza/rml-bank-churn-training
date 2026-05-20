"""
Compare saved ANN checkpoint metrics against saved non-ANN baseline metrics.

Default behavior does NOT retrain ANN models or non-ANN baselines.
It reads:
  - ANN metrics from artifacts/models/{tiny,small,medium,large}.pt
  - Non-ANN metrics from an existing JSON file, searched in this order:
      report/benchmark_results.json
      report/baseline_metrics.json
      reports/baseline_metrics.json

Optional fallback:
  --recompute-baselines can train Logistic Regression, Random Forest, and XGBoost
  if no saved baseline metric file exists.

Outputs:
  report/saved_model_comparison_results.json
  report/saved_model_comparison_results.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import recall_at_top_k, precision_at_top_k

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


DEFAULT_MODEL_DIR = Path("artifacts/models")
DEFAULT_OUTPUT_PREFIX = Path("report/saved_model_comparison_results")
DEFAULT_BASELINE_PATHS = [
    Path("report/benchmark_results.json"),
    Path("report/baseline_metrics.json"),
    Path("reports/baseline_metrics.json"),
]
DEFAULT_DATA_PATH = Path("data/Bank_Churn.csv")
TARGET = "Exited"
TOP_K = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved ANN checkpoint metrics against saved non-ANN baseline metrics."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing saved ANN .pt checkpoints.",
    )
    parser.add_argument(
        "--ann-models",
        nargs="+",
        default=["tiny", "small", "medium", "large"],
        help="Checkpoint stems to read, e.g. tiny small medium large.",
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="Saved non-ANN metrics JSON. If omitted, common report paths are searched.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path prefix without extension.",
    )
    parser.add_argument(
        "--sort-by",
        nargs="+",
        default=["recall_at_top_20_pct", "precision_at_top_20_pct", "roc_auc"],
        help="Metric columns to sort by, in priority order.",
    )
    parser.add_argument(
        "--recompute-baselines",
        action="store_true",
        help="Train non-ANN baselines only if saved baseline results are unavailable.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Dataset path used only with --recompute-baselines.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=314,
        help="Split/model seed used only with --recompute-baselines.",
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip XGBoost if --recompute-baselines is used.",
    )
    return parser.parse_args()


def safe_torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_size_mb(path: Path) -> Optional[float]:
    return float(path.stat().st_size / (1024**2)) if path.exists() else None


def normalize_metric_row(row: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """Keep key metrics consistent while preserving extra metadata."""
    out = dict(row)
    out["model_type"] = model_type

    # Some older files may use alternate spellings; normalize where possible.
    aliases = {
        "average_precision": "pr_auc",
        "ap": "pr_auc",
        "recall_top_20_pct": "recall_at_top_20_pct",
        "precision_top_20_pct": "precision_at_top_20_pct",
    }
    for old, new in aliases.items():
        if new not in out and old in out:
            out[new] = out[old]

    return out


def load_ann_rows(model_dir: Path, ann_models: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for model_name in ann_models:
        checkpoint_path = model_dir / f"{model_name}.pt"
        if not checkpoint_path.exists():
            print(f"[ANN] skipped {model_name}: missing {checkpoint_path}")
            continue

        checkpoint = safe_torch_load(checkpoint_path)
        metrics = checkpoint.get("metrics")
        if not isinstance(metrics, dict):
            print(f"[ANN] skipped {model_name}: checkpoint has no metrics dict")
            continue

        row = normalize_metric_row(metrics, model_type="ann")
        row["model"] = row.get("model", model_name)
        row["checkpoint_path"] = str(checkpoint_path)
        row["checkpoint_size_mb"] = file_size_mb(checkpoint_path)
        row["checkpoint_seed"] = checkpoint.get("seed", row.get("seed"))
        row["checkpoint_epochs"] = checkpoint.get("epochs", row.get("epochs"))
        row["checkpoint_learning_rate"] = checkpoint.get("learning_rate", row.get("learning_rate"))
        rows.append(row)
        print(f"[ANN] loaded metrics: {model_name}")

    return rows


def find_baseline_results_path(explicit_path: Optional[Path]) -> Optional[Path]:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None

    for path in DEFAULT_BASELINE_PATHS:
        if path.exists():
            return path
    return None


def load_baseline_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        # Support either {"results": [...]} or {model_name: metrics_dict} shapes.
        if isinstance(payload.get("results"), list):
            raw_rows = payload["results"]
        else:
            raw_rows = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    row = dict(value)
                    row.setdefault("model", key)
                    raw_rows.append(row)
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raise ValueError(f"Unsupported baseline JSON shape in {path}")

    rows = [normalize_metric_row(dict(row), model_type="non_ann") for row in raw_rows]
    print(f"[non-ANN] loaded {len(rows)} saved baseline rows from {path}")
    return rows


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate_probabilities(model_name: str, y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, Any]:
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        "model": model_name,
        "model_type": "non_ann",
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob_clipped)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "recall_at_top_20_pct": float(recall_at_top_k(y_true, y_prob, k=TOP_K)),
        "precision_at_top_20_pct": float(precision_at_top_k(y_true, y_prob, k=TOP_K)),
    }


def serialized_size_mb(obj: Any) -> Optional[float]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=True) as f:
            pickle.dump(obj, f)
            f.flush()
            return float(Path(f.name).stat().st_size / (1024**2))
    except Exception:
        return None


def recompute_baseline_rows(data_path: Path, seed: int, skip_xgboost: bool) -> List[Dict[str, Any]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset for baseline recompute: {data_path}")

    df = pd.read_csv(data_path)
    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    linear_preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", make_one_hot_encoder(), categorical_cols),
        ]
    )
    tree_preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", make_one_hot_encoder(), categorical_cols),
        ]
    )

    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("prep", linear_preprocess),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("prep", tree_preprocess),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if not skip_xgboost and XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            steps=[
                ("prep", tree_preprocess),
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
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    rows: List[Dict[str, Any]] = []
    for name, model in models.items():
        print(f"[non-ANN] recomputing: {name}")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        row = evaluate_probabilities(name, y_test, y_prob)
        row["serialized_pipeline_size_mb"] = serialized_size_mb(model)
        row["seed"] = seed
        rows.append(row)

    return rows


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    non_ann = df[df["model_type"] == "non_ann"]
    if non_ann.empty:
        return df

    best_auc = non_ann.sort_values("roc_auc", ascending=False).iloc[0]
    best_recall = non_ann.sort_values("recall_at_top_20_pct", ascending=False).iloc[0]
    best_precision = non_ann.sort_values("precision_at_top_20_pct", ascending=False).iloc[0]

    df["best_non_ann_roc_auc_model"] = best_auc["model"]
    df["roc_auc_delta_vs_best_non_ann"] = df["roc_auc"] - float(best_auc["roc_auc"])
    df["best_non_ann_recall20_model"] = best_recall["model"]
    df["recall20_delta_vs_best_non_ann"] = df["recall_at_top_20_pct"] - float(best_recall["recall_at_top_20_pct"])
    df["best_non_ann_precision20_model"] = best_precision["model"]
    df["precision20_delta_vs_best_non_ann"] = df["precision_at_top_20_pct"] - float(best_precision["precision_at_top_20_pct"])
    return df


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, Any]] = []
    rows.extend(load_ann_rows(args.model_dir, args.ann_models))

    baseline_path = find_baseline_results_path(args.baseline_results)
    if baseline_path is not None:
        rows.extend(load_baseline_rows(baseline_path))
    elif args.recompute_baselines:
        print("[non-ANN] no saved baseline file found; recomputing because --recompute-baselines was set")
        rows.extend(recompute_baseline_rows(args.data, args.seed, args.skip_xgboost))
    else:
        searched = [str(p) for p in DEFAULT_BASELINE_PATHS]
        raise FileNotFoundError(
            "No saved non-ANN baseline metrics file found. "
            f"Searched: {searched}. Pass --baseline-results PATH, or use --recompute-baselines."
        )

    if not rows:
        raise RuntimeError("No model rows loaded.")

    df = pd.DataFrame(rows)
    df = add_deltas(df)

    valid_sort_cols = [col for col in args.sort_by if col in df.columns]
    if valid_sort_cols:
        df = df.sort_values(valid_sort_cols, ascending=[False] * len(valid_sort_cols))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    df.to_json(json_path, orient="records", indent=2)
    df.to_csv(csv_path, index=False)

    display_cols = [
        "model",
        "model_type",
        "roc_auc",
        "pr_auc",
        "recall_at_top_20_pct",
        "precision_at_top_20_pct",
        "brier_score",
        "log_loss",
        "checkpoint_size_mb",
        "serialized_pipeline_size_mb",
        "roc_auc_delta_vs_best_non_ann",
        "recall20_delta_vs_best_non_ann",
        "precision20_delta_vs_best_non_ann",
    ]
    display_cols = [col for col in display_cols if col in df.columns]

    print("\nSaved-metric comparison:")
    print(df[display_cols].to_string(index=False))
    print("\nSaved comparison artifacts:")
    print(f"- {json_path}")
    print(f"- {csv_path}")


if __name__ == "__main__":
    main()
