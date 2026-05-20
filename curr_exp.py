import argparse
import copy
import json
from operator import truediv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model import Tiny, Small, Medium, Large
from utils import recall_at_top_k, precision_at_top_k


DATA_PATH = Path("data/Bank_Churn.csv")
BENCHMARK_PATH = Path("report/benchmark_results.json")
OUTPUT_PATH = Path("report/output_results.json")
ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = ARTIFACT_DIR / "models"
EXPORT_DIR = ARTIFACT_DIR / "deployment"
# SEEDS=42,314,2718
SEED = 314
EPOCHS = 10000
LR = 1e-3
TARGET = "Exited"
TOP_K = 0.20
PROGRESS_EVERY = 10


torch.manual_seed(SEED)
np.random.seed(SEED)


def build_model(model_def, input_dim: int):
    """Instantiate a model class/factory with the processed feature dimension."""
    try:
        return model_def(input_dim)
    except TypeError:
        try:
            return model_def(input_dim=input_dim)
        except TypeError:
            return model_def


def predict_proba(model, X_tensor):
    """Return positive-class probabilities from a BCEWithLogits model."""
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    return probs


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def estimate_fp32_size_mb(model):
    return float(count_parameters(model) * 4 / (1024 ** 2))


def train_model(
    model_def,
    X_tr,
    y_tr,
    X_vl,
    y_vl,
    model_name="model",
    epochs=EPOCHS,
    lr=LR,
    progress=False,
    progress_every=PROGRESS_EVERY,
):
    """Train a PyTorch binary classifier and restore the best validation-loss state."""
    model = build_model(model_def, X_tr.shape[1])
    if isinstance(model, type):
        model = model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if progress:
        print(f"\n[{model_name}] starting training on {device} | epochs={epochs} | lr={lr}")

    model = model.to(device)
    X_tr = X_tr.to(device)
    y_tr = y_tr.to(device)
    X_vl = X_vl.to(device)
    y_vl = y_vl.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    train_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_loss = criterion(val_logits, y_vl).item()

        train_history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
            }
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        should_print = (
            progress
            and (
                epoch == 1
                or epoch == epochs
                or epoch % max(1, progress_every) == 0
                or improved
            )
        )
        if should_print:
            marker = "*" if improved else " "
            print(
                f"[{model_name}] epoch {epoch:>4}/{epochs} "
                f"train_loss={loss.item():.5f} "
                f"val_loss={val_loss:.5f} "
                f"best={best_val_loss:.5f}@{best_epoch}{marker}",
                flush=True,
            )

    if progress:
        print(f"[{model_name}] done | best_val_loss={best_val_loss:.5f} at epoch {best_epoch}")

    model.load_state_dict(best_state)
    return model.cpu(), {"best_val_loss": float(best_val_loss), "best_epoch": int(best_epoch), "history": train_history}


def safe_top_k(metric_fn, y_true, y_prob, k=TOP_K):
    """Call top-k metric utilities while tolerating either positional or keyword APIs."""
    try:
        return float(metric_fn(y_true, y_prob, k=k))
    except TypeError:
        try:
            return float(metric_fn(y_true=y_true, y_prob=y_prob, k=k))
        except TypeError:
            return float(metric_fn(y_true=y_true, y_score=y_prob, k=k))


def load_benchmark_best(path: Path):
    """Load the best benchmark row by ROC-AUC, if a benchmark file exists."""
    if not path.exists():
        return None

    with open(path, "r") as f:
        benchmark_rows = json.load(f)

    benchmark_df = pd.DataFrame(benchmark_rows) if benchmark_rows else pd.DataFrame()
    if benchmark_df.empty or "roc_auc" not in benchmark_df.columns:
        return None

    return benchmark_df.sort_values("roc_auc", ascending=False).iloc[0].to_dict()


def get_feature_names(preprocess):
    """Return transformed feature names from a fitted ColumnTransformer."""
    try:
        return preprocess.get_feature_names_out().tolist()
    except Exception:
        return []


def export_preprocessing_schema(preprocess, numeric_cols, categorical_cols, output_path):
    """Save enough preprocessing metadata for browser-side feature construction."""
    scaler = preprocess.named_transformers_["num"]
    encoder = preprocess.named_transformers_["cat"]

    schema = {
        "target": TARGET,
        "numeric_columns": list(numeric_cols),
        "categorical_columns": list(categorical_cols),
        "feature_names_out": get_feature_names(preprocess),
        "numeric_standardization": {
            "mean": [float(x) for x in scaler.mean_],
            "scale": [float(x) for x in scaler.scale_],
        },
        "categorical_encoding": {
            "handle_unknown": "ignore",
            "categories": {
                col: [str(v) for v in cats]
                for col, cats in zip(categorical_cols, encoder.categories_)
            },
        },
        "notes": "Apply numeric standardization first, then append one-hot categorical features in the listed category order. Unknown categories map to all zeros for that column group.",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)

    return schema


def save_torch_state(model, model_name, input_dim, metrics, train_info, epochs=EPOCHS, lr=LR):
    """Save a PyTorch checkpoint suitable for retraining or later export."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / f"{model_name}.pt"
    checkpoint = {
        "model_name": model_name,
        "input_dim": int(input_dim),
        "state_dict": model.state_dict(),
        "metrics": metrics,
        "train_info": train_info,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate":LR,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def export_onnx(model, model_name, input_dim):
    """Export model to ONNX for browser deployment with onnxruntime-web."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = EXPORT_DIR / f"{model_name}.onnx"
    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["features"],
        output_names=["logit"],
        dynamic_axes={"features": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=17,
    )
    return onnx_path


def file_size_mb(path):
    return float(path.stat().st_size / (1024 ** 2)) if path.exists() else None


def parse_args():
    parser = argparse.ArgumentParser(description="Train bank churn ANN models and export deployment artifacts.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs per model.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate for Adam.")
    parser.add_argument("--progress", action="store_true", help="Print training progress during each model run.")
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY, help="Print progress every N epochs, plus first/last/improvements.")
    parser.add_argument("--models", nargs="+", default=["tiny", "small", "medium", "large"], choices=["tiny", "small", "medium", "large"], help="Subset of models to train.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.progress = True
    if args.progress:
        print("Loading data and preparing train/validation/test split...", flush=True)

    df = pd.read_csv(DATA_PATH)

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
        random_state=SEED,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=SEED,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train_np = preprocess.fit_transform(X_train)
    X_val_np = preprocess.transform(X_val)
    X_test_np = preprocess.transform(X_test)

    if hasattr(X_train_np, "toarray"):
        X_train_np = X_train_np.toarray()
        X_val_np = X_val_np.toarray()
        X_test_np = X_test_np.toarray()

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)

    y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1)

    input_dim = int(X_train_t.shape[1])

    models = {
        "tiny": Tiny,
        "small": Small,
        "medium": Medium,
        "large": Large,
    }

    benchmark_best = load_benchmark_best(BENCHMARK_PATH)
    models = {name: model_def for name, model_def in models.items() if name in set(args.models)}

    if args.progress:
        print(f"Input dimension after preprocessing: {input_dim}", flush=True)
        print(f"Models selected: {', '.join(models.keys())}", flush=True)
        if benchmark_best is not None:
            print(f"Loaded benchmark best: {benchmark_best.get('model')} roc_auc={benchmark_best.get('roc_auc')}", flush=True)
    results = []
    trained_models = {}

    for name, model_def in models.items():
        trained_model, train_info = train_model(
            model_def,
            X_train_t,
            y_train_t,
            X_val_t,
            y_val_t,
            model_name=name,
            epochs=args.epochs,
            lr=args.lr,
            progress=args.progress,
            progress_every=args.progress_every,
        )
        y_prob = predict_proba(trained_model, X_test_t)

        result = {
            "model": name,
            "input_dim": input_dim,
            "parameter_count": count_parameters(trained_model),
            "estimated_fp32_size_mb": estimate_fp32_size_mb(trained_model),
            "best_epoch": train_info["best_epoch"],
            "best_val_loss": train_info["best_val_loss"],
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "brier_score": float(brier_score_loss(y_test, y_prob)),
            "log_loss": float(log_loss(y_test, np.clip(y_prob, 1e-7, 1 - 1e-7))),
            "recall_at_top_20_pct": safe_top_k(recall_at_top_k, y_test, y_prob, k=TOP_K),
            "precision_at_top_20_pct": safe_top_k(precision_at_top_k, y_test, y_prob, k=TOP_K),
        }

        if benchmark_best is not None:
            result["benchmark_model"] = benchmark_best.get("model")
            result["benchmark_roc_auc"] = float(benchmark_best.get("roc_auc", np.nan))
            result["roc_auc_delta_vs_benchmark"] = result["roc_auc"] - result["benchmark_roc_auc"]

        checkpoint_path = save_torch_state(trained_model, name, input_dim, result, train_info, epochs=args.epochs, lr=args.lr)
        result["checkpoint_path"] = str(checkpoint_path)
        result["checkpoint_size_mb"] = file_size_mb(checkpoint_path)

        trained_models[name] = trained_model
        results.append(result)

        if args.progress:
            print(
                f"[{name}] metrics | roc_auc={result['roc_auc']:.4f} "
                f"pr_auc={result['pr_auc']:.4f} "
                f"recall@20={result['recall_at_top_20_pct']:.4f} "
                f"precision@20={result['precision_at_top_20_pct']:.4f} "
                f"params={result['parameter_count']}",
                flush=True,
            )
            print(f"[{name}] saved checkpoint: {checkpoint_path}", flush=True)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print(results_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    preprocessing_schema_path = EXPORT_DIR / "preprocessing_schema.json"
    export_preprocessing_schema(preprocess, numeric_cols, categorical_cols, preprocessing_schema_path)

    best_row = results_df.iloc[0].to_dict()
    best_model_name = str(best_row["model"])
    best_model = trained_models[best_model_name]

    best_checkpoint_path = MODEL_DIR / "best_model.pt"
    torch.save(
        {
            "model_name": best_model_name,
            "input_dim": input_dim,
            "state_dict": best_model.state_dict(),
            "metrics": best_row,
            "seed": SEED,
            "epochs": args.epochs,
            "learning_rate": args.lr,
        },
        best_checkpoint_path,
    )

    try:
        onnx_path = export_onnx(best_model, "best_model", input_dim)
        best_row["onnx_path"] = str(onnx_path)
        best_row["onnx_size_mb"] = file_size_mb(onnx_path)
    except Exception as exc:
        best_row["onnx_export_error"] = str(exc)

    manifest = {
        "selected_model": best_model_name,
        "selection_metric": "roc_auc",
        "input_dim": input_dim,
        "target": TARGET,
        "seed": SEED,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "best_model_checkpoint": str(best_checkpoint_path),
        "preprocessing_schema": str(preprocessing_schema_path),
        "best_model_metrics": best_row,
        "all_results_path": str(OUTPUT_PATH),
    }

    manifest_path = EXPORT_DIR / "deployment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nSaved artifacts:")
    print(f"- All checkpoints: {MODEL_DIR}")
    print(f"- Best checkpoint: {best_checkpoint_path}")
    print(f"- Deployment manifest: {manifest_path}")
    print(f"- Preprocessing schema: {preprocessing_schema_path}")
    if "onnx_path" in best_row:
        print(f"- Best ONNX model: {best_row['onnx_path']}")


if __name__ == "__main__":
    main()
