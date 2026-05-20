#!/usr/bin/env python3
"""
Export the Tiny bank-churn ANN PyTorch checkpoint to ONNX for static browser inference.

Place this file in the training repo root and run, for example:

  python export_ann_to_onnx.py

Default behavior exports artifacts/models/tiny.pt to artifacts/deployment/best_model.onnx.
Use --allow-non-tiny only for debugging or intentionally exporting another architecture.

This script infers the ANN architecture from checkpoint tensor shapes, but by
default it REQUIRES the Tiny checkpoint:
  Tiny: input -> 32 -> 16 -> 1

This prevents accidentally publishing Small as the browser demo model just because
best_model.pt had slightly better validation ROC-AUC.

It exports logits, not sigmoid probabilities. The browser app should apply sigmoid.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class Linear3ANN(nn.Module):
    """ANN with explicit layer names matching project checkpoints."""

    def __init__(self, input_dim: int, hidden1: int, hidden2: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden1)
        self.linear2 = nn.Linear(hidden1, hidden2)
        self.linear3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def unwrap_checkpoint(obj: Any) -> Dict[str, torch.Tensor]:
    """Return a plain state_dict from common checkpoint formats."""
    if isinstance(obj, nn.Module):
        return obj.state_dict()

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)!r}")

    # Common wrappers used by training loops.
    for key in ("model_state_dict", "state_dict", "model", "net"):
        value = obj.get(key)
        if isinstance(value, dict):
            obj = value
            break
        if isinstance(value, nn.Module):
            return value.state_dict()

    # Strip DataParallel-style prefixes if present.
    state: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            nk = k
            for prefix in ("module.", "model.", "net."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            state[nk] = v

    if not state:
        raise ValueError("Could not find tensor weights in checkpoint.")

    return state


def infer_architecture(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """Infer input_dim, hidden1, hidden2 from named linear layer weights."""
    required = ["linear1.weight", "linear2.weight", "linear3.weight"]
    missing = [k for k in required if k not in state]
    if missing:
        raise KeyError(
            "Checkpoint is missing expected keys: "
            + ", ".join(missing)
            + "\nFound keys include: "
            + ", ".join(list(state.keys())[:20])
        )

    w1 = state["linear1.weight"]
    w2 = state["linear2.weight"]
    w3 = state["linear3.weight"]

    if w1.ndim != 2 or w2.ndim != 2 or w3.ndim != 2:
        raise ValueError("Expected all linear*.weight tensors to be 2D.")

    hidden1, input_dim = int(w1.shape[0]), int(w1.shape[1])
    hidden2, hidden1_from_w2 = int(w2.shape[0]), int(w2.shape[1])
    output_dim, hidden2_from_w3 = int(w3.shape[0]), int(w3.shape[1])

    if hidden1_from_w2 != hidden1:
        raise ValueError(f"linear2 input size {hidden1_from_w2} != linear1 output size {hidden1}")
    if hidden2_from_w3 != hidden2:
        raise ValueError(f"linear3 input size {hidden2_from_w3} != linear2 output size {hidden2}")
    if output_dim != 1:
        raise ValueError(f"Expected binary-classification logit output size 1; got {output_dim}")

    return input_dim, hidden1, hidden2


def architecture_label(hidden1: int, hidden2: int) -> str:
    if (hidden1, hidden2) == (32, 16):
        return "tiny"
    if (hidden1, hidden2) == (64, 32):
        return "small"
    if (hidden1, hidden2) == (128, 64):
        return "medium_or_partial_medium"
    return f"custom_{hidden1}_{hidden2}"


def infer_input_dim_from_schema(schema: Dict[str, Any]) -> int | None:
    for key in (
        "input_dim",
        "n_features",
        "num_features_after_preprocessing",
        "transformed_feature_count",
    ):
        value = schema.get(key)
        if isinstance(value, int) and value > 0:
            return value

    for key in (
        "feature_names_out",
        "transformed_feature_names",
        "encoded_feature_names",
        "input_features",
    ):
        value = schema.get(key)
        if isinstance(value, list) and value:
            return len(value)

    numeric = schema.get("numeric_features") or schema.get("numeric_columns") or []
    categories = schema.get("categories") or schema.get("categorical_categories") or {}
    if isinstance(numeric, list) and isinstance(categories, dict):
        return len(numeric) + sum(len(v) for v in categories.values() if isinstance(v, list))

    return None


def export_onnx(
    model: nn.Module,
    output_path: Path,
    input_dim: int,
    opset: int,
    dynamic_batch: bool,
) -> None:
    model.eval()
    dummy = torch.zeros((1, input_dim), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"features": {0: "batch_size"}, "logits": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )


def validate_export(model: nn.Module, output_path: Path, input_dim: int) -> Dict[str, Any]:
    """Validate ONNX numerically when onnxruntime is installed."""
    result: Dict[str, Any] = {"onnxruntime_available": False, "checked": False}
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception as exc:
        result["reason"] = f"onnxruntime validation skipped: {exc}"
        return result

    torch.manual_seed(123)
    x = torch.randn((5, input_dim), dtype=torch.float32)
    with torch.no_grad():
        torch_logits = model(x).detach().cpu().numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"features": x.cpu().numpy().astype(np.float32)})[0]
    max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))

    result.update(
        {
            "onnxruntime_available": True,
            "checked": True,
            "max_abs_diff": max_abs_diff,
            "passed": max_abs_diff < 1e-5,
        }
    )
    return result


def build_manifest(
    checkpoint: Path,
    schema: Path,
    output: Path,
    model_name: str,
    arch_label: str,
    input_dim: int,
    hidden1: int,
    hidden2: int,
    opset: int,
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "detected_architecture": arch_label,
        "architecture": {
            "input_dim": input_dim,
            "hidden_layers": [hidden1, hidden2],
            "output_dim": 1,
            "hidden_activation": "relu",
            "output": "single binary-classification logit",
            "browser_probability_rule": "sigmoid(logit)",
        },
        "source_checkpoint": str(checkpoint),
        "preprocessing_schema": str(schema),
        "onnx_model": str(output),
        "onnx": {
            "opset_version": opset,
            "input_name": "features",
            "output_name": "logits",
            "dynamic_batch_axis": True,
        },
        "validation": validation,
        "notes": [
            "The web app consumes ONNX and preprocessing_schema.json only.",
            "The ONNX model outputs logits; apply sigmoid in JavaScript for probability.",
            "Deployment is intentionally locked to Tiny by default; use artifacts/models/tiny.pt for the browser app.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Tiny bank-churn ANN checkpoint to ONNX.")
    # parser.add_argument("--checkpoint", default="artifacts/models/best_model.pt")
    parser.add_argument("--checkpoint", default="artifacts/models/tiny.pt")
    parser.add_argument("--schema", default="artifacts/deployment/preprocessing_schema.json")
    parser.add_argument("--output", default="artifacts/deployment/best_model.onnx")
    parser.add_argument("--manifest", default="artifacts/deployment/deployment_manifest.json")
    parser.add_argument("--model-name", default="tiny", help="Manifest model name. Defaults to tiny.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-dynamic-batch", action="store_true")
    parser.add_argument("--allow-schema-mismatch", action="store_true")
    parser.add_argument(
        "--allow-non-tiny",
        action="store_true",
        help="Allow exporting Small/other checkpoints. Default is to fail unless the checkpoint is Tiny.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    schema_path = Path(args.schema)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    print(f"[info] checkpoint: {checkpoint_path}")
    print(f"[info] schema:     {schema_path}")
    print(f"[info] output:     {output_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location="cpu")
    state = unwrap_checkpoint(raw)
    input_dim, hidden1, hidden2 = infer_architecture(state)
    arch = architecture_label(hidden1, hidden2)
    model_name = args.model_name or arch

    print(f"[info] detected architecture: {input_dim} -> {hidden1} -> {hidden2} -> 1 ({arch})")

    if schema_path.exists():
        schema = read_json(schema_path)
        schema_dim = infer_input_dim_from_schema(schema)
        if schema_dim is not None:
            print(f"[info] schema input_dim: {schema_dim}")
            if schema_dim != input_dim and not args.allow_schema_mismatch:
                raise RuntimeError(
                    f"Schema transformed feature count ({schema_dim}) does not match "
                    f"checkpoint input_dim ({input_dim}). Fix the schema or pass --allow-schema-mismatch."
                )
    else:
        print("[warn] preprocessing schema not found; continuing without schema validation")

    model = Linear3ANN(input_dim=input_dim, hidden1=hidden1, hidden2=hidden2)
    model.load_state_dict(state, strict=True)
    model.eval()

    if arch != "tiny" and not args.allow_non_tiny:
        raise RuntimeError(
            f"Refusing to export {arch!r} because the browser deployment is locked to Tiny. "
            "Use --checkpoint artifacts/models/tiny.pt, or pass --allow-non-tiny only if you intentionally want another architecture."
        )

    if arch != "tiny":
        print(f"[warn] exporting non-Tiny architecture because --allow-non-tiny was passed: {arch}")
    else:
        print("[ok] confirmed Tiny architecture for browser deployment")

    export_onnx(
        model=model,
        output_path=output_path,
        input_dim=input_dim,
        opset=args.opset,
        dynamic_batch=not args.no_dynamic_batch,
    )
    print(f"[ok] wrote ONNX: {output_path}")

    validation = validate_export(model, output_path, input_dim)
    if validation.get("checked"):
        print(f"[info] ONNX validation max_abs_diff: {validation['max_abs_diff']:.3e}")
        if not validation.get("passed"):
            raise RuntimeError("ONNX validation failed; max_abs_diff >= 1e-5")
    else:
        print(f"[warn] {validation.get('reason', 'ONNX validation skipped')}")

    manifest = build_manifest(
        checkpoint=checkpoint_path,
        schema=schema_path,
        output=output_path,
        model_name=model_name,
        arch_label=arch,
        input_dim=input_dim,
        hidden1=hidden1,
        hidden2=hidden2,
        opset=args.opset,
        validation=validation,
    )
    write_json(manifest_path, manifest)
    print(f"[ok] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
