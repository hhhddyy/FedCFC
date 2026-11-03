from __future__ import annotations

import os
from pathlib import Path
import tempfile
from collections.abc import Sequence
from typing import Any

import torch

from .model import FedCFCClassifier


def create_checkpoint(
    models: Sequence[FedCFCClassifier],
    *,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    class_names: list[str],
    run_config: dict[str, Any],
    round_index: int,
    metrics: dict[str, Any],
    client_samples: list[int],
    client_class_counts: list[list[int]],
) -> dict[str, Any]:
    """Snapshot client weights and evaluation metadata on CPU."""
    return {
        "format_version": 1,
        "purpose": "evaluation",
        "torch_version": str(torch.__version__),
        "model_config": dict(model_config),
        "preprocessing": {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in preprocessing.items()
        },
        "class_names": list(class_names),
        "run_config": dict(run_config),
        "round": round_index,
        "metrics": dict(metrics),
        "client_samples": list(client_samples),
        "client_class_counts": client_class_counts,
        "client_states": [
            {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            for model in models
        ],
    }


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Save a checkpoint using atomic replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate checkpoint data on CPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("unsupported FedCFC checkpoint format")
    required = {"model_config", "preprocessing", "class_names", "run_config",
                "round", "client_states"}
    if not required.issubset(checkpoint):
        raise ValueError("incomplete FedCFC checkpoint")
    if not checkpoint["client_states"]:
        raise ValueError("checkpoint contains no client models")
    if len(checkpoint["class_names"]) != checkpoint["model_config"]["num_classes"]:
        raise ValueError("checkpoint class mapping does not match its model")
    return checkpoint


def restore_model(checkpoint: dict[str, Any], client_id: int) -> FedCFCClassifier:
    """Recreate one client model from its saved architecture and weights."""
    if not 0 <= client_id < len(checkpoint["client_states"]):
        raise ValueError(f"client_id must be in [0, {len(checkpoint['client_states']) - 1}]")
    model = FedCFCClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["client_states"][client_id], strict=True)
    model.eval()
    return model
