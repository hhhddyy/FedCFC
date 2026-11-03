from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor, Tensor]],
    *,
    device: torch.device | str,
    num_classes: int,
) -> dict[str, Any]:
    """Evaluate without updates; confusion rows=true, columns=predicted."""
    model.to(device).eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total_loss = 0.0
    for inputs, targets, timespans in batches:
        targets = targets.to(device)
        logits = model(inputs.to(device), timespans.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("model produced non-finite test logits")
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
        predictions = logits.argmax(dim=1)
        confusion += torch.bincount(
            (targets * num_classes + predictions).cpu(), minlength=num_classes**2
        ).reshape(num_classes, num_classes)
    support = confusion.sum(dim=1)
    samples = int(support.sum())
    if samples == 0:
        raise ValueError("evaluation loader is empty")
    recall = [float(confusion[i, i] / support[i]) if support[i] else None
              for i in range(num_classes)]
    return {
        "samples": samples,
        "loss": total_loss / samples,
        "accuracy": float(confusion.diag().sum()) / samples,
        "class_support": support.tolist(),
        "per_class_recall": recall,
        "confusion_matrix": confusion.tolist(),
    }
