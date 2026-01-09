from __future__ import annotations

import torch


@torch.no_grad()
def count_correct_top1(logits: torch.Tensor, targets: torch.Tensor) -> int:
    preds = torch.argmax(logits, dim=1)
    return int((preds == targets).sum().item())


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    correct = count_correct_top1(logits, targets)
    return float(correct / max(int(targets.numel()), 1))
