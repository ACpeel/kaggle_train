from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from kaggle_train.utils.eval import count_correct_top1


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    amp: bool = True
    grad_clip_norm: float | None = 1.0


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loss(label_smoothing: float) -> nn.Module:
    try:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    except TypeError:  # pragma: no cover
        return nn.CrossEntropyLoss()


def _build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(cfg.warmup_epochs, 1)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.epochs - cfg.warmup_epochs, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
    )


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, targets)

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += count_correct_top1(logits, targets)
        total_count += int(batch_size)

    return {
        "val_loss": total_loss / max(total_count, 1),
        "val_acc": total_correct / max(total_count, 1),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    grad_clip_norm: float | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += count_correct_top1(logits, targets)
        total_count += int(batch_size)

    return {
        "train_loss": total_loss / max(total_count, 1),
        "train_acc": total_correct / max(total_count, 1),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    run_dir: Path,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, float]:
    device = _device()
    model.to(device)

    run_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = _build_loss(cfg.label_smoothing).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_val_acc = -1.0
    best_path = run_dir / "best.pt"

    model_to_save = model.module if hasattr(model, "module") else model

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler if scaler.is_enabled() else None,
            device=device,
            grad_clip_norm=cfg.grad_clip_norm,
        )
        val_metrics = validate(model=model, loader=val_loader, loss_fn=loss_fn, device=device)
        scheduler.step()

        metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        print(metrics)

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = float(val_metrics["val_acc"])
            state = {
                "model_state_dict": model_to_save.state_dict(),
                "metrics": metrics,
                "train_config": asdict(cfg),
            }
            if extra_state:
                state.update(extra_state)
            torch.save(state, best_path)

    return {"best_val_acc": best_val_acc}
