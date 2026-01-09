from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kaggle_train.train.model import ModelSpec, create_model
from kaggle_train.train.training import TrainConfig, fit


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _try_build_transforms(model):
    try:
        from timm.data import resolve_data_config  # type: ignore
        from timm.data.transforms_factory import create_transform  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `timm` for best-practice augmentation. Install it (`pip install timm`) or run on Kaggle."
        ) from exc

    cfg = resolve_data_config({}, model=model)
    train_tf = create_transform(**cfg, is_training=True)
    val_tf = create_transform(**cfg, is_training=False)
    return train_tf, val_tf, cfg


def _resolve_splits(data_dir: Path) -> tuple[Path, Path | None]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    valid_dir = data_dir / "valid"
    if train_dir.exists():
        if val_dir.exists():
            return train_dir, val_dir
        if valid_dir.exists():
            return train_dir, valid_dir
        return train_dir, None
    return data_dir, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an image classifier on the butterfly dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_project_root() / "datasets" / "butterfly-image-classification",
        help="Dataset root directory (supports ImageFolder, or data-dir/train + optional val/valid).",
    )
    parser.add_argument(
        "--model",
        default="convnextv2_base",
        help="timm model name (recommended: convnextv2_base, swinv2_base_window12_192_22k, vit_base_patch16_224).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1, help="Used when no val/valid folder exists.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-multi-gpu", action="store_true", help="Disable multi-GPU (DataParallel).")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_project_root() / "outputs" / "butterfly_run",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--force", action="store_true", help="Overwrite run-dir if it exists.")
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    data_dir = args.data_dir
    if not data_dir.exists():
        kaggle_candidate = Path("/kaggle/input") / data_dir.name
        if kaggle_candidate.exists():
            data_dir = kaggle_candidate

    train_root, val_root = _resolve_splits(data_dir)
    if not train_root.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {train_root}. Run `python kaggle_train/scripts/get_dataset.py` first "
            "or pass `--data-dir /kaggle/input/<dataset-slug>` on Kaggle."
        )

    if args.run_dir.exists():
        if not args.force:
            print(f"Run dir already exists: {args.run_dir} (use --force to overwrite)")
            return 2
        for p in args.run_dir.glob("*"):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    base_train = ImageFolder(root=str(train_root))
    spec = ModelSpec(name=args.model, num_classes=len(base_train.classes), pretrained=True)
    model = create_model(spec)
    if torch.cuda.device_count() > 1 and not args.no_multi_gpu:
        print(f"Using torch.nn.DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model_for_cfg = model.module if hasattr(model, "module") else model
    train_tf, val_tf, data_cfg = _try_build_transforms(model_for_cfg)

    if val_root is not None and val_root.exists():
        train_ds = ImageFolder(root=str(train_root), transform=train_tf)
        val_ds = ImageFolder(root=str(val_root), transform=val_tf)
    else:
        full = ImageFolder(root=str(train_root), transform=train_tf)
        val_count = max(int(len(full) * args.val_split), 1)
        train_count = max(len(full) - val_count, 1)
        train_ds, val_ds = random_split(
            full,
            lengths=[train_count, val_count],
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_ds.dataset.transform = val_tf

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        amp=(not args.no_amp),
    )

    extra_state = {
        "model_spec": asdict(spec),
        "data_config": data_cfg,
        "classes": base_train.classes,
        "class_to_idx": base_train.class_to_idx,
    }
    summary = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=args.run_dir,
        extra_state=extra_state,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
