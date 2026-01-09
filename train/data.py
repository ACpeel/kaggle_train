from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

_FILE_COL_CANDIDATES = ("image", "img", "file", "filename", "path", "filepath")
_LABEL_COL_CANDIDATES = ("label", "class", "category", "species", "target")


def _has_images(path: Path) -> bool:
    if not path.exists():
        return False
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            return True
    return False


def _is_imagefolder_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    for child in root.iterdir():
        if child.is_dir():
            for p in child.iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    return True
    return False


def _normalize_col(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def _infer_csv_columns(fieldnames: Iterable[str]) -> tuple[str | None, str | None]:
    normalized = [_normalize_col(c) for c in fieldnames]

    file_col = None
    label_col = None
    for cand in _FILE_COL_CANDIDATES:
        if cand in normalized:
            file_col = cand
            break
    for cand in _LABEL_COL_CANDIDATES:
        if cand in normalized:
            label_col = cand
            break

    if file_col is None or label_col is None:
        return None, None
    return file_col, label_col


def _pick_best_csv(data_dir: Path, split_hint: str) -> Path | None:
    candidates = sorted({*data_dir.glob("*.csv"), *data_dir.rglob("*.csv")})
    if not candidates:
        return None

    def score(path: Path) -> int:
        s = 0
        name = path.name.lower()
        if split_hint in name:
            s += 5
        if "label" in name or "train" in name:
            s += 2
        return s

    for csv_path in sorted(candidates, key=lambda p: (-score(p), str(p))):
        try:
            with csv_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                file_col, label_col = _infer_csv_columns(reader.fieldnames)
                if file_col and label_col:
                    return csv_path
        except OSError:
            continue

    return None


def _build_image_index(search_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in _IMAGE_EXTS:
                continue
            index.setdefault(p.name, p)
    return index


def _resolve_image_path(value: str, search_dirs: list[Path], index: dict[str, Path]) -> Path | None:
    p = Path(value)
    if p.is_absolute() and p.exists():
        return p

    for d in search_dirs:
        candidate = d / value
        if candidate.exists():
            return candidate

    return index.get(p.name)


class TransformDataset(Dataset):
    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, target = self.base[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class CsvImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


@dataclass(frozen=True)
class DatasetBundle:
    base: Dataset
    classes: list[str]
    class_to_idx: dict[str, int]


def load_trainable_dataset(
    data_dir: Path,
    split_dir: Path,
    split_hint: str,
    csv_path: Path | None = None,
    images_dir: Path | None = None,
) -> DatasetBundle:
    if _is_imagefolder_root(split_dir):
        base = ImageFolder(root=str(split_dir), transform=None)
        return DatasetBundle(base=base, classes=base.classes, class_to_idx=base.class_to_idx)

    if csv_path is None:
        csv_path = _pick_best_csv(data_dir, split_hint=split_hint)
    if csv_path is None:
        raise FileNotFoundError(
            f"Couldn't find ImageFolder class folders under {split_dir} and no suitable CSV labels file under {data_dir}."
        )

    search_dirs: list[Path] = []
    if images_dir is not None:
        search_dirs.append(images_dir)
    if split_dir != data_dir:
        search_dirs.append(split_dir)
    search_dirs.append(data_dir)

    if not any(_has_images(d) for d in search_dirs):
        raise FileNotFoundError(f"Couldn't find any images under: {', '.join(str(d) for d in search_dirs)}")

    with csv_path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        file_col, label_col = _infer_csv_columns(reader.fieldnames)
        if not file_col or not label_col:
            raise ValueError(f"CSV header must include filename + label columns: {csv_path} (got {reader.fieldnames})")

        index = _build_image_index(search_dirs)
        rows = list(reader)

    label_values: list[str] = []
    resolved: list[tuple[Path, str]] = []
    for row in rows:
        filename = row.get(file_col) or row.get(file_col.upper()) or row.get(file_col.title())
        label = row.get(label_col) or row.get(label_col.upper()) or row.get(label_col.title())
        if not filename or label is None:
            continue
        path = _resolve_image_path(str(filename), search_dirs=search_dirs, index=index)
        if path is None:
            continue
        label_str = str(label)
        resolved.append((path, label_str))
        label_values.append(label_str)

    if not resolved:
        raise FileNotFoundError(
            f"Found CSV {csv_path} but couldn't resolve image paths. "
            f"Try passing `--images-dir` or `--train-csv` explicitly."
        )

    classes = sorted(set(label_values))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = [(p, class_to_idx[lbl]) for (p, lbl) in resolved]
    base = CsvImageDataset(samples=samples, transform=None)
    return DatasetBundle(base=base, classes=classes, class_to_idx=class_to_idx)


def make_split_indices(n: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    val_count = max(int(n * val_split), 1)
    train_count = max(n - val_count, 1)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:train_count]
    val_idx = perm[train_count : train_count + val_count]
    return train_idx, val_idx

