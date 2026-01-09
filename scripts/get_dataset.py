from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_DATASET_HANDLE = "phucthaiv02/butterfly-image-classification"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_dataset_slug(handle: str) -> str:
    return handle.split("/", 1)[-1]


def _kaggle_input_path(slug: str) -> Path | None:
    candidate = Path("/kaggle/input") / slug
    return candidate if candidate.exists() else None


def _try_kagglehub_download(handle: str) -> Path | None:
    try:
        import kagglehub  # type: ignore
    except Exception:
        return None

    path_str = kagglehub.dataset_download(handle)
    return Path(path_str)


def _run_kaggle_cli_download(handle: str, out_dir: Path) -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "kaggle CLI not found. Install it with `pip install kaggle` and configure Kaggle API credentials."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        handle,
        "-p",
        str(out_dir),
        "--unzip",
    ]
    subprocess.run(cmd, check=True)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _materialize_dataset(src_dir: Path, dest_dir: Path, mode: str) -> None:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        try:
            os.symlink(src_dir, dest_dir, target_is_directory=True)
            return
        except OSError:
            mode = "copy"

    if mode == "copy":
        shutil.copytree(src_dir, dest_dir)
        return

    raise ValueError(f"Unsupported mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the butterfly image classification dataset into kaggle_train/datasets."
    )
    parser.add_argument(
        "--handle",
        default=DEFAULT_DATASET_HANDLE,
        help="Kaggle dataset handle, e.g. owner/dataset-slug",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: kaggle_train/datasets/<dataset-slug>)",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to place the dataset in out-dir when a source path exists (default: symlink).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite out-dir if it already exists.",
    )
    args = parser.parse_args(argv)

    slug = _default_dataset_slug(args.handle)
    out_dir = args.out_dir or (_project_root() / "datasets" / slug)

    if out_dir.exists():
        if not args.force:
            print(f"Dataset already exists: {out_dir}")
            return 0
        _remove_path(out_dir)

    kaggle_input = _kaggle_input_path(slug)
    if kaggle_input is not None:
        _materialize_dataset(kaggle_input, out_dir, args.mode)
        print(f"Dataset ready at: {out_dir}")
        print(f"Source: {kaggle_input}")
        return 0

    kagglehub_path = _try_kagglehub_download(args.handle)
    if kagglehub_path is not None:
        _materialize_dataset(kagglehub_path, out_dir, args.mode)
        print(f"Dataset ready at: {out_dir}")
        print(f"Source: {kagglehub_path}")
        return 0

    try:
        _run_kaggle_cli_download(args.handle, out_dir)
    except Exception as exc:
        print("Failed to download dataset.")
        print(f"Reason: {exc}")
        print(
            "Fix: install Kaggle CLI (`pip install kaggle`) and set up API credentials in `~/.kaggle/kaggle.json`, "
            "or run this on Kaggle with the dataset added as an input."
        )
        return 2

    print(f"Dataset downloaded at: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
