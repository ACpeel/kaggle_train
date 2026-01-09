from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_classes: int
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


def create_model(spec: ModelSpec):
    try:
        import timm  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `timm`. Install it (`pip install timm`) or run on Kaggle where it is preinstalled."
        ) from exc

    return timm.create_model(
        spec.name,
        pretrained=spec.pretrained,
        num_classes=spec.num_classes,
        drop_rate=spec.drop_rate,
        drop_path_rate=spec.drop_path_rate,
    )
