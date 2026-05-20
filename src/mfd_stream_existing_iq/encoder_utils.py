from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..common.checkpoint import read_state_dict_from_checkpoint

try:
    from src.ResNet.ResNet import MainModel
except Exception:  # pragma: no cover - local fallback
    from .ResNet import MainModel  # type: ignore



def _to_jsonable(value: Any):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return str(value)



def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)



def _infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> int:
    for key in ("net.fc.weight", "classifier.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])
    raise KeyError("Could not infer num_classes from checkpoint state_dict.")



def _infer_in_channels_from_state_dict(state_dict: dict[str, Any]) -> int:
    for key in ("net.conv1.weight", "backbone.0.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[1])
    return 2



def _infer_projection_dim_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    for key in ("classifier.weight", "model.classifier.weight"):
        value = state_dict.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[1])
    for key in ("proj.0.weight", "model.proj.0.weight"):
        value = state_dict.get(key)
        if value is not None and hasattr(value, "shape") and len(value.shape) == 2:
            return int(value.shape[0])
    return None



def _uses_projection_head(state_dict: dict[str, Any]) -> bool:
    return any(str(k).startswith("proj.") or str(k).startswith("classifier.") for k in state_dict.keys())



def build_frozen_dual_backbone(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    strict: bool = True,
) -> tuple[torch.nn.Module, int, dict[str, Any], torch.device]:
    device_obj = resolve_device(device)
    state_dict, meta = read_state_dict_from_checkpoint(str(checkpoint_path), map_location=device_obj)
    num_classes = _infer_num_classes_from_state_dict(state_dict)
    in_channels = _infer_in_channels_from_state_dict(state_dict)
    use_projection_head = _uses_projection_head(state_dict)
    projection_dim = _infer_projection_dim_from_state_dict(state_dict) or 128

    model = MainModel(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=False,
        auto_shape=True,
        force_3ch=False,
        use_embedding_head=use_projection_head,
        out_dim=projection_dim,
    )
    model.load_state_dict(state_dict, strict=strict)
    model.to(device_obj)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    emb_dim = int(getattr(model, "embedding_dim", getattr(model, "backbone_output_dim", 512)))
    info = {
        "checkpoint_path": str(checkpoint_path),
        "num_classes": int(num_classes),
        "in_channels": int(in_channels),
        "backbone_embedding_dim": int(getattr(model, "backbone_output_dim", emb_dim)),
        "sequence_embedding_dim": int(emb_dim),
        "uses_projection_head": bool(use_projection_head),
        "projection_dim": int(projection_dim) if use_projection_head else None,
        "checkpoint_meta": _to_jsonable(meta),
    }
    return model, emb_dim, info, device_obj



def encode_dual_spectrogram_batch(
    model: torch.nn.Module,
    dual_specs_u8: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    if dual_specs_u8.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    x = torch.from_numpy(np.asarray(dual_specs_u8, dtype=np.uint8)).to(device, non_blocking=True)
    with torch.no_grad():
        emb = model.extract_sequence_embedding(x)
    return emb.detach().cpu().to(torch.float32).numpy().copy()


def encode_dual_spectrogram_batch_with_logits(
    model: torch.nn.Module,
    dual_specs_u8: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if dual_specs_u8.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    x = torch.from_numpy(np.asarray(dual_specs_u8, dtype=np.uint8)).to(device, non_blocking=True)
    with torch.no_grad():
        emb, logits = model.extract_sequence_embedding_and_logits(x)
    emb_np = emb.detach().cpu().to(torch.float32).numpy().copy()
    logits_np = logits.detach().cpu().to(torch.float32).numpy().copy()
    return emb_np, logits_np
