from __future__ import annotations

# Standard library
import copy
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Third-party
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


# ---------- minimal, resume-friendly checkpointing ----------
def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _resolve_map_location(map_location: str | torch.device = "auto"):
    if map_location == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return map_location


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(isinstance(k, str) for k in obj.keys())


def read_state_dict_from_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Read a state_dict from several checkpoint bundle styles used in this project.

    Supported payloads include:
    - {"state_dict": ...}
    - {"model_state_dict": ...}
    - {"model": nn.Module} or {"model": state_dict}
    - inference-only bundles
    - raw state_dict dictionaries
    - direct nn.Module pickles
    """
    resolved_map = _resolve_map_location(map_location)
    try:
        payload = torch.load(path, map_location=resolved_map, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=resolved_map)

    meta: dict[str, Any] = {}
    state_dict: dict[str, Any] | None = None

    if isinstance(payload, nn.Module):
        state_dict = payload.state_dict()
    elif isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "model_weights", "weights"):
            if key not in payload:
                continue
            candidate = payload[key]
            if isinstance(candidate, nn.Module):
                state_dict = candidate.state_dict()
                break
            if _looks_like_state_dict(candidate):
                state_dict = candidate
                break

        if state_dict is None and _looks_like_state_dict(payload):
            tensorish = [hasattr(v, "shape") or torch.is_tensor(v) for v in payload.values()]
            if tensorish and all(tensorish):
                state_dict = payload

        meta = {k: v for k, v in payload.items() if k not in {"state_dict", "model_state_dict", "model", "model_weights", "weights"}}

    if state_dict is None:
        raise ValueError(f"Could not extract a model state_dict from checkpoint: {path}")

    return _strip_module_prefix(state_dict), meta


def load_model_weights_from_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device = "auto",
    strict: bool = True,
) -> dict[str, Any]:
    state_dict, meta = read_state_dict_from_checkpoint(path, map_location=map_location)
    model.load_state_dict(state_dict, strict=strict)
    return meta


def save_minimal_checkpoint(
    out_path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,  # torch.cuda.amp.GradScaler or None
    label_to_id: dict | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    best_val: float | None = None,  # e.g., best val accuracy so far
    last_train_loss: float | None = None,
    extra_cfg: dict | None = None,  # any other small bits you want
):
    """
    Saves only what’s needed to resume training or run inference.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ckpt = {
        "format_version": 1,
        "timestamp": _timestamp(),
        "model_class": model.__class__.__name__,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": (
            scheduler.state_dict()
            if (scheduler is not None and hasattr(scheduler, "state_dict"))
            else None
        ),
        "scaler_state": (
            scaler.state_dict()
            if (scaler is not None and hasattr(scaler, "state_dict"))
            else None
        ),
        "label_to_id": label_to_id,
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "last_train_loss": last_train_loss,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "extra": extra_cfg or {},
    }
    torch.save(ckpt, out_path)
    return out_path


def load_minimal_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "auto",
    strict: bool = True,
):
    """
    Loads model (and optionally optimizer/scheduler/scaler).
    Returns a dict of metadata (epoch, best_val, label_to_id, etc.).
    """
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(path, map_location=map_location)

    # Model
    model.load_state_dict(ckpt["state_dict"], strict=strict)

    # Optimizer / scheduler / scaler (only if provided)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if (
        scheduler is not None
        and ckpt.get("scheduler_state") is not None
        and hasattr(scheduler, "load_state_dict")
    ):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if (
        scaler is not None
        and ckpt.get("scaler_state") is not None
        and hasattr(scaler, "load_state_dict")
    ):
        scaler.load_state_dict(ckpt["scaler_state"])

    meta = {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "best_val": ckpt.get("best_val"),
        "last_train_loss": ckpt.get("last_train_loss"),
        "label_to_id": ckpt.get("label_to_id"),
        "env": ckpt.get("env", {}),
        "extra": ckpt.get("extra", {}),
    }
    return meta


def save_inference_only(
    out_path: str, model: torch.nn.Module, label_to_id: dict | None = None
):
    """
    Super tiny bundle: weights + label map. Perfect for inference-only usage.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_class": model.__class__.__name__,
            "state_dict": model.state_dict(),
            "label_to_id": label_to_id,
            "timestamp": _timestamp(),
        },
        out_path,
    )
    return out_path


def save_best_checkpoint(
    *,
    run_dir: str,
    ckpt_file: str,
    model: nn.Module,
    opt,
    ep: int,
    stats_by_snr: dict,
    val_loss_macro: float,
    acc_macro: float,
    snr_keys: list[int],
    backbone: str,
):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_state = copy.deepcopy(model.state_dict())
    torch.save(
        {
            "epoch": ep,
            "model_state_dict": best_state,
            "optimizer_state_dict": opt.state_dict(),
            "thresholds_by_snr": {
                s: stats_by_snr[s]["thresholds"].cpu() for s in snr_keys
            },
            "val_macro": {"loss": val_loss_macro, "accuracy": acc_macro},
            "val_by_snr": {s: stats_by_snr[s] for s in snr_keys},
            "notes": {
                "baseline": f"{backbone} single-frame (frame-level accuracy)",
                "image_mode": True,
            },
        },
        os.path.join(ckpt_dir, ckpt_file),
    )
