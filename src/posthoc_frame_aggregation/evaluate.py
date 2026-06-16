from __future__ import annotations

"""Post-hoc aggregation of single-frame ResNet predictions over MFD-Net T-frame chunks.

This module deliberately keeps the baseline separate from MFD-Net internals.
It uses the MFD sequence evaluation loader only to recover the same T-frame
chunks, then runs the original single-frame ResNet checkpoint on each frame.

Main reviewer baseline:
    - average raw logits over the selected T frames, then argmax
    - majority vote over per-frame argmax predictions
    - confidence-weighted vote using max softmax probability

The sequence target remains the original MFD chunk/stream label yb. If the
single-frame model predicts the auxiliary noise class after aggregation, that is
counted as an incorrect sequence-level modulation prediction under the default
noise_policy='allow_noise'. This is the most literal apples-to-apples baseline.
"""

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.ResNet.ResNet import MainModel
from src.common.checkpoint import read_state_dict_from_checkpoint
from src.mfd_stream_train.trainer import (
    ExperimentConfig,
    _build_valid_mask,
    _experiment_config_from_checkpoint,
    _move_sequence_batch_to_device,
    _torch_load_checkpoint,
    _unpack_sequence_batch,
    build_eval_loader,
)

METHODS = ("avg_logits", "majority_vote", "confidence_weighted_vote")
FRAME_MASK_MODES = ("valid", "signal")
NOISE_POLICIES = ("allow_noise", "force_signal", "drop_noise_frames")
INPUT_SCALES = ("uint8_float", "zero_one", "imagenet")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return {"type": "Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _infer_num_classes(sd: dict[str, Any]) -> int:
    for key in ("classifier.weight", "net.fc.weight"):
        v = sd.get(key)
        if v is not None and hasattr(v, "shape") and len(v.shape) == 2:
            return int(v.shape[0])
    raise KeyError("Could not infer num_classes from checkpoint. Expected classifier.weight or net.fc.weight.")


def _infer_in_channels(sd: dict[str, Any]) -> int:
    for key in ("backbone.0.weight", "net.conv1.weight"):
        v = sd.get(key)
        if v is not None and hasattr(v, "shape") and len(v.shape) == 4:
            return int(v.shape[1])
    # The old project default is a 3-channel image model.
    return 3


def _uses_projection_head(sd: dict[str, Any]) -> bool:
    return "classifier.weight" in sd or any(str(k).startswith("proj.") for k in sd.keys())


def _infer_projection_dim(sd: dict[str, Any]) -> int:
    v = sd.get("classifier.weight")
    if v is not None and hasattr(v, "shape") and len(v.shape) == 2:
        return int(v.shape[1])
    v = sd.get("proj.0.weight")
    if v is not None and hasattr(v, "shape") and len(v.shape) == 2:
        return int(v.shape[0])
    return 128


def _load_singleframe_mainmodel(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    strict: bool = True,
    l2norm_embedding: bool = False,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the single-frame checkpoint using the same MainModel class as src/ResNet/ResNet_MemMap.py.

    This is intentionally not JanFusedBufferModel and not the Power/Phase wrapper.
    The normal single-frame memmap evaluation path calls MainModel(x), so this
    baseline does the same after flattening the T-frame chunks.
    """
    sd, meta = read_state_dict_from_checkpoint(str(checkpoint_path), map_location=device)
    num_classes = _infer_num_classes(sd)
    in_channels = _infer_in_channels(sd)
    use_head = _uses_projection_head(sd)
    out_dim = _infer_projection_dim(sd)

    model = MainModel(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=False,
        auto_shape=True,
        force_3ch=(in_channels == 3),
        use_embedding_head=use_head,
        out_dim=out_dim,
        l2norm_embedding=bool(l2norm_embedding),
    )
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint did not load cleanly: {checkpoint_path}\n"
            f"missing={missing}\nunexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    info = {
        "checkpoint_path": str(checkpoint_path),
        "model_class": "src.ResNet.ResNet.MainModel",
        "num_classes": int(num_classes),
        "in_channels": int(in_channels),
        "use_embedding_head": bool(use_head),
        "projection_dim": int(out_dim) if use_head else None,
        "force_3ch": bool(in_channels == 3),
        "l2norm_embedding": bool(l2norm_embedding),
        "checkpoint_meta": _jsonable(meta),
    }
    return model, info


def _resolve_device(device_name: str) -> torch.device:
    if str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def _override_eval_paths(
    cfg: ExperimentConfig,
    *,
    split: str,
    memmap_dir: str | None,
    noise_bank_dir: str | None,
) -> None:
    if split == "test":
        if memmap_dir is not None:
            cfg.test_memmap_dir = memmap_dir
            cfg.data_root_dir = None
        if noise_bank_dir is not None:
            cfg.test_noise_bank_dir = noise_bank_dir
            cfg.data_root_dir = None
    elif split == "val":
        if memmap_dir is not None:
            cfg.val_memmap_dir = memmap_dir
            cfg.data_root_dir = None
        if noise_bank_dir is not None:
            cfg.val_noise_bank_dir = noise_bank_dir
            cfg.data_root_dir = None
    else:
        raise ValueError("split must be 'val' or 'test'.")


def _normalise_frame_mask_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode not in FRAME_MASK_MODES:
        raise ValueError(f"frame_mask_mode must be one of {FRAME_MASK_MODES}, got {mode!r}.")
    return mode


def _normalise_noise_policy(policy: str) -> str:
    policy = str(policy).lower().strip()
    if policy not in NOISE_POLICIES:
        raise ValueError(f"noise_policy must be one of {NOISE_POLICIES}, got {policy!r}.")
    return policy


def _normalise_input_scale(scale: str) -> str:
    scale = str(scale).lower().strip()
    if scale not in INPUT_SCALES:
        raise ValueError(f"input_scale must be one of {INPUT_SCALES}, got {scale!r}.")
    return scale


def _select_frames_for_source(xb: Any, source: str) -> torch.Tensor:
    if isinstance(xb, dict):
        frames = xb.get("frames")
        if frames is None:
            raise ValueError("Hybrid MFD batch is missing the 'frames' entry.")
    else:
        frames = xb
    if frames.ndim != 5:
        raise ValueError(f"Expected sequence images [B,T,2,H,W], got {tuple(frames.shape)}")
    if frames.shape[2] < 2:
        raise ValueError(f"Expected at least power+phase channels at dim=2, got shape {tuple(frames.shape)}")

    source = str(source).lower()
    if source == "power":
        return frames[:, :, 0:1]
    if source == "phase":
        return frames[:, :, 1:2]
    if source == "dual":
        return frames[:, :, 0:2]
    raise ValueError("source must be 'power', 'phase', or 'dual'.")


def _prepare_flat_frames(frames_btchw: torch.Tensor, *, input_scale: str) -> torch.Tensor:
    b, t, c, h, w = frames_btchw.shape
    x = frames_btchw.reshape(b * t, c, h, w).contiguous().float()

    # MFD loader returns 0..1 floats when normalise_to_float=True. The current
    # single-frame memmap training/eval code casts uint8 images to float before
    # calling MainModel, so those checkpoints normally saw 0..255 float input.
    if input_scale == "uint8_float":
        # Do not double-scale if a future loader returns uint8-like floats.
        if x.numel() > 0 and float(x.detach().max().cpu()) <= 1.5:
            x = x * 255.0
        return x

    if input_scale == "zero_one":
        if x.numel() > 0 and float(x.detach().max().cpu()) > 2.0:
            x = x / 255.0
        return x

    if input_scale == "imagenet":
        if x.numel() > 0 and float(x.detach().max().cpu()) > 2.0:
            x = x / 255.0
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c == 2:
            x = torch.cat([x, x.mean(dim=1, keepdim=True)], dim=1)
        elif c != 3:
            raise ValueError(f"ImageNet normalisation expects 1, 2, or 3 channels, got {c}.")
        mean = torch.tensor((0.485, 0.456, 0.406), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    raise ValueError(f"Unexpected input_scale={input_scale!r}")


def _selected_frame_mask(
    *,
    valid_mask: torch.Tensor,
    signal_mask: torch.Tensor,
    frame_mask_mode: str,
) -> torch.Tensor:
    if frame_mask_mode == "valid":
        return valid_mask.bool()
    if frame_mask_mode == "signal":
        return valid_mask.bool() & signal_mask.bool()
    raise ValueError(f"Unknown frame_mask_mode={frame_mask_mode!r}")


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    """Guarantee at least one selected frame per row to avoid empty reductions."""
    out = mask.bool()
    empty = ~out.any(dim=1)
    if empty.any():
        out = out.clone()
        out[empty, 0] = True
    return out


def _aggregate_logits(
    logits: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    method: str,
    signal_class_count: int,
    noise_policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pred_all, effective_mask).

    pred_all is in 0..signal_class_count for checkpoints with a noise class at
    index signal_class_count. If noise_policy='force_signal', predictions are
    restricted to 0..signal_class_count-1.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,T,C], got {tuple(logits.shape)}")
    if selected_mask.shape != logits.shape[:2]:
        raise ValueError(f"selected_mask shape {tuple(selected_mask.shape)} does not match logits {tuple(logits.shape[:2])}")

    method = str(method).lower()
    b, t, c_all = logits.shape
    has_noise_class = c_all > signal_class_count
    noise_idx = signal_class_count if has_noise_class else None

    mask = selected_mask.bool()
    if noise_policy == "drop_noise_frames" and has_noise_class:
        frame_pred_all = logits.argmax(dim=-1)
        mask = mask & frame_pred_all.ne(int(noise_idx))
    mask = _safe_mask(mask)

    if method == "avg_logits":
        weights = mask.float().unsqueeze(-1)
        avg = (logits.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if noise_policy == "force_signal":
            pred = avg[:, :signal_class_count].argmax(dim=1)
        else:
            pred = avg.argmax(dim=1)
        return pred, mask

    if noise_policy == "force_signal":
        probs = torch.softmax(logits[:, :, :signal_class_count].float(), dim=-1)
        frame_conf, frame_pred = probs.max(dim=-1)
        vote_classes = signal_class_count
    else:
        probs = torch.softmax(logits.float(), dim=-1)
        frame_conf, frame_pred = probs.max(dim=-1)
        vote_classes = c_all

    votes = torch.zeros((b, vote_classes), dtype=logits.dtype, device=logits.device)
    if method == "majority_vote":
        vote_weight = mask.float()
    elif method == "confidence_weighted_vote":
        vote_weight = frame_conf * mask.float()
    else:
        raise ValueError(f"Unknown method={method!r}. Expected one of {METHODS}.")

    votes.scatter_add_(dim=1, index=frame_pred, src=vote_weight)
    return votes.argmax(dim=1), mask


def _rectangular_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    signal_class_count: int,
    pred_class_count: int,
) -> np.ndarray:
    cm = np.zeros((signal_class_count, pred_class_count), dtype=np.int64)
    for t, p in zip(y_true.reshape(-1), y_pred.reshape(-1)):
        ti = int(t)
        pi = int(p)
        if 0 <= ti < signal_class_count and 0 <= pi < pred_class_count:
            cm[ti, pi] += 1
    return cm


def _primary_accuracy(cm: np.ndarray, *, signal_class_count: int) -> float:
    total = float(cm.sum())
    if total <= 0:
        return 0.0
    correct = float(sum(cm[i, i] for i in range(signal_class_count) if i < cm.shape[1]))
    return correct / total


def _primary_macro_f1(cm: np.ndarray, *, signal_class_count: int) -> float:
    vals: list[float] = []
    for c in range(signal_class_count):
        tp = float(cm[c, c]) if c < cm.shape[1] else 0.0
        fp = float(cm[:, c].sum() - tp) if c < cm.shape[1] else 0.0
        fn = float(cm[c, :].sum() - tp)
        denom = 2.0 * tp + fp + fn
        vals.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _save_confusion_csv(cm: np.ndarray, row_labels: list[str], col_labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + col_labels)
        for i, label in enumerate(row_labels):
            writer.writerow([label] + [int(x) for x in cm[i]])


def _available_sources(cfg: ExperimentConfig, requested: Iterable[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for source in requested:
        s = str(source).lower().strip()
        if s == "power" and cfg.power_checkpoint:
            out.append(("power", str(cfg.power_checkpoint)))
        elif s == "phase" and cfg.phase_checkpoint:
            out.append(("phase", str(cfg.phase_checkpoint)))
        elif s == "dual" and cfg.dual_checkpoint:
            out.append(("dual", str(cfg.dual_checkpoint)))
        elif s in {"power", "phase", "dual"}:
            print(f"[skip] no checkpoint configured for source={s!r}")
        else:
            raise ValueError("sources must contain only 'power', 'phase', and/or 'dual'.")
    return out


def _apply_path_overrides(
    cfg: ExperimentConfig,
    *,
    power_checkpoint: str | None,
    phase_checkpoint: str | None,
    dual_checkpoint: str | None,
) -> None:
    if power_checkpoint is not None:
        cfg.power_checkpoint = power_checkpoint
    if phase_checkpoint is not None:
        cfg.phase_checkpoint = phase_checkpoint
    if dual_checkpoint is not None:
        cfg.dual_checkpoint = dual_checkpoint


@torch.no_grad()
def run_posthoc_sequence_baseline(
    *,
    mfd_checkpoint_path: str | Path,
    output_dir: str | Path,
    split: str = "test",
    sources: tuple[str, ...] = ("power", "phase"),
    frame_mask_mode: str = "valid",
    noise_policy: str = "allow_noise",
    input_scale: str = "uint8_float",
    device_name: str = "auto",
    batch_size: int | None = None,
    num_workers: int | None = 0,
    strict_checkpoint_load: bool = True,
    l2norm_embedding: bool = False,
    test_memmap_dir: str | None = None,
    test_noise_bank_dir: str | None = None,
    val_memmap_dir: str | None = None,
    val_noise_bank_dir: str | None = None,
    power_checkpoint: str | None = None,
    phase_checkpoint: str | None = None,
    dual_checkpoint: str | None = None,
) -> Path:
    """Evaluate post-hoc single-frame aggregation on MFD-Net sequence chunks."""
    split = str(split).lower().strip()
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'.")
    frame_mask_mode = _normalise_frame_mask_mode(frame_mask_mode)
    noise_policy = _normalise_noise_policy(noise_policy)
    input_scale = _normalise_input_scale(input_scale)
    device = _resolve_device(device_name)

    payload = _torch_load_checkpoint(mfd_checkpoint_path, map_location="cpu")
    cfg = _experiment_config_from_checkpoint(payload)
    _apply_path_overrides(
        cfg,
        power_checkpoint=power_checkpoint,
        phase_checkpoint=phase_checkpoint,
        dual_checkpoint=dual_checkpoint,
    )
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    if num_workers is not None:
        cfg.num_workers = int(num_workers)

    if split == "test":
        _override_eval_paths(cfg, split=split, memmap_dir=test_memmap_dir, noise_bank_dir=test_noise_bank_dir)
    else:
        _override_eval_paths(cfg, split=split, memmap_dir=val_memmap_dir, noise_bank_dir=val_noise_bank_dir)

    ds, loader = build_eval_loader(cfg, split_name=split)
    signal_labels = [str(x) for x in ds.label_names]
    signal_class_count = len(signal_labels)
    noise_label = "noise"

    selected_sources = _available_sources(cfg, sources)
    if not selected_sources:
        raise ValueError("No usable single-frame checkpoints were found for the requested sources.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    source_infos: dict[str, Any] = {}

    print("[posthoc sequence baseline]")
    print(f"  mfd checkpoint  = {mfd_checkpoint_path}")
    print(f"  split           = {split}")
    print(f"  frame_mask      = {frame_mask_mode}")
    print(f"  noise_policy    = {noise_policy}")
    print(f"  input_scale     = {input_scale}")
    print(f"  device          = {device}")
    print(f"  signal classes  = {signal_class_count}")
    print(f"  samples         = {len(ds)}")

    for source_name, ckpt_path in selected_sources:
        print(f"\n[source={source_name}] loading {ckpt_path}")
        model, info = _load_singleframe_mainmodel(
            ckpt_path,
            device=device,
            strict=strict_checkpoint_load,
            l2norm_embedding=l2norm_embedding,
        )
        source_infos[source_name] = info
        pred_class_count = int(info["num_classes"])
        has_noise = pred_class_count > signal_class_count
        col_labels = signal_labels + ([noise_label] if has_noise else [])
        if pred_class_count != len(col_labels):
            # Fallback if a checkpoint has an unexpected class count. The metrics
            # still work, but the CSV columns need labels.
            col_labels = [f"class_{i}" for i in range(pred_class_count)]
            for i, label in enumerate(signal_labels[: min(signal_class_count, pred_class_count)]):
                col_labels[i] = label
            if pred_class_count > signal_class_count:
                col_labels[signal_class_count] = noise_label

        preds_by_method: dict[str, list[torch.Tensor]] = {m: [] for m in METHODS}
        y_parts: list[torch.Tensor] = []
        effective_frame_counts: dict[str, list[torch.Tensor]] = {m: [] for m in METHODS}
        raw_selected_counts: list[torch.Tensor] = []
        noise_pred_frame_counts: list[torch.Tensor] = []

        for batch in tqdm(loader, desc=f"posthoc {split}/{source_name}", leave=False):
            xb, yb, idxb, sigb, window_classb = _unpack_sequence_batch(batch)
            valid_mask = _build_valid_mask(idxb, sigb, cfg.sequence_mask_mode).to(device, non_blocking=True)
            signal_mask = sigb.to(device, non_blocking=True).bool()
            xb = _move_sequence_batch_to_device(xb, device)
            yb = yb.to(device, non_blocking=True).long()

            frames = _select_frames_for_source(xb, source_name)
            b, t, c, h, w = frames.shape
            selected_mask = _selected_frames = _selected_frame_mask(
                valid_mask=valid_mask,
                signal_mask=signal_mask,
                frame_mask_mode=frame_mask_mode,
            )
            selected_mask = _safe_mask(selected_mask)
            raw_selected_counts.append(selected_mask.sum(dim=1).detach().cpu())

            x_flat = _prepare_flat_frames(frames, input_scale=input_scale)
            logits_flat = model(x_flat)
            if isinstance(logits_flat, (tuple, list)):
                logits_flat = logits_flat[0]
            if not torch.is_tensor(logits_flat):
                raise TypeError(f"Model returned unsupported output type: {type(logits_flat)!r}")
            logits = logits_flat.view(b, t, -1)

            if logits.shape[-1] <= signal_class_count:
                noise_frame_count = torch.zeros((b,), dtype=torch.long, device=device)
            else:
                noise_frame_count = ((logits.argmax(dim=-1) == signal_class_count) & selected_mask).sum(dim=1)
            noise_pred_frame_counts.append(noise_frame_count.detach().cpu())

            for method in METHODS:
                pred, effective_mask = _aggregate_logits(
                    logits,
                    selected_mask,
                    method=method,
                    signal_class_count=signal_class_count,
                    noise_policy=noise_policy,
                )
                preds_by_method[method].append(pred.detach().cpu())
                effective_frame_counts[method].append(effective_mask.sum(dim=1).detach().cpu())
            y_parts.append(yb.detach().cpu())

        y_true = torch.cat(y_parts, dim=0).numpy()
        raw_counts = torch.cat(raw_selected_counts, dim=0).float()
        noise_frame_counts_all = torch.cat(noise_pred_frame_counts, dim=0).float()

        for method in METHODS:
            y_pred = torch.cat(preds_by_method[method], dim=0).numpy()
            cm = _rectangular_confusion(
                y_true,
                y_pred,
                signal_class_count=signal_class_count,
                pred_class_count=pred_class_count if noise_policy != "force_signal" else signal_class_count,
            )
            method_col_labels = col_labels[: cm.shape[1]]
            acc = _primary_accuracy(cm, signal_class_count=signal_class_count)
            macro_f1 = _primary_macro_f1(cm, signal_class_count=signal_class_count)
            eff_counts = torch.cat(effective_frame_counts[method], dim=0).float()
            predicted_noise = int(np.sum(y_pred == signal_class_count)) if pred_class_count > signal_class_count else 0
            cm_path = output_dir / f"cm_{split}_{source_name}_{frame_mask_mode}_{noise_policy}_{method}.csv"
            _save_confusion_csv(cm, signal_labels, method_col_labels, cm_path)

            row = {
                "split": split,
                "source": source_name,
                "method": method,
                "frame_mask_mode": frame_mask_mode,
                "noise_policy": noise_policy,
                "input_scale": input_scale,
                "samples": int(y_true.size),
                "num_signal_classes": int(signal_class_count),
                "num_checkpoint_classes": int(pred_class_count),
                "primary_acc": float(acc),
                "primary_macro_f1": float(macro_f1),
                "avg_selected_frames": float(raw_counts.mean().item()) if raw_counts.numel() else 0.0,
                "avg_effective_frames": float(eff_counts.mean().item()) if eff_counts.numel() else 0.0,
                "avg_noise_argmax_frames": float(noise_frame_counts_all.mean().item()) if noise_frame_counts_all.numel() else 0.0,
                "sequence_predictions_as_noise": predicted_noise,
                "cm_csv": str(cm_path),
            }
            all_rows.append(row)
            print(
                f"{source_name:>8s} | {method:<24s} | mask={frame_mask_mode:<6s} | "
                f"noise={noise_policy:<16s} | acc={acc:.4f} | macro_f1={macro_f1:.4f} | "
                f"avg_frames={row['avg_effective_frames']:.2f} | seq_noise={predicted_noise}"
            )

    metrics_json = output_dir / f"posthoc_sequence_metrics_{split}_{frame_mask_mode}_{noise_policy}.json"
    metrics_json.write_text(
        json.dumps(
            {
                "mfd_checkpoint_path": str(mfd_checkpoint_path),
                "split": split,
                "frame_mask_mode": frame_mask_mode,
                "noise_policy": noise_policy,
                "input_scale": input_scale,
                "labels": signal_labels,
                "source_infos": source_infos,
                "experiment_config": _jsonable(asdict(cfg)),
                "results": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_csv = output_dir / f"posthoc_sequence_metrics_{split}_{frame_mask_mode}_{noise_policy}.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n[done] wrote {metrics_json}")
    print(f"[done] wrote {metrics_csv}")
    return output_dir
