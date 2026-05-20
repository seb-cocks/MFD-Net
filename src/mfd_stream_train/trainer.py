from __future__ import annotations

import csv
import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mfd_stream_existing_iq.loader import MFDStreamMemmapDataset
from src.mfd_stream_existing_iq.config import (
    LoaderConfig,
    WINDOW_CLASS_IGNORE,
    build_class_subset_tag,
    canonicalise_label_subset,
    path_contains_class_subset_tag,
)
from .jan_fused_buffer_model import ModelConfig as JanModelConfig
from .jan_fused_buffer_model import build_model as build_jan_model
from .jan_fused_buffer_model import set_encoder_trainability

try:
    from src.ResNet.ResNet_phase import _amp_cfg_for_device as _project_amp_cfg_for_device
except Exception:
    def _project_amp_cfg_for_device(device_str: str):
        if device_str == "cuda" and torch.cuda.is_available():
            return (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)), torch.amp.GradScaler("cuda")
        return nullcontext, torch.amp.GradScaler("cpu", enabled=False)


@dataclass
class ExperimentConfig:
    train_memmap_dir: str
    val_memmap_dir: str
    test_memmap_dir: str | None = None
    data_root_dir: str | None = None
    class_subset_names: tuple[str, ...] | None = None

    modality_mode: str = "power_phase"  # power | phase | power_phase | dual
    power_checkpoint: str | None = None
    phase_checkpoint: str | None = None
    dual_checkpoint: str | None = None

    train_noise_bank_dir: str | None = None
    val_noise_bank_dir: str | None = None
    test_noise_bank_dir: str | None = None
    noise_handling_mode: str = "pulse_tail_pad"
    deterministic_val_noise: bool = True
    deterministic_test_noise: bool = True

    out_dir: str = "out/mfd_stream_train"
    run_name: str = "jan_fused_buffer"

    input_representation: str = "images"  # images | hybrid
    chunk_len: int = 16
    chunk_stride_train: int = 4
    chunk_stride_val: int = 16
    train_pad_short_streams: bool = True
    val_pad_short_streams: bool = True

    batch_size: int = 8
    num_workers: int = 0
    max_epochs: int = 25
    patience: int = 5
    min_delta: float = 0.001
    learning_rate: float = 1e-4
    backbone_learning_rate: float | None = None
    weight_decay: float = 1e-4
    seed: int = 42

    freeze_backbone: bool = True
    train_encoder_proj: bool = True
    train_encoder_classifier: bool = True

    power_embedding_dim: int = 64
    phase_embedding_dim: int = 64
    dual_embedding_dim: int = 64
    d_model: int = 256
    nhead: int = 4
    num_decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_time_windows: int | None = None

    use_frame_encoder_logits: bool = True
    add_aux_logit_token: bool = True
    aux_logit_proj_dim: int = 64
    aux_last_weight: float = 0.3

    use_class_weights: bool = False  # unused by default in BCE mode
    channels_last: bool = True
    print_every: int = 20
    sequence_mask_mode: str = "valid"  # valid | signal_only
    min_frames_for_buffer_label: int = 1
    threshold_mode: str = "per_class"  # per_class | global | 0.5
    key_metric: str = "micro_f1"  # micro_f1 | macro_f1 | primary_acc

    # Retained compatibility flags so old drivers still import.
    temporal_arch: str = "jan_fused"
    use_input_projector: bool = False
    backbone_embedding_dim: int = 0
    use_window_state_aux: bool = False
    use_window_modulation_aux: bool = False
    window_state_aux_weight: float = 0.0
    window_modulation_aux_weight: float = 0.0
    window_state_ignore_index: int = -100
    window_modulation_ignore_index: int = WINDOW_CLASS_IGNORE


class CSVLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.header_written = csv_path.exists()

    def append(self, row: dict[str, Any]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)


class ConfusionMeter:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        yt = y_true.detach().cpu().numpy().reshape(-1)
        yp = y_pred.detach().cpu().numpy().reshape(-1)
        for t, p in zip(yt, yp):
            self.cm[int(t), int(p)] += 1

    def accuracy(self) -> float:
        total = self.cm.sum()
        return float(np.trace(self.cm) / total) if total > 0 else 0.0

    def macro_f1(self) -> float:
        vals: list[float] = []
        for c in range(self.num_classes):
            tp = float(self.cm[c, c])
            fp = float(self.cm[:, c].sum() - tp)
            fn = float(self.cm[c, :].sum() - tp)
            denom = 2.0 * tp + fp + fn
            vals.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(vals)) if vals else 0.0


def save_confusion_csv(cm: np.ndarray, labels: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + [int(x) for x in cm[i]])


class BCEWithLogitsLossWithPosWeight(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _expected_class_subset(cfg: ExperimentConfig) -> tuple[str, ...]:
    return canonicalise_label_subset(cfg.class_subset_names)


def _resolve_data_paths(cfg: ExperimentConfig) -> dict[str, str | None]:
    expected = _expected_class_subset(cfg)
    if cfg.data_root_dir:
        subset_root = Path(cfg.data_root_dir) / build_class_subset_tag(expected)
        return {
            "train_memmap_dir": str(subset_root / "MFD_Net_train_stream_memmap"),
            "val_memmap_dir": str(subset_root / "MFD_Net_val_stream_memmap"),
            "test_memmap_dir": str(subset_root / "MFD_Net_test_stream_memmap"),
            "train_noise_bank_dir": str(subset_root / "MFD_Net_train_stream_noise_bank"),
            "val_noise_bank_dir": str(subset_root / "MFD_Net_val_stream_noise_bank"),
            "test_noise_bank_dir": str(subset_root / "MFD_Net_test_stream_noise_bank"),
        }
    return {
        "train_memmap_dir": cfg.train_memmap_dir,
        "val_memmap_dir": cfg.val_memmap_dir,
        "test_memmap_dir": cfg.test_memmap_dir,
        "train_noise_bank_dir": cfg.train_noise_bank_dir,
        "val_noise_bank_dir": cfg.val_noise_bank_dir,
        "test_noise_bank_dir": cfg.test_noise_bank_dir,
    }


def _verify_subset_path(path_value: str | None, expected_labels: tuple[str, ...], *, kind: str) -> None:
    if not path_value or not expected_labels:
        return
    if not path_contains_class_subset_tag(path_value, expected_labels):
        expected_tag = build_class_subset_tag(expected_labels)
        raise ValueError(
            f"{kind} path does not contain the expected class subset tag. expected_tag={expected_tag} | path={path_value}"
        )


def _verify_dataset_labels(ds: MFDStreamMemmapDataset, expected_labels: tuple[str, ...], *, split_name: str) -> None:
    if not expected_labels:
        return
    actual = tuple(str(x) for x in ds.label_names)
    if set(actual) != set(expected_labels):
        raise ValueError(
            f"{split_name} memmap labels do not match the requested class subset. expected={list(expected_labels)} | actual={list(actual)}"
        )


def build_loaders(cfg: ExperimentConfig):
    resolved_paths = _resolve_data_paths(cfg)
    expected_labels = _expected_class_subset(cfg)
    _verify_subset_path(resolved_paths["train_memmap_dir"], expected_labels, kind="train_memmap_dir")
    _verify_subset_path(resolved_paths["val_memmap_dir"], expected_labels, kind="val_memmap_dir")
    _verify_subset_path(resolved_paths["test_memmap_dir"], expected_labels, kind="test_memmap_dir")

    if str(cfg.input_representation).lower() not in {"images", "hybrid"}:
        raise ValueError("The Jan-style trainer currently supports input_representation='images' or 'hybrid' only.")

    train_ds = MFDStreamMemmapDataset(
        LoaderConfig(
            memmap_dir=resolved_paths["train_memmap_dir"],
            chunk_len=cfg.chunk_len,
            chunk_stride=cfg.chunk_stride_train,
            pad_short_streams=cfg.train_pad_short_streams,
            return_window_indices=True,
            return_signal_mask=True,
            return_window_state_targets=False,
            return_window_class_targets=True,
            normalise_to_float=True,
            noise_handling_mode=cfg.noise_handling_mode,
            noise_bank_dir=resolved_paths["train_noise_bank_dir"],
            deterministic_noise=False,
            embedding_dim=max(cfg.power_embedding_dim, cfg.phase_embedding_dim, 1),
            input_representation=cfg.input_representation,
        )
    )
    val_ds = MFDStreamMemmapDataset(
        LoaderConfig(
            memmap_dir=resolved_paths["val_memmap_dir"],
            chunk_len=cfg.chunk_len,
            chunk_stride=cfg.chunk_stride_val,
            pad_short_streams=cfg.val_pad_short_streams,
            return_window_indices=True,
            return_signal_mask=True,
            return_window_state_targets=False,
            return_window_class_targets=True,
            normalise_to_float=True,
            noise_handling_mode=cfg.noise_handling_mode,
            noise_bank_dir=resolved_paths["val_noise_bank_dir"],
            deterministic_noise=cfg.deterministic_val_noise,
            embedding_dim=max(cfg.power_embedding_dim, cfg.phase_embedding_dim, 1),
            input_representation=cfg.input_representation,
        )
    )
    test_ds = MFDStreamMemmapDataset(
        LoaderConfig(
            memmap_dir=resolved_paths["test_memmap_dir"],
            chunk_len=cfg.chunk_len,
            chunk_stride=cfg.chunk_stride_val,
            pad_short_streams=cfg.val_pad_short_streams,
            return_window_indices=True,
            return_signal_mask=True,
            return_window_state_targets=False,
            return_window_class_targets=True,
            normalise_to_float=True,
            noise_handling_mode=cfg.noise_handling_mode,
            noise_bank_dir=resolved_paths["test_noise_bank_dir"],
            deterministic_noise=cfg.deterministic_test_noise,
            embedding_dim=max(cfg.power_embedding_dim, cfg.phase_embedding_dim, 1),
            input_representation=cfg.input_representation,
        )
    )
    _verify_dataset_labels(train_ds, expected_labels, split_name="train")
    _verify_dataset_labels(val_ds, expected_labels, split_name="val")
    _verify_dataset_labels(test_ds, expected_labels, split_name="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_ds, train_loader, val_ds, val_loader, test_ds, test_loader


def _class_name_counts(ds: MFDStreamMemmapDataset) -> dict[str, int]:
    counts: dict[str, int] = {str(name): 0 for name in ds.label_names}
    for stream_id, _, _ in ds.samples:
        label = int(ds.stream_labels[int(stream_id)])
        counts[str(ds.label_names[label])] = counts.get(str(ds.label_names[label]), 0) + 1
    return counts


def build_scheduler(
    opt: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int = 3,
    hold_epochs: int = 2,
    start_factor: float = 0.3,
    min_lr_ratio: float = 0.2,
):
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            if warmup_epochs == 1:
                return 1.0
            progress = epoch / max(warmup_epochs - 1, 1)
            return start_factor + (1.0 - start_factor) * progress
        if epoch < warmup_epochs + hold_epochs:
            return 1.0
        decay_epochs = max_epochs - warmup_epochs - hold_epochs
        if decay_epochs <= 0:
            return 1.0
        progress = (epoch - warmup_epochs - hold_epochs) / max(decay_epochs - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def _unpack_sequence_batch(batch):
    if len(batch) != 5:
        raise ValueError(f"Expected batch structure (x, y, idx, signal_mask, window_class), got len={len(batch)}")
    return batch


def _extract_frames(xb: Any) -> torch.Tensor:
    if isinstance(xb, dict):
        frames = xb.get("frames")
        if frames is None:
            raise ValueError("Hybrid batch payload is missing 'frames'.")
        return frames
    return xb


def _move_sequence_batch_to_device(xb: Any, device: torch.device):
    if isinstance(xb, dict):
        out: dict[str, Any] = {}
        for k, v in xb.items():
            out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        return out
    return xb.to(device, non_blocking=True)


def _build_valid_mask(idxb: torch.Tensor, sigb: torch.Tensor | None, mode: str) -> torch.Tensor:
    if str(mode).lower() == "signal_only":
        if sigb is None:
            raise ValueError("signal_only mask mode requested but signal mask is missing from the loader batch.")
        return sigb.bool()
    return idxb.ge(0)


def _compute_buffer_targets(
    window_classb: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
    min_frames: int = 1,
) -> torch.Tensor:
    y = torch.zeros((window_classb.size(0), num_classes), device=window_classb.device, dtype=torch.float32)
    valid = valid_mask.bool() & window_classb.ne(ignore_index)
    for c in range(num_classes):
        counts = ((window_classb == c) & valid).sum(dim=1)
        y[:, c] = (counts >= int(min_frames)).float()
    return y


def _compute_last_targets(
    window_classb: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    b, _ = window_classb.shape
    y = torch.zeros((b, num_classes), device=window_classb.device, dtype=torch.float32)
    valid_counts = valid_mask.long().sum(dim=1)
    for i in range(b):
        if int(valid_counts[i]) <= 0:
            continue
        last_pos = int(valid_counts[i].item()) - 1
        cls = int(window_classb[i, last_pos].item())
        if cls == ignore_index or cls < 0 or cls >= num_classes:
            continue
        y[i, cls] = 1.0
    return y


def _find_per_class_thresholds(probs: torch.Tensor, targets: torch.Tensor, grid=None) -> torch.Tensor:
    if grid is None:
        grid = torch.linspace(0.05, 0.95, 19)
    n, c = probs.shape
    taus = torch.full((c,), 0.5)
    for k in range(c):
        p = probs[:, k]
        y = targets[:, k].bool()
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred = p > t
            tp = (pred & y).sum().item()
            fp = (pred & ~y).sum().item()
            fn = (~pred & y).sum().item()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        taus[k] = best_t
    return taus


def _find_global_threshold(probs: torch.Tensor, targets: torch.Tensor, grid=None) -> float:
    if grid is None:
        grid = torch.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    y = targets.bool()
    for t in grid:
        pred = probs > t
        tp = (pred & y).sum().item()
        fp = (pred & ~y).sum().item()
        fn = (~pred & y).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _apply_thresholds(probs: torch.Tensor, thresholds: str | float | torch.Tensor) -> torch.Tensor:
    if isinstance(thresholds, torch.Tensor):
        return probs > thresholds.view(1, -1)
    return probs > float(thresholds)


def _compute_binary_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    yb = y.bool()
    tp = (pred & yb).sum().item()
    fp = (pred & ~yb).sum().item()
    fn = (~pred & yb).sum().item()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    micro_f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    per_class_f1: list[float] = []
    for c in range(y.shape[1]):
        tp_c = (pred[:, c] & yb[:, c]).sum().item()
        fp_c = (pred[:, c] & ~yb[:, c]).sum().item()
        fn_c = (~pred[:, c] & yb[:, c]).sum().item()
        denom = 2 * tp_c + fp_c + fn_c
        per_class_f1.append((2 * tp_c / denom) if denom > 0 else 0.0)
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def _build_primary_confusion(signal_probs: torch.Tensor, y_primary: torch.Tensor, labels: list[str]) -> tuple[ConfusionMeter, torch.Tensor]:
    pred = signal_probs.argmax(dim=1)
    meter = ConfusionMeter(num_classes=len(labels))
    meter.update(y_primary.cpu(), pred.cpu())
    return meter, pred


def _build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("power_encoder.backbone") or name.startswith("phase_encoder.backbone"):
            backbone_params.append(p)
        else:
            head_params.append(p)
    if backbone_params and cfg.backbone_learning_rate is not None and not cfg.freeze_backbone:
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.backbone_learning_rate},
                {"params": head_params, "lr": cfg.learning_rate},
            ],
            weight_decay=cfg.weight_decay,
        )
    return torch.optim.AdamW(head_params + backbone_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def _compute_primary_class_weights(train_ds: MFDStreamMemmapDataset, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(train_ds.stream_labels, dtype=np.int64), minlength=len(train_ds.label_names)).astype(np.float32)
    total = float(counts.sum())
    weights = total / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, device=device, dtype=torch.float32)


def _compute_multilabel_pos_weight(train_ds: MFDStreamMemmapDataset, *, num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(train_ds.stream_labels, dtype=np.int64), minlength=max(num_classes - 1, 1)).astype(np.float32)
    total = float(counts.sum())
    pos_weight = np.ones((num_classes,), dtype=np.float32)
    if counts.size:
        signal_pw = (total - counts) / np.maximum(counts, 1.0)
        pos_weight[: counts.size] = np.clip(signal_pw, 1.0, 20.0)
    pos_weight[-1] = 1.0  # noise class
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ctx,
    loss_fn: nn.Module,
    epoch: int,
    max_epochs: int,
    cfg: ExperimentConfig,
    signal_class_count: int,
) -> dict[str, float]:
    model.train()
    running = 0.0
    running_main = 0.0
    running_last = 0.0
    n_seen = 0
    cm = ConfusionMeter(num_classes=signal_class_count)
    pbar = tqdm(loader, desc=f"train {epoch}/{max_epochs}", leave=False)
    ema = None

    for i, batch in enumerate(pbar):
        xb, yb, idxb, sigb, window_classb = _unpack_sequence_batch(batch)
        valid_mask = _build_valid_mask(idxb, sigb, cfg.sequence_mask_mode).to(device, non_blocking=True)
        xb = _move_sequence_batch_to_device(xb, device)
        yb = yb.to(device, non_blocking=True).long()
        window_classb = window_classb.to(device, non_blocking=True).long()

        ybuf = _compute_buffer_targets(
            window_classb,
            valid_mask,
            num_classes=signal_class_count + 1,
            ignore_index=cfg.window_modulation_ignore_index,
            min_frames=cfg.min_frames_for_buffer_label,
        )
        ylast = _compute_last_targets(
            window_classb,
            valid_mask,
            num_classes=signal_class_count + 1,
            ignore_index=cfg.window_modulation_ignore_index,
        )

        opt.zero_grad(set_to_none=True)
        with ctx():
            out = model.forward_sequence_images(xb, mask=valid_mask, return_last=(cfg.aux_last_weight > 0.0))
            loss_main = loss_fn(out["logits"], ybuf)
            loss = loss_main
            loss_last = torch.tensor(0.0, device=device)
            if cfg.aux_last_weight > 0.0 and out.get("logits_last") is not None:
                loss_last = loss_fn(out["logits_last"], ylast)
                loss = loss + (cfg.aux_last_weight * loss_last)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        bs = yb.size(0)
        running += loss.item() * bs
        running_main += loss_main.item() * bs
        running_last += float(loss_last.item()) * bs
        n_seen += bs

        signal_probs = torch.sigmoid(out["logits"][:, :signal_class_count])
        pred_primary = signal_probs.argmax(dim=1)
        cm.update(yb, pred_primary)

        ema = loss.item() if ema is None else (0.9 * ema + 0.1 * loss.item())
        if (i % max(1, cfg.print_every)) == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", ema=f"{ema:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

    return {
        "loss": running / max(1, n_seen),
        "buffer_loss": running_main / max(1, n_seen),
        "aux_last_loss": running_last / max(1, n_seen),
        "primary_acc": cm.accuracy(),
        "primary_macro_f1": cm.macro_f1(),
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    cfg: ExperimentConfig,
    signal_class_count: int,
    labels: list[str],
    thresholds_override: str | float | torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_main = 0.0
    total_last = 0.0
    n_seen = 0
    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_signal_probs: list[torch.Tensor] = []
    all_y_primary: list[torch.Tensor] = []

    pbar = tqdm(loader, desc="val", leave=False)
    for batch in pbar:
        xb, yb, idxb, sigb, window_classb = _unpack_sequence_batch(batch)
        valid_mask = _build_valid_mask(idxb, sigb, cfg.sequence_mask_mode).to(device, non_blocking=True)
        xb = _move_sequence_batch_to_device(xb, device)
        yb = yb.to(device, non_blocking=True).long()
        window_classb = window_classb.to(device, non_blocking=True).long()

        ybuf = _compute_buffer_targets(
            window_classb,
            valid_mask,
            num_classes=signal_class_count + 1,
            ignore_index=cfg.window_modulation_ignore_index,
            min_frames=cfg.min_frames_for_buffer_label,
        )
        ylast = _compute_last_targets(
            window_classb,
            valid_mask,
            num_classes=signal_class_count + 1,
            ignore_index=cfg.window_modulation_ignore_index,
        )
        out = model.forward_sequence_images(xb, mask=valid_mask, return_last=(cfg.aux_last_weight > 0.0))
        loss_main = loss_fn(out["logits"], ybuf)
        loss = loss_main
        loss_last = torch.tensor(0.0, device=device)
        if cfg.aux_last_weight > 0.0 and out.get("logits_last") is not None:
            loss_last = loss_fn(out["logits_last"], ylast)
            loss = loss + (cfg.aux_last_weight * loss_last)

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_main += loss_main.item() * bs
        total_last += float(loss_last.item()) * bs
        n_seen += bs

        probs = torch.sigmoid(out["logits"]).detach().cpu()
        all_probs.append(probs)
        all_targets.append(ybuf.detach().cpu())
        all_signal_probs.append(probs[:, :signal_class_count])
        all_y_primary.append(yb.detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    signal_probs = torch.cat(all_signal_probs, dim=0)
    y_primary = torch.cat(all_y_primary, dim=0)

    if thresholds_override is not None:
        thresholds: str | float | torch.Tensor = thresholds_override
    elif str(cfg.threshold_mode).lower() == "per_class":
        thresholds = _find_per_class_thresholds(probs, targets)
    elif str(cfg.threshold_mode).lower() == "global":
        thresholds = _find_global_threshold(probs, targets)
    else:
        thresholds = float(cfg.threshold_mode)

    pred = _apply_thresholds(probs, thresholds)
    metrics = _compute_binary_metrics(pred, targets)
    cm, pred_primary = _build_primary_confusion(signal_probs, y_primary, labels)

    return {
        "loss": total_loss / max(1, n_seen),
        "buffer_loss": total_main / max(1, n_seen),
        "aux_last_loss": total_last / max(1, n_seen),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "micro_f1": metrics["micro_f1"],
        "macro_f1": metrics["macro_f1"],
        "primary_acc": cm.accuracy(),
        "primary_macro_f1": cm.macro_f1(),
        "thresholds": thresholds,
        "cm": cm.cm,
        "pred_primary": pred_primary,
        "y_primary": y_primary,
    }


def run_experiment(cfg: ExperimentConfig) -> Path:
    mode = str(cfg.modality_mode).lower()
    if mode == "power_phase":
        if not cfg.power_checkpoint or not cfg.phase_checkpoint:
            raise ValueError("modality_mode='power_phase' requires both power_checkpoint and phase_checkpoint.")
    elif mode == "power":
        if not cfg.power_checkpoint:
            raise ValueError("modality_mode='power' requires power_checkpoint.")
    elif mode == "phase":
        if not cfg.phase_checkpoint:
            raise ValueError("modality_mode='phase' requires phase_checkpoint.")
    elif mode == "dual":
        if not cfg.dual_checkpoint:
            raise ValueError("modality_mode='dual' requires dual_checkpoint.")
    else:
        raise ValueError(f"Unsupported modality_mode={cfg.modality_mode!r}. Expected power, phase, power_phase, or dual.")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{cfg.run_name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, train_loader, val_ds, val_loader, test_ds, test_loader = build_loaders(cfg)
    signal_labels = list(train_ds.label_names)
    signal_class_count = len(signal_labels)
    all_labels = signal_labels + ["noise"]

    model_cfg = JanModelConfig(
        num_classes=signal_class_count + 1,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_time_windows=int(cfg.max_time_windows or cfg.chunk_len),
        add_aux_logit_token=cfg.add_aux_logit_token,
        aux_logit_proj_dim=cfg.aux_logit_proj_dim,
        use_frame_encoder_logits=cfg.use_frame_encoder_logits,
        modality_mode=mode,
        power_embedding_dim=cfg.power_embedding_dim,
        phase_embedding_dim=cfg.phase_embedding_dim,
        dual_embedding_dim=cfg.dual_embedding_dim,
    )
    model, encoder_meta = build_jan_model(
        model_cfg,
        modality_mode=mode,
        power_checkpoint=cfg.power_checkpoint,
        phase_checkpoint=cfg.phase_checkpoint,
        dual_checkpoint=cfg.dual_checkpoint,
    )
    set_encoder_trainability(
        getattr(model, "power_encoder", None),
        train_backbone=not cfg.freeze_backbone,
        train_proj=cfg.train_encoder_proj,
        train_classifier=cfg.train_encoder_classifier,
    )
    set_encoder_trainability(
        getattr(model, "phase_encoder", None),
        train_backbone=not cfg.freeze_backbone,
        train_proj=cfg.train_encoder_proj,
        train_classifier=cfg.train_encoder_classifier,
    )
    set_encoder_trainability(
        getattr(model, "dual_encoder", None),
        train_backbone=not cfg.freeze_backbone,
        train_proj=cfg.train_encoder_proj,
        train_classifier=cfg.train_encoder_classifier,
    )
    model = model.to(device)

    # Contract smoke test.
    first_batch = next(iter(train_loader))
    xb0, yb0, idxb0, sigb0, window_classb0 = _unpack_sequence_batch(first_batch)
    valid0 = _build_valid_mask(idxb0, sigb0, cfg.sequence_mask_mode).to(device)
    xb0 = _move_sequence_batch_to_device(xb0, device)
    with torch.no_grad():
        out0 = model.forward_sequence_images(xb0, mask=valid0, return_last=(cfg.aux_last_weight > 0.0))
    if tuple(out0["logits"].shape) != (yb0.shape[0], signal_class_count + 1):
        raise ValueError(
            f"Model returned logits with shape {tuple(out0['logits'].shape)}; expected {(yb0.shape[0], signal_class_count + 1)}"
        )

    pos_weight = _compute_multilabel_pos_weight(train_ds, num_classes=signal_class_count + 1).to(device) if cfg.use_class_weights else None
    loss_fn = BCEWithLogitsLossWithPosWeight(pos_weight=pos_weight)
    opt = _build_optimizer(model, cfg)
    sched = build_scheduler(opt, max_epochs=cfg.max_epochs)
    ctx, scaler = _project_amp_cfg_for_device(device_str)

    config_dump = {
        "experiment": asdict(cfg),
        "resolved_paths": _resolve_data_paths(cfg),
        "labels_signal": signal_labels,
        "labels_all": all_labels,
        "device": str(device),
        "train_len": len(train_ds),
        "val_len": len(val_ds),
        "test_len": len(test_ds),
        "encoder_meta": encoder_meta,
    }
    (run_dir / "config.json").write_text(json.dumps(config_dump, indent=2, default=str), encoding="utf-8")
    csv_logger = CSVLogger(run_dir / "metrics.csv")

    print(
        f"[data] train samples={len(train_ds)} | val samples={len(val_ds)} | test samples={len(test_ds)} | chunk_len={cfg.chunk_len} | "
        f"noise_mode={cfg.noise_handling_mode} | input={cfg.input_representation}"
    )
    print(f"[data] train class counts: {_class_name_counts(train_ds)}")
    print(f"[data] val class counts:   {_class_name_counts(val_ds)}")
    print(f"[data] test class counts:  {_class_name_counts(test_ds)}")

    best_score = float("-inf")
    best_epoch = -1
    stale = 0

    for epoch in range(1, cfg.max_epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            device,
            opt,
            scaler,
            ctx,
            loss_fn,
            epoch,
            cfg.max_epochs,
            cfg,
            signal_class_count,
        )
        val_stats = evaluate_epoch(
            model,
            val_loader,
            device,
            loss_fn,
            cfg,
            signal_class_count,
            signal_labels,
        )
        sched.step()

        row = {
            "epoch": epoch,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lr": float(opt.param_groups[0]["lr"]),
            "train_loss": float(train_stats["loss"]),
            "train_buffer_loss": float(train_stats["buffer_loss"]),
            "train_aux_last_loss": float(train_stats["aux_last_loss"]),
            "train_primary_acc": float(train_stats["primary_acc"]),
            "train_primary_macro_f1": float(train_stats["primary_macro_f1"]),
            "val_loss": float(val_stats["loss"]),
            "val_buffer_loss": float(val_stats["buffer_loss"]),
            "val_aux_last_loss": float(val_stats["aux_last_loss"]),
            "val_precision": float(val_stats["precision"]),
            "val_recall": float(val_stats["recall"]),
            "val_micro_f1": float(val_stats["micro_f1"]),
            "val_macro_f1": float(val_stats["macro_f1"]),
            "val_primary_acc": float(val_stats["primary_acc"]),
            "val_primary_macro_f1": float(val_stats["primary_macro_f1"]),
        }
        csv_logger.append(row)

        cm_epoch_path = run_dir / f"cm_ep{epoch:02d}.csv"
        save_confusion_csv(val_stats["cm"], signal_labels, cm_epoch_path)

        thresholds = val_stats["thresholds"]
        if isinstance(thresholds, torch.Tensor):
            thr_dump = {name: float(thresholds[i].item()) for i, name in enumerate(all_labels)}
        else:
            thr_dump = {"global": float(thresholds)}
        (run_dir / f"thresholds_ep{epoch:02d}.json").write_text(json.dumps(thr_dump, indent=2), encoding="utf-8")

        score = {
            "micro_f1": float(val_stats["micro_f1"]),
            "macro_f1": float(val_stats["macro_f1"]),
            "primary_acc": float(val_stats["primary_acc"]),
        }.get(cfg.key_metric, float(val_stats["micro_f1"]))
        print(
            f"epoch {epoch:02d} | train_loss={train_stats['loss']:.4f} | val_micro_f1={val_stats['micro_f1']:.4f} | "
            f"val_macro_f1={val_stats['macro_f1']:.4f} | val_primary_acc={val_stats['primary_acc']:.4f}"
        )

        if score > best_score + cfg.min_delta:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_confusion_csv(val_stats["cm"], signal_labels, run_dir / "cm_best.csv")
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "score": float(score),
                "val_stats": {k: v for k, v in val_stats.items() if k not in {"cm", "pred_primary", "y_primary"}},
                "thresholds": val_stats["thresholds"].clone() if isinstance(val_stats["thresholds"], torch.Tensor) else val_stats["thresholds"],
                "config": asdict(cfg),
            }
            torch.save(ckpt, run_dir / "best.pt")
            print(f"[earlystop] new best {cfg.key_metric}={best_score:.4f} at epoch {epoch}")
        else:
            stale += 1
            print(f"[earlystop] no improvement for {stale}/{cfg.patience} epochs (best at {best_epoch}: {best_score:.4f})")
            if stale >= cfg.patience:
                print(f"[earlystop] stopping at epoch {epoch}")
                break

    best_ckpt_path = run_dir / "best.pt"
    test_thresholds_override = None
    if best_ckpt_path.is_file():
        best_payload = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_payload["model_state_dict"])
        test_thresholds_override = best_payload.get("thresholds")
        print(f"[test] loaded best checkpoint from: {best_ckpt_path}")
    else:
        print(f"[test] best checkpoint not found at {best_ckpt_path}; evaluating current model state instead")

    test_stats = evaluate_epoch(
        model,
        test_loader,
        device,
        loss_fn,
        cfg,
        signal_class_count,
        signal_labels,
        thresholds_override=test_thresholds_override,
    )
    save_confusion_csv(test_stats["cm"], signal_labels, run_dir / "cm_test_best.csv")
    test_thresholds = test_stats["thresholds"]
    if isinstance(test_thresholds, torch.Tensor):
        test_thr_dump = {name: float(test_thresholds[i].item()) for i, name in enumerate(all_labels)}
    else:
        test_thr_dump = {"global": float(test_thresholds)}
    (run_dir / "thresholds_test_best.json").write_text(json.dumps(test_thr_dump, indent=2), encoding="utf-8")

    test_summary = {
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_score),
        "key_metric": str(cfg.key_metric),
        "test_loss": float(test_stats["loss"]),
        "test_buffer_loss": float(test_stats["buffer_loss"]),
        "test_aux_last_loss": float(test_stats["aux_last_loss"]),
        "test_precision": float(test_stats["precision"]),
        "test_recall": float(test_stats["recall"]),
        "test_micro_f1": float(test_stats["micro_f1"]),
        "test_macro_f1": float(test_stats["macro_f1"]),
        "test_primary_acc": float(test_stats["primary_acc"]),
        "test_primary_macro_f1": float(test_stats["primary_macro_f1"]),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    print(
        f"[test] micro_f1={test_stats['micro_f1']:.4f} | macro_f1={test_stats['macro_f1']:.4f} | "
        f"primary_acc={test_stats['primary_acc']:.4f}"
    )

    return run_dir
