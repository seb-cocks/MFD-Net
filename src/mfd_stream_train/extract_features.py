from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .trainer import (
    ExperimentConfig,
    _build_valid_mask,
    _move_sequence_batch_to_device,
    _unpack_sequence_batch,
    build_loaders,
)
from .jan_fused_buffer_model import ModelConfig as JanModelConfig
from .jan_fused_buffer_model import build_model as build_jan_model


def _load_experiment_config(run_dir: Path) -> ExperimentConfig:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find run config at: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_cfg = payload.get("experiment")
    if not isinstance(experiment_cfg, dict):
        raise ValueError(f"config.json at {config_path} does not contain an 'experiment' section.")
    return ExperimentConfig(**experiment_cfg)


def _build_model_from_run(cfg: ExperimentConfig, checkpoint_path: Path):
    mode = str(cfg.modality_mode).lower()
    signal_class_count = len(cfg.class_subset_names or ())
    if signal_class_count <= 0:
        raise ValueError("class_subset_names must be present in the saved run config so the model output size can be rebuilt.")

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

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain model_state_dict/state_dict.")
    model.load_state_dict(state_dict, strict=True)
    return model, checkpoint, encoder_meta


def _stack_optional_tensors(items: list[torch.Tensor] | None) -> torch.Tensor | None:
    if not items:
        return None
    return torch.cat(items, dim=0)


def extract_features(
    run_dir: Path,
    *,
    split: str,
    checkpoint_name: str,
    output_path: Path | None,
    limit_batches: int | None,
) -> Path:
    cfg = _load_experiment_config(run_dir)
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_meta, encoder_meta = _build_model_from_run(cfg, checkpoint_path)
    model = model.to(device)
    model.eval()

    train_ds, train_loader, val_ds, val_loader, test_ds, test_loader = build_loaders(cfg)
    if str(split).lower() == "train":
        ds = train_ds
        loader = train_loader
    elif str(split).lower() == "val":
        ds = val_ds
        loader = val_loader
    elif str(split).lower() == "test":
        ds = test_ds
        loader = test_loader
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")

    frame_features_all: list[torch.Tensor] = []
    valid_mask_all: list[torch.Tensor] = []
    window_indices_all: list[torch.Tensor] = []
    primary_labels_all: list[torch.Tensor] = []
    window_class_all: list[torch.Tensor] = []
    modality_feature_buckets: dict[str, list[torch.Tensor]] = {}
    modality_logit_buckets: dict[str, list[torch.Tensor]] = {}
    buffer_logits_all: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Extracting {split} features", leave=False)):
            if limit_batches is not None and batch_idx >= int(limit_batches):
                break
            xb, yb, idxb, sigb, window_classb = _unpack_sequence_batch(batch)
            valid_mask = _build_valid_mask(idxb, sigb, cfg.sequence_mask_mode).to(device)
            xb = _move_sequence_batch_to_device(xb, device)

            feature_out = model.extract_frame_features(xb, mask=valid_mask, include_logits=True)
            forward_out = model.forward_sequence_images(xb, mask=valid_mask, return_last=False)

            frame_features_all.append(feature_out["frame_features"].detach().cpu().to(torch.float32))
            valid_mask_all.append(feature_out["valid_mask"].detach().cpu())
            window_indices_all.append(idxb.detach().cpu())
            primary_labels_all.append(yb.detach().cpu())
            window_class_all.append(window_classb.detach().cpu())
            buffer_logits_all.append(forward_out["logits_buffer"].detach().cpu().to(torch.float32))

            for name, feats in feature_out["frame_features_by_modality"].items():
                modality_feature_buckets.setdefault(name, []).append(feats.detach().cpu().to(torch.float32))

            for name, logits in feature_out.get("frame_logits_by_modality", {}).items():
                modality_logit_buckets.setdefault(name, []).append(logits.detach().cpu().to(torch.float32))

    payload: dict[str, Any] = {
        "split": str(split).lower(),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "label_names": [str(x) for x in ds.label_names],
        "modality_mode": str(cfg.modality_mode),
        "frame_features": torch.cat(frame_features_all, dim=0),
        "valid_mask": torch.cat(valid_mask_all, dim=0),
        "window_indices": torch.cat(window_indices_all, dim=0),
        "primary_labels": torch.cat(primary_labels_all, dim=0),
        "window_class_targets": torch.cat(window_class_all, dim=0),
        "buffer_logits": torch.cat(buffer_logits_all, dim=0),
        "frame_features_by_modality": {
            name: torch.cat(parts, dim=0) for name, parts in modality_feature_buckets.items()
        },
        "frame_logits_by_modality": {
            name: torch.cat(parts, dim=0) for name, parts in modality_logit_buckets.items()
        },
        "experiment_config": cfg.__dict__,
        "checkpoint_meta": checkpoint_meta,
        "encoder_meta": encoder_meta,
    }

    if output_path is None:
        output_path = run_dir / f"features_{str(split).lower()}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-window frame features from a trained MFD-Net run.")
    parser.add_argument("--run-dir", required=True, help="Path to the MFD training run directory containing config.json and best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split to export")
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint filename inside the run directory")
    parser.add_argument("--output", default=None, help="Optional output .pt path")
    parser.add_argument("--limit-batches", type=int, default=None, help="Optional number of batches to export for a quick test")
    args = parser.parse_args()

    out_path = extract_features(
        Path(args.run_dir),
        split=args.split,
        checkpoint_name=args.checkpoint,
        output_path=Path(args.output) if args.output else None,
        limit_batches=args.limit_batches,
    )
    print(f"[done] features written to: {out_path}")


if __name__ == "__main__":
    main()
