from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mfd_stream_existing_iq.config import LoaderConfig
from mfd_stream_existing_iq.loader import MFDStreamMemmapDataset
from mfd_stream_train.model import ModelConfig, EmbeddingOnlySequenceClassifier


class DummyFrameEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(2, embedding_dim, bias=False)

    def extract_sequence_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        # images: [N,2,H,W]
        pooled = images.mean(dim=(2, 3))
        return self.proj(pooled)


def _write_stream_memmap(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    H, W = 8, 8
    N = 4
    S = 2

    power = np.memmap(root / "power_u8.bin", mode="w+", dtype=np.uint8, shape=(N, H, W))
    phase = np.memmap(root / "phase_u8.bin", mode="w+", dtype=np.uint8, shape=(N, H, W))
    labels = np.memmap(root / "labels_i16.bin", mode="w+", dtype=np.int16, shape=(N,))
    window_indices = np.memmap(root / "window_indices_i32.bin", mode="w+", dtype=np.int32, shape=(N,))
    stream_offsets = np.memmap(root / "stream_offsets_i64.bin", mode="w+", dtype=np.int64, shape=(S,))
    stream_lengths = np.memmap(root / "stream_lengths_i32.bin", mode="w+", dtype=np.int32, shape=(S,))
    stream_labels = np.memmap(root / "stream_labels_i16.bin", mode="w+", dtype=np.int16, shape=(S,))
    stream_snr = np.memmap(root / "stream_snr_f32.bin", mode="w+", dtype=np.float32, shape=(S,))
    window_states = np.memmap(root / "window_states_u8.bin", mode="w+", dtype=np.uint8, shape=(N,))

    power[0] = 10
    power[1] = 20
    power[2] = 30
    power[3] = 40
    phase[0] = 1
    phase[1] = 2
    phase[2] = 3
    phase[3] = 4
    labels[:] = [0, 0, 1, 1]
    window_indices[:] = [1, 3, 0, 2]
    stream_offsets[:] = [0, 2]
    stream_lengths[:] = [2, 2]
    stream_labels[:] = [0, 1]
    stream_snr[:] = [0.0, 0.0]
    window_states[:] = [1, 2, 1, 2]

    for mm in [power, phase, labels, window_indices, stream_offsets, stream_lengths, stream_labels, stream_snr, window_states]:
        mm.flush()

    meta = {
        "num_valid_windows": N,
        "spectrogram_height": H,
        "spectrogram_width": W,
        "num_streams": S,
        "window_count": 4,
        "label_names": ["unmodulated", "lfm_up"],
        "class_subset_tag": "lfm_up__unmodulated",
        "memmaps": {
            "power": "power_u8.bin",
            "phase": "phase_u8.bin",
            "labels": "labels_i16.bin",
            "window_indices": "window_indices_i32.bin",
            "stream_offsets": "stream_offsets_i64.bin",
            "stream_lengths": "stream_lengths_i32.bin",
            "stream_labels": "stream_labels_i16.bin",
            "stream_snr_db": "stream_snr_f32.bin",
            "window_states": "window_states_u8.bin",
        },
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_noise_bank(root: Path, embedding_dim: int = 512) -> None:
    root.mkdir(parents=True, exist_ok=True)
    H, W = 8, 8
    N = 2
    power = np.memmap(root / "noise_power_u8.bin", mode="w+", dtype=np.uint8, shape=(N, H, W))
    phase = np.memmap(root / "noise_phase_u8.bin", mode="w+", dtype=np.uint8, shape=(N, H, W))
    embedding = np.memmap(root / "noise_embedding_f32.bin", mode="w+", dtype=np.float32, shape=(N, embedding_dim))
    snr = np.memmap(root / "noise_snr_f32.bin", mode="w+", dtype=np.float32, shape=(N,))
    power[0] = 5
    power[1] = 6
    phase[0] = 7
    phase[1] = 8
    embedding[:] = 0.5
    snr[:] = 0.0
    for mm in [power, phase, embedding, snr]:
        mm.flush()
    meta = {
        "num_samples": N,
        "spectrogram_height": H,
        "spectrogram_width": W,
        "export_images": True,
        "export_embeddings": True,
        "embedding_dim": embedding_dim,
        "memmaps": {
            "power": "noise_power_u8.bin",
            "phase": "noise_phase_u8.bin",
            "embedding": "noise_embedding_f32.bin",
            "snr_db": "noise_snr_f32.bin",
        },
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _run_case(noise_mode: str, stream_root: Path, noise_root: Path) -> None:
    cfg = LoaderConfig(
        memmap_dir=stream_root,
        chunk_len=4,
        chunk_stride=1,
        pad_short_streams=True,
        return_window_indices=True,
        return_signal_mask=True,
        return_window_state_targets=True,
        return_window_class_targets=True,
        noise_handling_mode=noise_mode,
        noise_bank_dir=noise_root,
        embedding_dim=512,
    )
    ds = MFDStreamMemmapDataset(cfg)
    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False)))
    xb, yb, idxb, sigb, stateb, classb = batch
    mask = idxb >= 0
    model = EmbeddingOnlySequenceClassifier(
        ModelConfig(
            num_classes=2,
            backbone_embedding_dim=512,
            d_model=64,
            dim_feedforward=128,
            temporal_arch="tcn_mil",
            use_window_state_head=True,
            use_window_modulation_head=True,
            window_modulation_classes=3,
        ),
        frame_encoder=DummyFrameEncoder(embedding_dim=512),
    )
    out = model.forward_sequence_images(xb, mask=mask)
    assert tuple(out["logits"].shape) == (2, 2)
    assert tuple(out["temporal_seq"].shape[:2]) == (2, 4)
    assert tuple(out["window_state_logits"].shape) == (2, 4, 3)
    assert tuple(out["window_modulation_logits"].shape) == (2, 4, 3)
    assert tuple(stateb.shape) == (2, 4)
    assert tuple(classb.shape) == (2, 4)


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="mfd_stream_smoke_"))
    stream_root = tmpdir / "stream"
    noise_root = tmpdir / "noise"
    _write_stream_memmap(stream_root)
    _write_noise_bank(noise_root)
    for noise_mode in ["pulse_tail_pad", "gap_padding", "gap_noise_embeddings", "gap_noise_images"]:
        _run_case(noise_mode, stream_root, noise_root)
    print(f"All smoke tests passed. Temporary files kept at: {tmpdir}")


if __name__ == "__main__":
    main()
