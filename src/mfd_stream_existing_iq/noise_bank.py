from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import NoiseBankConfig, build_class_subset_tag, ensure_empty_or_create
from .encoder_utils import _to_jsonable, build_frozen_dual_backbone, encode_dual_spectrogram_batch_with_logits
from .noise_utils import add_noise
from .spectrogram_adapter import SpectrogramAdapter


class NoiseBankBuilder:
    def __init__(self, cfg: NoiseBankConfig):
        self.cfg = cfg
        self.output_dir = ensure_empty_or_create(cfg.output_dir, cfg.overwrite)
        self.adapter = SpectrogramAdapter(cfg.spectrogram)
        self.encoder = None
        self.encoder_device = None
        self.embedding_dim: int | None = None
        self.encoder_meta: dict[str, object] = {}
        if self.cfg.export_embeddings:
            if self.cfg.resnet_checkpoint_path is None:
                raise ValueError("NoiseBankConfig.export_embeddings=True requires resnet_checkpoint_path.")
            self.encoder, self.embedding_dim, self.encoder_meta, self.encoder_device = build_frozen_dual_backbone(
                self.cfg.resnet_checkpoint_path,
                device=self.cfg.device,
                strict=self.cfg.checkpoint_strict,
            )

    def build(self) -> Path:
        snrs = np.asarray(self.cfg.snr_values_db, dtype=np.float32)
        total = int(len(snrs) * self.cfg.samples_per_snr)
        H = int(self.cfg.spectrogram.out_height)
        W = int(self.cfg.spectrogram.out_width)

        power_mm = phase_mm = emb_mm = logit_mm = None
        if self.cfg.export_images:
            power_mm = np.memmap(self.output_dir / "noise_power_u8.bin", mode="w+", dtype=np.uint8, shape=(total, H, W))
            phase_mm = np.memmap(self.output_dir / "noise_phase_u8.bin", mode="w+", dtype=np.uint8, shape=(total, H, W))
        if self.cfg.export_embeddings:
            emb_dim = int(self.embedding_dim or 0)
            logit_dim = int(self.encoder_meta.get("num_classes") or 1)
            emb_mm = np.memmap(self.output_dir / "noise_embedding_f32.bin", mode="w+", dtype=np.float32, shape=(total, emb_dim))
            logit_mm = np.memmap(self.output_dir / "noise_encoder_logits_f32.bin", mode="w+", dtype=np.float32, shape=(total, logit_dim))
        snr_mm = np.memmap(self.output_dir / "noise_snr_f32.bin", mode="w+", dtype=np.float32, shape=(total,))

        rng = np.random.default_rng(self.cfg.base_seed)
        cursor = 0
        zeros = np.zeros(self.cfg.window_len + 2 * self.cfg.spectrogram.context_samples_each_side, dtype=np.float32)
        pending_dual_specs: list[np.ndarray] = []
        pending_rows: list[int] = []

        def flush_pending_embeddings() -> None:
            nonlocal pending_dual_specs, pending_rows
            if emb_mm is None or logit_mm is None or not pending_dual_specs:
                pending_dual_specs = []
                pending_rows = []
                return
            batch = np.stack(pending_dual_specs, axis=0)
            batch_emb, batch_logits = encode_dual_spectrogram_batch_with_logits(self.encoder, batch, device=self.encoder_device)
            rows = np.asarray(pending_rows, dtype=np.int64)
            emb_mm[rows] = batch_emb
            logit_mm[rows] = batch_logits
            emb_mm.flush()
            logit_mm.flush()
            pending_dual_specs = []
            pending_rows = []

        for snr_db in snrs:
            for _ in range(self.cfg.samples_per_snr):
                i_noisy, q_noisy = add_noise(
                    rng,
                    zeros,
                    zeros,
                    snr_db=float(snr_db),
                    noise_power_db=self.cfg.noise_power_db,
                    apply_hard_clip=self.cfg.apply_hard_clip,
                    clip_min=self.cfg.clip_min,
                    clip_max=self.cfg.clip_max,
                )
                iq = (i_noisy + 1j * q_noisy).astype(np.complex64, copy=False)
                rendered = self.adapter.render_window(iq, self.cfg.window_len)

                if power_mm is not None and phase_mm is not None:
                    power_mm[cursor] = rendered.power_u8
                    phase_mm[cursor] = rendered.phase_u8
                if emb_mm is not None:
                    pending_dual_specs.append(rendered.dual_u8)
                    pending_rows.append(cursor)
                    if len(pending_dual_specs) >= max(1, int(self.cfg.encoder_batch_size)):
                        flush_pending_embeddings()

                snr_mm[cursor] = float(snr_db)
                cursor += 1

            flush_pending_embeddings()
            if power_mm is not None:
                power_mm.flush()
            if phase_mm is not None:
                phase_mm.flush()
            snr_mm.flush()
            print(f"[{self.cfg.split_name}] noise bank bucket SNR={snr_db:g} done -> total={cursor}/{total}")

        flush_pending_embeddings()
        if power_mm is not None:
            power_mm.flush()
        if phase_mm is not None:
            phase_mm.flush()
        if emb_mm is not None:
            emb_mm.flush()
        if logit_mm is not None:
            logit_mm.flush()
        snr_mm.flush()

        memmaps = {"snr_db": "noise_snr_f32.bin"}
        if self.cfg.export_images:
            memmaps["power"] = "noise_power_u8.bin"
            memmaps["phase"] = "noise_phase_u8.bin"
        if self.cfg.export_embeddings:
            memmaps["embedding"] = "noise_embedding_f32.bin"
            memmaps["encoder_logits"] = "noise_encoder_logits_f32.bin"

        parent_name = self.output_dir.parent.name
        meta = {
            "split_name": self.cfg.split_name,
            "class_subset_tag": parent_name if parent_name else build_class_subset_tag(None),
            "output_dir": str(self.output_dir),
            "sample_rate_hz": float(self.cfg.sample_rate_hz),
            "window_len": int(self.cfg.window_len),
            "samples_per_snr": int(self.cfg.samples_per_snr),
            "snr_values_db": [float(v) for v in snrs.tolist()],
            "noise_power_db": float(self.cfg.noise_power_db),
            "num_samples": total,
            "spectrogram_height": H,
            "spectrogram_width": W,
            "spectrogram_context_samples_each_side": int(self.cfg.spectrogram.context_samples_each_side),
            "export_images": bool(self.cfg.export_images),
            "export_embeddings": bool(self.cfg.export_embeddings),
            "embedding_source": "resnet_sequence_embedding" if self.cfg.export_embeddings else None,
            "embedding_dim": int(self.embedding_dim) if self.embedding_dim is not None else None,
            "frame_encoder_logit_dim": int(self.encoder_meta.get("num_classes")) if self.cfg.export_embeddings else None,
            "encoder": self.encoder_meta if self.cfg.export_embeddings else None,
            "memmaps": memmaps,
        }
        with open(self.output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(meta), f, indent=2)
        return self.output_dir



def build_noise_bank(cfg: NoiseBankConfig) -> Path:
    return NoiseBankBuilder(cfg).build()
