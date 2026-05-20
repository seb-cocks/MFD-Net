from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    LoaderConfig,
    WINDOW_CLASS_IGNORE,
    WINDOW_STATE_IGNORE,
    WINDOW_STATE_NOISE,
    to_path,
)


class MFDStreamMemmapDataset(Dataset):
    """Chunk sparse pulse-only stream memmaps with optional gap/noise reconstruction.

    The exporter stores only windows that passed the overlap threshold. This loader
    rebuilds temporal chunks using the preserved original dense window indices.

    Returned representation is controlled by ``LoaderConfig.input_representation``:
    - images: [T, 2, H, W]
    - embeddings: [T, D]
    - hybrid: dict with ``frames``, ``backbone_embeddings``, ``embedding_mask`` and, when available,
      ``frame_encoder_logits`` plus ``logit_mask``
    """

    VALID_NOISE_HANDLING_MODES = {
        "pulse_tail_pad",
        "gap_padding",
        "gap_noise_images",
        "gap_noise_embeddings",
    }
    VALID_INPUT_REPRESENTATIONS = {"images", "embeddings", "hybrid"}

    def __init__(self, cfg: LoaderConfig):
        self.cfg = cfg
        self.root = to_path(cfg.memmap_dir)
        with open(self.root / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        N = int(self.meta["num_valid_windows"])
        H = int(self.meta["spectrogram_height"])
        W = int(self.meta["spectrogram_width"])
        S = int(self.meta["num_streams"])

        self.height = H
        self.width = W
        self.full_window_count = int(self.meta.get("window_count", 0))
        self.label_names = list(self.meta["label_names"])
        self.noise_handling_mode = str(cfg.noise_handling_mode).lower()
        if self.noise_handling_mode not in self.VALID_NOISE_HANDLING_MODES:
            raise ValueError(
                f"noise_handling_mode must be one of {sorted(self.VALID_NOISE_HANDLING_MODES)}, got {cfg.noise_handling_mode!r}"
            )
        self.input_representation = str(cfg.input_representation).lower()
        if self.input_representation not in self.VALID_INPUT_REPRESENTATIONS:
            raise ValueError(
                f"input_representation must be one of {sorted(self.VALID_INPUT_REPRESENTATIONS)}, got {cfg.input_representation!r}"
            )

        memmaps = dict(self.meta.get("memmaps", {}))
        self.export_images = bool(self.meta.get("export_images", False))
        self.export_embeddings = bool(self.meta.get("export_embeddings", False))
        if self.input_representation in {"images", "hybrid"} and not self.export_images:
            raise ValueError("This stream memmap does not contain exported images.")
        if self.input_representation in {"embeddings", "hybrid"} and not self.export_embeddings:
            raise ValueError("This stream memmap does not contain exported pulse embeddings.")

        self.power = None
        self.phase = None
        if self.export_images:
            self.power = np.memmap(self.root / memmaps["power"], mode="r", dtype=np.uint8, shape=(N, H, W))
            self.phase = np.memmap(self.root / memmaps["phase"], mode="r", dtype=np.uint8, shape=(N, H, W))

        self.backbone_embeddings = None
        self.frame_encoder_logits = None
        self.frame_encoder_logit_dim = int(self.meta.get("frame_encoder_logit_dim") or 0)
        if self.export_embeddings:
            emb_dim = int(self.meta.get("embedding_dim") or cfg.embedding_dim)
            self.backbone_embeddings = np.memmap(
                self.root / memmaps["backbone_embeddings"], mode="r", dtype=np.float32, shape=(N, emb_dim)
            )
            logits_name = memmaps.get("frame_encoder_logits")
            if logits_name and self.frame_encoder_logit_dim > 0:
                self.frame_encoder_logits = np.memmap(
                    self.root / logits_name, mode="r", dtype=np.float32, shape=(N, self.frame_encoder_logit_dim)
                )
            self.embedding_dim = emb_dim
        else:
            self.embedding_dim = int(cfg.embedding_dim)

        self.labels = np.memmap(self.root / memmaps["labels"], mode="r", dtype=np.int16, shape=(N,))
        self.window_indices = np.memmap(self.root / memmaps["window_indices"], mode="r", dtype=np.int32, shape=(N,))
        self.stream_offsets = np.memmap(self.root / memmaps["stream_offsets"], mode="r", dtype=np.int64, shape=(S,))
        self.stream_lengths = np.memmap(self.root / memmaps["stream_lengths"], mode="r", dtype=np.int32, shape=(S,))
        self.stream_labels = np.memmap(self.root / memmaps["stream_labels"], mode="r", dtype=np.int16, shape=(S,))

        window_states_name = memmaps.get("window_states")
        self.window_states = None
        if window_states_name:
            self.window_states = np.memmap(self.root / window_states_name, mode="r", dtype=np.uint8, shape=(N,))
        if self.cfg.return_window_state_targets and self.window_states is None:
            raise ValueError(
                "This stream memmap does not contain per-window auxiliary state labels. Re-export the memmap with the updated exporter."
            )
        self.window_class_noise_index = int(cfg.window_class_noise_index) if cfg.window_class_noise_index is not None else int(len(self.label_names))
        self.window_class_ignore_index = int(cfg.window_class_ignore_index)

        snr_name = memmaps.get("stream_snr_db")
        self.stream_snr = None
        if snr_name:
            self.stream_snr = np.memmap(self.root / snr_name, mode="r", dtype=np.float32, shape=(S,))

        self.uses_dense_timeline = self.noise_handling_mode != "pulse_tail_pad"
        self.noise_bank = None
        if self.noise_handling_mode in {"gap_noise_images", "gap_noise_embeddings"}:
            if cfg.noise_bank_dir is None:
                raise ValueError(f"noise_handling_mode={self.noise_handling_mode!r} requires noise_bank_dir to be set.")
            bank_mode = "images" if self.noise_handling_mode == "gap_noise_images" else "embeddings"
            self.noise_bank = NoiseBankMemmap(
                cfg.noise_bank_dir,
                mode=bank_mode,
                normalise_images_to_float=cfg.normalise_to_float,
            )
            if self.noise_bank.height != H or self.noise_bank.width != W:
                raise ValueError(
                    f"Noise bank spectrogram shape {(self.noise_bank.height, self.noise_bank.width)} != stream shape {(H, W)}"
                )
            if self.noise_handling_mode == "gap_noise_embeddings" and int(self.noise_bank.embedding_dim or 0) != self.embedding_dim:
                raise ValueError(
                    f"Noise bank embedding dim {self.noise_bank.embedding_dim} != stream embedding dim {self.embedding_dim}"
                )

        self._rng = np.random.default_rng(int(cfg.base_seed))
        self.samples: list[tuple[int, int, int]] = []
        self._build_samples(S)

    def _build_samples(self, num_streams: int) -> None:
        chunk_len = int(self.cfg.chunk_len)
        stride = max(1, int(self.cfg.chunk_stride))
        for stream_id in range(num_streams):
            sparse_len = int(self.stream_lengths[stream_id])
            if sparse_len <= 0:
                continue
            if not self.uses_dense_timeline:
                if sparse_len < chunk_len:
                    if self.cfg.pad_short_streams:
                        self.samples.append((stream_id, 0, sparse_len))
                    continue
                for start in range(0, sparse_len - chunk_len + 1, stride):
                    self.samples.append((stream_id, start, chunk_len))
                continue

            off = int(self.stream_offsets[stream_id])
            sparse_idx = np.asarray(self.window_indices[off : off + sparse_len], dtype=np.int32)
            if sparse_idx.size == 0:
                continue

            full_len = int(self.full_window_count)
            def _append_dense_sample(start_sparse: int) -> None:
                dense_start = int(sparse_idx[start_sparse])
                effective_len = max(0, min(chunk_len, full_len - dense_start))
                if effective_len > 0:
                    self.samples.append((stream_id, dense_start, effective_len))

            if sparse_len < chunk_len:
                if self.cfg.pad_short_streams:
                    _append_dense_sample(0)
                continue
            for start_sparse in range(0, sparse_len - chunk_len + 1, stride):
                _append_dense_sample(start_sparse)

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_noise_indices(self, dataset_idx: int, stream_id: int, count: int) -> np.ndarray:
        if self.noise_bank is None or count <= 0:
            return np.empty((0,), dtype=np.int64)

        candidate_indices = None
        if self.stream_snr is not None:
            target_snr = float(self.stream_snr[stream_id])
            if np.isfinite(target_snr):
                candidate_indices = np.flatnonzero(np.isclose(self.noise_bank.snr, np.float32(target_snr)))
                if candidate_indices.size == 0:
                    candidate_indices = None
        if candidate_indices is None:
            candidate_indices = np.arange(len(self.noise_bank), dtype=np.int64)

        if self.cfg.deterministic_noise:
            seed = (int(self.cfg.base_seed) + 1000003 * int(dataset_idx) + 9176 * int(stream_id)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
        else:
            rng = self._rng

        if self.cfg.noise_fill_mode == "per_chunk":
            chosen = int(candidate_indices[int(rng.integers(0, len(candidate_indices)))])
            return np.full((count,), chosen, dtype=np.int64)
        return candidate_indices[rng.integers(0, len(candidate_indices), size=count)].astype(np.int64, copy=False)

    def _to_image_array(self, power: np.ndarray, phase: np.ndarray) -> np.ndarray:
        x = np.stack([power, phase], axis=1)
        if self.cfg.normalise_to_float:
            x = x.astype(np.float32) / 255.0
        return x

    def _build_window_class_targets(
        self,
        *,
        stream_label: int,
        signal_mask: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        window_class = np.full((int(self.cfg.chunk_len),), self.window_class_ignore_index, dtype=np.int64)
        if np.any(valid_mask):
            window_class[valid_mask] = np.int64(self.window_class_noise_index)
        if np.any(signal_mask):
            window_class[signal_mask] = np.int64(stream_label)
        return window_class

    def _getitem_pulse_tail_pad(self, idx: int):
        stream_id, start_sparse, effective_len = self.samples[idx]
        T = int(self.cfg.chunk_len)
        off = int(self.stream_offsets[stream_id]) + int(start_sparse)
        label = int(self.stream_labels[stream_id])

        signal_mask = np.zeros((T,), dtype=np.bool_)
        signal_mask[:effective_len] = True
        dense_time = np.full((T,), -1, dtype=np.int32)
        dense_time[:effective_len] = np.asarray(self.window_indices[off : off + effective_len], dtype=np.int32)
        window_state = np.full((T,), WINDOW_STATE_IGNORE, dtype=np.int64)
        if self.window_states is not None:
            window_state[:effective_len] = np.asarray(self.window_states[off : off + effective_len], dtype=np.int64)
        window_class = self._build_window_class_targets(
            stream_label=label,
            signal_mask=signal_mask,
            valid_mask=signal_mask,
        )

        if self.input_representation == "embeddings":
            emb = np.zeros((T, self.embedding_dim), dtype=np.float32)
            emb[:effective_len] = np.asarray(self.backbone_embeddings[off : off + effective_len], dtype=np.float32)
            x_t: Any = torch.from_numpy(np.array(emb, copy=True))
        elif self.input_representation == "images":
            power = np.zeros((T, self.height, self.width), dtype=np.uint8)
            phase = np.zeros((T, self.height, self.width), dtype=np.uint8)
            power[:effective_len] = self.power[off : off + effective_len]
            phase[:effective_len] = self.phase[off : off + effective_len]
            x_t = torch.from_numpy(np.array(self._to_image_array(power, phase), copy=True))
        else:
            power = np.zeros((T, self.height, self.width), dtype=np.uint8)
            phase = np.zeros((T, self.height, self.width), dtype=np.uint8)
            emb = np.zeros((T, self.embedding_dim), dtype=np.float32)
            emb_mask = np.zeros((T,), dtype=np.bool_)
            logit = None
            logit_mask = None
            if self.frame_encoder_logits is not None and self.frame_encoder_logit_dim > 0:
                logit = np.zeros((T, self.frame_encoder_logit_dim), dtype=np.float32)
                logit_mask = np.zeros((T,), dtype=np.bool_)
            power[:effective_len] = self.power[off : off + effective_len]
            phase[:effective_len] = self.phase[off : off + effective_len]
            emb[:effective_len] = np.asarray(self.backbone_embeddings[off : off + effective_len], dtype=np.float32)
            emb_mask[:effective_len] = True
            payload = {
                "frames": torch.from_numpy(np.array(self._to_image_array(power, phase), copy=True)),
                "backbone_embeddings": torch.from_numpy(np.array(emb, copy=True)),
                "embedding_mask": torch.from_numpy(np.array(emb_mask, copy=True)),
            }
            if logit is not None and logit_mask is not None:
                logit[:effective_len] = np.asarray(self.frame_encoder_logits[off : off + effective_len], dtype=np.float32)
                logit_mask[:effective_len] = True
                payload["frame_encoder_logits"] = torch.from_numpy(np.array(logit, copy=True))
                payload["logit_mask"] = torch.from_numpy(np.array(logit_mask, copy=True))
            x_t = payload

        y_t = torch.tensor(label, dtype=torch.long)
        idx_t = torch.from_numpy(np.array(dense_time, dtype=np.int32, copy=True))
        sig_t = torch.from_numpy(np.array(signal_mask, dtype=np.bool_, copy=True))
        state_t = torch.from_numpy(np.array(window_state, dtype=np.int64, copy=True))
        class_t = torch.from_numpy(np.array(window_class, dtype=np.int64, copy=True))
        return self._format_output(x_t, y_t, idx_t, sig_t, state_t, class_t, stream_id)

    def _getitem_dense_gap_mode(self, idx: int):
        stream_id, start_full, effective_len = self.samples[idx]
        T = int(self.cfg.chunk_len)
        label = int(self.stream_labels[stream_id])

        dense_time = np.arange(start_full, start_full + T, dtype=np.int32)
        valid_mask = np.zeros((T,), dtype=np.bool_)
        if effective_len < T:
            valid_mask[:effective_len] = True
            dense_time[effective_len:] = -1
        else:
            valid_mask[:] = True

        power = phase = None
        if self.input_representation in {"images", "hybrid"}:
            power = np.zeros((T, self.height, self.width), dtype=np.uint8)
            phase = np.zeros((T, self.height, self.width), dtype=np.uint8)
        emb = None
        emb_mask = None
        logit = None
        logit_mask = None
        if self.input_representation in {"embeddings", "hybrid"}:
            emb = np.zeros((T, self.embedding_dim), dtype=np.float32)
            emb_mask = np.zeros((T,), dtype=np.bool_)
            if self.frame_encoder_logits is not None and self.frame_encoder_logit_dim > 0:
                logit = np.zeros((T, self.frame_encoder_logit_dim), dtype=np.float32)
                logit_mask = np.zeros((T,), dtype=np.bool_)

        signal_mask = np.zeros((T,), dtype=np.bool_)
        window_state = np.full((T,), WINDOW_STATE_IGNORE, dtype=np.int16)
        window_state[valid_mask] = WINDOW_STATE_NOISE

        off = int(self.stream_offsets[stream_id])
        sparse_len = int(self.stream_lengths[stream_id])
        sparse_idx = np.asarray(self.window_indices[off : off + sparse_len], dtype=np.int32)
        target = dense_time[valid_mask]
        if sparse_idx.size > 0 and target.size > 0:
            pos = np.searchsorted(sparse_idx, target)
            in_bounds = pos < sparse_idx.size
            hit = np.zeros_like(in_bounds, dtype=np.bool_)
            if np.any(in_bounds):
                hit[in_bounds] = sparse_idx[pos[in_bounds]] == target[in_bounds]
            if np.any(hit):
                dense_hit_pos = np.where(valid_mask)[0][hit]
                sparse_hit_pos = pos[hit]
                src_rows = off + sparse_hit_pos
                signal_mask[dense_hit_pos] = True
                if power is not None and phase is not None:
                    power[dense_hit_pos] = self.power[src_rows]
                    phase[dense_hit_pos] = self.phase[src_rows]
                if emb is not None and emb_mask is not None:
                    emb[dense_hit_pos] = np.asarray(self.backbone_embeddings[src_rows], dtype=np.float32)
                    emb_mask[dense_hit_pos] = True
                    if logit is not None and logit_mask is not None and self.frame_encoder_logits is not None:
                        logit[dense_hit_pos] = np.asarray(self.frame_encoder_logits[src_rows], dtype=np.float32)
                        logit_mask[dense_hit_pos] = True
                if self.window_states is not None:
                    window_state[dense_hit_pos] = np.asarray(self.window_states[src_rows], dtype=np.int16)

        gap_pos = np.where(valid_mask & (~signal_mask))[0]
        if gap_pos.size > 0:
            if self.noise_handling_mode == "gap_noise_images":
                noise_ids = self._choose_noise_indices(idx, stream_id, int(gap_pos.size))
                if power is None or phase is None or self.noise_bank is None:
                    raise ValueError("gap_noise_images requested but image noise path is unavailable.")
                power[gap_pos] = self.noise_bank.power[noise_ids]
                phase[gap_pos] = self.noise_bank.phase[noise_ids]
            elif self.noise_handling_mode == "gap_noise_embeddings":
                noise_ids = self._choose_noise_indices(idx, stream_id, int(gap_pos.size))
                if emb is None or emb_mask is None or self.noise_bank is None:
                    raise ValueError("gap_noise_embeddings requested but embedding noise path is unavailable.")
                emb[gap_pos] = np.asarray(self.noise_bank.embedding[noise_ids], dtype=np.float32)
                emb_mask[gap_pos] = True
                if logit is not None and logit_mask is not None and getattr(self.noise_bank, "encoder_logits", None) is not None:
                    logit[gap_pos] = np.asarray(self.noise_bank.encoder_logits[noise_ids], dtype=np.float32)
                    logit_mask[gap_pos] = True
            elif self.noise_handling_mode == "gap_padding":
                # Leave zero features in the dense timeline.
                pass

        if self.input_representation == "images":
            x_t: Any = torch.from_numpy(np.array(self._to_image_array(power, phase), copy=True))
        elif self.input_representation == "embeddings":
            x_t = torch.from_numpy(np.array(emb, copy=True))
        else:
            payload = {
                "frames": torch.from_numpy(np.array(self._to_image_array(power, phase), copy=True)),
                "backbone_embeddings": torch.from_numpy(np.array(emb, copy=True)),
                "embedding_mask": torch.from_numpy(np.array(emb_mask, copy=True)),
            }
            if logit is not None and logit_mask is not None:
                payload["frame_encoder_logits"] = torch.from_numpy(np.array(logit, copy=True))
                payload["logit_mask"] = torch.from_numpy(np.array(logit_mask, copy=True))
            x_t = payload

        window_class = self._build_window_class_targets(
            stream_label=label,
            signal_mask=signal_mask,
            valid_mask=valid_mask,
        )
        y_t = torch.tensor(label, dtype=torch.long)
        idx_t = torch.from_numpy(np.array(dense_time, dtype=np.int32, copy=True))
        sig_t = torch.from_numpy(np.array(signal_mask, dtype=np.bool_, copy=True))
        state_t = torch.from_numpy(np.array(window_state, dtype=np.int64, copy=True))
        class_t = torch.from_numpy(np.array(window_class, dtype=np.int64, copy=True))
        return self._format_output(x_t, y_t, idx_t, sig_t, state_t, class_t, stream_id)

    def _format_output(
        self,
        x_t: Any,
        y_t: torch.Tensor,
        idx_t: torch.Tensor,
        sig_t: torch.Tensor,
        state_t: torch.Tensor,
        class_t: torch.Tensor,
        stream_id: int,
    ):
        out: list[Any] = [x_t, y_t]
        if self.cfg.return_window_indices:
            out.append(idx_t)
        if self.cfg.return_signal_mask:
            out.append(sig_t)
        if self.cfg.return_window_state_targets:
            out.append(state_t)
        if self.cfg.return_window_class_targets:
            out.append(class_t)
        if self.cfg.return_stream_id:
            out.append(torch.tensor(stream_id, dtype=torch.long))
        return tuple(out) if len(out) > 2 else (x_t, y_t)

    def __getitem__(self, idx: int) -> Any:
        if self.noise_handling_mode == "pulse_tail_pad":
            return self._getitem_pulse_tail_pad(idx)
        return self._getitem_dense_gap_mode(idx)



def create_mfd_stream_loader(
    cfg: LoaderConfig,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    ds = MFDStreamMemmapDataset(cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class NoiseBankMemmap:
    VALID_MODES = {"images", "embeddings", "both"}

    def __init__(self, noise_bank_dir: str | Any, *, mode: str = "images", normalise_images_to_float: bool = True):
        self.root = to_path(noise_bank_dir)
        self.mode = str(mode).lower()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self.VALID_MODES)}, got: {mode!r}")
        self.normalise_images_to_float = bool(normalise_images_to_float)

        with open(self.root / "meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.num_samples = int(self.meta["num_samples"])
        self.height = int(self.meta["spectrogram_height"])
        self.width = int(self.meta["spectrogram_width"])
        self.export_images = bool(self.meta.get("export_images", False))
        self.export_embeddings = bool(self.meta.get("export_embeddings", False))
        self.embedding_dim = self.meta.get("embedding_dim")
        self.memmaps = dict(self.meta.get("memmaps", {}))

        if self.mode in {"images", "both"} and not self.export_images:
            raise ValueError("This noise bank does not contain exported noise images.")
        if self.mode in {"embeddings", "both"} and not self.export_embeddings:
            raise ValueError("This noise bank does not contain exported noise embeddings.")

        self.snr = np.memmap(self.root / self.memmaps["snr_db"], mode="r", dtype=np.float32, shape=(self.num_samples,))
        self.power = None
        self.phase = None
        self.embedding = None
        self.encoder_logits = None
        self.frame_encoder_logit_dim = int(self.meta.get("frame_encoder_logit_dim") or 0)
        if self.export_images:
            self.power = np.memmap(self.root / self.memmaps["power"], mode="r", dtype=np.uint8, shape=(self.num_samples, self.height, self.width))
            self.phase = np.memmap(self.root / self.memmaps["phase"], mode="r", dtype=np.uint8, shape=(self.num_samples, self.height, self.width))
        if self.export_embeddings:
            emb_dim = int(self.embedding_dim)
            self.embedding = np.memmap(self.root / self.memmaps["embedding"], mode="r", dtype=np.float32, shape=(self.num_samples, emb_dim))
            logits_name = self.memmaps.get("encoder_logits")
            if logits_name and self.frame_encoder_logit_dim > 0:
                self.encoder_logits = np.memmap(self.root / logits_name, mode="r", dtype=np.float32, shape=(self.num_samples, self.frame_encoder_logit_dim))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        snr_db = float(self.snr[idx])
        if self.mode == "images":
            x_img = np.stack([self.power[idx], self.phase[idx]], axis=0)
            if self.normalise_images_to_float:
                x_img = x_img.astype(np.float32) / 255.0
            return x_img, snr_db
        if self.mode == "embeddings":
            return np.asarray(self.embedding[idx], dtype=np.float32), snr_db
        x_img = np.stack([self.power[idx], self.phase[idx]], axis=0)
        if self.normalise_images_to_float:
            x_img = x_img.astype(np.float32) / 255.0
        return x_img, np.asarray(self.embedding[idx], dtype=np.float32), snr_db
