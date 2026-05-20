from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    ExportConfig,
    WINDOW_STATE_FULL_PULSE,
    WINDOW_STATE_NOISE,
    WINDOW_STATE_PARTIAL_PULSE,
    build_class_subset_tag,
    canonicalise_label_subset,
    ensure_empty_or_create,
    to_path,
)
from .encoder_utils import build_frozen_dual_backbone, encode_dual_spectrogram_batch_with_logits
from .iq_dataset import PulseRecord, build_pulse_index, save_index_summary
from .spectrogram_adapter import RenderedSpectrogram, SpectrogramAdapter
from .stream_builder import BuiltStream, StreamBuilder, WindowPlan


class StreamMemmapExporter:
    def __init__(self, cfg: ExportConfig):
        self.cfg = cfg
        self.input_root = to_path(cfg.input_root)
        self.output_dir = ensure_empty_or_create(cfg.output_dir, cfg.overwrite)
        self.records, self.label_map = build_pulse_index(self.input_root, cfg.scan)
        if not self.records:
            raise ValueError(f"No pulse records found under: {self.input_root}")
        save_index_summary(self.output_dir, cfg.split_name, self.records, self.label_map)
        self.records_by_label: dict[int, list[PulseRecord]] = defaultdict(list)
        for rec in self.records:
            self.records_by_label[rec.label_index].append(rec)
        self.adapter = SpectrogramAdapter(cfg.spectrogram)
        self.builder = StreamBuilder(self.records_by_label, cfg.stream, cfg.scan, split_name=cfg.split_name)
        self.label_names = [label for label, _ in sorted(self.label_map.items(), key=lambda kv: kv[1])]
        self.encoder = None
        self.encoder_device = None
        self.embedding_dim: int | None = None
        self.encoder_meta: dict[str, object] = {}
        if self.cfg.export_embeddings:
            if self.cfg.resnet_checkpoint_path is None:
                raise ValueError("export_embeddings=True requires resnet_checkpoint_path to be set.")
            self.encoder, self.embedding_dim, self.encoder_meta, self.encoder_device = build_frozen_dual_backbone(
                self.cfg.resnet_checkpoint_path,
                device=self.cfg.device,
                strict=self.cfg.checkpoint_strict,
            )

    def _streams_requested_for_label(self, label_index: int) -> int:
        spc = self.cfg.stream.streams_per_class
        if spc is None or int(spc) == -1:
            return len(self.records_by_label[label_index])
        return int(spc)

    def _should_save_debug(self, plan: WindowPlan, *, rejected: bool) -> bool:
        base_prob = self.cfg.qa_rejected_probability if rejected else self.cfg.qa_keep_probability
        if base_prob <= 0.0:
            return False
        # Deterministic probability per stream/window so QA exports are stable.
        seed = (
            int(self.cfg.stream.base_seed)
            + 1009 * int(plan.label_index)
            + 9176 * int(plan.stream_index)
            + 13 * int(plan.window_index)
            + (1 if rejected else 0)
        ) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        return bool(rng.random() < float(base_prob))

    @staticmethod
    def _normalise_for_png(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array for PNG export, got shape={arr.shape!r}")
        if arr.dtype == np.uint8:
            return arr.copy()
        arr = np.asarray(arr, dtype=np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            return np.zeros(arr.shape, dtype=np.uint8)
        valid = arr[finite]
        lo = float(valid.min())
        hi = float(valid.max())
        if hi <= lo:
            out = np.zeros(arr.shape, dtype=np.uint8)
            out[finite] = 255
            return out
        scaled = (arr - lo) / (hi - lo)
        scaled = np.clip(scaled, 0.0, 1.0)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
        return np.asarray(np.round(scaled * 255.0), dtype=np.uint8)

    def _save_png(self, path: Path, arr: np.ndarray, *, min_visual_width: int | None = None, min_visual_height: int | None = None) -> None:
        img = Image.fromarray(self._normalise_for_png(arr), mode="L")
        if min_visual_width is not None or min_visual_height is not None:
            target_w = max(int(min_visual_width or img.width), int(img.width))
            target_h = max(int(min_visual_height or img.height), int(img.height))
            if target_w != img.width or target_h != img.height:
                img = img.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        img.save(path)

    def _save_dual_png(self, path: Path, dual_u8: np.ndarray) -> None:
        dual_u8 = np.asarray(dual_u8, dtype=np.uint8)
        if dual_u8.ndim != 3 or dual_u8.shape[0] != 2:
            raise ValueError(f"Expected dual spectrogram with shape (2, H, W), got {dual_u8.shape!r}")
        h, w = int(dual_u8.shape[1]), int(dual_u8.shape[2])
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = dual_u8[0]
        rgb[..., 1] = dual_u8[1]
        Image.fromarray(rgb, mode="RGB").save(path)

    def _save_debug_assets(self, plan: WindowPlan, rendered: RenderedSpectrogram, *, rejected: bool) -> None:
        subset = "rejected_p_to_n" if rejected else "kept_p"
        root = self.output_dir / "qa_examples" / subset / plan.label_name
        root.mkdir(parents=True, exist_ok=True)
        stem = (
            f"stream{plan.stream_index:05d}_win{plan.window_index:04d}_"
            f"emitter_{plan.emitter_id}_full{int(plan.full_pulse)}_"
            f"ov{plan.overlap_fraction_window:.4f}"
        )
        # Raw STFT grids can be very narrow in time (for example ~7 bins for a 1000-sample
        # window with hop=128). Upscale only the saved PNG view so humans can inspect it easily;
        # keep the true raw array shape in the sidecar metadata.
        self._save_png(root / f"{stem}__raw_power.png", rendered.raw_power, min_visual_width=224)
        self._save_png(root / f"{stem}__raw_phase.png", rendered.raw_phase, min_visual_width=224)
        self._save_png(root / f"{stem}__power_224.png", rendered.power_u8)
        self._save_png(root / f"{stem}__phase_224.png", rendered.phase_u8)
        self._save_dual_png(root / f"{stem}__dual_224.png", rendered.dual_u8)
        meta = {
            "label_name": plan.label_name,
            "label_index": int(plan.label_index),
            "dataset_split": self.cfg.split_name,
            "stream_index": int(plan.stream_index),
            "window_index": int(plan.window_index),
            "source_record_index": int(plan.source_record_index),
            "emitter_id": plan.emitter_id,
            "full_pulse": bool(plan.full_pulse),
            "window_state": int(plan.window_state),
            "overlap_samples": int(plan.overlap_samples),
            "overlap_fraction_window": float(plan.overlap_fraction_window),
            "overlap_fraction_pulse": float(plan.overlap_fraction_pulse),
            "pw_samples": int(plan.pw_samples),
            "pri_samples": int(plan.pri_samples),
            "toa_samples": int(plan.toa_samples),
            "rf_hz": None if plan.rf_hz is None else float(plan.rf_hz),
            "snr_db": None if plan.snr_db is None else float(plan.snr_db),
            "raw_power_shape": list(rendered.raw_power.shape),
            "raw_phase_shape": list(rendered.raw_phase.shape),
            "resized_shape": [int(rendered.power_u8.shape[0]), int(rendered.power_u8.shape[1])],
        }
        with open(root / f"{stem}__meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def export(self) -> Path:
        plan = self._plan_streams()
        if plan["total_kept_windows"] <= 0:
            raise ValueError("No valid non-noise windows were generated during the planning pass.")

        H = int(self.cfg.spectrogram.out_height)
        W = int(self.cfg.spectrogram.out_width)
        N = int(plan["total_kept_windows"])
        S = int(plan["num_streams"])

        power_mm = phase_mm = emb_mm = logit_mm = None
        if self.cfg.export_images:
            power_mm = np.memmap(self.output_dir / "power_u8.bin", mode="w+", dtype=np.uint8, shape=(N, H, W))
            phase_mm = np.memmap(self.output_dir / "phase_u8.bin", mode="w+", dtype=np.uint8, shape=(N, H, W))
        if self.cfg.export_embeddings:
            emb_dim = int(self.embedding_dim or 0)
            logit_dim = int(self.encoder_meta.get("num_classes") or len(self.label_names))
            emb_mm = np.memmap(self.output_dir / "backbone_embedding_f32.bin", mode="w+", dtype=np.float32, shape=(N, emb_dim))
            logit_mm = np.memmap(self.output_dir / "frame_encoder_logits_f32.bin", mode="w+", dtype=np.float32, shape=(N, logit_dim))

        labels_mm = np.memmap(self.output_dir / "labels_i16.bin", mode="w+", dtype=np.int16, shape=(N,))
        stream_ids_mm = np.memmap(self.output_dir / "stream_ids_i32.bin", mode="w+", dtype=np.int32, shape=(N,))
        source_record_mm = np.memmap(self.output_dir / "source_record_i32.bin", mode="w+", dtype=np.int32, shape=(N,))
        window_indices_mm = np.memmap(self.output_dir / "window_indices_i32.bin", mode="w+", dtype=np.int32, shape=(N,))
        window_states_mm = np.memmap(self.output_dir / "window_states_u8.bin", mode="w+", dtype=np.uint8, shape=(N,))
        full_pulse_mm = np.memmap(self.output_dir / "full_pulse_u8.bin", mode="w+", dtype=np.uint8, shape=(N,))
        overlap_samples_mm = np.memmap(self.output_dir / "overlap_samples_i32.bin", mode="w+", dtype=np.int32, shape=(N,))
        overlap_window_mm = np.memmap(self.output_dir / "overlap_fraction_window_f32.bin", mode="w+", dtype=np.float32, shape=(N,))
        overlap_pulse_mm = np.memmap(self.output_dir / "overlap_fraction_pulse_f32.bin", mode="w+", dtype=np.float32, shape=(N,))
        pw_mm = np.memmap(self.output_dir / "pw_samples_i32.bin", mode="w+", dtype=np.int32, shape=(N,))
        pri_mm = np.memmap(self.output_dir / "pri_samples_i32.bin", mode="w+", dtype=np.int32, shape=(N,))
        toa_mm = np.memmap(self.output_dir / "toa_samples_i32.bin", mode="w+", dtype=np.int32, shape=(N,))
        rf_mm = np.memmap(self.output_dir / "rf_hz_f32.bin", mode="w+", dtype=np.float32, shape=(N,))
        snr_mm = np.memmap(self.output_dir / "snr_db_f32.bin", mode="w+", dtype=np.float32, shape=(N,))

        stream_offsets_mm = np.memmap(self.output_dir / "stream_offsets_i64.bin", mode="w+", dtype=np.int64, shape=(S,))
        stream_lengths_mm = np.memmap(self.output_dir / "stream_lengths_i32.bin", mode="w+", dtype=np.int32, shape=(S,))
        stream_labels_mm = np.memmap(self.output_dir / "stream_labels_i16.bin", mode="w+", dtype=np.int16, shape=(S,))
        stream_source_record_mm = np.memmap(self.output_dir / "stream_source_record_i32.bin", mode="w+", dtype=np.int32, shape=(S,))
        stream_snr_mm = np.memmap(self.output_dir / "stream_snr_f32.bin", mode="w+", dtype=np.float32, shape=(S,))

        pending_dual_specs: list[np.ndarray] = []
        pending_rows: list[int] = []
        cursor = 0
        global_stream_id = 0

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

        for label_index in range(len(self.label_names)):
            for local_stream_index in range(self._streams_requested_for_label(label_index)):
                built = self.builder.build_stream(
                    label_index,
                    local_stream_index,
                    context_samples_each_side=self.cfg.spectrogram.context_samples_each_side,
                )
                length = int(len(built.kept_windows))
                if length < self.cfg.stream.min_valid_windows_per_stream:
                    continue
                stream_offsets_mm[global_stream_id] = cursor
                stream_lengths_mm[global_stream_id] = length
                stream_labels_mm[global_stream_id] = built.label_index
                stream_source_record_mm[global_stream_id] = built.source_record_index
                stream_snr_mm[global_stream_id] = np.nan if built.snr_db is None else float(built.snr_db)

                for plan_item in built.kept_windows:
                    rendered = self.adapter.render_window(plan_item.pulse_iq_with_context, self.cfg.stream.window_len)
                    if power_mm is not None and phase_mm is not None:
                        power_mm[cursor] = rendered.power_u8
                        phase_mm[cursor] = rendered.phase_u8
                    if emb_mm is not None:
                        pending_dual_specs.append(rendered.dual_u8)
                        pending_rows.append(cursor)
                        if len(pending_dual_specs) >= max(1, int(self.cfg.encoder_batch_size)):
                            flush_pending_embeddings()
                    labels_mm[cursor] = plan_item.label_index
                    stream_ids_mm[cursor] = global_stream_id
                    source_record_mm[cursor] = plan_item.source_record_index
                    window_indices_mm[cursor] = int(plan_item.window_index)
                    window_states_mm[cursor] = np.uint8(plan_item.window_state)
                    full_pulse_mm[cursor] = np.uint8(plan_item.full_pulse)
                    overlap_samples_mm[cursor] = int(plan_item.overlap_samples)
                    overlap_window_mm[cursor] = float(plan_item.overlap_fraction_window)
                    overlap_pulse_mm[cursor] = float(plan_item.overlap_fraction_pulse)
                    pw_mm[cursor] = int(plan_item.pw_samples)
                    pri_mm[cursor] = int(plan_item.pri_samples)
                    toa_mm[cursor] = int(plan_item.toa_samples)
                    rf_mm[cursor] = np.nan if plan_item.rf_hz is None else float(plan_item.rf_hz)
                    snr_mm[cursor] = np.nan if plan_item.snr_db is None else float(plan_item.snr_db)
                    if self._should_save_debug(plan_item, rejected=False):
                        self._save_debug_assets(plan_item, rendered, rejected=False)
                    cursor += 1

                for plan_item in built.rejected_windows:
                    if not self._should_save_debug(plan_item, rejected=True):
                        continue
                    rendered = self.adapter.render_window(plan_item.pulse_iq_with_context, self.cfg.stream.window_len)
                    self._save_debug_assets(plan_item, rendered, rejected=True)

                global_stream_id += 1
                if global_stream_id % max(1, self.cfg.flush_every_streams) == 0:
                    flush_pending_embeddings()
                    self._flush_all(
                        power_mm, phase_mm, emb_mm, logit_mm,
                        labels_mm, stream_ids_mm, source_record_mm,
                        window_indices_mm, window_states_mm, full_pulse_mm,
                        overlap_samples_mm, overlap_window_mm, overlap_pulse_mm,
                        pw_mm, pri_mm, toa_mm, rf_mm, snr_mm,
                        stream_offsets_mm, stream_lengths_mm, stream_labels_mm,
                        stream_source_record_mm, stream_snr_mm,
                    )
                    print(f"[{self.cfg.split_name}] exported streams={global_stream_id}/{S} written_windows={cursor}/{N}")

        flush_pending_embeddings()
        self._flush_all(
            power_mm, phase_mm, emb_mm, logit_mm,
            labels_mm, stream_ids_mm, source_record_mm,
            window_indices_mm, window_states_mm, full_pulse_mm,
            overlap_samples_mm, overlap_window_mm, overlap_pulse_mm,
            pw_mm, pri_mm, toa_mm, rf_mm, snr_mm,
            stream_offsets_mm, stream_lengths_mm, stream_labels_mm,
            stream_source_record_mm, stream_snr_mm,
        )

        requested_labels = canonicalise_label_subset(self.cfg.scan.modulation_list)
        meta = {
            "split_name": self.cfg.split_name,
            "dataset_split": self.cfg.split_name,
            "class_subset_tag": build_class_subset_tag(requested_labels),
            "requested_modulation_list": list(requested_labels),
            "input_root": str(self.input_root),
            "output_dir": str(self.output_dir),
            "num_streams": int(global_stream_id),
            "num_valid_windows": int(cursor),
            "label_names": self.label_names,
            "sample_rate_hz": float(self.cfg.stream.sample_rate_hz),
            "window_len": int(self.cfg.stream.window_len),
            "window_count": int(self.cfg.stream.resolved_window_count()),
            "overlap_threshold_fraction": float(self.cfg.stream.overlap_threshold_fraction),
            "spectrogram_height": H,
            "spectrogram_width": W,
            "spectrogram_context_samples_each_side": int(self.cfg.spectrogram.context_samples_each_side),
            "streams_per_class_requested": self.cfg.stream.streams_per_class,
            "use_recorded_pri": bool(self.cfg.stream.use_recorded_pri),
            "use_recorded_toa": bool(self.cfg.stream.use_recorded_toa),
            "add_noise_to_stream": bool(self.cfg.stream.add_noise_to_stream),
            "noise_power_db": float(self.cfg.stream.noise_power_db),
            "dropped_noise_windows": True,
            "window_indices_preserved": True,
            "export_images": bool(self.cfg.export_images),
            "export_embeddings": bool(self.cfg.export_embeddings),
            "embedding_dim": int(self.embedding_dim) if self.embedding_dim is not None else None,
            "frame_encoder_logit_dim": int(self.encoder_meta.get("num_classes")) if self.cfg.export_embeddings else None,
            "encoder": self.encoder_meta if self.cfg.export_embeddings else None,
            "window_state_labels": {
                "0": "gap_or_noise",
                "1": "full_pulse",
                "2": "partial_pulse",
            },
            "window_state_value_map": {
                "gap_or_noise": int(WINDOW_STATE_NOISE),
                "full_pulse": int(WINDOW_STATE_FULL_PULSE),
                "partial_pulse": int(WINDOW_STATE_PARTIAL_PULSE),
            },
            "memmaps": {
                **({"power": "power_u8.bin", "phase": "phase_u8.bin"} if self.cfg.export_images else {}),
                **({
                    "backbone_embeddings": "backbone_embedding_f32.bin",
                    "frame_encoder_logits": "frame_encoder_logits_f32.bin",
                } if self.cfg.export_embeddings else {}),
                "labels": "labels_i16.bin",
                "stream_ids": "stream_ids_i32.bin",
                "source_record": "source_record_i32.bin",
                "window_indices": "window_indices_i32.bin",
                "window_states": "window_states_u8.bin",
                "full_pulse": "full_pulse_u8.bin",
                "overlap_samples": "overlap_samples_i32.bin",
                "overlap_fraction_window": "overlap_fraction_window_f32.bin",
                "overlap_fraction_pulse": "overlap_fraction_pulse_f32.bin",
                "pw_samples": "pw_samples_i32.bin",
                "pri_samples": "pri_samples_i32.bin",
                "toa_samples": "toa_samples_i32.bin",
                "rf_hz": "rf_hz_f32.bin",
                "snr_db": "snr_db_f32.bin",
                "stream_offsets": "stream_offsets_i64.bin",
                "stream_lengths": "stream_lengths_i32.bin",
                "stream_labels": "stream_labels_i16.bin",
                "stream_source_record": "stream_source_record_i32.bin",
                "stream_snr_db": "stream_snr_f32.bin",
            },
        }
        with open(self.output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[{self.cfg.split_name}] done: streams={global_stream_id}, valid_windows={cursor}, shape=({cursor}, {H}, {W})")
        return self.output_dir

    @staticmethod
    def _flush_all(*arrays) -> None:
        for arr in arrays:
            if arr is not None:
                arr.flush()

    def _plan_streams(self) -> dict[str, int]:
        total_kept_windows = 0
        total_rejected_windows = 0
        num_streams = 0
        counts_per_label: dict[str, int] = {}
        for label_index, label_name in enumerate(self.label_names):
            kept = 0
            for local_stream_index in range(self._streams_requested_for_label(label_index)):
                built = self.builder.build_stream(
                    label_index,
                    local_stream_index,
                    context_samples_each_side=self.cfg.spectrogram.context_samples_each_side,
                )
                length = int(len(built.kept_windows))
                if length < self.cfg.stream.min_valid_windows_per_stream:
                    continue
                total_kept_windows += length
                total_rejected_windows += int(len(built.rejected_windows))
                num_streams += 1
                kept += 1
            counts_per_label[label_name] = kept
        print("=" * 70)
        print(f"STREAM EXPORT PLAN [{self.cfg.split_name}]")
        print(f"  labels                : {self.label_names}")
        print(f"  streams kept per label: {counts_per_label}")
        print(f"  total streams         : {num_streams}")
        print(f"  kept pulse windows    : {total_kept_windows}")
        print(f"  rejected P=>N windows : {total_rejected_windows}")
        print("=" * 70)
        return {
            "total_kept_windows": int(total_kept_windows),
            "total_rejected_windows": int(total_rejected_windows),
            "num_streams": int(num_streams),
        }



def export_stream_memmap(cfg: ExportConfig) -> Path:
    return StreamMemmapExporter(cfg).export()
