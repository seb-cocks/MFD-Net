from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SingleFrameExportConfig, build_class_subset_tag, canonicalise_label_subset, ensure_empty_or_create, to_path
from .iq_dataset import PulseRecord, build_pulse_index, load_pulse_iq, save_index_summary
from .noise_utils import add_noise
from .spectrogram_adapter import SpectrogramAdapter


@dataclass(slots=True)
class _FramePlan:
    stream_id: int
    label_name: str
    label_index: int
    snr_db: float
    frame_idx: int
    frames_in_stream: int
    source_record_index: int
    window_index: int
    overlap_samples: int
    full_pulse: bool
    pw_samples: int
    toa_samples: int
    iq_with_context: np.ndarray


class SingleFrameMemmapExporter:
    """Export pulse-covering frames for single-frame ResNet training.

    Each source H5 pulse becomes one short stream containing only the minimum number
    of 20 us windows needed to cover the *first* pulse instance. Windows that do not
    overlap that first pulse are not exported. Noise-only examples are appended as a
    normal class so the single-frame trainer can learn an explicit noise label.
    """

    VALID_IMAGE_MODES = {"power", "phase", "dual"}

    def __init__(self, cfg: SingleFrameExportConfig):
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
        self.image_mode = str(cfg.image_mode).lower()
        if self.image_mode not in self.VALID_IMAGE_MODES:
            raise ValueError(f"image_mode must be one of {sorted(self.VALID_IMAGE_MODES)}, got {cfg.image_mode!r}")
        self.window_len = int(cfg.frame.window_len)
        self.context = int(cfg.spectrogram.context_samples_each_side)
        self.label_names = [label for label, _ in sorted(self.label_map.items(), key=lambda kv: kv[1])]
        if cfg.include_noise_class and cfg.noise_label not in self.label_names:
            self.label_names = self.label_names + [cfg.noise_label]
        self.class_to_id = {name: i for i, name in enumerate(self.label_names)}

    def _record_rng(self, label_index: int, record_index: int) -> np.random.Generator:
        return np.random.default_rng(int(self.cfg.frame.base_seed) + 1000003 * int(label_index) + 9176 * int(record_index))

    def _resolve_toa_samples(self, rng: np.random.Generator, rec: PulseRecord, pulse_len: int) -> int:
        if self.cfg.frame.use_recorded_toa and int(rec.toa_idx) > 0:
            return int(rec.toa_idx % self.window_len)
        max_start = max(0, self.window_len - 1)
        if self.cfg.frame.bias_small_pw_initial_toa and pulse_len <= int(self.cfg.frame.short_pulse_max_samples):
            margin = int(max(1, self.cfg.frame.short_pulse_safe_margin_samples))
            safe_low = margin
            safe_high = max(safe_low, self.window_len - pulse_len - margin)
            if safe_high >= safe_low:
                return int(rng.integers(safe_low, safe_high + 1))
        return int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

    def _resolve_stream_snr(self, rng: np.random.Generator, rec: PulseRecord) -> float:
        if self.cfg.frame.use_signal_noise_attr and np.isfinite(rec.signal_noise):
            return float(rec.signal_noise)
        return float(rng.choice(np.asarray(self.cfg.frame.snr_db_choices, dtype=np.float32)))

    def _select_image(self, rendered):
        if self.image_mode == "power":
            return rendered.power_u8[np.newaxis, ...]
        if self.image_mode == "phase":
            return rendered.phase_u8[np.newaxis, ...]
        return rendered.dual_u8

    def _build_record_frames(self, rec: PulseRecord, source_record_index: int, stream_id: int) -> list[_FramePlan]:
        rng = self._record_rng(rec.label_index, source_record_index)
        pulse = load_pulse_iq(rec, self.cfg.scan)
        pulse_len = int(rec.pw) if rec.pw is not None and int(rec.pw) > 0 else int(len(pulse))
        pulse_len = min(pulse_len, int(len(pulse)))
        if pulse_len <= 0:
            return []
        pulse = pulse[:pulse_len].astype(np.complex64, copy=False)
        toa_samples = self._resolve_toa_samples(rng, rec, pulse_len)
        pulse_start = int(toa_samples)
        pulse_end = int(pulse_start + pulse_len)
        total_windows = max(1, int(math.ceil(pulse_end / self.window_len)))
        total_samples = int(total_windows * self.window_len)

        clean_stream = np.zeros((total_samples,), dtype=np.complex64)
        clean_stream[pulse_start:pulse_end] = pulse
        snr_db = self._resolve_stream_snr(rng, rec) if self.cfg.frame.add_noise_to_stream else 0.0
        noisy_stream = clean_stream
        if self.cfg.frame.add_noise_to_stream:
            i_noisy, q_noisy = add_noise(
                rng,
                clean_stream.real.astype(np.float32, copy=False),
                clean_stream.imag.astype(np.float32, copy=False),
                snr_db=float(snr_db),
                noise_power_db=float(self.cfg.frame.noise_power_db),
                apply_hard_clip=self.cfg.frame.apply_hard_clip,
                clip_min=self.cfg.frame.clip_min,
                clip_max=self.cfg.frame.clip_max,
            )
            noisy_stream = (i_noisy + 1j * q_noisy).astype(np.complex64, copy=False)

        kept: list[_FramePlan] = []
        threshold = float(self.cfg.frame.overlap_threshold_fraction)
        for window_index in range(total_windows):
            win_start = int(window_index * self.window_len)
            win_end = int(win_start + self.window_len)
            overlap = max(0, min(pulse_end, win_end) - max(pulse_start, win_start))
            if overlap <= 0:
                continue
            overlap_fraction_window = float(overlap / self.window_len)
            if overlap_fraction_window < threshold:
                continue
            context_left = int(max(0, min(self.context, win_start)))
            context_right = int(max(0, min(self.context, total_samples - win_end)))
            iq_with_context = noisy_stream[win_start - context_left : win_end + context_right].astype(np.complex64, copy=True)
            kept.append(_FramePlan(
                stream_id=stream_id,
                label_name=rec.label_name,
                label_index=int(self.class_to_id[rec.label_name]),
                snr_db=float(snr_db),
                frame_idx=0,
                frames_in_stream=0,
                source_record_index=int(source_record_index),
                window_index=int(window_index),
                overlap_samples=int(overlap),
                full_pulse=bool(overlap == pulse_len),
                pw_samples=int(pulse_len),
                toa_samples=int(toa_samples),
                iq_with_context=iq_with_context,
            ))
        for idx, plan in enumerate(kept):
            plan.frame_idx = int(idx)
            plan.frames_in_stream = int(len(kept))
        return kept

    def _iter_pulse_frame_plans(self):
        stream_id = 0
        source_record_index = 0
        for label_name, label_old in sorted(self.label_map.items(), key=lambda kv: kv[1]):
            for rec in self.records_by_label[label_old]:
                frames = self._build_record_frames(rec, source_record_index=source_record_index, stream_id=stream_id)
                source_record_index += 1
                if not frames:
                    continue
                yield frames
                stream_id += 1

    def _iter_noise_frame_specs(self, starting_stream_id: int):
        if not self.cfg.include_noise_class or int(self.cfg.noise_samples_per_snr) <= 0:
            return
        total_len = self.window_len + 2 * self.context
        zeros = np.zeros((total_len,), dtype=np.float32)
        rng = np.random.default_rng(int(self.cfg.frame.base_seed) + 77777)
        noise_label_id = int(self.class_to_id[self.cfg.noise_label])
        stream_id = int(starting_stream_id)
        for snr_db in self.cfg.frame.snr_db_choices:
            for _ in range(int(self.cfg.noise_samples_per_snr)):
                i_noisy, q_noisy = add_noise(
                    rng,
                    zeros,
                    zeros,
                    snr_db=float(snr_db),
                    noise_power_db=float(self.cfg.frame.noise_power_db),
                    apply_hard_clip=self.cfg.frame.apply_hard_clip,
                    clip_min=self.cfg.frame.clip_min,
                    clip_max=self.cfg.frame.clip_max,
                )
                iq = (i_noisy + 1j * q_noisy).astype(np.complex64, copy=False)
                yield _FramePlan(
                    stream_id=stream_id,
                    label_name=self.cfg.noise_label,
                    label_index=noise_label_id,
                    snr_db=float(snr_db),
                    frame_idx=0,
                    frames_in_stream=1,
                    source_record_index=-1,
                    window_index=0,
                    overlap_samples=0,
                    full_pulse=False,
                    pw_samples=0,
                    toa_samples=0,
                    iq_with_context=iq.copy(),
                )
                stream_id += 1

    def export(self) -> Path:
        pulse_plans = list(self._iter_pulse_frame_plans())
        pulse_frame_count = int(sum(len(x) for x in pulse_plans))
        noise_stream_start = int(len(pulse_plans))
        noise_plans = list(self._iter_noise_frame_specs(noise_stream_start))
        N = pulse_frame_count + len(noise_plans)
        if N <= 0:
            raise ValueError("No single-frame samples were generated.")

        H = int(self.cfg.spectrogram.out_height)
        W = int(self.cfg.spectrogram.out_width)
        Cimg = 2 if self.image_mode == "dual" else 1
        C = int(len(self.label_names))

        image_mm = np.memmap(self.output_dir / "image_u8.dat", mode="w+", dtype=np.uint8, shape=(N, Cimg, H, W))
        yend_mm = np.memmap(self.output_dir / "yend_u8.dat", mode="w+", dtype=np.uint8, shape=(N, C))
        snr_mm = np.memmap(self.output_dir / "snr_i16.dat", mode="w+", dtype=np.int16, shape=(N,))
        mod_mm = np.memmap(self.output_dir / "mod_i16.dat", mode="w+", dtype=np.int16, shape=(N,))
        stream_mm = np.memmap(self.output_dir / "stream_i32.dat", mode="w+", dtype=np.int32, shape=(N,))
        frame_idx_mm = np.memmap(self.output_dir / "frame_idx_i16.dat", mode="w+", dtype=np.int16, shape=(N,))
        frames_in_stream_mm = np.memmap(self.output_dir / "frames_in_stream_i16.dat", mode="w+", dtype=np.int16, shape=(N,))
        k_mm = np.memmap(self.output_dir / "k_u8.dat", mode="w+", dtype=np.uint8, shape=(N,))

        cursor = 0
        for frames in pulse_plans:
            for plan in frames:
                rendered = self.adapter.render_window(plan.iq_with_context, self.window_len)
                image_mm[cursor] = self._select_image(rendered)
                yend_mm[cursor] = 0
                yend_mm[cursor, plan.label_index] = 1
                snr_mm[cursor] = int(round(plan.snr_db))
                mod_mm[cursor] = int(plan.label_index)
                stream_mm[cursor] = int(plan.stream_id)
                frame_idx_mm[cursor] = int(plan.frame_idx)
                frames_in_stream_mm[cursor] = int(plan.frames_in_stream)
                k_mm[cursor] = 1
                cursor += 1

        for plan in noise_plans:
            rendered = self.adapter.render_window(plan.iq_with_context, self.window_len)
            image_mm[cursor] = self._select_image(rendered)
            yend_mm[cursor] = 0
            yend_mm[cursor, plan.label_index] = 1
            snr_mm[cursor] = int(round(plan.snr_db))
            mod_mm[cursor] = int(plan.label_index)
            stream_mm[cursor] = int(plan.stream_id)
            frame_idx_mm[cursor] = 0
            frames_in_stream_mm[cursor] = 1
            k_mm[cursor] = 1
            cursor += 1

        for arr in (image_mm, yend_mm, snr_mm, mod_mm, stream_mm, frame_idx_mm, frames_in_stream_mm, k_mm):
            arr.flush()

        requested_labels = canonicalise_label_subset(self.cfg.scan.modulation_list)
        meta = {
            "split_name": self.cfg.split_name,
            "dataset_split": self.cfg.split_name,
            "class_subset_tag": build_class_subset_tag(requested_labels),
            "requested_modulation_list": list(requested_labels),
            "modulation_list": list(self.label_names),
            "image_mode": self.image_mode,
            "input_root": str(self.input_root),
            "output_dir": str(self.output_dir),
            "sample_rate_hz": float(self.cfg.frame.sample_rate_hz),
            "window_len": int(self.window_len),
            "threshold_time_frac": float(self.cfg.frame.overlap_threshold_fraction),
            "spectrogram_context_samples_each_side": int(self.context),
            "N": int(N),
            "Cimg": int(Cimg),
            "H": int(H),
            "W": int(W),
            "C": int(C),
            "num_pulse_streams": int(len(pulse_plans)),
            "num_pulse_frames": int(pulse_frame_count),
            "num_noise_frames": int(len(noise_plans)),
            "files": {
                "image_u8": {"path": "image_u8.dat", "shape": [N, Cimg, H, W]},
                "yend_u8": {"path": "yend_u8.dat", "shape": [N, C]},
                "snr_i16": {"path": "snr_i16.dat", "shape": [N]},
                "mod_i16": {"path": "mod_i16.dat", "shape": [N]},
                "stream_i32": {"path": "stream_i32.dat", "shape": [N]},
                "frame_idx_i16": {"path": "frame_idx_i16.dat", "shape": [N]},
                "frames_in_stream_i16": {"path": "frames_in_stream_i16.dat", "shape": [N]},
                "k_u8": {"path": "k_u8.dat", "shape": [N]},
            },
        }
        with open(self.output_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[{self.cfg.split_name}] single-frame export done: N={N}, pulse_frames={pulse_frame_count}, noise_frames={len(noise_plans)}, mode={self.image_mode}")
        return self.output_dir


def export_singleframe_memmap(cfg: SingleFrameExportConfig) -> Path:
    return SingleFrameMemmapExporter(cfg).export()
