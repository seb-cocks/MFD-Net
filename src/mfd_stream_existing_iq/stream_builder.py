from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    IQScanConfig,
    StreamBuildConfig,
    WINDOW_STATE_FULL_PULSE,
    WINDOW_STATE_PARTIAL_PULSE,
)
from .iq_dataset import PulseRecord, load_pulse_iq
from .noise_utils import add_noise


@dataclass(slots=True)
class WindowPlan:
    stream_index: int
    dataset_split: str
    label_name: str
    label_index: int
    source_record_index: int
    emitter_id: str
    window_index: int
    overlap_samples: int
    overlap_fraction_window: float
    overlap_fraction_pulse: float
    full_pulse: bool
    keep_window: bool
    pulse_iq_with_context: np.ndarray
    pw_samples: int
    pri_samples: int
    toa_samples: int
    rf_hz: float | None
    raw_mod_name: str | None
    snr_db: float | None
    window_state: int


@dataclass(slots=True)
class BuiltStream:
    label_name: str
    label_index: int
    stream_index: int
    source_record_index: int
    emitter_id: str
    snr_db: float | None
    total_windows: int
    kept_windows: list[WindowPlan]
    rejected_windows: list[WindowPlan]


class StreamBuilder:
    def __init__(
        self,
        records_by_label: dict[int, list[PulseRecord]],
        stream_cfg: StreamBuildConfig,
        scan_cfg: IQScanConfig,
        *,
        split_name: str,
    ):
        self.records_by_label = records_by_label
        self.stream_cfg = stream_cfg
        self.scan_cfg = scan_cfg
        self.split_name = str(split_name)
        self.total_samples = stream_cfg.total_samples()
        self.window_count = stream_cfg.resolved_window_count()
        self.window_len = stream_cfg.window_len

    def _stream_rng(self, label_index: int, stream_index: int) -> np.random.Generator:
        return np.random.default_rng(self.stream_cfg.base_seed + (label_index * 1_000_003) + stream_index)

    def _choose_record(self, label_index: int, stream_index: int, rng: np.random.Generator) -> tuple[int, PulseRecord]:
        records = self.records_by_label[label_index]
        if not records:
            raise ValueError(f"No pulse records for label index {label_index}")
        requested = self.stream_cfg.streams_per_class
        if requested is None or int(requested) == -1:
            record_index = int(stream_index % len(records))
        else:
            record_index = int(rng.integers(0, len(records)))
        return record_index, records[record_index]

    def _choose_stream_snr(self, rng: np.random.Generator, record: PulseRecord) -> float | None:
        if not self.stream_cfg.add_noise_to_stream:
            return None
        if self.stream_cfg.use_signal_noise_attr and np.isfinite(record.signal_noise):
            return float(record.signal_noise)
        return float(rng.choice(np.asarray(self.stream_cfg.snr_db_choices, dtype=np.float32)))

    def _resolve_pri_samples(self, rng: np.random.Generator, rec: PulseRecord, pulse_len: int) -> int:
        if self.stream_cfg.use_recorded_pri and int(rec.pri) > 0:
            return max(int(rec.pri), pulse_len)
        pri_s = float(rng.uniform(self.stream_cfg.fallback_pri_min_s, self.stream_cfg.fallback_pri_max_s))
        pri_samples = int(round(pri_s * self.stream_cfg.sample_rate_hz))
        return max(1, pri_samples, pulse_len)

    def _resolve_initial_toa_samples(self, rng: np.random.Generator, rec: PulseRecord, pulse_len: int) -> int:
        max_start = max(0, self.window_len - 1)
        if self.stream_cfg.use_recorded_toa and int(rec.toa_idx) > 0:
            return int(rec.toa_idx % self.window_len)
        if (
            self.stream_cfg.bias_small_pw_initial_toa
            and pulse_len <= int(self.stream_cfg.short_pulse_max_samples)
        ):
            margin = int(max(1, self.stream_cfg.short_pulse_safe_margin_samples))
            safe_low = margin
            safe_high = max(safe_low, self.window_len - pulse_len - margin)
            if safe_high >= safe_low:
                return int(rng.integers(safe_low, safe_high + 1))
        return int(rng.integers(0, max_start + 1)) if max_start > 0 else 0

    def _pulse_positions(self, toa_samples: int, pri_samples: int, pulse_len: int) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        pulse_idx = 0
        while True:
            start = int(toa_samples + pulse_idx * pri_samples)
            if start >= self.total_samples:
                break
            end = int(start + pulse_len)
            if self.stream_cfg.require_full_pulse_inside_stream and end > self.total_samples:
                break
            positions.append((start, min(end, self.total_samples)))
            pulse_idx += 1
        return positions

    def _build_clean_stream(self, pulse: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
        stream = np.zeros(self.total_samples, dtype=np.complex64)
        pulse_len = int(len(pulse))
        for start, end in positions:
            usable_len = min(pulse_len, end - start)
            if usable_len <= 0:
                continue
            stream[start : start + usable_len] += pulse[:usable_len]
        return stream

    def build_stream(self, label_index: int, stream_index: int, *, context_samples_each_side: int) -> BuiltStream:
        rng = self._stream_rng(label_index, stream_index)
        source_record_index, record = self._choose_record(label_index, stream_index, rng)
        pulse = load_pulse_iq(record, self.scan_cfg)
        pulse_len = int(record.pw) if record.pw is not None and int(record.pw) > 0 else int(len(pulse))
        pulse_len = min(pulse_len, int(len(pulse)))
        if pulse_len <= 0:
            raise ValueError(f"Pulse length resolved to zero for {record.file_path}:{record.emitter_key}")
        pulse = pulse[:pulse_len].astype(np.complex64, copy=False)

        pri_samples = self._resolve_pri_samples(rng, record, pulse_len)
        toa_samples = self._resolve_initial_toa_samples(rng, record, pulse_len)
        positions = self._pulse_positions(toa_samples, pri_samples, pulse_len)
        clean_stream = self._build_clean_stream(pulse, positions)
        snr_db = self._choose_stream_snr(rng, record)

        noisy_stream = clean_stream
        if snr_db is not None:
            i_noisy, q_noisy = add_noise(
                rng,
                clean_stream.real.astype(np.float32, copy=False),
                clean_stream.imag.astype(np.float32, copy=False),
                snr_db=snr_db,
                noise_power_db=self.stream_cfg.noise_power_db,
                apply_hard_clip=self.stream_cfg.apply_hard_clip,
                clip_min=self.stream_cfg.clip_min,
                clip_max=self.stream_cfg.clip_max,
            )
            noisy_stream = (i_noisy + 1j * q_noisy).astype(np.complex64, copy=False)

        overlap_samples = np.zeros((self.window_count,), dtype=np.int32)
        pulse_hits = np.zeros((self.window_count,), dtype=np.int16)
        for start, end in positions:
            start_window = int(start // self.window_len)
            end_window = int((max(start + 1, end) - 1) // self.window_len)
            for window_index in range(start_window, min(self.window_count, end_window + 1)):
                win_start = int(window_index * self.window_len)
                win_end = int(win_start + self.window_len)
                overlap = max(0, min(end, win_end) - max(start, win_start))
                if overlap <= 0:
                    continue
                overlap_samples[window_index] += int(overlap)
                pulse_hits[window_index] += 1

        kept_windows: list[WindowPlan] = []
        rejected_windows: list[WindowPlan] = []
        threshold = float(self.stream_cfg.overlap_threshold_fraction)
        for window_index in range(self.window_count):
            overlap = int(overlap_samples[window_index])
            if overlap <= 0:
                continue
            overlap_fraction_window = float(overlap / self.window_len)
            overlap_fraction_pulse = float(overlap / max(1, pulse_len))
            full_pulse = bool(pulse_hits[window_index] == 1 and overlap == pulse_len)
            keep_window = overlap_fraction_window >= threshold
            state = WINDOW_STATE_FULL_PULSE if full_pulse else WINDOW_STATE_PARTIAL_PULSE

            window_start = int(window_index * self.window_len)
            window_end = int(window_start + self.window_len)
            context_left = int(max(0, min(context_samples_each_side, window_start)))
            context_right = int(max(0, min(context_samples_each_side, self.total_samples - window_end)))
            iq_with_context = noisy_stream[window_start - context_left : window_end + context_right].astype(np.complex64, copy=True)

            plan = WindowPlan(
                stream_index=stream_index,
                dataset_split=self.split_name,
                label_name=record.label_name,
                label_index=record.label_index,
                source_record_index=source_record_index,
                emitter_id=record.emitter_key,
                window_index=window_index,
                overlap_samples=overlap,
                overlap_fraction_window=overlap_fraction_window,
                overlap_fraction_pulse=overlap_fraction_pulse,
                full_pulse=full_pulse,
                keep_window=keep_window,
                pulse_iq_with_context=iq_with_context,
                pw_samples=pulse_len,
                pri_samples=pri_samples,
                toa_samples=toa_samples,
                rf_hz=record.rf,
                raw_mod_name=record.raw_mod_name,
                snr_db=snr_db,
                window_state=state,
            )
            if keep_window:
                kept_windows.append(plan)
            else:
                rejected_windows.append(plan)

        return BuiltStream(
            label_name=record.label_name,
            label_index=record.label_index,
            stream_index=stream_index,
            source_record_index=source_record_index,
            emitter_id=record.emitter_key,
            snr_db=snr_db,
            total_windows=self.window_count,
            kept_windows=kept_windows,
            rejected_windows=rejected_windows,
        )
