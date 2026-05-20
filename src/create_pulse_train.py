#!/usr/bin/env python3
"""
Standalone driver script for creating one MFD-Net style IQ pulse train.

This version is NOT a command-line script. Edit the DRIVER CONFIG block below,
then run the file directly from VS Code / PyCharm / terminal / whatever.

It loads one pulse from an H5 modulation library, repeats it into a pulse train,
adds optional AWGN/clipping, and saves:

    output/example.npz
    output/example_windows.csv
    output/example_pulses.csv
    output/example_meta.json
    output/example_plot.png    # only if PLOT = True
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import h5py  # type: ignore
except Exception as exc:
    raise SystemExit(
        "This script requires h5py. Install it with: pip install h5py"
    ) from exc

import os
from contextlib import contextmanager

# =============================================================================
# DRIVER CONFIG
# =============================================================================
# Edit this section only for normal use.

INPUT_ROOT = Path(r"C:\Apps\Code\MFD_Net_train")

# Final modulation label to use.
# Examples:
#   "unmodulated"
#   "lfm_up"
#   "lfm_down"
#   "bpsk"
#   "qpsk"
#   "barker_11"
#   "barker_13"
#   "polyphase"
MODULATION = "lfm_up"

# For normal classes, leave as None.
# For polyphase, you can force one raw source file:
#   RAW_MODULATION = "p1"
#   RAW_MODULATION = "p2"
#   RAW_MODULATION = "p3"
#   RAW_MODULATION = "p4"
RAW_MODULATION: str | None = None

# Choose which emitter_x record to load from the H5 file.
# RECORD_INDEX is used unless EMITTER_KEY is set.
RECORD_INDEX = 0
EMITTER_KEY: str | None = None
# Example:
# EMITTER_KEY = "emitter_42"

OUTPUT_PREFIX = Path(r"output\lfm_up_pulse_train")

SAMPLE_RATE_HZ = 50e6

# 1000 samples at 50 MHz = 20 us windows.
WINDOW_LEN = 1000
WINDOW_COUNT = 128

# Noise / clipping settings.
ADD_NOISE = True
SNR_DB = 0.0
NOISE_POWER_DB = 0.0

# Set True to mimic the real-data-style I/Q clipping behaviour.
APPLY_HARD_CLIP = True
CLIP_MIN = -1.0
CLIP_MAX = 1.0

# If True, uses the signal_noise attribute from the H5 record instead of SNR_DB.
USE_SIGNAL_NOISE_ATTR = False

# Pulse placement.
USE_RECORDED_PRI = True
USE_RECORDED_TOA = True

# Used only if the H5 record has no valid PRI.
FALLBACK_PRI_MIN_S = 40e-6
FALLBACK_PRI_MAX_S = 300e-6

# Keeps only fully contained pulses inside the stream.
REQUIRE_FULL_PULSE_INSIDE_STREAM = True

# Label windows as pulse if overlap is at least 2% of the window.
OVERLAP_THRESHOLD_FRACTION = 0.02

# Removes leading/trailing zero padding from loaded pulse before use.
TRIM_ZERO_PADDING = True
ZERO_TRIM_EPS = 1e-10

# For short pulses, avoid placing the first ToA right on the edge of the first window
# when recorded ToA is unavailable.
BIAS_SMALL_PW_INITIAL_TOA = True
SHORT_PULSE_MAX_SAMPLES = 128
SHORT_PULSE_SAFE_MARGIN_SAMPLES = 20

SEED = 12345

# Save a quick magnitude plot.
PLOT = True


# =============================================================================
# IMPLEMENTATION
# =============================================================================

WINDOW_STATE_NOISE = 0
WINDOW_STATE_FULL_PULSE = 1
WINDOW_STATE_PARTIAL_PULSE = 2

DEFAULT_MERGE_MAP: dict[str, tuple[str, ...]] = {
    "polyphase": ("p1", "p2", "p3", "p4"),
}


@dataclass(slots=True)
class PulseRecord:
    file_path: Path
    emitter_key: str
    label_name: str
    label_index: int
    pri: int
    toa_idx: int
    signal_noise: float
    pw: int | None = None
    rf: float | None = None
    raw_mod_name: str | None = None


@dataclass(slots=True)
class PulseTrainConfig:
    input_root: Path
    modulation: str
    raw_modulation: str | None
    record_index: int
    emitter_key: str | None
    output_prefix: Path
    sample_rate_hz: float
    window_len: int
    window_count: int
    trim_zero_padding: bool
    zero_trim_eps: float
    use_recorded_pri: bool
    use_recorded_toa: bool
    fallback_pri_min_s: float
    fallback_pri_max_s: float
    require_full_pulse_inside_stream: bool
    add_noise: bool
    use_signal_noise_attr: bool
    snr_db: float
    noise_power_db: float
    apply_hard_clip: bool
    clip_min: float
    clip_max: float
    overlap_threshold_fraction: float
    bias_small_pw_initial_toa: bool
    short_pulse_max_samples: int
    short_pulse_safe_margin_samples: int
    seed: int
    plot: bool

    @property
    def total_samples(self) -> int:
        return int(self.window_len * self.window_count)


@dataclass(slots=True)
class PulsePlacement:
    pulse_number: int
    start_sample: int
    end_sample: int
    length_samples: int
    start_us: float
    end_us: float


@dataclass(slots=True)
class WindowMetadata:
    window_index: int
    start_sample: int
    end_sample: int
    start_us: float
    end_us: float
    overlap_samples: int
    overlap_fraction_window: float
    overlap_fraction_pulse: float
    pulse_hits: int
    window_state: int
    window_state_name: str
    window_label: str
    keep_for_mfd: bool


def build_driver_config() -> PulseTrainConfig:
    return PulseTrainConfig(
        input_root=INPUT_ROOT,
        modulation=MODULATION,
        raw_modulation=RAW_MODULATION,
        record_index=RECORD_INDEX,
        emitter_key=EMITTER_KEY,
        output_prefix=OUTPUT_PREFIX,
        sample_rate_hz=SAMPLE_RATE_HZ,
        window_len=WINDOW_LEN,
        window_count=WINDOW_COUNT,
        trim_zero_padding=TRIM_ZERO_PADDING,
        zero_trim_eps=ZERO_TRIM_EPS,
        use_recorded_pri=USE_RECORDED_PRI,
        use_recorded_toa=USE_RECORDED_TOA,
        fallback_pri_min_s=FALLBACK_PRI_MIN_S,
        fallback_pri_max_s=FALLBACK_PRI_MAX_S,
        require_full_pulse_inside_stream=REQUIRE_FULL_PULSE_INSIDE_STREAM,
        add_noise=ADD_NOISE,
        use_signal_noise_attr=USE_SIGNAL_NOISE_ATTR,
        snr_db=SNR_DB,
        noise_power_db=NOISE_POWER_DB,
        apply_hard_clip=APPLY_HARD_CLIP,
        clip_min=CLIP_MIN,
        clip_max=CLIP_MAX,
        overlap_threshold_fraction=OVERLAP_THRESHOLD_FRACTION,
        bias_small_pw_initial_toa=BIAS_SMALL_PW_INITIAL_TOA,
        short_pulse_max_samples=SHORT_PULSE_MAX_SAMPLES,
        short_pulse_safe_margin_samples=SHORT_PULSE_SAFE_MARGIN_SAMPLES,
        seed=SEED,
        plot=PLOT,
    )


def _sorted_emitter_keys(h5_file: Any) -> list[str]:
    def key_fn(name: str) -> Any:
        text = str(name)
        if text.startswith("emitter_"):
            suffix = text.split("_")[-1]
            if suffix.isdigit():
                return int(suffix)
        return text

    return sorted(h5_file.keys(), key=key_fn)


def _read_optional_numeric_attr(group: Any, *names: str) -> float | int | None:
    for name in names:
        if name not in group.attrs:
            continue

        value = group.attrs[name]

        try:
            item = np.asarray(value).item() if np.asarray(value).shape == () else value

            if isinstance(item, (int, np.integer)):
                return int(item)

            if isinstance(item, (float, np.floating)):
                return float(item)

        except Exception:
            continue

    return None


def _candidate_raw_modulations(
    modulation: str,
    raw_modulation: str | None,
) -> tuple[str, ...]:
    if raw_modulation:
        return (raw_modulation,)

    return DEFAULT_MERGE_MAP.get(modulation, (modulation,))


def extract_records_from_h5(
    file_path: str | Path,
    *,
    raw_mod_name: str,
    label_name: str,
    label_index: int = 0,
) -> list[PulseRecord]:
    file_path = Path(file_path)
    records: list[PulseRecord] = []

    with h5py.File(file_path, "r") as f:
        for emitter_key in _sorted_emitter_keys(f):
            group = f[emitter_key]

            pri = _read_optional_numeric_attr(group, "pri_idx", "pri")
            toa_idx = _read_optional_numeric_attr(group, "toa_idx", "toa")
            signal_noise = _read_optional_numeric_attr(group, "signal_noise")
            pw = _read_optional_numeric_attr(group, "pw_idx", "pw")
            rf = _read_optional_numeric_attr(group, "rf_idx", "rf", "carrier_freq_hz")

            records.append(
                PulseRecord(
                    file_path=file_path,
                    emitter_key=str(emitter_key),
                    label_name=label_name,
                    label_index=label_index,
                    pri=int(pri) if pri is not None else 0,
                    toa_idx=int(toa_idx) if toa_idx is not None else 0,
                    signal_noise=float(signal_noise) if signal_noise is not None else float("nan"),
                    pw=int(pw) if pw is not None else None,
                    rf=float(rf) if rf is not None else None,
                    raw_mod_name=raw_mod_name,
                )
            )

    return records


def find_pulse_records(
    input_root: str | Path,
    modulation: str,
    raw_modulation: str | None,
) -> list[PulseRecord]:
    root = Path(input_root)

    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    records: list[PulseRecord] = []
    missing: list[Path] = []

    for raw_name in _candidate_raw_modulations(modulation, raw_modulation):
        path = root / f"{raw_name}.h5"

        if not path.is_file():
            missing.append(path)
            continue
        
        records.extend(
            extract_records_from_h5(
                path,
                raw_mod_name=raw_name,
                label_name=modulation,
                label_index=0,
            )
        )

    if not records:
        missing_str = "\n  ".join(str(p) for p in missing) if missing else f"{root}/{modulation}.h5"
        raise FileNotFoundError(
            f"No records found for modulation={modulation!r}. Tried:\n  {missing_str}"
        )

    return records


def choose_record(
    records: list[PulseRecord],
    record_index: int,
    emitter_key: str | None,
) -> tuple[int, PulseRecord]:
    if emitter_key:
        matches = [
            (idx, rec)
            for idx, rec in enumerate(records)
            if rec.emitter_key == emitter_key
        ]

        if not matches:
            available = ", ".join(rec.emitter_key for rec in records[:10])
            raise KeyError(
                f"No emitter_key={emitter_key!r}. First available keys: {available}"
            )

        return matches[0]

    if not 0 <= int(record_index) < len(records):
        raise IndexError(
            f"record_index={record_index} out of range for {len(records)} records"
        )

    return int(record_index), records[int(record_index)]


def trim_zero_padding(iq: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    magnitude = np.abs(iq)
    non_zero = np.flatnonzero(magnitude > eps)

    if len(non_zero) == 0:
        return iq[:1].copy()

    return iq[non_zero[0] : non_zero[-1] + 1].copy()


@contextmanager
def temporary_cwd(path: Path):
    """
    Temporarily change working directory.

    Some HDF5 files store datasets using external/raw backing files.
    In those cases, h5py/HDF5 may resolve the external file path relative to
    the current working directory rather than the .h5 file location.

    This makes dataset reads behave as if the script was run from the H5 folder.
    """
    old_cwd = Path.cwd()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)


def _read_h5_float_array(group: Any, dataset_name: str, h5_path: Path, emitter_key: str) -> np.ndarray:
    """
    Read one H5 dataset and provide a useful error if the H5 file uses external
    storage that cannot be found/opened.
    """
    if dataset_name not in group:
        available = list(group.keys())
        raise KeyError(
            f"Dataset {dataset_name!r} not found in {h5_path}:{emitter_key}. "
            f"Available keys: {available}"
        )

    dataset = group[dataset_name]

    try:
        return np.asarray(dataset[()], dtype=np.float32).reshape(-1)

    except OSError as exc:
        external_info = None

        try:
            external_info = dataset.external
        except Exception:
            external_info = "not available"

        raise OSError(
            "\nFailed to read an HDF5 dataset.\n"
            f"  File:        {h5_path}\n"
            f"  Emitter:     {emitter_key}\n"
            f"  Dataset:     {dataset_name}\n"
            f"  Current cwd: {Path.cwd()}\n"
            f"  H5 folder:   {h5_path.parent}\n"
            f"  Shape:       {getattr(dataset, 'shape', None)}\n"
            f"  Dtype:       {getattr(dataset, 'dtype', None)}\n"
            f"  External:    {external_info}\n\n"
            "This usually means the H5 dataset points to an external/raw backing "
            "file that HDF5 cannot find from the current working directory. "
            "Make sure the full extracted H5/data folder is present, not just the "
            ".h5 file by itself."
        ) from exc

def load_pulse_iq(
    record: PulseRecord,
    *,
    trim: bool,
    zero_trim_eps: float,
) -> np.ndarray:
    h5_path = Path(record.file_path).resolve()

    # Important for HDF5 files that use external/raw storage.
    # The dataset read happens while cwd is the H5 file's folder.
    with temporary_cwd(h5_path.parent):
        with h5py.File(h5_path, "r") as f:
            if record.emitter_key not in f:
                available = list(f.keys())[:20]
                raise KeyError(
                    f"Emitter key {record.emitter_key!r} not found in {h5_path}. "
                    f"First available keys: {available}"
                )

            group = f[record.emitter_key]

            # Normal expected layout: emitter_x/I and emitter_x/Q
            if "I" in group and "Q" in group:
                i = _read_h5_float_array(group, "I", h5_path, record.emitter_key)
                q = _read_h5_float_array(group, "Q", h5_path, record.emitter_key)

                if i.shape != q.shape:
                    raise ValueError(
                        f"I/Q length mismatch in {h5_path}:{record.emitter_key} "
                        f"-> I={i.shape}, Q={q.shape}"
                    )

                iq = (i + 1j * q).astype(np.complex64)

            # Fallback layout: complex IQ stored directly as one dataset.
            elif "IQ" in group:
                raw = np.asarray(group["IQ"][()]).reshape(-1)

                if np.iscomplexobj(raw):
                    iq = raw.astype(np.complex64)
                else:
                    raw = np.asarray(raw, dtype=np.float32)

                    if raw.ndim == 2 and raw.shape[-1] == 2:
                        iq = (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64)
                    else:
                        raise ValueError(
                            f"Dataset 'IQ' exists in {h5_path}:{record.emitter_key}, "
                            f"but it is not complex and does not have shape [N, 2]. "
                            f"Actual shape: {raw.shape}"
                        )

            else:
                available = list(group.keys())
                raise KeyError(
                    f"Could not find I/Q datasets in {h5_path}:{record.emitter_key}. "
                    f"Expected 'I' and 'Q', or fallback 'IQ'. "
                    f"Available keys: {available}"
                )

    if trim:
        iq = trim_zero_padding(iq, zero_trim_eps)

    return np.asarray(iq, dtype=np.complex64).copy()

def add_awgn_like_stream_builder(
    rng: np.random.Generator,
    signal: np.ndarray,
    *,
    snr_db: float,
    noise_power_db: float = 0.0,
    apply_hard_clip: bool = False,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex64)

    noise_linear = 10.0 ** (noise_power_db / 10.0)

    non_zero = (np.abs(signal.real) > 0) | (np.abs(signal.imag) > 0)

    if bool(non_zero.any()):
        signal_power = float(np.mean(np.abs(signal[non_zero]) ** 2))
        desired_signal_power = noise_linear * (10.0 ** (float(snr_db) / 10.0))
        scale = np.sqrt(max(desired_signal_power, eps) / max(signal_power, eps))
        scaled_signal = signal * scale
    else:
        scaled_signal = np.zeros_like(signal)

    noise = np.sqrt(noise_linear / 2.0) * (
        rng.normal(size=signal.size) + 1j * rng.normal(size=signal.size)
    )

    noisy = scaled_signal + noise.astype(np.complex64, copy=False)

    real_out = noisy.real.astype(np.float32, copy=False)
    imag_out = noisy.imag.astype(np.float32, copy=False)

    if apply_hard_clip:
        real_out = np.clip(real_out, clip_min, clip_max).astype(np.float32, copy=False)
        imag_out = np.clip(imag_out, clip_min, clip_max).astype(np.float32, copy=False)

    return (real_out + 1j * imag_out).astype(np.complex64, copy=False)


def resolve_pri_samples(
    cfg: PulseTrainConfig,
    rng: np.random.Generator,
    record: PulseRecord,
    pulse_len: int,
) -> int:
    if cfg.use_recorded_pri and int(record.pri) > 0:
        return max(int(record.pri), int(pulse_len))

    pri_s = float(rng.uniform(cfg.fallback_pri_min_s, cfg.fallback_pri_max_s))

    return max(
        1,
        int(round(pri_s * cfg.sample_rate_hz)),
        int(pulse_len),
    )


def resolve_initial_toa_samples(
    cfg: PulseTrainConfig,
    rng: np.random.Generator,
    record: PulseRecord,
    pulse_len: int,
) -> int:
    max_start = max(0, int(cfg.window_len) - 1)

    if cfg.use_recorded_toa and int(record.toa_idx) > 0:
        return int(record.toa_idx % cfg.window_len)

    if cfg.bias_small_pw_initial_toa and pulse_len <= int(cfg.short_pulse_max_samples):
        margin = int(max(1, cfg.short_pulse_safe_margin_samples))
        safe_low = margin
        safe_high = max(safe_low, int(cfg.window_len) - pulse_len - margin)

        if safe_high >= safe_low:
            return int(rng.integers(safe_low, safe_high + 1))

    return int(rng.integers(0, max_start + 1)) if max_start > 0 else 0


def sample_to_us(sample: int | float, sample_rate_hz: float) -> float:
    return float(sample) / float(sample_rate_hz) * 1e6


def pulse_positions(
    *,
    toa_samples: int,
    pri_samples: int,
    pulse_len: int,
    total_samples: int,
    require_full_pulse_inside_stream: bool,
    sample_rate_hz: float,
) -> list[PulsePlacement]:
    placements: list[PulsePlacement] = []
    pulse_number = 0

    while True:
        start = int(toa_samples + pulse_number * pri_samples)

        if start >= total_samples:
            break

        end = int(start + pulse_len)

        if require_full_pulse_inside_stream and end > total_samples:
            break

        clipped_end = int(min(end, total_samples))

        placements.append(
            PulsePlacement(
                pulse_number=pulse_number,
                start_sample=start,
                end_sample=clipped_end,
                length_samples=max(0, clipped_end - start),
                start_us=sample_to_us(start, sample_rate_hz),
                end_us=sample_to_us(clipped_end, sample_rate_hz),
            )
        )

        pulse_number += 1

    return placements


def build_clean_stream(
    pulse: np.ndarray,
    placements: Iterable[PulsePlacement],
    total_samples: int,
) -> np.ndarray:
    stream = np.zeros(int(total_samples), dtype=np.complex64)
    pulse_len = int(len(pulse))

    for placement in placements:
        start = int(placement.start_sample)
        usable_len = min(pulse_len, int(placement.end_sample) - start)

        if usable_len <= 0:
            continue

        stream[start : start + usable_len] += pulse[:usable_len]

    return stream


def compute_window_metadata(
    *,
    placements: list[PulsePlacement],
    pulse_len: int,
    window_len: int,
    window_count: int,
    sample_rate_hz: float,
    overlap_threshold_fraction: float,
) -> list[WindowMetadata]:
    overlap_samples = np.zeros((window_count,), dtype=np.int32)
    pulse_hits = np.zeros((window_count,), dtype=np.int16)

    for placement in placements:
        start = int(placement.start_sample)
        end = int(placement.end_sample)

        start_window = int(start // window_len)
        end_window = int((max(start + 1, end) - 1) // window_len)

        for window_index in range(start_window, min(window_count, end_window + 1)):
            win_start = int(window_index * window_len)
            win_end = int(win_start + window_len)
            overlap = max(0, min(end, win_end) - max(start, win_start))

            if overlap <= 0:
                continue

            overlap_samples[window_index] += int(overlap)
            pulse_hits[window_index] += 1

    rows: list[WindowMetadata] = []
    threshold = float(overlap_threshold_fraction)

    for window_index in range(window_count):
        win_start = int(window_index * window_len)
        win_end = int(win_start + window_len)

        overlap = int(overlap_samples[window_index])
        hits = int(pulse_hits[window_index])

        overlap_fraction_window = float(overlap / window_len)
        overlap_fraction_pulse = float(overlap / max(1, pulse_len))

        full_pulse = bool(hits == 1 and overlap == pulse_len)
        keep = bool(overlap_fraction_window >= threshold)

        if overlap <= 0:
            state = WINDOW_STATE_NOISE
            state_name = "noise"
            label = "N"
        elif full_pulse:
            state = WINDOW_STATE_FULL_PULSE
            state_name = "full_pulse"
            label = "P"
        else:
            state = WINDOW_STATE_PARTIAL_PULSE
            state_name = "partial_pulse"
            label = "P" if keep else "N"

        rows.append(
            WindowMetadata(
                window_index=window_index,
                start_sample=win_start,
                end_sample=win_end,
                start_us=sample_to_us(win_start, sample_rate_hz),
                end_us=sample_to_us(win_end, sample_rate_hz),
                overlap_samples=overlap,
                overlap_fraction_window=overlap_fraction_window,
                overlap_fraction_pulse=overlap_fraction_pulse,
                pulse_hits=hits,
                window_state=state,
                window_state_name=state_name,
                window_label=label,
                keep_for_mfd=keep,
            )
        )

    return rows


def write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(asdict(rows[0]).keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(asdict(row))


def build_pulse_train(cfg: PulseTrainConfig) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed))

    records = find_pulse_records(
        cfg.input_root,
        cfg.modulation,
        cfg.raw_modulation,
    )

    source_record_index, record = choose_record(
        records,
        cfg.record_index,
        cfg.emitter_key,
    )

    pulse = load_pulse_iq(
        record,
        trim=cfg.trim_zero_padding,
        zero_trim_eps=cfg.zero_trim_eps,
    )

    if record.pw is not None and int(record.pw) > 0:
        pulse_len = int(record.pw)
    else:
        pulse_len = int(len(pulse))

    pulse_len = min(pulse_len, int(len(pulse)))

    if pulse_len <= 0:
        raise ValueError(
            f"Pulse length resolved to zero for {record.file_path}:{record.emitter_key}"
        )

    pulse = pulse[:pulse_len].astype(np.complex64, copy=False)

    pri_samples = resolve_pri_samples(cfg, rng, record, pulse_len)
    toa_samples = resolve_initial_toa_samples(cfg, rng, record, pulse_len)

    placements = pulse_positions(
        toa_samples=toa_samples,
        pri_samples=pri_samples,
        pulse_len=pulse_len,
        total_samples=cfg.total_samples,
        require_full_pulse_inside_stream=cfg.require_full_pulse_inside_stream,
        sample_rate_hz=cfg.sample_rate_hz,
    )

    clean_stream = build_clean_stream(
        pulse,
        placements,
        cfg.total_samples,
    )

    resolved_snr_db = float(cfg.snr_db)

    if cfg.use_signal_noise_attr and np.isfinite(record.signal_noise):
        resolved_snr_db = float(record.signal_noise)

    if cfg.add_noise:
        noisy_stream = add_awgn_like_stream_builder(
            rng,
            clean_stream,
            snr_db=resolved_snr_db,
            noise_power_db=cfg.noise_power_db,
            apply_hard_clip=cfg.apply_hard_clip,
            clip_min=cfg.clip_min,
            clip_max=cfg.clip_max,
        )
    else:
        noisy_stream = clean_stream.astype(np.complex64, copy=True)

    windows = compute_window_metadata(
        placements=placements,
        pulse_len=pulse_len,
        window_len=cfg.window_len,
        window_count=cfg.window_count,
        sample_rate_hz=cfg.sample_rate_hz,
        overlap_threshold_fraction=cfg.overlap_threshold_fraction,
    )

    out_prefix = Path(cfg.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = out_prefix.with_suffix(".npz")
    pulses_csv = out_prefix.with_name(out_prefix.name + "_pulses.csv")
    windows_csv = out_prefix.with_name(out_prefix.name + "_windows.csv")
    meta_json = out_prefix.with_name(out_prefix.name + "_meta.json")

    np.savez_compressed(
        npz_path,
        iq=noisy_stream.astype(np.complex64),
        clean_iq=clean_stream.astype(np.complex64),
        pulse_iq=pulse.astype(np.complex64),
        sample_rate_hz=np.float64(cfg.sample_rate_hz),
        window_len=np.int32(cfg.window_len),
        window_count=np.int32(cfg.window_count),
        pulse_len_samples=np.int32(pulse_len),
        pri_samples=np.int32(pri_samples),
        toa_samples=np.int32(toa_samples),
        snr_db=np.float32(resolved_snr_db if cfg.add_noise else np.nan),
    )

    write_csv(pulses_csv, placements)
    write_csv(windows_csv, windows)

    kept_windows = [row.window_index for row in windows if row.keep_for_mfd]
    pulse_windows = [row.window_index for row in windows if row.window_label == "P"]
    noise_windows = [row.window_index for row in windows if row.window_label == "N"]

    metadata = {
        "config": {
            **asdict(cfg),
            "input_root": str(cfg.input_root),
            "output_prefix": str(cfg.output_prefix),
        },
        "source_record_index": int(source_record_index),
        "source_record": {
            "file_path": str(record.file_path),
            "emitter_key": record.emitter_key,
            "label_name": record.label_name,
            "raw_mod_name": record.raw_mod_name,
            "pri_attr": int(record.pri),
            "toa_idx_attr": int(record.toa_idx),
            "pw_attr": None if record.pw is None else int(record.pw),
            "rf_attr": record.rf,
            "signal_noise_attr": record.signal_noise,
        },
        "resolved": {
            "sample_rate_hz": float(cfg.sample_rate_hz),
            "total_samples": int(cfg.total_samples),
            "duration_us": sample_to_us(cfg.total_samples, cfg.sample_rate_hz),
            "pulse_len_samples": int(pulse_len),
            "pulse_width_us": sample_to_us(pulse_len, cfg.sample_rate_hz),
            "pri_samples": int(pri_samples),
            "pri_us": sample_to_us(pri_samples, cfg.sample_rate_hz),
            "toa_samples": int(toa_samples),
            "toa_us": sample_to_us(toa_samples, cfg.sample_rate_hz),
            "snr_db": resolved_snr_db if cfg.add_noise else None,
            "noise_added": bool(cfg.add_noise),
            "hard_clipped": bool(cfg.apply_hard_clip and cfg.add_noise),
            "pulse_count": len(placements),
            "kept_window_count": len(kept_windows),
            "pulse_window_count": len(pulse_windows),
            "noise_window_count": len(noise_windows),
        },
        "outputs": {
            "npz": str(npz_path),
            "pulses_csv": str(pulses_csv),
            "windows_csv": str(windows_csv),
            "meta_json": str(meta_json),
        },
        "kept_window_indices": kept_windows,
        "pulse_window_indices": pulse_windows,
        "noise_window_indices": noise_windows,
    }

    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if cfg.plot:
        plot_path = out_prefix.with_name(out_prefix.name + "_plot.png")
        plot_pulse_train(noisy_stream, clean_stream, placements, cfg, plot_path)

        metadata["outputs"]["plot_png"] = str(plot_path)

        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return metadata


def plot_pulse_train(
    iq: np.ndarray,
    clean_iq: np.ndarray,
    placements: list[PulsePlacement],
    cfg: PulseTrainConfig,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    t_us = np.arange(iq.size, dtype=np.float64) / float(cfg.sample_rate_hz) * 1e6

    magnitude = np.abs(iq)
    clean_magnitude = np.abs(clean_iq)

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(t_us, magnitude, linewidth=0.7, label="noisy |IQ|")

    if cfg.add_noise:
        ax.plot(t_us, clean_magnitude, linewidth=0.7, label="clean |IQ|", alpha=0.75)

    for placement in placements[:200]:
        ax.axvspan(placement.start_us, placement.end_us, alpha=0.10)

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Pulse train: {cfg.modulation}")
    ax.legend(loc="upper right")

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    cfg = build_driver_config()
    metadata = build_pulse_train(cfg)

    resolved = metadata["resolved"]
    outputs = metadata["outputs"]

    print("\nPulse train created")
    print("=" * 80)
    print(f"Source file:        {metadata['source_record']['file_path']}")
    print(f"Emitter key:        {metadata['source_record']['emitter_key']}")
    print(f"Modulation:         {cfg.modulation}")
    print(f"Raw modulation:     {metadata['source_record']['raw_mod_name']}")
    print(f"Pulse count:        {resolved['pulse_count']}")
    print(f"Pulse windows:      {resolved['pulse_window_count']}")
    print(f"Noise windows:      {resolved['noise_window_count']}")
    print(f"Kept MFD windows:   {resolved['kept_window_count']}")
    print(f"PW:                 {resolved['pulse_len_samples']} samples / {resolved['pulse_width_us']:.3f} us")
    print(f"PRI:                {resolved['pri_samples']} samples / {resolved['pri_us']:.3f} us")
    print(f"ToA:                {resolved['toa_samples']} samples / {resolved['toa_us']:.3f} us")
    print(f"SNR:                {resolved['snr_db']}")
    print(f"Hard clipped:       {resolved['hard_clipped']}")
    print("-" * 80)
    print(f"Wrote NPZ:          {outputs['npz']}")
    print(f"Wrote windows CSV:  {outputs['windows_csv']}")
    print(f"Wrote pulses CSV:   {outputs['pulses_csv']}")
    print(f"Wrote metadata:     {outputs['meta_json']}")

    if "plot_png" in outputs:
        print(f"Wrote plot:         {outputs['plot_png']}")

    print("=" * 80)


if __name__ == "__main__":
    main()