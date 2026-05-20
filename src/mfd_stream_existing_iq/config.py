from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Per-window auxiliary supervision states used by the exporter/loader.
#
# Meaning:
# - 0: gap / noise-only window in the reconstructed dense timeline
# - 1: a window that contains an entire pulse
# - 2: a window that contains only part of a pulse
# - -100: padded tail / ignore index during training
WINDOW_STATE_NOISE = 0
WINDOW_STATE_FULL_PULSE = 1
WINDOW_STATE_PARTIAL_PULSE = 2
WINDOW_STATE_SPLIT_PULSE = WINDOW_STATE_PARTIAL_PULSE  # backwards-compatible alias
WINDOW_STATE_IGNORE = -100
WINDOW_CLASS_IGNORE = -100


@dataclass(slots=True)
class SpectrogramAdapterConfig:
    module_name: str = "src.common.spectrogram"
    power_method_name: str = "iq_to_power_spectrogram"
    phase_method_name: str = "iq_to_phase_spectrogram"
    resize_method_name: str | None = "resize_spec01_for_save"
    # Use a shorter STFT so the 20 us / 1000-sample window keeps a denser
    # time axis before resizing. The earlier 256/256/128 settings produced only
    # a handful of time bins after centre-cropping, which then stretched badly
    # when resized to 224x224.
    power_n_fft: int = 64
    power_win_len: int = 32
    power_hop_len: int = 8
    # Keep the phase STFT aligned with the power STFT so both channels
    # describe the same time/frequency support. The actual phase image is now
    # delegated to the selected helper in src.common.spectrogram; by default
    # that should be a soft-gated temporal phase-difference image, not a custom
    # wrapped-phase frequency-gradient image.
    phase_n_fft: int = 64
    phase_win_len: int = 32
    phase_hop_len: int = 8
    phase_power_gate_db: float | None = None
    phase_soft_gate_width_db: float = 6.0
    out_height: int = 224
    out_width: int = 224
    # Context samples appended to each side of the labelled 20 us / 1000-sample
    # window before the STFT. The spectrogram is then centre-cropped back to the
    # original window support to reduce edge artefacts.
    context_samples_each_side: int = 128


@dataclass(slots=True)
class IQScanConfig:
    modulation_list: tuple[str, ...] | None = None
    file_suffix: str = ".h5"
    recursive: bool = False
    trim_zero_padding: bool = True
    zero_trim_eps: float = 1e-10
    strict_modulation_list: bool = True
    merge_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
    balance_merged_groups: bool = True


@dataclass(slots=True)
class StreamBuildConfig:
    sample_rate_hz: float
    window_len: int
    window_count: int | None = 64
    sim_time_s: float | None = None
    streams_per_class: int | None = 100
    require_full_pulse_inside_stream: bool = True
    min_valid_windows_per_stream: int = 1
    use_recorded_pri: bool = True
    use_recorded_toa: bool = True
    fallback_pri_min_s: float = 40e-6
    fallback_pri_max_s: float = 300e-6
    add_noise_to_stream: bool = True
    use_signal_noise_attr: bool = False
    snr_db_choices: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    noise_power_db: float = 0.0
    apply_hard_clip: bool = False
    clip_min: float = -1.0
    clip_max: float = 1.0
    base_seed: int = 12345
    # Keep a pulse window only if its pulse overlap occupies at least this
    # fraction of the 20 us window.
    overlap_threshold_fraction: float = 0.02
    # For very small pulse widths, bias the initial ToA away from window
    # boundaries so the shortest pulses are not disproportionately rejected by
    # the overlap threshold.
    bias_small_pw_initial_toa: bool = True
    short_pulse_max_samples: int = 128
    short_pulse_safe_margin_samples: int = 20

    def total_samples(self) -> int:
        if self.window_count is not None and self.sim_time_s is not None:
            raise ValueError("Specify only one of window_count or sim_time_s.")
        if self.window_count is None and self.sim_time_s is None:
            raise ValueError("Specify one of window_count or sim_time_s.")
        if self.window_count is not None:
            return int(self.window_count * self.window_len)
        return int(round(self.sim_time_s * self.sample_rate_hz))

    def resolved_window_count(self) -> int:
        total = self.total_samples()
        return total // self.window_len


@dataclass(slots=True)
class ExportConfig:
    input_root: str | Path
    output_dir: str | Path
    split_name: str
    stream: StreamBuildConfig
    scan: IQScanConfig = field(default_factory=IQScanConfig)
    spectrogram: SpectrogramAdapterConfig = field(default_factory=SpectrogramAdapterConfig)
    flush_every_streams: int = 50
    overwrite: bool = True
    export_images: bool = True
    export_embeddings: bool = True
    # Frozen dual ResNet18 checkpoint used to cache backbone embeddings.
    resnet_checkpoint_path: str | Path | None = None
    checkpoint_strict: bool = True
    device: str = "auto"
    encoder_batch_size: int = 128
    # QA export knobs for retained pulse windows and rejected pulse fragments.
    qa_keep_probability: float = 0.05
    qa_rejected_probability: float = 0.10


@dataclass(slots=True)
class NoiseBankConfig:
    output_dir: str | Path
    split_name: str
    sample_rate_hz: float
    window_len: int
    samples_per_snr: int = 500
    snr_values_db: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    noise_power_db: float = 0.0
    apply_hard_clip: bool = False
    clip_min: float = -1.0
    clip_max: float = 1.0
    base_seed: int = 54321
    spectrogram: SpectrogramAdapterConfig = field(default_factory=SpectrogramAdapterConfig)
    overwrite: bool = True
    export_images: bool = True
    export_embeddings: bool = False
    resnet_checkpoint_path: str | Path | None = None
    checkpoint_strict: bool = True
    device: str = "auto"
    encoder_batch_size: int = 128


@dataclass(slots=True)
class LoaderConfig:
    memmap_dir: str | Path
    chunk_len: int
    chunk_stride: int = 1
    pad_short_streams: bool = False
    return_window_indices: bool = True
    return_stream_id: bool = False
    return_signal_mask: bool = False
    return_window_state_targets: bool = False
    return_window_class_targets: bool = False
    window_class_ignore_index: int = WINDOW_CLASS_IGNORE
    window_class_noise_index: int | None = None
    normalise_to_float: bool = True
    noise_handling_mode: str = "pulse_tail_pad"
    noise_bank_dir: str | Path | None = None
    deterministic_noise: bool = False
    noise_fill_mode: str = "per_position"  # per_position | per_chunk
    base_seed: int = 12345
    embedding_dim: int = 512
    # Controls what representation the loader returns.
    # - images: [T,2,H,W]
    # - embeddings: [T,D]
    # - hybrid: dict with image frames plus cached embeddings and an embedding mask
    input_representation: str = "hybrid"


@dataclass(slots=True)
class DriverConfig:
    split_exports: dict[str, ExportConfig]
    split_noise_banks: dict[str, NoiseBankConfig | None] = field(default_factory=dict)


def to_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def ensure_empty_or_create(path: str | Path, overwrite: bool) -> Path:
    path = to_path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return path
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
    return path


def label_names_to_map(labels: Iterable[str]) -> dict[str, int]:
    unique = list(dict.fromkeys(labels))
    return {label: idx for idx, label in enumerate(unique)}



def canonicalise_label_subset(labels: Iterable[str] | None) -> tuple[str, ...]:
    if labels is None:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        label = str(raw).strip()
        if not label:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return tuple(out)


def build_class_subset_tag(labels: Iterable[str] | None) -> str:
    """Return a deterministic directory-safe tag for a class subset."""
    canon = canonicalise_label_subset(labels)
    if not canon:
        return "all_classes"

    def _safe_token(name: str) -> str:
        cleaned = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in name.strip())
        while '__' in cleaned:
            cleaned = cleaned.replace('__', '_')
        return cleaned.strip('_') or 'unnamed'

    tokens = sorted((_safe_token(name) for name in canon), key=lambda x: x.casefold())
    return '__'.join(tokens)


def build_class_subset_root(parent_dir: str | Path, labels: Iterable[str] | None) -> Path:
    return to_path(parent_dir) / build_class_subset_tag(labels)


def path_contains_class_subset_tag(path: str | Path, labels: Iterable[str] | None) -> bool:
    expected = build_class_subset_tag(labels)
    return expected in {part for part in to_path(path).parts}


@dataclass(slots=True)
class SingleFrameBuildConfig:
    sample_rate_hz: float
    window_len: int
    use_recorded_toa: bool = True
    add_noise_to_stream: bool = True
    use_signal_noise_attr: bool = False
    snr_db_choices: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    noise_power_db: float = 0.0
    apply_hard_clip: bool = False
    clip_min: float = -1.0
    clip_max: float = 1.0
    base_seed: int = 12345
    overlap_threshold_fraction: float = 0.02
    bias_small_pw_initial_toa: bool = True
    short_pulse_max_samples: int = 128
    short_pulse_safe_margin_samples: int = 20


@dataclass(slots=True)
class SingleFrameExportConfig:
    input_root: str | Path
    output_dir: str | Path
    split_name: str
    frame: SingleFrameBuildConfig
    scan: IQScanConfig = field(default_factory=IQScanConfig)
    spectrogram: SpectrogramAdapterConfig = field(default_factory=SpectrogramAdapterConfig)
    overwrite: bool = True
    image_mode: str = "dual"   # power | phase | dual
    include_noise_class: bool = True
    noise_label: str = "noise"
    noise_samples_per_snr: int = 500
    qa_keep_probability: float = 0.0
