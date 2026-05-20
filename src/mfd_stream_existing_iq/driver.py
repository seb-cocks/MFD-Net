from __future__ import annotations

from .config import (
    DriverConfig,
    ExportConfig,
    IQScanConfig,
    NoiseBankConfig,
    SingleFrameBuildConfig,
    SingleFrameExportConfig,
    SpectrogramAdapterConfig,
    StreamBuildConfig,
    build_class_subset_root,
    build_class_subset_tag,
)
from .memmap_export import export_stream_memmap
from .noise_bank import build_noise_bank
from .resnet_memmap_export import export_singleframe_memmap

DEFAULT_MODULATION_LIST = (
    "unmodulated",
    "lfm_up",
    "lfm_down",
    "dlfm_up_down",
    "NLFM_up",
    "NLFM_down",
    "mlfm",
    "dlfm_down_up",
    "bpsk",
    "qpsk",
    "barker_11",
    "barker_13",
    "polyphase",
)
MERGE_MAP = {"polyphase": ("p1", "p2", "p3", "p4")}
SELECTED_MODULATION_LIST = DEFAULT_MODULATION_LIST

VERSION = ""  # _real | *blank*
DATA_SPLITS: tuple[str, ...] = ("train", "val", "test")


def _common_spectrogram() -> SpectrogramAdapterConfig:
    return SpectrogramAdapterConfig(
        module_name="src.common.spectrogram",
        power_method_name="iq_to_power_spectrogram",
        phase_method_name="iq_to_phase_spectrogram",  # "iq_to_phase_edge_spectrogram" or "iq_to_phase_spectrogram"
        resize_method_name="resize_spec01_for_save",
        # power_n_fft=64,
        # power_win_len=32,
        # power_hop_len=8,
        # phase_n_fft=64,
        # phase_win_len=32,
        # phase_hop_len=8,
        #
        power_n_fft=256,
        power_win_len=64,
        power_hop_len=8,
        #
        phase_n_fft=256,
        phase_win_len=64,
        phase_hop_len=8,
        phase_power_gate_db=-12.0,
        phase_soft_gate_width_db=6.0,
        # phase_power_gate_db=None,
        #
        out_height=224,
        out_width=224,
        context_samples_each_side=128,
    )


def _common_scan() -> IQScanConfig:
    return IQScanConfig(
        modulation_list=SELECTED_MODULATION_LIST,
        file_suffix=".h5",
        recursive=False,
        trim_zero_padding=True,
        zero_trim_eps=1e-10,
        strict_modulation_list=False,
        merge_map=MERGE_MAP,
        balance_merged_groups=True,
    )


def _common_stream() -> StreamBuildConfig:
    return StreamBuildConfig(
        sample_rate_hz=50e6,
        window_len=1000,
        # window_count=32,
        window_count=128,
        streams_per_class=-1,
        # streams_per_class=100,
        require_full_pulse_inside_stream=True,
        min_valid_windows_per_stream=1,
        use_recorded_pri=True,
        use_recorded_toa=True,
        fallback_pri_min_s=40e-6,
        fallback_pri_max_s=300e-6,
        add_noise_to_stream=True,
        use_signal_noise_attr=False,
        snr_db_choices=(-6.0, -3.0, 0.0, 3.0, 6.0),
        noise_power_db=0.0,
        apply_hard_clip=True,
        clip_min=-1.0,
        clip_max=1.0,
        base_seed=12345,
        overlap_threshold_fraction=0.02,
        bias_small_pw_initial_toa=True,
        short_pulse_max_samples=128,
        short_pulse_safe_margin_samples=20,
    )


def _common_singleframe() -> SingleFrameBuildConfig:
    return SingleFrameBuildConfig(
        sample_rate_hz=50e6,
        window_len=1000,
        use_recorded_toa=True,
        add_noise_to_stream=True,
        use_signal_noise_attr=False,
        snr_db_choices=(-6.0, -3.0, 0.0, 3.0, 6.0),
        noise_power_db=0.0,
        apply_hard_clip=True,
        clip_min=-1.0,
        clip_max=1.0,
        base_seed=12345,
        overlap_threshold_fraction=0.02,
        bias_small_pw_initial_toa=True,
        short_pulse_max_samples=128,
        short_pulse_safe_margin_samples=20,
    )


def build_default_driver_config() -> DriverConfig:
    dual_resnet_ckpt = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\out\resnet_dual\20260403_165139_None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6\checkpoints\best__None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6.pt"
    common_spectrogram = _common_spectrogram()
    common_scan = _common_scan()
    common_stream = _common_stream()

    # output_dir_parent = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\data"
    output_dir_parent = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\data" + VERSION

    input_dir = r"C:\Apps\Code\MFD_Net" + VERSION + "_"
    # input_dir = r"C:\Apps\Code\MFD_Net_"

    subset_root = build_class_subset_root(output_dir_parent, common_scan.modulation_list)
    subset_tag = build_class_subset_tag(common_scan.modulation_list)
    print(f"[subset] class_subset_tag={subset_tag}")
    print(f"[subset] output_root={subset_root}")

    split_exports: dict[str, ExportConfig] = {}
    split_noise_banks: dict[str, NoiseBankConfig] = {}

    for split_name in DATA_SPLITS:
        split_exports[split_name] = ExportConfig(
            input_root=input_dir + split_name,
            output_dir=str(subset_root / f"MFD_Net_{split_name}_stream_memmap"),
            split_name=split_name,
            stream=common_stream,
            scan=common_scan,
            spectrogram=common_spectrogram,
            overwrite=True,
            export_images=True,
            export_embeddings=True,
            resnet_checkpoint_path=dual_resnet_ckpt,
            checkpoint_strict=True,
            device="auto",
            encoder_batch_size=128,
            qa_keep_probability=0.05,
            qa_rejected_probability=0.10,
        )
        split_noise_banks[split_name] = NoiseBankConfig(
            output_dir=str(subset_root / f"MFD_Net_{split_name}_stream_noise_bank"),
            split_name=f"{split_name}_noise",
            sample_rate_hz=common_stream.sample_rate_hz,
            window_len=common_stream.window_len,
            samples_per_snr=500,
            snr_values_db=(-6.0, -3.0, 0.0, 3.0, 6.0),
            noise_power_db=common_stream.noise_power_db,
            apply_hard_clip=common_stream.apply_hard_clip,
            clip_min=common_stream.clip_min,
            clip_max=common_stream.clip_max,
            spectrogram=common_spectrogram,
            overwrite=True,
            export_images=True,
            export_embeddings=True,
            resnet_checkpoint_path=dual_resnet_ckpt,
            checkpoint_strict=True,
            device="auto",
            encoder_batch_size=128,
        )

    return DriverConfig(
        split_exports=split_exports,
        split_noise_banks=split_noise_banks,
    )


def build_default_singleframe_export_configs() -> list[SingleFrameExportConfig]:
    common_scan = _common_scan()
    common_spectrogram = _common_spectrogram()
    common_single = _common_singleframe()
    # output_dir_parent = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\data"
    # output_dir_parent = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\data_strong"
    output_dir_parent = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\data" + VERSION
    input_dir = r"C:\Apps\Code\MFD_Net" + VERSION + "_"
    # input_dir = r"C:\Apps\Code\MFD_Net_tiny_"
    # input_dir = r"C:\Apps\Code\MFD_Net_"
    subset_root = build_class_subset_root(output_dir_parent, common_scan.modulation_list)
    cfgs: list[SingleFrameExportConfig] = []
    for mode in ("power", "phase", "dual"):
        for split_name in DATA_SPLITS:
            cfgs.append(
                SingleFrameExportConfig(
                    input_root=input_dir + split_name,
                    output_dir=str(subset_root / f"{mode}_singleframe" / split_name),
                    split_name=split_name,
                    frame=common_single,
                    scan=common_scan,
                    spectrogram=common_spectrogram,
                    overwrite=True,
                    image_mode=mode,
                    include_noise_class=True,
                    noise_label="noise",
                    noise_samples_per_snr=500,
                )
            )
    return cfgs


def main() -> None:
    build_stream_memmaps = True
    build_noise_banks = True
    build_singleframe_memmaps = True

    if build_stream_memmaps or build_noise_banks:
        cfg = build_default_driver_config()
        if build_stream_memmaps:
            for split_name, split_cfg in cfg.split_exports.items():
                print("=" * 70)
                print(f"BUILDING {split_name.upper()} STREAM MEMMAP")
                print("=" * 70)
                export_stream_memmap(split_cfg)

        if build_noise_banks:
            for split_name, noise_cfg in cfg.split_noise_banks.items():
                if noise_cfg is None:
                    continue
                print("=" * 70)
                print(f"BUILDING {split_name.upper()} NOISE BANK")
                print("=" * 70)
                build_noise_bank(noise_cfg)

    if build_singleframe_memmaps:
        for split_cfg in build_default_singleframe_export_configs():
            print("=" * 70)
            print(f"BUILDING SINGLE-FRAME MEMMAP [{split_cfg.image_mode}] {split_cfg.split_name.upper()}")
            print("=" * 70)
            export_singleframe_memmap(split_cfg)


if __name__ == "__main__":
    main()
