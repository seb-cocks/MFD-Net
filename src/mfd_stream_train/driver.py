from __future__ import annotations

from .trainer import ExperimentConfig, run_experiment
from src.mfd_stream_existing_iq.config import build_class_subset_tag

PARENT_DIR = r"C:\Users\scocks\BitBucket\spectral_labs\GOOD-VERSION-MFD-NET\HPC-aimc-spec-v2 - data gen v2 - no noise in model\\"

def main():
    # output_dir_parent = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\data_strong"
    output_dir_parent = PARENT_DIR + r"data"
    # output_dir_parent = r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\data_real"

    class_subset_names = (
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
    subset_tag = build_class_subset_tag(class_subset_names)
    print(f"[subset] class_subset_tag={subset_tag}")

    modality_mode="power_phase"  # power | phase | power_phase | dual

    noise_handling_mode="gap_noise_images" # pulse_tail_pad | gap_padding | gap_noise_images | gap_noise_embeddings

    chunky = 64

    cfg = ExperimentConfig(
        train_memmap_dir=output_dir_parent + r"/MFD_Net_train_stream_memmap",
        val_memmap_dir=output_dir_parent + r"/MFD_Net_val_stream_memmap",
        test_memmap_dir=output_dir_parent + r"/MFD_Net_test_stream_memmap",
        data_root_dir=output_dir_parent,
        class_subset_names=class_subset_names,
        modality_mode=modality_mode,

        # Checkpoints used according to modality_mode:
        #   power       -> power_checkpoint
        #   phase       -> phase_checkpoint
        #   power_phase -> power_checkpoint + phase_checkpoint
        #   dual        -> dual_checkpoint

        # OG Ckpt
        # phase_checkpoint=r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\out\resnet_phase\20260403_234649_None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6\checkpoints\best__None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6.pt",
        # power_checkpoint=r"C:\Users\scocks\BitBucket\spec_labs_2\SL_mfd-net-v2-tests-22-03-2026\HPC-aimc-spec-v2 - data gen v2 - no noise in model\out\resnet_power\20260403_234052_None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6\checkpoints\best__None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6.pt",
        
        # Real version ckpt
        phase_checkpoint=PARENT_DIR + r"out\resnet_phase\20260417_115620_None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6\checkpoints\best__None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6.pt",
        power_checkpoint=PARENT_DIR + r"out\resnet_power\20260417_121336_None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6\checkpoints\best__None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6.pt",
        # Dual not retrained on the real version
        dual_checkpoint=PARENT_DIR + r"out\resnet_dual\20260408_091805_None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6\checkpoints\best__None_singleframe_e50_wc1_h1_lr0.0006_snr630-3-6.pt",
        train_noise_bank_dir=output_dir_parent + r"/MFD_Net_train_stream_noise_bank",
        val_noise_bank_dir=output_dir_parent + r"/MFD_Net_val_stream_noise_bank",
        test_noise_bank_dir=output_dir_parent + r"/MFD_Net_test_stream_noise_bank",
        noise_handling_mode=noise_handling_mode, # pulse_tail_pad | gap_padding | gap_noise_images | gap_noise_embeddings
        out_dir=r"out/mfd_stream_train/",
        
        run_name=f"{modality_mode}_{noise_handling_mode}",

        input_representation="images",
        chunk_len=chunky,
        chunk_stride_train=4,
        chunk_stride_val=chunky,
        train_pad_short_streams=True,
        val_pad_short_streams=True,
        batch_size=16,
        num_workers=0,
        # max_epochs=25,
        max_epochs=50,
        # patience=5,
        patience=10,
        min_delta=0.001,
        learning_rate=1e-4,
        backbone_learning_rate=None,
        weight_decay=1e-4,
        seed=42,
        freeze_backbone=True,
        train_encoder_proj=True,
        train_encoder_classifier=True,
        power_embedding_dim=256,
        phase_embedding_dim=256,
        dual_embedding_dim=256,
        d_model=256,
        nhead=4,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.1,

        # use_frame_encoder_logits=True,
        # add_aux_logit_token=True,
        use_frame_encoder_logits=False,
        add_aux_logit_token=False,

        aux_logit_proj_dim=64,
        aux_last_weight=0.3,
        sequence_mask_mode="valid",
        min_frames_for_buffer_label=1,
        threshold_mode="per_class",
        key_metric="micro_f1",
        use_class_weights=False,
        channels_last=True,
        print_every=20,
    )

    print("[task setup]")
    print("  main task          = multi-label buffer classification (Jan-style)")
    print(f"  modality_mode      = {cfg.modality_mode}")
    print("  input              = precomputed power/phase images")
    print(f"  freeze_backbone    = {cfg.freeze_backbone}")
    print(
        f"  use_logits_token   = {cfg.use_frame_encoder_logits and cfg.add_aux_logit_token}"
    )
    print(f"  aux_last_weight    = {cfg.aux_last_weight}")
    run_dir = run_experiment(cfg)
    print(f"[done] outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
 