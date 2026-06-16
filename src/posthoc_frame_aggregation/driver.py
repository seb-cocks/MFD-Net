from __future__ import annotations

"""Driver for the reviewer-requested post-hoc single-frame aggregation baseline.

Place this folder at:
    src/posthoc_frame_aggregation/

Run from the repository root:
    python -m src.posthoc_frame_aggregation.driver

What this does:
    1. Loads an already-trained MFD-Net checkpoint only to recover the exact
       ExperimentConfig and MFD sequence test/val loader.
    2. Loads the original single-frame ResNet checkpoint(s) using the same
       MainModel class used by the single-frame memmap evaluation path.
    3. Flattens each MFD T-frame chunk to B*T frames, runs the single-frame
       model, reshapes logits back to [B,T,C], and aggregates them.

Main baseline settings:
    FRAME_MASK_MODES = ("valid",)
    NOISE_POLICIES = ("allow_noise",)

That means: use the same T-frame buffer as MFD-Net. If the single-frame model's
post-hoc sequence prediction is the auxiliary noise class, it is counted as
wrong against the modulation sequence label.
"""

from pathlib import Path

from .evaluate import run_posthoc_sequence_baseline


# -----------------------------------------------------------------------------
# Edit these paths/settings
# -----------------------------------------------------------------------------

MFD_CHECKPOINT_PATH = Path(
    r"out\mfd_stream_train\power_phase_gap_noise_images_t16_posthoc_20260616_161924\best.pt"
)

# Usually leave as None so outputs go beside the MFD checkpoint.
OUTPUT_ROOT: Path | None = None

# Use test for the final held-out comparison. Use ("val", "test") if needed.
SPLITS: tuple[str, ...] = ("test",)

# Available sources depend on checkpoint paths stored in the MFD config.
# For the thesis T=16 power_phase run, ("power", "phase") is the clean comparison.
SOURCES: tuple[str, ...] = ("power", "phase")

# "valid" = same dense T-frame buffer as MFD-Net.
# "signal" = diagnostic/oracle mask that only aggregates known signal frames.
FRAME_MASK_MODES: tuple[str, ...] = ("valid", "signal")

# "allow_noise"       = literal baseline. Aggregate all classes; if sequence pred is noise, count wrong.
# "force_signal"      = diagnostic. Aggregate all frames but force final argmax over modulation classes only.
# "drop_noise_frames" = diagnostic. Drop frames whose per-frame argmax is noise before aggregation.
NOISE_POLICIES: tuple[str, ...] = ("allow_noise",)

# This matches the current single-frame memmap training/eval path, where uint8
# images are cast to float before MainModel sees them. The MFD loader returns
# 0..1 floats, so this multiplies them back to 0..255.
# Other options: "zero_one", "imagenet".
INPUT_SCALE = "uint8_float"

DEVICE = "auto"
BATCH_SIZE_OVERRIDE: int | None = None
NUM_WORKERS_OVERRIDE: int | None = 0

# Optional path overrides. Leave as None to use the paths/checkpoints stored in best.pt.
POWER_CHECKPOINT_OVERRIDE: str | None = None
PHASE_CHECKPOINT_OVERRIDE: str | None = None
DUAL_CHECKPOINT_OVERRIDE: str | None = None
VAL_MEMMAP_DIR_OVERRIDE: str | None = None
VAL_NOISE_BANK_DIR_OVERRIDE: str | None = None
TEST_MEMMAP_DIR_OVERRIDE: str | None = None
TEST_NOISE_BANK_DIR_OVERRIDE: str | None = None

# Keep False for checkpoints trained by src/ResNet/ResNet_MemMap.py/MainModel.
# Set True only for a legacy checkpoint that normalised the projection embedding
# before its classifier.
L2NORM_EMBEDDING = False
STRICT_CHECKPOINT_LOAD = True


def _clean_tuple(values: tuple[str, ...], *, allowed: set[str], name: str) -> tuple[str, ...]:
    out = tuple(str(v).lower().strip() for v in values)
    bad = [v for v in out if v not in allowed]
    if bad:
        raise ValueError(f"{name} contains invalid values {bad}. Allowed={sorted(allowed)}")
    if not out:
        raise ValueError(f"{name} must not be empty.")
    return out


def _out_dir(root: Path, split: str, frame_mask: str, noise_policy: str) -> Path:
    if len(SPLITS) == 1 and len(FRAME_MASK_MODES) == 1 and len(NOISE_POLICIES) == 1:
        return root
    return root / f"{split}_{frame_mask}_{noise_policy}"


def main() -> None:
    ckpt = MFD_CHECKPOINT_PATH
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Could not find MFD checkpoint: {ckpt}\n"
            "Update MFD_CHECKPOINT_PATH in src/posthoc_frame_aggregation/driver.py."
        )

    splits = _clean_tuple(SPLITS, allowed={"val", "test"}, name="SPLITS")
    frame_masks = _clean_tuple(FRAME_MASK_MODES, allowed={"valid", "signal"}, name="FRAME_MASK_MODES")
    noise_policies = _clean_tuple(
        NOISE_POLICIES,
        allowed={"allow_noise", "force_signal", "drop_noise_frames"},
        name="NOISE_POLICIES",
    )
    sources = _clean_tuple(SOURCES, allowed={"power", "phase", "dual"}, name="SOURCES")

    root = OUTPUT_ROOT if OUTPUT_ROOT is not None else ckpt.parent / "posthoc_frame_aggregation"
    root.mkdir(parents=True, exist_ok=True)

    print("[driver setup]")
    print(f"  mfd checkpoint = {ckpt}")
    print(f"  output root    = {root}")
    print(f"  splits         = {splits}")
    print(f"  sources        = {sources}")
    print(f"  frame masks    = {frame_masks}")
    print(f"  noise policies = {noise_policies}")
    print(f"  input scale    = {INPUT_SCALE}")

    written: list[Path] = []
    for split in splits:
        for frame_mask in frame_masks:
            for noise_policy in noise_policies:
                out_dir = _out_dir(root, split, frame_mask, noise_policy)
                written.append(
                    run_posthoc_sequence_baseline(
                        mfd_checkpoint_path=ckpt,
                        output_dir=out_dir,
                        split=split,
                        sources=sources,
                        frame_mask_mode=frame_mask,
                        noise_policy=noise_policy,
                        input_scale=INPUT_SCALE,
                        device_name=DEVICE,
                        batch_size=BATCH_SIZE_OVERRIDE,
                        num_workers=NUM_WORKERS_OVERRIDE,
                        strict_checkpoint_load=STRICT_CHECKPOINT_LOAD,
                        l2norm_embedding=L2NORM_EMBEDDING,
                        test_memmap_dir=TEST_MEMMAP_DIR_OVERRIDE,
                        test_noise_bank_dir=TEST_NOISE_BANK_DIR_OVERRIDE,
                        val_memmap_dir=VAL_MEMMAP_DIR_OVERRIDE,
                        val_noise_bank_dir=VAL_NOISE_BANK_DIR_OVERRIDE,
                        power_checkpoint=POWER_CHECKPOINT_OVERRIDE,
                        phase_checkpoint=PHASE_CHECKPOINT_OVERRIDE,
                        dual_checkpoint=DUAL_CHECKPOINT_OVERRIDE,
                    )
                )

    print("\n[done] post-hoc frame aggregation outputs:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
