# Post-hoc frame aggregation baseline

This module evaluates the simple reviewer/examiner baseline:

1. Load the same MFD-Net sequence chunks used by the T-frame model.
2. Flatten each chunk from `[B,T,C,H,W]` to `[B*T,C,H,W]`.
3. Run the original single-frame ResNet checkpoint on each frame.
4. Reshape logits back to `[B,T,C]`.
5. Aggregate the frame predictions/logits into one sequence prediction using:
   - averaged logits,
   - majority vote,
   - confidence-weighted vote.

The target is still the MFD sequence label `yb`, i.e. one modulation class per T-frame group.

Default settings use:

```python
FRAME_MASK_MODES = ("valid", "signal")
NOISE_POLICIES = ("allow_noise",)
INPUT_SCALE = "uint8_float"
```

For the main thesis comparison, use the `valid` + `allow_noise` rows. The `signal` rows are diagnostic because they use the known signal-frame mask.

Run:

```powershell
python -m src.posthoc_frame_aggregation.driver
```
