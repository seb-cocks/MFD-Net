# mfd_stream_existing_iq

Updated sparse-stream exporter for MFD-Net-v2.

Core behaviour:
- Each exported stream is built from **one source pulse/emitter record** repeated through time using its PW/PRI/ToA metadata.
- PW, PRI, and ToA are resolved in **samples** at 50 MHz.
- Pulse windows are kept using an **exact overlap threshold** rather than a simple occupancy mask.
- Gap/noise windows are still dropped from the main stream memmap, but their original dense positions are preserved in `window_indices_i32.bin`.

What changed:
- Short pulses can use a mild **initial-ToA bias** so they are less likely to be lost to boundary splits.
- The overlap threshold is a **window-overlap rule** (`overlap_fraction_window >= 0.02` by default).
- Per-window metadata now includes:
  - `full_pulse`
  - `overlap_samples`
  - `overlap_fraction_window`
  - `overlap_fraction_pulse`
  - `pw_samples`, `pri_samples`, `toa_samples`
  - `source_record`
  - `window_states`
- Spectrogram export uses **128 context samples on each side** by default, computes the STFT on the expanded segment, then centre-crops back to the labelled 20 us window to reduce edge artefacts.
- The main stream memmap can now cache **both images and frozen ResNet18 backbone embeddings** for kept pulse windows.
- The noise bank can also cache **both images and frozen ResNet18 backbone embeddings**.
- QA exports now save a deterministic subset of:
  - kept pulse windows
  - rejected `P => N` windows
  storing both raw cropped spectrograms and final 224x224 outputs.

Auxiliary window-state targets:
- `0 = gap_or_noise`
- `1 = full_pulse`
- `2 = partial_pulse`
- padded tail positions use `-100` during training

Important:
- Re-export the stream memmaps after applying this patch.
- The training loader can now return **images**, **embeddings**, or a **hybrid payload** depending on `LoaderConfig.input_representation`.


Single-frame / ResNet memmap export:
- Added a parallel exporter that writes memmaps for single-frame classification.
- Each source H5 pulse becomes only the minimum number of 20 us windows needed to cover the first pulse.
- Noise is exported as a normal class using the same spectrogram pipeline.
- Output layout matches the common single-frame memmap loader used by ResNet_MemMap.py.
