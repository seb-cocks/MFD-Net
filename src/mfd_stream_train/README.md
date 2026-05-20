# mfd_stream_train

Updated MFD-Net-v2 temporal training path.

What it does:
- Trains on sparse stream memmaps exported by `mfd_stream_existing_iq`.
- Supports three input forms from the loader:
  - `images`
  - `embeddings`
  - `hybrid`
- Supports dense gap-aware reconstruction with:
  - `pulse_tail_pad`
  - `gap_padding`
  - `gap_noise_images`
  - `gap_noise_embeddings`

Model path:
- `dual image -> frozen / optionally trainable ResNet18 backbone -> 512d embedding -> optional front projector -> TCN/MIL temporal head -> class logits`
- The front projector is configurable with:
  - `use_input_projector=True/False`
  - `d_model=...`
- The model can consume:
  - direct image chunks `[B,T,2,H,W]`
  - direct embedding chunks `[B,T,D]`
  - hybrid payloads with cached embeddings plus an embedding mask

Temporal options:
- `temporal_arch="tcn_mil"`
- `temporal_arch="tcn_only"`
- `temporal_arch="mil_only"`

Auxiliary supervision:
- Optional per-window auxiliary head predicts:
  - `0 = gap_or_noise`
  - `1 = full_pulse`
  - `2 = partial_pulse`
- padded positions use ignore index `-100`

Recommended cached-embedding starting point:
- `input_representation="embeddings"`
- `noise_handling_mode="gap_noise_embeddings"`
- `freeze_backbone=True`
- `use_input_projector=True`
- `d_model=256`
- `temporal_arch="tcn_mil"`
