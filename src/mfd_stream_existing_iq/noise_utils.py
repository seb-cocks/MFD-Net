from __future__ import annotations

import numpy as np


def add_noise(
    rng,
    real,
    imag=0,
    snr_db=0.0,
    noise_power_db=0.0,
    verify=False,
    eps=1e-12,
    apply_hard_clip: bool = False,
    clip_min: float = -1.0,
    clip_max: float = 1.0,
):
    """
    Robust AWGN adder with pulse-only power estimate.

    - Compute signal power ONLY over non-zero samples (|I|+|Q| > 0), so padding zeros
      don't taint the SNR. If there are no non-zero samples, fall back to noise-only.
    - `snr_db` is the desired SNR in dB relative to `noise_power_db`.
    - If `apply_hard_clip` is True, clip I and Q independently to [clip_min, clip_max]
      after noise injection to mimic recorder saturation.
    - Returns noisy (real, imag) as float32.
    """
    real = np.asarray(real, dtype=np.float32)
    imag = np.asarray(imag, dtype=np.float32)
    signal = real.astype(np.float32) + 1j * imag.astype(np.float32)

    N = signal.size
    noise_lin = 10.0 ** (noise_power_db / 10.0)

    nz = (np.abs(signal.real) > 0) | (np.abs(signal.imag) > 0)
    has_pulse = bool(nz.any())

    if not has_pulse:
        noise = np.sqrt(noise_lin / 2.0) * (
            rng.normal(size=N) + 1j * rng.normal(size=N)
        )
        noisy = noise
        if verify:
            act_noise_p = float(np.mean(np.abs(noise) ** 2))
            act_noise_db = 10.0 * np.log10(max(act_noise_p, eps))
            print(
                f"[noise-only] target_noise={noise_power_db:.2f} dB | "
                f"actual_noise={act_noise_db:.2f} dB"
            )
    else:
        sig_power = float(np.mean(np.abs(signal[nz]) ** 2))
        desired_sig_power = noise_lin * (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(max(desired_sig_power, eps) / max(sig_power, eps))
        signal_scaled = signal * scale

        noise = np.sqrt(noise_lin / 2.0) * (
            rng.normal(size=N) + 1j * rng.normal(size=N)
        )
        noisy = signal_scaled + noise

        if verify:
            act_sig_p = float(np.mean(np.abs(signal_scaled[nz]) ** 2))
            act_noise_p = float(np.mean(np.abs(noise) ** 2))
            act_snr_db = 10.0 * np.log10(max(act_sig_p, eps) / max(act_noise_p, eps))
            print(f"Desired SNR={snr_db:.2f} dB | Actual SNR={act_snr_db:.2f} dB")

    real_out = np.real(noisy).astype(np.float32, copy=False)
    imag_out = np.imag(noisy).astype(np.float32, copy=False)

    if apply_hard_clip:
        real_out = np.clip(real_out, clip_min, clip_max).astype(np.float32, copy=False)
        imag_out = np.clip(imag_out, clip_min, clip_max).astype(np.float32, copy=False)

    return real_out, imag_out
