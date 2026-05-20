from __future__ import annotations

import numpy as np


def _normal_sample(rng, size):
    """Support both Generator and RandomState style RNGs."""
    return rng.normal(size=size)


def _mildly_coloured_complex_noise(
    rng,
    n_samples: int,
    *,
    color_alpha: float = 0.18,
    drift_std: float = 0.10,
    impulse_prob: float = 0.0025,
    impulse_scale: float = 2.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Create mildly more challenging complex noise than plain iid AWGN.

    The goal is not to make the background unrealistic or dominated by artefacts,
    but to move it a little closer to imperfect receiver-like noise by adding:
    - light temporal correlation / colouring
    - slow gain drift across the sample
    - occasional impulsive outliers

    The returned noise is normalised back to unit average power so the caller can
    scale it to the desired noise level cleanly.
    """
    w = (_normal_sample(rng, n_samples) + 1j * _normal_sample(rng, n_samples)).astype(np.complex64)

    # 1) Mild temporal colouring: simple first-order recursion.
    if color_alpha is not None and abs(color_alpha) > 0:
        y = np.empty_like(w)
        y[0] = w[0]
        a = np.float32(color_alpha)
        for i in range(1, n_samples):
            y[i] = w[i] + a * y[i - 1]
    else:
        y = w.copy()

    # 2) Slow gain drift so the background is not perfectly stationary.
    if drift_std is not None and drift_std > 0:
        n_ctrl = max(4, min(16, n_samples // 128 + 2))
        ctrl_x = np.linspace(0, n_samples - 1, n_ctrl, dtype=np.float32)
        ctrl_y = _normal_sample(rng, n_ctrl).astype(np.float32)
        drift = np.interp(np.arange(n_samples, dtype=np.float32), ctrl_x, ctrl_y)
        drift = (drift - drift.mean()) / (drift.std() + eps)
        envelope = 1.0 + np.float32(drift_std) * drift
        envelope = np.clip(envelope, 0.70, 1.35).astype(np.float32)
        y = y * envelope

    # 3) Rare impulsive bursts / hot samples.
    if impulse_prob is not None and impulse_prob > 0 and impulse_scale is not None and impulse_scale > 0:
        hit = _normal_sample(rng, n_samples) < 0  # placeholder to keep dtype simple
        # use uniform randoms if available; otherwise map normals to approx uniform via sigmoid-ish thresholding
        if hasattr(rng, "random"):
            hit = rng.random(n_samples) < float(impulse_prob)
        else:
            hit = np.abs(_normal_sample(rng, n_samples)) > 10.0  # effectively never unless no random() exists
        if np.any(hit):
            bursts = (
                _normal_sample(rng, int(np.sum(hit))) + 1j * _normal_sample(rng, int(np.sum(hit)))
            ).astype(np.complex64)
            y[hit] = y[hit] + np.float32(impulse_scale) * bursts

    # Renormalise to unit average power so the caller still controls absolute noise level.
    power = np.mean(np.abs(y) ** 2)
    y = y / np.sqrt(max(power, eps))
    return y.astype(np.complex64, copy=False)


def add_noise(
    rng,
    real,
    imag=0,
    snr_db=0.0,
    noise_power_db=0.0,
    verify=False,
    eps=1e-12,
    *,
    color_alpha: float = 0.18,
    drift_std: float = 0.10,
    impulse_prob: float = 0.0025,
    impulse_scale: float = 2.5,
):
    """
    AWGN-like adder with pulse-only power estimate plus mild non-ideal noise.

    Compared with pure iid AWGN, this version makes the background slightly more
    challenging by introducing:
    - mild temporal correlation
    - slow gain drift
    - rare impulsive outliers

    The final noise is still renormalised to the requested noise power, so SNR
    remains controlled in a consistent way.

    - Compute signal power ONLY over non-zero samples (|I|+|Q| > 0), so padding
      zeros don't taint the SNR.
    - `snr_db` is the desired SNR in dB relative to `noise_power_db`.
    - Returns noisy (real, imag) as float32.
    """
    real = np.asarray(real, dtype=np.float32)
    imag = np.asarray(imag, dtype=np.float32)
    signal = real.astype(np.float32) + 1j * imag.astype(np.float32)

    n_samples = signal.size
    noise_lin = 10.0 ** (noise_power_db / 10.0)

    nz = (np.abs(signal.real) > 0) | (np.abs(signal.imag) > 0)
    has_pulse = bool(nz.any())

    # Build unit-power challenging noise, then scale to the requested absolute noise power.
    unit_noise = _mildly_coloured_complex_noise(
        rng,
        n_samples,
        color_alpha=color_alpha,
        drift_std=drift_std,
        impulse_prob=impulse_prob,
        impulse_scale=impulse_scale,
        eps=eps,
    )
    noise = np.sqrt(noise_lin) * unit_noise

    if not has_pulse:
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

        noisy = signal_scaled + noise

        if verify:
            act_sig_p = float(np.mean(np.abs(signal_scaled[nz]) ** 2))
            act_noise_p = float(np.mean(np.abs(noise) ** 2))
            act_snr_db = 10.0 * np.log10(max(act_sig_p, eps) / max(act_noise_p, eps))
            print(f"Desired SNR={snr_db:.2f} dB | Actual SNR={act_snr_db:.2f} dB")

    return (
        np.real(noisy).astype(np.float32, copy=False),
        np.imag(noisy).astype(np.float32, copy=False),
    )
