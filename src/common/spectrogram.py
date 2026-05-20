import math
import torch
import torch.nn.functional as F
import numpy as np


def _complex_stft_batched(
    x_b2n: torch.Tensor,
    *,
    n_fft: int,
    win_len: int,
    hop_len: int,
    center_dc: bool = True,
) -> torch.Tensor:
    """
    Shared complex STFT kernel used by both power and phase features.

    Input:
        x_b2n: (B, 2, N) real/imag channels
    Output:
        S: (B, F, T) complex STFT with full complex spectrum (complex baseband)
    """
    assert x_b2n.dim() == 3 and x_b2n.size(1) == 2, "expected (B,2,N)"
    B, _, N = x_b2n.shape
    I, Q = x_b2n[:, 0, :], x_b2n[:, 1, :]
    z = torch.complex(I, Q)

    if N < win_len:
        z = F.pad(z, (0, win_len - N))

    win = torch.hann_window(int(win_len), periodic=True, dtype=torch.float32, device=z.device)
    frames = z.unfold(-1, int(win_len), int(hop_len))  # (B, T, win_len)
    frames = frames * win.view(1, 1, -1)

    if n_fft > win_len:
        frames = F.pad(frames, (0, int(n_fft) - int(win_len)))

    S = torch.fft.fft(frames, n=int(n_fft), dim=-1)    # (B, T, F)
    if center_dc:
        S = torch.fft.fftshift(S, dim=-1)
    return S.transpose(1, 2)                            # (B, F, T)


def _soft_support_from_power_db(
    power_db_bft: torch.Tensor,
    *,
    relative_threshold_db: float,
    width_db: float,
) -> torch.Tensor:
    """
    Build a soft support mask in [0,1] from per-time relative power.

    Each time slice is referenced to its own max over frequency, which is robust
    to varying SNR and keeps the support tied to bins that actually carry signal.
    """
    frame_max = power_db_bft.amax(dim=1, keepdim=True)
    thr = frame_max + float(relative_threshold_db)
    width = max(float(width_db), 1e-6)
    return ((power_db_bft - thr) / width).clamp_(0.0, 1.0)


def _legacy_phase_return_mode(mode: str) -> str:
    """
    Backwards-compat shim: some older code passed 'snr01' for phase images even
    though phase is not an SNR quantity. Treat it as 'abs01'.
    """
    return "abs01" if mode == "snr01" else mode


# =========================
# Batched POWER spectrogram
# =========================
@torch.no_grad()
def batched_iq_to_power_spectrogram(
    x_b2n: torch.Tensor,            # (B, 2, N)
    *,
    n_fft: int = 64,
    win_len: int = 64,
    hop_len: int = 32,
    center_dc: bool = True,
    eps: float = 1e-12,
    dyn_range_db: float | None = None,
    normalise_noise: bool = True,
    noise_floor_mode: str = "fixed",  # 'percentile'|'median'|'fixed'|'none'
    noise_percentile: float = 50.0,
    noise_floor_db: float = 1.0,
    to_db: bool = True,
    db_min: float = -100.0,
    db_max: float = 20.0,
    return_mode: str = "snr01",       # 'db'|'snr_db'|'snr01' (when to_db=True)
) -> torch.Tensor:
    """
    Returns: (B, F, T) float32

    Power is derived from the shared complex STFT so the power and phase channels
    use the same underlying time/frequency lattice.
    """
    S = _complex_stft_batched(
        x_b2n,
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        center_dc=center_dc,
    )
    P = (S.abs() ** 2).clamp_min_(eps)                    # (B,F,T)

    if dyn_range_db is not None:
        X_db = 10.0 * torch.log10(P)
        frame_max = X_db.amax(dim=1, keepdim=True)
        X_rel = (X_db - frame_max).clamp(min=-dyn_range_db, max=0.0)
        out = (X_rel + dyn_range_db) / dyn_range_db
        return out.to(torch.float32)

    if not to_db:
        if not normalise_noise or noise_floor_mode == "none":
            return P.to(torch.float32)
        if noise_floor_mode in ("percentile", "median"):
            q = (noise_percentile / 100.0) if noise_floor_mode == "percentile" else 0.5
            floor = torch.quantile(P.reshape(P.shape[0], -1), q, dim=1, keepdim=True).view(P.shape[0], 1, 1)
        elif noise_floor_mode == "fixed":
            floor = torch.tensor(10.0 ** (noise_floor_db / 10.0), device=P.device, dtype=P.dtype).view(1, 1, 1)
        else:
            floor = torch.zeros(1, 1, 1, device=P.device, dtype=P.dtype)
        return torch.clamp(P - floor, min=0.0).to(torch.float32)

    X_db = 10.0 * torch.log10(P)
    if not normalise_noise:
        return X_db.to(torch.float32)

    B = x_b2n.shape[0]
    if noise_floor_mode == "percentile":
        nf = torch.quantile(X_db.reshape(B, -1), noise_percentile / 100.0, dim=1, keepdim=True)
        floor_db = nf.view(B, 1, 1)
    elif noise_floor_mode == "median":
        floor_db = X_db.median(dim=1, keepdim=True).values.view(B, 1, 1)
    elif noise_floor_mode == "fixed":
        floor_db = torch.full((B, 1, 1), float(noise_floor_db), device=X_db.device, dtype=X_db.dtype)
    elif noise_floor_mode == "none":
        floor_db = torch.zeros((B, 1, 1), device=X_db.device, dtype=X_db.dtype)
    else:
        raise ValueError("noise_floor_mode must be 'percentile'|'median'|'fixed'|'none'")

    SNR_db = torch.clamp(X_db - floor_db, db_min, db_max)
    if return_mode == "db":
        out = X_db
    elif return_mode == "snr_db":
        out = SNR_db
    elif return_mode == "snr01":
        out = (SNR_db - db_min) / max(db_max - db_min, 1e-6)
    else:
        raise ValueError("return_mode must be 'db'|'snr_db'|'snr01'")
    return out.clamp_(0.0, 1.0).to(torch.float32)


# ==================================
# Batched PHASE-DELTA spectrogram
# ==================================
@torch.no_grad()
def batched_iq_to_phase_spectrogram(
    x_b2n: torch.Tensor,
    *,
    n_fft: int = 64,
    win_len: int = 32,
    hop_len: int = 8,
    center_dc: bool = True,
    power_gate_db: float | None = None,
    soft_gate_width_db: float = 6.0,
    eps: float = 1e-12,
    return_mode: str = "abs01",
) -> torch.Tensor:
    """
    Soft-gated temporal phase-difference spectrogram.

    Definition:
        S(f,t)   = STFT{x}(f,t)
        Δφ(f,t)  = angle( S(f,t) * conj(S(f,t-1)) )

    This is the wrapped phase increment across adjacent STFT frames. It is much
    more stable and interpretable than raw wrapped phase, and it avoids the
    frequency-derivative zebra artefacts caused by taking differences directly
    across neighbouring frequency bins.

    A soft relative-power gate is applied because STFT phase becomes unstable
    near zeros / very low-magnitude bins.
    """
    return_mode = _legacy_phase_return_mode(return_mode)

    S = _complex_stft_batched(
        x_b2n,
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        center_dc=center_dc,
    )

    dphi = torch.angle(S[:, :, 1:] * torch.conj(S[:, :, :-1]))  # (B,F,T-1)

    if return_mode == "raw":
        out = dphi
    elif return_mode == "abs":
        out = dphi.abs()
    elif return_mode == "abs01":
        out = (dphi.abs() / np.pi).clamp_(0.0, 1.0)
    else:
        raise ValueError("phase return_mode must be 'raw'|'abs'|'abs01'|'snr01'")

    if power_gate_db is not None:
        P_db = 10.0 * torch.log10((S.abs() ** 2) + eps)
        w = _soft_support_from_power_db(
            P_db,
            relative_threshold_db=float(power_gate_db),
            width_db=float(soft_gate_width_db),
        )
        # A phase increment is only trustworthy if both adjacent STFT frames are
        # supported. Use min rather than product so the gate stays soft.
        w_pair = torch.minimum(w[:, :, 1:], w[:, :, :-1])
        out = out * w_pair

    return out.to(torch.float32)


# =================================
# Batched PHASE-EDGE spectrogram
# =================================
@torch.no_grad()
def batched_iq_to_phase_edge_spectrogram(
    x_b2n: torch.Tensor,
    *,
    n_fft: int = 64,
    win_len: int = 32,
    hop_len: int = 8,
    center_dc: bool = True,
    power_gate_db: float | None = -12.0,
    soft_gate_width_db: float = 6.0,
    eps: float = 1e-12,
    smooth_kernel_size: int = 3,
    return_mode: str = "edge01",
) -> torch.Tensor:
    """
    Edge map of the temporal phase-difference spectrogram.

    There is no single canonical DSP object universally called a "phase-edge
    spectrogram". In this codebase we define it explicitly as:

        1) compute the soft-gated temporal phase-difference magnitude
           A(f,t) = |angle(S(f,t) * conj(S(f,t-1)))| / pi
        2) lightly smooth A to reduce single-bin speckle
        3) compute the 2-D gradient magnitude of A

    This is intentionally different from taking finite differences of raw wrapped
    phase across frequency, which tends to create zebra stripes around STFT zeros.
    """
    mode = _legacy_phase_return_mode(return_mode)
    if mode not in ("edge", "edge01", "abs", "abs01"):
        raise ValueError("phase-edge return_mode must be 'edge'|'edge01'|'abs'|'abs01'|'snr01'")

    base = batched_iq_to_phase_spectrogram(
        x_b2n,
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        center_dc=center_dc,
        power_gate_db=power_gate_db,
        soft_gate_width_db=soft_gate_width_db,
        eps=eps,
        return_mode="abs01",
    )  # (B,F,T-1) in [0,1]

    if int(smooth_kernel_size) > 1:
        k = int(smooth_kernel_size)
        if k % 2 == 0:
            raise ValueError("smooth_kernel_size must be odd")
        pad = k // 2
        base = F.avg_pool2d(base.unsqueeze(1), kernel_size=k, stride=1, padding=pad)[:, 0]

    df = torch.diff(base, dim=1, prepend=base[:, :1, :])
    dt = torch.diff(base, dim=2, prepend=base[:, :, :1])
    edge = torch.sqrt(df.square() + dt.square()) / math.sqrt(2.0)

    if mode in ("abs", "abs01"):
        # convenience: allow callers to use the edge helper but retrieve the
        # underlying smoothed phase-delta magnitude image instead
        return base.to(torch.float32)
    return edge.clamp_(0.0, 1.0).to(torch.float32)


# ===================================
# Single-sample wrappers (API parity)
# ===================================
@torch.no_grad()
def iq_to_power_spectrogram(
    x_2_by_T: torch.Tensor,
    n_fft: int = 256,
    win_len: int = 256,
    hop_len: int = 128,
    window: torch.Tensor | None = None,  # kept for signature compatibility
    power: float = 2.0,                  # unused here; kept for compatibility
    center_dc: bool = True,
    one_sided: bool = False,             # ignored (complex baseband)
    normalise_noise: bool = True,
    to_db: bool = True,
    noise_floor_mode: str = "fixed",
    noise_percentile: float = 50.0,
    noise_floor_db: float = 1.0,
    db_min: float = -100.0,
    db_max: float = 20.0,
    return_mode: str = "snr01",
    eps: float = 1e-12,
    make_contiguous_final: bool = False,
):
    x = x_2_by_T.unsqueeze(0)
    out = batched_iq_to_power_spectrogram(
        x,
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        center_dc=center_dc,
        eps=eps,
        dyn_range_db=None,
        normalise_noise=normalise_noise,
        noise_floor_mode=noise_floor_mode,
        noise_percentile=noise_percentile,
        noise_floor_db=noise_floor_db,
        to_db=to_db,
        db_min=db_min,
        db_max=db_max,
        return_mode=return_mode,
    )[0]
    return out.contiguous() if make_contiguous_final else out


@torch.no_grad()
def iq_to_phase_spectrogram(
    iq: torch.Tensor,
    n_fft: int = 64,
    win_len: int = 32,
    hop_len: int = 8,
    window: torch.Tensor | None = None,
    center_dc: bool = True,
    power_gate_db: float | None = -12.0,
    soft_gate_width_db: float = 6.0,
    return_mode: str = "abs01",
    make_contiguous_final: bool = False,
) -> torch.Tensor:
    x = iq.unsqueeze(0)
    out = batched_iq_to_phase_spectrogram(
        x,
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        center_dc=center_dc,
        power_gate_db=power_gate_db,
        soft_gate_width_db=soft_gate_width_db,
        return_mode=return_mode,
    )[0]
    return out.contiguous() if make_contiguous_final else out


@torch.no_grad()
def iq_to_phase_edge_spectrogram(
    iq: torch.Tensor,
    n_fft: int = 64,
    win_len: int = 32,
    hop_len: int = 8,
    window: torch.Tensor | None = None,
    center_dc: bool = True,
    power_gate_db: float | None = -12.0,
    soft_gate_width_db: float = 6.0,
    smooth_kernel_size: int = 3,
    return_mode: str = "edge01",
    make_contiguous_final: bool = False,
) -> torch.Tensor:
    x = iq.unsqueeze(0)
    out = batched_iq_to_phase_edge_spectrogram(
        x,
        n_fft=n_fft,
        win_len=win_len,
        hop_len=hop_len,
        center_dc=center_dc,
        power_gate_db=power_gate_db,
        soft_gate_width_db=soft_gate_width_db,
        smooth_kernel_size=smooth_kernel_size,
        return_mode=return_mode,
    )[0]
    return out.contiguous() if make_contiguous_final else out


# ===========================
# Shared image prep utilities
# ===========================
@torch.no_grad()
def to_resnet_batch(
    spec_bft: torch.Tensor,
    size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> torch.Tensor:
    x = spec_bft.unsqueeze(1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def resize_spec01_for_save(
    spec01_bft: torch.Tensor, size: int = 224
) -> torch.Tensor:
    x = spec01_bft.unsqueeze(1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def spec_to_resnet_input(spec_ft: torch.Tensor, size: int = 224, channels: str = "tile3") -> torch.Tensor:
    if not torch.is_tensor(spec_ft):
        spec_ft = torch.as_tensor(spec_ft, dtype=torch.float32)
    if spec_ft.dim() != 2:
        raise ValueError(f"expected (F,T) tensor, got {tuple(spec_ft.shape)}")
    x = to_resnet_batch(spec_ft.unsqueeze(0), size=size)
    img = x[0]
    if img.dim() == 4:
        img = img.contiguous(memory_format=torch.channels_last)
    else:
        img = img.contiguous()
    return img
