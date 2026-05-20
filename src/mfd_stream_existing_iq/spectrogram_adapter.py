from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import SpectrogramAdapterConfig


@dataclass(slots=True)
class RenderedSpectrogram:
    raw_power: np.ndarray
    raw_phase: np.ndarray
    power_u8: np.ndarray
    phase_u8: np.ndarray

    @property
    def dual_u8(self) -> np.ndarray:
        return np.stack([self.power_u8, self.phase_u8], axis=0)


class SpectrogramAdapter:
    """Render power and phase-derived spectrograms for a single labelled window.

    The adapter accepts a central labelled window plus optional context samples on
    each side. The feature transform is computed on the expanded support, then
    centre-cropped back to the original window before final resize. This preserves
    alignment to the label while reducing STFT edge artefacts.

    Important: the phase channel is delegated to the configured helper in
    ``src.common.spectrogram``. That keeps the maths explicit and prevents the
    adapter from silently inventing its own incompatible phase representation.
    """

    def __init__(self, cfg: SpectrogramAdapterConfig):
        self.cfg = cfg
        self.module = importlib.import_module(cfg.module_name)
        self.power_fn = self._resolve_required(cfg.power_method_name)
        self.phase_fn = self._resolve_required(cfg.phase_method_name)
        self.resize_fn = self._resolve_optional(cfg.resize_method_name)

    def _resolve_required(self, name: str) -> Callable[..., Any]:
        if not hasattr(self.module, name):
            raise AttributeError(f"{self.cfg.module_name} has no attribute {name!r}")
        fn = getattr(self.module, name)
        if not callable(fn):
            raise TypeError(f"{self.cfg.module_name}.{name} is not callable")
        return fn

    def _resolve_optional(self, name: str | None) -> Callable[..., Any] | None:
        if name is None:
            return None
        if not hasattr(self.module, name):
            return None
        fn = getattr(self.module, name)
        return fn if callable(fn) else None

    @staticmethod
    def _iq_to_tensor(iq: np.ndarray) -> torch.Tensor:
        iq = np.asarray(iq, dtype=np.complex64).reshape(-1)
        x = np.stack([iq.real.astype(np.float32), iq.imag.astype(np.float32)], axis=0)
        return torch.from_numpy(x)

    def _resize_spec(self, spec_ft: torch.Tensor) -> torch.Tensor:
        spec_ft = spec_ft.to(torch.float32)
        if spec_ft.dim() != 2:
            spec_ft = torch.squeeze(spec_ft)
        if spec_ft.dim() != 2:
            raise ValueError(f"Expected (F,T) spectrogram, got {tuple(spec_ft.shape)}")

        if self.resize_fn is not None and self.cfg.out_height == self.cfg.out_width:
            out = self.resize_fn(spec_ft.unsqueeze(0), size=self.cfg.out_height)
            out = out[0, 0]
        else:
            out = F.interpolate(
                spec_ft.unsqueeze(0).unsqueeze(0),
                size=(self.cfg.out_height, self.cfg.out_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        return out.clamp(0.0, 1.0)

    def _to_u8(self, spec_ft: torch.Tensor) -> np.ndarray:
        spec01 = self._resize_spec(spec_ft)
        img = torch.round(spec01 * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        return img.cpu().numpy().copy()

    @staticmethod
    def _centre_crop_time(spec: torch.Tensor, raw_len: int, target_len: int) -> torch.Tensor:
        if spec.dim() != 2:
            raise ValueError(f"Expected 2D spectrogram before crop, got {tuple(spec.shape)}")
        if raw_len <= 0 or target_len <= 0 or raw_len <= target_len:
            return spec
        total_bins = int(spec.shape[1])
        if total_bins <= 1:
            return spec
        start_ratio = max(0.0, (raw_len - target_len) * 0.5 / raw_len)
        end_ratio = min(1.0, (raw_len + target_len) * 0.5 / raw_len)
        start_bin = int(round(start_ratio * total_bins))
        end_bin = int(round(end_ratio * total_bins))
        end_bin = max(start_bin + 1, min(total_bins, end_bin))
        return spec[:, start_bin:end_bin]

    def _compute_power_raw(self, iq: np.ndarray) -> torch.Tensor:
        x = self._iq_to_tensor(iq)
        return self.power_fn(
            x,
            n_fft=self.cfg.power_n_fft,
            win_len=self.cfg.power_win_len,
            hop_len=self.cfg.power_hop_len,
            center_dc=True,
            normalise_noise=True,
            to_db=True,
            noise_floor_mode="fixed",
            noise_floor_db=1.0,
            db_min=-100.0,
            db_max=20.0,
            return_mode="snr01",
            make_contiguous_final=False,
        ).to(torch.float32)

    # def _compute_phase_raw(self, iq: np.ndarray) -> torch.Tensor:
    #     x = self._iq_to_tensor(iq)
    #     kwargs: dict[str, Any] = {
    #         "n_fft": self.cfg.phase_n_fft,
    #         "win_len": self.cfg.phase_win_len,
    #         "hop_len": self.cfg.phase_hop_len,
    #         "center_dc": True,
    #         "power_gate_db": self.cfg.phase_power_gate_db,
    #         "return_mode": "abs01",
    #         "make_contiguous_final": False,
    #     }

    #     sig = inspect.signature(self.phase_fn)
    #     if "soft_gate_width_db" in sig.parameters and hasattr(self.cfg, "phase_soft_gate_width_db"):
    #         kwargs["soft_gate_width_db"] = getattr(self.cfg, "phase_soft_gate_width_db")

    #     return self.phase_fn(x, **kwargs).to(torch.float32)

    def _compute_phase_raw(self, iq: np.ndarray) -> torch.Tensor:
        x = self._iq_to_tensor(iq)
        return self.phase_fn(
            x,
            n_fft=self.cfg.phase_n_fft,
            win_len=self.cfg.phase_win_len,
            hop_len=self.cfg.phase_hop_len,
            center_dc=True,
            power_gate_db=self.cfg.phase_power_gate_db,
            return_mode="abs01",
            make_contiguous_final=False,
        ).to(torch.float32)


    def render_window(self, iq_with_context: np.ndarray, target_window_len: int) -> RenderedSpectrogram:
        iq = np.asarray(iq_with_context, dtype=np.complex64).reshape(-1)
        raw_power = self._compute_power_raw(iq)
        raw_phase = self._compute_phase_raw(iq)
        cropped_power = self._centre_crop_time(raw_power, raw_len=len(iq), target_len=int(target_window_len))
        cropped_phase = self._centre_crop_time(raw_phase, raw_len=len(iq), target_len=int(target_window_len))
        return RenderedSpectrogram(
            raw_power=np.asarray(cropped_power.cpu().numpy(), dtype=np.float32).copy(),
            raw_phase=np.asarray(cropped_phase.cpu().numpy(), dtype=np.float32).copy(),
            power_u8=self._to_u8(cropped_power),
            phase_u8=self._to_u8(cropped_phase),
        )
