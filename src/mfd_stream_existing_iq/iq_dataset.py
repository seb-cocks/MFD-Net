from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import IQScanConfig, label_names_to_map, to_path

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


@dataclass(slots=True)
class PulseRecord:
    file_path: Path
    emitter_key: str
    label_name: str
    label_index: int
    pri: int
    toa_idx: int
    signal_noise: float
    pw: int | None = None
    rf: float | None = None
    raw_mod_name: str | None = None



def _sorted_emitter_keys(h5_file: Any) -> list[str]:
    return sorted(
        h5_file.keys(),
        key=lambda x: int(x.split("_")[-1]) if str(x).startswith("emitter_") else str(x),
    )



def _read_optional_numeric_attr(group: Any, *names: str) -> float | int | None:
    for name in names:
        if name in group.attrs:
            value = group.attrs[name]
            try:
                if np.asarray(value).shape == ():
                    item = np.asarray(value).item()
                else:
                    item = value
                if isinstance(item, (int, np.integer)):
                    return int(item)
                if isinstance(item, (float, np.floating)):
                    return float(item)
            except Exception:
                continue
    return None



def extract_emitters_from_file_with_mod(
    file_path: str | Path,
    raw_mod_name: str,
    *,
    label_name: str,
    label_index: int,
) -> list[PulseRecord]:
    if h5py is None:
        raise ImportError("h5py is required to load .h5 modulation files")

    file_path = to_path(file_path)
    out: list[PulseRecord] = []
    with h5py.File(file_path, "r") as f:
        for ek in _sorted_emitter_keys(f):
            em = f[ek]
            pri = _read_optional_numeric_attr(em, "pri_idx", "pri")
            toa_idx = _read_optional_numeric_attr(em, "toa_idx", "toa")
            sig_noise = _read_optional_numeric_attr(em, "signal_noise")
            pw = _read_optional_numeric_attr(em, "pw_idx", "pw")
            rf = _read_optional_numeric_attr(em, "rf_idx", "rf", "carrier_freq_hz")
            out.append(
                PulseRecord(
                    file_path=file_path,
                    emitter_key=str(ek),
                    label_name=label_name,
                    label_index=label_index,
                    pri=int(pri) if pri is not None else 0,
                    toa_idx=int(toa_idx) if toa_idx is not None else 0,
                    signal_noise=float(sig_noise) if sig_noise is not None else 0.0,
                    pw=int(pw) if pw is not None else None,
                    rf=float(rf) if rf is not None else None,
                    raw_mod_name=raw_mod_name,
                )
            )
    return out



def _balanced_merged_records(blocks: list[list[PulseRecord]]) -> list[PulseRecord]:
    non_empty = [list(block) for block in blocks if block]
    if not non_empty:
        return []
    if len(non_empty) == 1:
        return non_empty[0]

    target_total = min(len(block) for block in non_empty)
    if target_total <= 0:
        return []

    block_count = len(non_empty)
    base_take = target_total // block_count
    remainder = target_total % block_count

    trimmed_blocks: list[list[PulseRecord]] = []
    for idx, block in enumerate(non_empty):
        take = base_take + (1 if idx < remainder else 0)
        trimmed_blocks.append(block[:take] if take > 0 else [])

    balanced: list[PulseRecord] = []
    max_len = max((len(block) for block in trimmed_blocks), default=0)
    for pos in range(max_len):
        for block in trimmed_blocks:
            if pos < len(block):
                balanced.append(block[pos])
    return balanced



def build_pulse_index(root: str | Path, cfg: IQScanConfig) -> tuple[list[PulseRecord], dict[str, int]]:
    root = to_path(root)
    suffix = cfg.file_suffix.lower()
    final_labels = list(cfg.modulation_list) if cfg.modulation_list is not None else []
    if not final_labels:
        stems = [
            p.stem
            for p in sorted((root.rglob(f"*{suffix}") if cfg.recursive else root.glob(f"*{suffix}")))
            if p.is_file()
        ]
        if not stems:
            raise FileNotFoundError(f"No {suffix} files found under: {root}")
        final_labels = stems

    initial_map = label_names_to_map(final_labels)
    records: list[PulseRecord] = []
    for final_label in final_labels:
        raw_names = cfg.merge_map.get(final_label, (final_label,))
        blocks: list[list[PulseRecord]] = []
        for raw_name in raw_names:
            path = root / f"{raw_name}{suffix}"
            if path.exists():
                blocks.append(
                    extract_emitters_from_file_with_mod(
                        path,
                        raw_name,
                        label_name=final_label,
                        label_index=initial_map[final_label],
                    )
                )
            elif cfg.strict_modulation_list:
                raise FileNotFoundError(f"Missing modulation file: {path}")
        if not blocks:
            continue

        if len(blocks) > 1 and cfg.balance_merged_groups:
            records.extend(_balanced_merged_records(blocks))
        else:
            for block in blocks:
                records.extend(block)

    present_labels = [lbl for lbl in final_labels if any(r.label_name == lbl for r in records)]
    label_map = label_names_to_map(present_labels)
    for rec in records:
        rec.label_index = label_map[rec.label_name]
    return records, label_map


@lru_cache(maxsize=8192)
def _cached_read_pulse(file_path: str, emitter_key: str) -> np.ndarray:
    if h5py is None:
        raise ImportError("h5py is required to load .h5 modulation files")
    with h5py.File(file_path, "r") as f:
        em = f[emitter_key]
        i = np.asarray(em["I"][:], dtype=np.float32).reshape(-1)
        q = np.asarray(em["Q"][:], dtype=np.float32).reshape(-1)
    if i.shape != q.shape:
        raise ValueError(f"I/Q length mismatch in {file_path}:{emitter_key} -> {i.shape} vs {q.shape}")
    return (i + 1j * q).astype(np.complex64)



def trim_zero_padding(iq: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    mag = np.abs(iq)
    nz = np.flatnonzero(mag > eps)
    if len(nz) == 0:
        return iq[:1].copy()
    return iq[nz[0] : nz[-1] + 1].copy()



def load_pulse_iq(record: PulseRecord, cfg: IQScanConfig) -> np.ndarray:
    iq = _cached_read_pulse(str(record.file_path), record.emitter_key)
    if cfg.trim_zero_padding:
        iq = trim_zero_padding(iq, cfg.zero_trim_eps)
    return np.asarray(iq, dtype=np.complex64).copy()



def save_index_summary(output_dir: str | Path, split_name: str, records: list[PulseRecord], label_map: dict[str, int]) -> None:
    output_dir = to_path(output_dir)
    counts_per_label = {label: 0 for label in sorted(label_map.keys(), key=lambda x: label_map[x])}
    counts_per_raw_mod: dict[str, int] = {}
    rf_available = 0
    pw_available = 0
    pri_available = 0
    for rec in records:
        counts_per_label[rec.label_name] += 1
        raw = rec.raw_mod_name or rec.label_name
        counts_per_raw_mod[raw] = counts_per_raw_mod.get(raw, 0) + 1
        rf_available += int(rec.rf is not None)
        pw_available += int(rec.pw is not None)
        pri_available += int(int(rec.pri) > 0)
    payload = {
        "split_name": split_name,
        "num_pulses": len(records),
        "labels": label_map,
        "counts_per_label": counts_per_label,
        "counts_per_raw_mod": dict(sorted(counts_per_raw_mod.items())),
        "metadata_coverage": {
            "rf_available": rf_available,
            "pw_available": pw_available,
            "pri_available": pri_available,
        },
    }
    with open(output_dir / f"{split_name}_pulse_index_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
