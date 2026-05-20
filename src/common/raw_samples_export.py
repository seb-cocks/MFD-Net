from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.common.spectrogram import (
    batched_iq_to_power_spectrogram,
    resize_spec01_for_save,
)


# -----------------------------
# Stable key across SNRs
# -----------------------------
@dataclass(frozen=True)
class PulseKey:
    modulation: str
    seed: int
    rel_path: str          # relative to the SNR root
    emitter_index: int     # fallback if seed lookup fails


# -----------------------------
# Filesystem helpers
# -----------------------------
def _norm(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("-", "").replace("+", "")


def _find_snr_root(raw_root: str, snr_db: int) -> str:
    """
    Try to locate a subdirectory corresponding to a given SNR.
    Falls back to raw_root if nothing matches.
    """
    snr = int(snr_db)
    raw_root = os.path.normpath(raw_root)

    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    # Search one level deep for something like: snr_6, snr+6, +6db, 6db, db6, etc.
    candidates = []
    for name in os.listdir(raw_root):
        p = os.path.join(raw_root, name)
        if not os.path.isdir(p):
            continue
        n = _norm(name)
        # extremely permissive matching
        if (f"snr{snr}" in n) or (f"{snr}db" in n) or (n == str(snr)) or (f"db{snr}" in n):
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]

    # If multiple, prefer ones that literally contain "snr"
    if candidates:
        snr_pref = [c for c in candidates if "snr" in _norm(os.path.basename(c))]
        return snr_pref[0] if snr_pref else candidates[0]

    # No SNR subfolders found; assume raw_root already corresponds to an SNR
    return raw_root


def _list_candidate_files(snr_root: str, modulation: str) -> list[str]:
    """
    Collect candidate files under snr_root that likely contain the modulation.
    """
    mod_key = _norm(modulation)
    exts = (".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle", ".h5")

    hits: list[str] = []
    for dirpath, _, filenames in os.walk(snr_root):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            full = os.path.join(dirpath, fn)
            if mod_key in _norm(full):
                hits.append(full)

    return hits


# -----------------------------
# Data loading helpers
# -----------------------------
def _extract_emitters(obj: Any) -> list[dict]:
    """
    Accepts common container shapes and returns a list of emitter dicts.
    """
    if obj is None:
        return []

    # If it's already a list of emitters
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if isinstance(obj[0], dict) and ("I" in obj[0] and "Q" in obj[0]):
            return obj
        # sometimes list wrapping a dict
        if len(obj) == 1 and isinstance(obj[0], dict) and "emitters" in obj[0]:
            return _extract_emitters(obj[0]["emitters"])
        return []

    # Dict containers
    if isinstance(obj, dict):
        # most common
        if "emitters" in obj and isinstance(obj["emitters"], list):
            return _extract_emitters(obj["emitters"])

        # sometimes stored under other keys
        for k in ("data", "samples", "items"):
            if k in obj:
                em = _extract_emitters(obj[k])
                if em:
                    return em

        # Or it might *be* a single emitter
        if ("I" in obj) and ("Q" in obj):
            return [obj]

    return []


def _load_container(path: str) -> Any:
    """
    Load file into Python object. Supports npz/npy, pt/pth, pkl, json.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        # if there is only one array, return it
        if len(z.files) == 1:
            return z[z.files[0]].tolist() if hasattr(z[z.files[0]], "tolist") else z[z.files[0]]
        # try common keys
        for k in ("emitters", "data"):
            if k in z.files:
                arr = z[k]
                return arr.tolist() if hasattr(arr, "tolist") else arr
        # else return dict-like
        return {k: (z[k].tolist() if hasattr(z[k], "tolist") else z[k]) for k in z.files}

    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        return arr.tolist() if hasattr(arr, "tolist") else arr

    if ext in (".pt", ".pth"):
        return torch.load(path, map_location="cpu")

    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported file extension: {ext}")


def _emitter_iq_to_2n(em: dict) -> np.ndarray:
    I = np.asarray(em["I"], dtype=np.float32)
    Q = np.asarray(em["Q"], dtype=np.float32)
    if I.ndim != 1 or Q.ndim != 1 or I.shape[0] != Q.shape[0]:
        raise ValueError(f"Bad I/Q shapes: I={I.shape}, Q={Q.shape}")
    return np.stack([I, Q], axis=0)  # (2,N)


# -----------------------------
# Roulette sampling + cross-SNR fetch by seed
# -----------------------------
def list_pulse_keys_for_mod_snr(
    raw_root: str,
    *,
    modulation: str,
    snr_db: int,
    rng: np.random.Generator,
    pool_limit_files: int = 200,
) -> tuple[str, list[PulseKey]]:
    """
    Build a candidate pool of PulseKeys at the given snr_db.
    Returns: (snr_root, keys)
    """
    snr_root = _find_snr_root(raw_root, snr_db)
    files = _list_candidate_files(snr_root, modulation)

    if not files:
        raise RuntimeError(
            f"No candidate files found for modulation='{modulation}' under snr_root='{snr_root}'.\n"
            "Your folder naming probably doesn't include the modulation string. "
            "If so, we need a different file discovery rule."
        )

    # cap to avoid walking/loading an insane number of files
    if len(files) > pool_limit_files:
        files = list(rng.choice(files, size=pool_limit_files, replace=False))

    keys: list[PulseKey] = []
    for fp in files:
        try:
            obj = _load_container(fp)
            emitters = _extract_emitters(obj)
            if not emitters:
                continue

            rel = os.path.relpath(fp, snr_root)
            # Add all seeds from this file into the pool
            for ei, em in enumerate(emitters):
                if not isinstance(em, dict):
                    continue
                if "seed" not in em:
                    continue
                keys.append(
                    PulseKey(
                        modulation=str(modulation),
                        seed=int(em["seed"]),
                        rel_path=rel,
                        emitter_index=int(ei),
                    )
                )
        except Exception:
            # ignore “not actually emitters” files
            continue

    if not keys:
        raise RuntimeError(
            f"Found files for modulation='{modulation}' but none yielded emitter dicts with I/Q/seed.\n"
            f"snr_root={snr_root}"
        )

    return snr_root, keys


def load_iq_for_pulse(
    raw_root: str,
    *,
    key: PulseKey,
    snr_db: int,
) -> np.ndarray:
    """
    Load IQ for a given PulseKey at a specified SNR.
    Prefers matching by seed; falls back to emitter_index.
    """
    snr_root = _find_snr_root(raw_root, snr_db)
    path = os.path.join(snr_root, key.rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found for SNR {snr_db}: {path}")

    obj = _load_container(path)
    emitters = _extract_emitters(obj)
    if not emitters:
        raise RuntimeError(f"No emitters in file: {path}")

    # best: match by seed
    for em in emitters:
        if isinstance(em, dict) and int(em.get("seed", -1)) == int(key.seed):
            return _emitter_iq_to_2n(em)

    # fallback: same index
    ei = int(key.emitter_index)
    if 0 <= ei < len(emitters) and isinstance(emitters[ei], dict):
        return _emitter_iq_to_2n(emitters[ei])

    raise RuntimeError(
        f"Could not find emitter by seed={key.seed} (or emitter_index={ei}) in {path}"
    )


# -----------------------------
# Image save helper (no title)
# -----------------------------
def _save_spec01_png(
    spec01_hw: np.ndarray,
    out_path: str,
    *,
    cmap: str = "gray",
    flip_vertical: bool = True,
):
    if flip_vertical:
        spec01_hw = np.flipud(spec01_hw)

    plt.figure(figsize=(5, 5))
    plt.imshow(spec01_hw, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


@torch.no_grad()
def export_random_pulse_bundles_all_snrs(
    raw_root: str,
    *,
    modulation: str,
    n_samples: int = 20,
    snrs: Iterable[int] = (6, 3, 0, -3, -6),
    out_root: str | None = None,
    seed: int = 42,
    device: str = "cpu",
    # spectrogram params
    n_fft: int = 64,
    win_len: int = 64,
    hop_len: int = 32,
    dyn_range_db: float | None = None,
    # output style
    cmap: str = "turbo",
    flip_vertical: bool = True,
) -> dict[str, Any]:
    """
    Randomly samples N emitters at +6 dB (or snrs[0]) for the modulation, then
    regenerates and saves the 5 SNR spectrograms for the same emitter seed.
    """
    snrs = [int(s) for s in snrs]
    anchor_snr = int(snrs[0])

    if out_root is None:
        out_root = os.path.join(
            raw_root, "_sample_bundles", f"{modulation}_anchor_{anchor_snr:+d}dB"
        )
    os.makedirs(out_root, exist_ok=True)

    rng = np.random.default_rng(seed)

    anchor_root, pool = list_pulse_keys_for_mod_snr(
        raw_root,
        modulation=modulation,
        snr_db=anchor_snr,
        rng=rng,
    )

    k = min(int(n_samples), len(pool))
    chosen = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]

    torch_device = torch.device(device)

    bundles = []
    for bi, key in enumerate(chosen):
        # folder name includes seed so you can eyeball it
        bundle_dir = os.path.join(out_root, f"{modulation}_seed_{key.seed}_b{bi:03d}")
        os.makedirs(bundle_dir, exist_ok=True)

        for snr in snrs:
            iq_2n = load_iq_for_pulse(raw_root, key=key, snr_db=snr)  # (2,N) float32
            x = torch.from_numpy(iq_2n).to(torch.float32, device=torch_device).unsqueeze(0)  # (1,2,N)

            spec01 = batched_iq_to_power_spectrogram(
                x,
                n_fft=n_fft,
                win_len=win_len,
                hop_len=hop_len,
                center_dc=True,
                dyn_range_db=dyn_range_db,
                normalise_noise=True,
                noise_floor_mode="fixed",
                noise_floor_db=1.0,
                to_db=True,
                db_min=-100.0,
                db_max=20.0,
                return_mode="snr01",
            )  # (1,F,T) in [0,1]

            spec01_224 = resize_spec01_for_save(spec01, size=224)[0, 0]  # (224,224)

            out_path = os.path.join(bundle_dir, f"power_snr_{snr:+d}.png")
            _save_spec01_png(
                spec01_224.detach().cpu().numpy(),
                out_path,
                cmap=cmap,
                flip_vertical=flip_vertical,
            )

        bundles.append({"seed": int(key.seed), "dir": bundle_dir})

    summary = {
        "raw_root": raw_root,
        "out_root": out_root,
        "modulation": modulation,
        "anchor_snr": anchor_snr,
        "snrs": snrs,
        "requested": int(n_samples),
        "saved_bundles": len(bundles),
        "bundles": bundles,
    }

    print(f"Saved {len(bundles)} bundles to: {out_root}")
    return summary