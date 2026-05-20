from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_meta(memmap_dir: str) -> dict:
    meta_path = os.path.join(memmap_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in: {memmap_dir}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _open_memmap(memmap_dir: str, rel_path: str, dtype, shape):
    path = os.path.join(memmap_dir, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"memmap file missing: {path}")
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))


@dataclass
class MemmapImageStore:
    memmap_dir: str
    N: int
    Cimg: int
    H: int
    W: int
    C: int
    image_u8: np.memmap
    yend_u8: np.memmap
    snr_i16: np.memmap
    mod_i16: np.memmap
    stream_i32: np.memmap
    frame_idx_i16: np.ndarray
    frames_in_stream_i16: np.ndarray
    k_u8: np.memmap
    meta: dict = field(default_factory=dict)

    @staticmethod
    def open(memmap_dir: str) -> "MemmapImageStore":
        meta = _load_meta(memmap_dir)

        N = int(meta["N"])
        H = int(meta["H"])
        W = int(meta["W"])
        C = int(meta["C"])
        files = meta["files"]

        # New generic export
        if "image_u8" in files:
            Cimg = int(meta["Cimg"])
            image_u8 = _open_memmap(
                memmap_dir,
                files["image_u8"]["path"],
                np.uint8,
                files["image_u8"]["shape"],
            )

        # Backwards compatibility with old power-only export
        elif "power_u8" in files:
            Cimg = 1
            image_u8 = _open_memmap(
                memmap_dir,
                files["power_u8"]["path"],
                np.uint8,
                files["power_u8"]["shape"],
            )

        else:
            raise RuntimeError(
                f"No supported image memmap found in meta.json for {memmap_dir}. "
                f"Expected 'image_u8' or 'power_u8'."
            )

        yend_u8 = _open_memmap(
            memmap_dir, files["yend_u8"]["path"], np.uint8, files["yend_u8"]["shape"]
        )
        snr_i16 = _open_memmap(
            memmap_dir, files["snr_i16"]["path"], np.int16, files["snr_i16"]["shape"]
        )
        mod_i16 = _open_memmap(
            memmap_dir, files["mod_i16"]["path"], np.int16, files["mod_i16"]["shape"]
        )
        stream_i32 = _open_memmap(
            memmap_dir, files["stream_i32"]["path"], np.int32, files["stream_i32"]["shape"]
        )
        k_u8 = _open_memmap(
            memmap_dir, files["k_u8"]["path"], np.uint8, files["k_u8"]["shape"]
        )

        # Optional new metadata for stream-aware exports
        if "frame_idx_i16" in files:
            frame_idx_i16 = _open_memmap(
                memmap_dir,
                files["frame_idx_i16"]["path"],
                np.int16,
                files["frame_idx_i16"]["shape"],
            )
        else:
            frame_idx_i16 = np.zeros((N,), dtype=np.int16)

        if "frames_in_stream_i16" in files:
            frames_in_stream_i16 = _open_memmap(
                memmap_dir,
                files["frames_in_stream_i16"]["path"],
                np.int16,
                files["frames_in_stream_i16"]["shape"],
            )
        else:
            frames_in_stream_i16 = np.ones((N,), dtype=np.int16)

        return MemmapImageStore(
            memmap_dir=memmap_dir,
            N=N,
            Cimg=Cimg,
            H=H,
            W=W,
            C=C,
            image_u8=image_u8,
            yend_u8=yend_u8,
            snr_i16=snr_i16,
            mod_i16=mod_i16,
            stream_i32=stream_i32,
            frame_idx_i16=frame_idx_i16,
            frames_in_stream_i16=frames_in_stream_i16,
            k_u8=k_u8,
            meta=meta,
        )


def _filter_indices_by_snr(
    store: MemmapImageStore, idx: np.ndarray, snr_value: int
) -> np.ndarray:
    snr_value = int(snr_value)
    return idx[(store.snr_i16[idx] == snr_value)]


class MemmapImageFrameDataset(Dataset):
    def __init__(self, store: MemmapImageStore):
        self.store = store
        self.C = store.C
        self.class_to_id: dict[str, int] = {}
        self.id_to_class: dict[int, str] = {}

    def __len__(self) -> int:
        return self.store.N

    def __getitem__(self, i: int):
        img = np.asarray(self.store.image_u8[i])

        # image_u8 may be:
        #   old format: (H,W)
        #   new format: (Cimg,H,W)
        if img.ndim == 2:
            xb = torch.from_numpy(img).unsqueeze(0)  # -> (1,H,W)
        elif img.ndim == 3:
            xb = torch.from_numpy(img)               # -> (C,H,W)
        else:
            raise ValueError(f"Unexpected image shape at index {i}: {img.shape}")

        y = np.asarray(self.store.yend_u8[i])
        yb = torch.from_numpy(y).to(torch.float32).unsqueeze(0)
        yend = yb[0].clone()
        msk = torch.ones((1,), dtype=torch.bool)

        meta = {
            "snr": int(self.store.snr_i16[i]),
            "mod_id": int(self.store.mod_i16[i]),
            "stream_id": int(self.store.stream_i32[i]),
            "frame_idx_in_stream": int(self.store.frame_idx_i16[i]),
            "frames_in_stream": int(self.store.frames_in_stream_i16[i]),
            "K": int(self.store.k_u8[i]),
        }
        return xb, yb, yend, msk, meta


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, j: int):
        i = int(self.indices[j])
        return self.base[i]


def collate_memmap_singleframe(batch):
    x_list, yb_list, yend_list, msk_list, meta_list = zip(*batch)
    xb = torch.stack(x_list, dim=0)
    yb = torch.stack(yb_list, dim=0)
    yend = torch.stack(yend_list, dim=0)
    msk = torch.stack(msk_list, dim=0)
    meta = list(meta_list)
    return xb, yb, yend, msk, meta


def build_loaders_memmap(ds_cfg, train_cfg, rng: np.random.Generator):
    root = os.path.normpath(ds_cfg.data_path)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    for split_name, split_dir in (("train", train_dir), ("val", val_dir), ("test", test_dir)):
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Expected {split_name} directory at: {split_dir}")

    train_store = MemmapImageStore.open(train_dir)
    val_store = MemmapImageStore.open(val_dir)
    test_store = MemmapImageStore.open(test_dir)

    def _assert_store_compatible(ref: MemmapImageStore, other: MemmapImageStore, *, ref_name: str, other_name: str) -> None:
        if ref.C != other.C:
            raise RuntimeError(
                f"{ref_name}/{other_name} memmap C mismatch: {ref_name}={ref.C}, {other_name}={other.C}"
            )
        if ref.H != other.H or ref.W != other.W:
            raise RuntimeError(f"{ref_name}/{other_name} memmap spatial shape mismatch.")
        if ref.Cimg != other.Cimg:
            raise RuntimeError(f"{ref_name}/{other_name} memmap image channel count mismatch.")

    _assert_store_compatible(train_store, val_store, ref_name="train", other_name="val")
    _assert_store_compatible(train_store, test_store, ref_name="train", other_name="test")

    Cfull = int(train_store.C)

    wanted_snrs = set(int(s) for s in ds_cfg.snr_range)
    for split_name, store in (("val", val_store), ("test", test_store)):
        present_snrs = set(int(x) for x in np.unique(store.snr_i16))
        missing = wanted_snrs - present_snrs
        if missing:
            raise RuntimeError(
                f"{split_name.capitalize()} memmap missing SNRs {sorted(missing)}. Present={sorted(present_snrs)}"
            )

    stored_mods = None
    for st in (train_store, val_store, test_store):
        if isinstance(st.meta, dict) and isinstance(st.meta.get("modulation_list"), list):
            stored_mods = st.meta["modulation_list"]
            break

    if stored_mods is None:
        if Cfull != len(ds_cfg.modulation_list):
            raise RuntimeError(
                "meta.json does not contain 'modulation_list', so selected-mod remapping "
                "cannot be done safely."
            )
        stored_mods = list(ds_cfg.modulation_list)

    if len(stored_mods) != Cfull:
        raise RuntimeError(
            f"meta['modulation_list'] length mismatch: len={len(stored_mods)} but C={Cfull}."
        )

    stored_mod_to_id = {str(m): int(i) for i, m in enumerate(stored_mods)}

    selected_mods = list(ds_cfg.modulation_list)
    if getattr(ds_cfg, "use_noise", False):
        if ds_cfg.noise_label not in selected_mods:
            selected_mods = selected_mods + [ds_cfg.noise_label]

    if len(selected_mods) == 0:
        raise RuntimeError("ds_cfg.modulation_list is empty.")

    missing_mods = [m for m in selected_mods if str(m) not in stored_mod_to_id]
    if missing_mods:
        raise RuntimeError(
            f"Selected modulations not present in memmap meta: {missing_mods}\n"
            f"Memmap contains: {stored_mods}"
        )

    selected_stored_ids = np.array(
        [stored_mod_to_id[str(m)] for m in selected_mods], dtype=np.int64
    )
    selected_stored_ids_sorted = np.sort(selected_stored_ids)

    class_to_id = {str(m): int(i) for i, m in enumerate(selected_mods)}
    id_to_class = {int(i): str(m) for m, i in class_to_id.items()}

    old_to_new = {
        int(stored_mod_to_id[str(m)]): int(class_to_id[str(m)]) for m in selected_mods
    }

    def _filter_idx_by_mod(store: MemmapImageStore, idx: np.ndarray) -> np.ndarray:
        mid = store.mod_i16[idx].astype(np.int64, copy=False)
        keep = np.isin(mid, selected_stored_ids_sorted, assume_unique=False)
        return idx[keep]

    train_base = MemmapImageFrameDataset(train_store)
    val_base = MemmapImageFrameDataset(val_store)
    test_base = MemmapImageFrameDataset(test_store)

    for base_ds in (train_base, val_base, test_base):
        base_ds.class_to_id = class_to_id
        base_ds.id_to_class = id_to_class

    class RemapAndSliceDataset(Dataset):
        def __init__(
            self,
            base: MemmapImageFrameDataset,
            y_cols: np.ndarray,
            old_to_new_map: dict[int, int],
        ):
            self.base = base
            self.y_cols = np.asarray(y_cols, dtype=np.int64)
            self.old_to_new_map = {int(k): int(v) for k, v in old_to_new_map.items()}
            self.class_to_id = base.class_to_id
            self.id_to_class = base.id_to_class

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i: int):
            xb, yb, yend, msk, meta = self.base[i]

            yb = yb[:, self.y_cols]
            yend = yend[self.y_cols]

            old = int(meta.get("mod_id", -1))
            meta = dict(meta)
            meta["mod_id_full"] = old
            meta["mod_id"] = int(self.old_to_new_map.get(old, -1))

            return xb, yb, yend, msk, meta

    y_cols_in_user_order = np.array(
        [stored_mod_to_id[str(m)] for m in selected_mods], dtype=np.int64
    )

    train_base2 = RemapAndSliceDataset(train_base, y_cols_in_user_order, old_to_new)
    val_base2 = RemapAndSliceDataset(val_base, y_cols_in_user_order, old_to_new)
    test_base2 = RemapAndSliceDataset(test_base, y_cols_in_user_order, old_to_new)

    train_idx_all = _filter_idx_by_mod(
        train_store, np.arange(train_store.N, dtype=np.int64)
    )
    train_ds = IndexedDataset(train_base2, train_idx_all)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size_train,
        shuffle=True,
        num_workers=getattr(train_cfg, "num_workers", 0),
        pin_memory=getattr(train_cfg, "pin_memory", False),
        persistent_workers=(
            getattr(train_cfg, "num_workers", 0) > 0
            and getattr(train_cfg, "persistent_workers", False)
        ),
        prefetch_factor=(
            getattr(train_cfg, "prefetch_factor", 2)
            if getattr(train_cfg, "num_workers", 0) > 0
            else None
        ),
        collate_fn=collate_memmap_singleframe,
    )

    def _build_eval_loaders_by_snr(store: MemmapImageStore, base_ds: Dataset) -> dict[int, DataLoader]:
        idx_all = _filter_idx_by_mod(store, np.arange(store.N, dtype=np.int64))
        loaders_by_snr: dict[int, DataLoader] = {}
        for s in ds_cfg.snr_range:
            idx_s = _filter_indices_by_snr(store, idx_all, int(s))
            eval_ds = IndexedDataset(base_ds, idx_s)
            loaders_by_snr[int(s)] = DataLoader(
                eval_ds,
                batch_size=train_cfg.batch_size_val,
                shuffle=False,
                num_workers=getattr(train_cfg, "num_workers", 0),
                pin_memory=getattr(train_cfg, "pin_memory", False),
                persistent_workers=(
                    getattr(train_cfg, "num_workers", 0) > 0
                    and getattr(train_cfg, "persistent_workers", False)
                ),
                prefetch_factor=(
                    getattr(train_cfg, "prefetch_factor", 2)
                    if getattr(train_cfg, "num_workers", 0) > 0
                    else None
                ),
                collate_fn=collate_memmap_singleframe,
            )
        return loaders_by_snr

    val_loaders_by_snr = _build_eval_loaders_by_snr(val_store, val_base2)
    test_loaders_by_snr = _build_eval_loaders_by_snr(test_store, test_base2)

    return train_base2, train_loader, val_loaders_by_snr, test_loaders_by_snr
