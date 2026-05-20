from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.common.config import DatasetConfig, TrainConfig
from src.common.run import run_experiment
from src.ResNet.ResNet import MainModel

# ==========================================================#

DATA_PATH = r"C:\Users\scocks\BitBucket\spectral_labs\mfd-net\data"


def main():
    print("|=====| ResNet Single-Frame |=====|")
    modulation_list = [
        "unmodulated",
        "lfm_up",
        "lfm_down",
        "dlfm_up_down",
        "NLFM_up",
        "NLFM_down",
        "mlfm",
        "dlfm_down_up",
        "bpsk",
        "qpsk",
        "barker_11",
        "barker_13",
        "polyphase",
    ]

    # Choose one of: power | phase | dual
    spectro = "phase"

    try:
        from src.mfd_stream_existing_iq.config import build_class_subset_tag

        subset_tag = build_class_subset_tag(modulation_list)
    except Exception:
        subset_tag = "__".join(sorted(modulation_list))

    data_root_parent = DATA_PATH

    data_path = os.path.join(data_root_parent, subset_tag, f"{spectro}_singleframe")
    print(f"[singleframe] spectro={spectro} | data_path={data_path}")

    ds_cfg = DatasetConfig(
        modulation_list=modulation_list,
        noise_label="noise",
        snr_range=[6, 3, 0, -3, -6],
        data_path=data_path,
        use_noise=True,
        window_count=1,
        window_size=1000,
        window_step_size=1000,
        k_ratio={1: 1.0},
        history_length=1,
        stride=1,
        threshold_time_frac=0.02,
        train_noise_pos_ratio=0.5,
        val_noise_pos_ratio=0.05,
    )

    model_name = f"resnet_{spectro}"

    train_cfg = TrainConfig(
        seed=42,
        max_epochs=50,
        batch_size_train=32,
        batch_size_val=64,
        learning_rate=6e-4,
        patience=10,
        min_delta=0.001,
        update_every=10,
        ema_alpha=0.1,
        backbone=None,
        pretrained_pt=None,
        out_dir=f"out\\{model_name}\\",
        ckpt_dir=f"checkpoints\\{model_name}\\",
        cm_every_epochs=5,
        criterion=nn.CrossEntropyLoss(label_smoothing=0.0),  # was 0.05
    )

    num_classes = len(modulation_list) + (1 if ds_cfg.use_noise else 0)

    # model = MainModel(
    #     num_classes=num_classes,
    #     pretrained=True,
    #     auto_shape=True,  # safe if memmap returns (B,224,224) etc.
    #     use_embedding_head=False,  # simplest: pure resnet classifier
    # )

    is_dual = spectro == "dual"
    use_embedding_head = True
    embedding_out_dim = 256  # set to 128 or 256 when the ResNet should learn the sequence embedding internally

    model = MainModel(
        num_classes=num_classes,
        in_channels=2 if is_dual else 3,
        pretrained=True,
        auto_shape=True,
        force_3ch=not is_dual,
        use_embedding_head=use_embedding_head,
        out_dim=embedding_out_dim,
    )

    train_cfg.optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=1e-2,
    )

    # new
    train_cfg.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        train_cfg.optimizer,
        T_max=train_cfg.max_epochs,
    )

    run_experiment(model=model, ds_cfg=ds_cfg, train_cfg=train_cfg)


if __name__ == "__main__":
    main()
