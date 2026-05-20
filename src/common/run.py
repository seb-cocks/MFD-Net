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

from src.common.config import DatasetConfig, TrainConfig, build_model_and_amp, write_run_config_txt
from src.common.confusion_matrix import confusion_matrix_frames, save_confusion_matrix_csv
from src.common.metrics import make_run_dir, MetricsWriter
from src.common.memmap import build_loaders_memmap
from src.common.training import train_one_epoch
from src.common.evaluation import validate_across_snrs
from src.common.checkpoint import save_best_checkpoint

#==========================================================#

def run_experiment(model: nn.Module, ds_cfg: DatasetConfig, train_cfg: TrainConfig):
    print(f"Started setup at {datetime.now().time().strftime('%H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(train_cfg.seed)

    # OLD
    # train_ds, train_loader, val_loaders_by_snr = build_loaders_memmap(
    #     ds_cfg, train_cfg, rng
    # )

    # NEW
    train_ds, train_loader, val_loaders_by_snr, test_loaders_by_snr = build_loaders_memmap(ds_cfg, train_cfg, rng)

    class_count = len(train_ds.class_to_id)
    id_to_class = train_ds.id_to_class
    model, ctx, scaler = build_model_and_amp(
        model=model,
        device=device,
    )

    opt = train_cfg.optimizer
    sched = train_cfg.scheduler
    loss_fn_ce = train_cfg.criterion

    snr_keys = sorted(val_loaders_by_snr.keys())
    suffix_info = (
        f"_{train_cfg.backbone}_singleframe_e{train_cfg.max_epochs}"
        f"_wc{ds_cfg.window_count}_h{ds_cfg.history_length}_lr{train_cfg.learning_rate}_snr{ds_cfg.snr_str}"
    )

    run_dir, ts = make_run_dir(train_cfg.out_dir, suffix_info)

    cfg_path = write_run_config_txt(
        run_dir,
        ds_cfg,
        train_cfg,
        extra={
            "timestamp": ts,
            "device": str(device),
            "class_count": class_count,
            "snr_keys": snr_keys,
            "test_snr_keys": sorted(test_loaders_by_snr.keys()),
        },
    )
    print(f"[run] configs saved to: {cfg_path}")
    print(f"[run] outputs will be written to: {run_dir}")

    writer = MetricsWriter(run_dir, suffix_info, snr_keys)
    ckpt_file = "best_" + suffix_info + ".pt"

    best_macro = float("-inf")
    best_epoch = -1
    stale = 0

    print(f"[{datetime.now().time().strftime('%H:%M:%S')}] - Finished setup")

    for ep in range(1, train_cfg.max_epochs + 1):
        print(f"[{datetime.now().time().strftime('%H:%M:%S')}] - Epoch {ep} started")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            opt=opt,
            loss_fn_ce=loss_fn_ce,
            ctx=ctx,
            scaler=scaler,
            update_every=train_cfg.update_every,
            ema_alpha=train_cfg.ema_alpha,
            ep=ep,
            max_epochs=train_cfg.max_epochs,
        )
        # sched.step()

        print(
            f"[{datetime.now().time().strftime('%H:%M:%S')}] - Epoch {ep} finished training and started validation"
        )
        print(f"[{datetime.now().time().strftime('%H:%M:%S')}] - Loss: {train_loss}")

        stats_by_snr, val_loss_macro, acc_macro = validate_across_snrs(
            model=model,
            val_loaders_by_snr=val_loaders_by_snr,
            device=device,
            loss_fn_ce=loss_fn_ce,
            ep=ep,
        )

        # sched.step(acc_macro)  # <===  NEW
        sched.step()

        writer.append(ep, train_loss, stats_by_snr, val_loss_macro, acc_macro)

        # Periodic confusion matrix snapshots
        if train_cfg.cm_every_epochs and train_cfg.cm_every_epochs > 0:
            if (ep % train_cfg.cm_every_epochs) == 0:
                cm = np.zeros((class_count, class_count), dtype=np.int64)
                for s in snr_keys:
                    cm += confusion_matrix_frames(
                        model,
                        val_loaders_by_snr[s],
                        device,
                        class_count,
                        prefer="power",
                    )
                cm_path = save_confusion_matrix_csv(
                    cm,
                    run_dir=run_dir,
                    filename=f"cm_best_ep{ep:02d}.csv",  # must be str
                    id_to_class=id_to_class,
                )
                print(f"[cm] saved periodic confusion matrix: {cm_path}")

        score = acc_macro
        if score > best_macro + train_cfg.min_delta:
            best_macro = score
            best_epoch = ep
            stale = 0
            save_best_checkpoint(
                run_dir=run_dir,  # NEW
                ckpt_file=ckpt_file,
                model=model,
                opt=opt,
                ep=ep,
                stats_by_snr=stats_by_snr,
                val_loss_macro=val_loss_macro,
                acc_macro=acc_macro,
                snr_keys=snr_keys,
                backbone=train_cfg.backbone,
            )
            cm_loader = (
                val_loaders_by_snr[snr_keys[0]]
                if len(snr_keys) == 1
                else (
                    val_loaders_by_snr[0]
                    if 0 in val_loaders_by_snr
                    else val_loaders_by_snr[snr_keys[0]]
                )
            )
            cm = confusion_matrix_frames(
                model=model,
                loader=cm_loader,
                device=device,
                num_classes=class_count,
                prefer="power",
            )
            cm_path = save_confusion_matrix_csv(
                cm,
                run_dir=run_dir,
                filename=f"cm_best_ep{ep:02d}.csv",
                id_to_class=id_to_class,
            )
            print(f"[cm] saved best confusion matrix: {cm_path}")
            print(f"[earlystop] new best acc(macro)={best_macro:.4f} at epoch {ep}")
        else:
            stale += 1
            print(
                f"[earlystop] no improvement for {stale}/{train_cfg.patience} epochs (best={best_macro:.4f} at ep {best_epoch})"
            )
            if stale >= train_cfg.patience:
                print(f"[earlystop] stopping at epoch {ep} (best at {best_epoch})")
                break

    best_ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_file)
    if os.path.isfile(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[test] loaded best checkpoint from: {best_ckpt_path}")
    else:
        print(f"[test] best checkpoint not found at {best_ckpt_path}; evaluating current model state instead")

    test_stats_by_snr, test_loss_macro, test_acc_macro = validate_across_snrs(
        model=model,
        val_loaders_by_snr=test_loaders_by_snr,
        device=device,
        loss_fn_ce=loss_fn_ce,
        ep=best_epoch if best_epoch > 0 else 0,
    )
    test_summary = {
        "best_epoch": int(best_epoch),
        "best_val_accuracy_macro": float(best_macro),
        "test_loss_macro": float(test_loss_macro),
        "test_accuracy_macro": float(test_acc_macro),
        "test_by_snr": {
            f"{int(s):+d}dB": {
                "loss": float(test_stats_by_snr[s]["loss"]),
                "accuracy": float(test_stats_by_snr[s]["accuracy"]),
            }
            for s in sorted(test_stats_by_snr.keys())
        },
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    test_cm = np.zeros((class_count, class_count), dtype=np.int64)
    for s in sorted(test_loaders_by_snr.keys()):
        test_cm += confusion_matrix_frames(
            model,
            test_loaders_by_snr[s],
            device,
            class_count,
            prefer="power",
        )
    test_cm_path = save_confusion_matrix_csv(
        test_cm,
        run_dir=run_dir,
        filename="cm_test_best.csv",
        id_to_class=id_to_class,
    )
    print(
        f"[test] macro_loss={test_loss_macro:.4f} | macro_acc={test_acc_macro:.4f} | cm={test_cm_path}"
    )

    print(f"Finished at {datetime.now().time().strftime('%H:%M:%S')}")
