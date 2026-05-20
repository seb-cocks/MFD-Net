# phase_image_encoder_synced.py
# Matches functionality of your working Power encoder (mix SNRs, per-SNR val, CSV append,
# excl-noise checkpoint rule, mixup, label smoothing, confusion matrices, etc.)

# Standard library
import os, math
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
from src.common.checkpoint import save_minimal_checkpoint
from src.common.MultiHotStreamLoader import prepMultiHotStreamLoader
from src.common.spectrogram import (
    batched_iq_to_phase_spectrogram as _batch_phase_edge,
    to_resnet_batch as _to_resnet_batch,
)

# ---------------- Model ----------------
class PhaseImageEncoder_ResNet18(nn.Module):
    def __init__(self, out_dim=64, num_classes=2, dropout=0.2, pretrained=True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feat = base.fc.in_features  # 512
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(in_feat, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(out_dim, num_classes)
        self.out_dim = int(out_dim)
        self.num_classes = int(num_classes)

    def _prepare_image_batch(self, x_img: torch.Tensor) -> torch.Tensor:
        if x_img.ndim == 3:
            x_img = x_img.unsqueeze(1)
        if x_img.ndim != 4:
            raise ValueError(f"Expected image batch [B,C,H,W] or [B,H,W], got {tuple(x_img.shape)}")
        if x_img.dtype == torch.uint8:
            x_img = x_img.float().div_(255.0)
        else:
            x_img = x_img.float()
        if x_img.shape[1] == 1:
            x_img = x_img.repeat(1, 3, 1, 1)
        elif x_img.shape[1] == 2:
            x_img = torch.cat([x_img, x_img[:, :1]], dim=1)
        elif x_img.shape[1] != 3:
            raise ValueError(f"Expected 1, 2, or 3 channels, got {x_img.shape[1]}")
        return x_img

    def forward_backbone_features(self, x_img: torch.Tensor) -> torch.Tensor:
        img = self._prepare_image_batch(x_img)
        return self.backbone(img).flatten(1)

    @torch.no_grad()
    def extract_backbone_features(self, x_img: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        try:
            self.eval()
            return self.forward_backbone_features(x_img)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def extract_image_embedding(self, x_img: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        try:
            self.eval()
            feat = self.forward_backbone_features(x_img)
            return F.normalize(self.proj(feat), dim=1)
        finally:
            self.train(was_training)

    def classify_from_image_embedding(self, emb: torch.Tensor, return_embedding: bool = False):
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits

    def forward_images(
        self,
        x_img: torch.Tensor,
        return_embedding: bool = False,
        return_backbone_features: bool = False,
    ):
        feat = self.forward_backbone_features(x_img)
        emb = F.normalize(self.proj(feat), dim=1)
        logits = self.classifier(emb)

        outputs = [logits]
        if return_backbone_features:
            outputs.append(feat)
        if return_embedding:
            outputs.append(emb)

        if len(outputs) == 1:
            return outputs[0]
        if return_embedding and not return_backbone_features:
            return outputs[0], outputs[1]
        return tuple(outputs)

    def forward(self, x, return_embedding=False):
        # x: (B,2,N) IQ or (2,N)
        if x.ndim == 2 and x.size(0) == 2:
            x = x.unsqueeze(0)
        spec = _batch_phase_edge(
            x,
            n_fft=64,
            win_len=32,
            hop_len=8,
            normalise_noise=True,
            noise_floor_db=1.0,
            to_db=True,
            db_min=-100.0,
            db_max=20.0,
            center_dc=True,
        )
        img = _to_resnet_batch(spec, size=224)
        feat = self.backbone(img).flatten(1)
        emb = F.normalize(self.proj(feat), dim=1)
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits



# ---------- Confusion Matrix utils (single-label, C classes) ----------
@torch.no_grad()
def _collect_preds_targets(model, loader, device, *, max_batches=None):
    """
    Returns (y_true[N], y_pred[N]) on CPU. Optionally limit #batches.
    """
    model.eval().to(device)
    ys, ps = [], []
    for bi, (xb, yb) in enumerate(loader):
        if (max_batches is not None) and (bi >= max_batches):
            break
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).long()
        logits = model(xb)
        pred = logits.argmax(1)
        ys.append(yb.detach().cpu())
        ps.append(pred.detach().cpu())
    return torch.cat(ys, 0), torch.cat(ps, 0)

def _confusion_from_preds(y_true_cpu: torch.Tensor, y_pred_cpu: torch.Tensor, num_classes: int):
    c = num_classes
    idx = y_true_cpu.to(torch.int64) * c + y_pred_cpu.to(torch.int64)
    cm = torch.bincount(idx, minlength=c * c).reshape(c, c)
    return cm

def _save_confusion_csv_png(cm: torch.Tensor, class_names, out_csv, out_png, normalize="row"):
    """
    normalize: None | 'row' (recall) | 'col' (precision) | 'all'
    """
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    # CSV (raw counts)
    df = pd.DataFrame(cm.numpy(), index=class_names, columns=class_names)
    df.to_csv(out_csv, index=True)

    # Normalised view for the figure
    M = cm.float()
    if normalize == "row":
        denom = M.sum(1, keepdim=True).clamp_min(1)
        M = M / denom
        title = "Confusion (row-norm: recall)"
    elif normalize == "col":
        denom = M.sum(0, keepdim=True).clamp_min(1)
        M = M / denom
        title = "Confusion (col-norm: precision)"
    elif normalize == "all":
        s = M.sum().clamp_min(1)
        M = M / s
        title = "Confusion (global-norm)"
    else:
        title = "Confusion (counts)"

    plt.figure(figsize=(8, 7))
    plt.imshow(M.numpy(), interpolation="nearest", aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    C = M.shape[0]
    step = 1 if C <= 12 else max(1, math.ceil(C / 16))
    for i in range(0, C, step):
        for j in range(0, C, step):
            v = M[i, j].item()
            plt.text(j, i, f"{v:.2f}" if v < 1000 else f"{int(v)}",
                     ha="center", va="center", fontsize=8,
                     color="white" if v > M.max()/2 else "black")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ---------------- Utils ----------------
def prep_batch(xb, yb, device):
    if xb.ndim == 4 and xb.shape[1] not in (1, 3) and xb.shape[-1] in (1, 3):
        xb = xb.permute(0, 3, 1, 2).contiguous()
    xb = xb.float()
    return xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

def _amp_cfg_for_device(device: str):
    if device == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = amp.GradScaler(enabled=(amp_dtype is torch.float16))
        ctx = lambda: amp.autocast(device_type="cuda", dtype=amp_dtype)
    elif device == "cpu":
        scaler = amp.GradScaler(enabled=False)
        ctx = nullcontext
    else:
        scaler = amp.GradScaler(enabled=False)
        ctx = lambda: amp.autocast(device_type=device)
    return ctx, scaler

@torch.no_grad()
def evaluate(model, loader, device, snr):
    """Standard accuracy evaluation (all classes)."""
    model.eval().to(device)
    ctx, _ = _amp_cfg_for_device(device)
    correct = total = 0
    pbar = tqdm(loader, desc=f"Evaluating {snr}", leave=False)
    for xb, yb in pbar:
        xb, yb = prep_batch(xb, yb, device)
        with ctx():
            logits = model(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        acc = correct / max(1, total)
        pbar.set_postfix(acc=f"{acc:.4f}")
    return correct / max(1, total)

@torch.no_grad()
def eval_excl_noise(model, loader, device, noise_id, snr):
    """Accuracy excluding noise samples."""
    model.eval().to(device)
    c = t = 0
    pbar = tqdm(loader, desc=f"Evaluating {snr} (excl noise)", leave=False)
    for xb, yb in pbar:
        m = (yb != noise_id)
        if not m.any():
            continue
        logits = model(xb.to(device))[m]
        pred = logits.argmax(1).cpu()
        y = yb[m].cpu()
        c += (pred == y).sum().item()
        t += len(y)
        acc = c / max(1, t)
        pbar.set_postfix(acc=f"{acc:.4f}")
    return c / max(1, t)

def mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y, None, None, None
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[perm], y, (perm, lam), lam, perm


def train(
    model,
    train_loader,
    val_loaders_by_snr,  # dict[int, DataLoader]
    label_to_id,
    epochs=5,
    lr=1e-3,
    wd=1e-4,
    ckpt_dir="checkpoints",
    ckpt_file="best.pt",
    csv_file="out\\metrics.csv",
    # --- confusion control ---
    save_confusion=True,
    cm_every=1,
    cm_train_max_batches=200,
    cm_val_max_batches=None,
    cm_norm_mode="row",
    out_dir_cm="out\\phase_encoder\\cm\\",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    ctx, scaler = _amp_cfg_for_device(device)

    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(csv_file) or ".").mkdir(parents=True, exist_ok=True)

    snr_keys = sorted(val_loaders_by_snr.keys())
    best_metric = float("-inf")   # checkpoint on excl-noise macro
    global_step = 0

    # CSV columns
    base_cols = ["epoch", "time", "global_step", "lr", "train_loss", "val_macro", "metric"]
    snr_cols  = [f"val_snr_{s:+d}dB" for s in snr_keys]
    all_cols  = base_cols + snr_cols

    def _append_metrics_row(ep, train_loss, accs_dict, macro, metric, lr_now):
        row = {
            "epoch": ep,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "global_step": global_step,
            "lr": lr_now,
            "train_loss": float(train_loss),
            "val_macro": float(macro),
            "metric": float(metric),
        }
        for s in snr_keys:
            row[f"val_snr_{s:+d}dB"] = float(accs_dict.get(s, np.nan))
        df = pd.DataFrame([row], columns=all_cols)
        write_header = not os.path.exists(csv_file)
        df.to_csv(csv_file, mode="a", header=write_header, index=False)

    for ep in range(1, epochs + 1):
        print(f"[{datetime.now().time().strftime('%H:%M:%S')}] - Epoch {ep} - Started Training")
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}", leave=False)

        for xb, yb in pbar:
            xb, yb = prep_batch(xb, yb, device)
            opt.zero_grad(set_to_none=True)

            with ctx():
                use_mixup = ep >= 2
                if use_mixup:
                    xb_in, yb_in, _, lam, perm = mixup(xb, yb, alpha=0.2)
                else:
                    xb_in, yb_in, lam, perm = xb, yb, None, None
                logits = model(xb_in)
                if lam is None:
                    loss = loss_fn(logits, yb_in)
                else:
                    loss = lam * loss_fn(logits, yb_in) + (1 - lam) * loss_fn(logits, yb_in[perm])

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            bs = yb.size(0)
            running += loss.item() * bs
            n_seen += bs
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        sched.step()

        print(f"[{datetime.now().time().strftime('%H:%M:%S')}] - Epoch {ep} - Started Testing")

        # per-SNR validation + macro
        train_loss = running / max(1, n_seen)
        accs = {s: evaluate(model, vldr, device, s) for s, vldr in val_loaders_by_snr.items()}
        macro = float(np.mean(list(accs.values()))) if accs else 0.0

        # Confusion matrices (optional)
        if save_confusion and (ep % cm_every == 0):
            id_to_label = {v: k for k, v in label_to_id.items()}
            class_names = [id_to_label[i] for i in range(len(label_to_id))]

            # TRAIN
            y_true, y_pred = _collect_preds_targets(model, train_loader, device, max_batches=cm_train_max_batches)
            cm = _confusion_from_preds(y_true, y_pred, num_classes=len(class_names))

            ep_tag = f"ep{ep:02d}"
            os.makedirs(out_dir_cm, exist_ok=True)
            snr_str = "".join(str(s) for s in snr_keys)

            train_csv = os.path.join(out_dir_cm, f"train_cm_{ep_tag}_snr{snr_str}.csv")
            train_png = os.path.join(out_dir_cm, f"train_cm_{ep_tag}_snr{snr_str}.png")
            _save_confusion_csv_png(cm, class_names, train_csv, train_png, normalize=cm_norm_mode)

            # VAL per SNR
            for s in snr_keys:
                vldr = val_loaders_by_snr[s]
                y_true_v, y_pred_v = _collect_preds_targets(model, vldr, device, max_batches=cm_val_max_batches)
                cm_v = _confusion_from_preds(y_true_v, y_pred_v, num_classes=len(class_names))
                val_csv = os.path.join(out_dir_cm, f"val_cm_{ep_tag}_{s:+d}dB_snr{snr_str}.csv")
                val_png = os.path.join(out_dir_cm, f"val_cm_{ep_tag}_{s:+d}dB_snr{snr_str}.png")
                _save_confusion_csv_png(cm_v, class_names, val_csv, val_png, normalize=cm_norm_mode)

        print(f"[{datetime.now().time().strftime('%H:%M:%S')}] - Epoch {ep} - Finished Testing")

        # excl-noise metric for checkpointing
        metric_accs = {s: eval_excl_noise(model, vldr, device, label_to_id["noise"], s)
                       for s, vldr in val_loaders_by_snr.items()}
        metric = float(np.mean(list(metric_accs.values()))) if metric_accs else 0.0

        tqdm.write(
            "epoch {:02d}  loss {:.4f}  ".format(ep, train_loss)
            + "  ".join([f"val@{s:+d}dB {accs[s]:.4f}" for s in snr_keys])
            + f"  macro {macro:.4f}  metric {metric:.4f}"
        )

        # append CSV row now
        _append_metrics_row(
            ep=ep,
            train_loss=train_loss,
            accs_dict=accs,
            macro=macro,
            metric=metric,
            lr_now=float(opt.param_groups[0]["lr"]),
        )

        # checkpoint on excl-noise macro
        if metric > best_metric:
            best_metric = metric
            save_minimal_checkpoint(
                os.path.join(ckpt_dir, ckpt_file),
                model=model,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
                label_to_id=label_to_id,
                epoch=ep,
                global_step=global_step,
                best_val=best_metric,
                last_train_loss=train_loss,
                extra_cfg={
                    "lr": opt.param_groups[0]["lr"],
                    "batch_size": train_loader.batch_size,
                    "selection_rule": "excl_noise_macro",
                },
            )

    return model


# ---------------- Single run (mirrors your Power main) ----------------
if __name__ == "__main__":
    # base config (match power file)
    epochs = 40
    learning_rate = 2e-4
    weight_decay = 1e-4

    batch_size = 16
    window_count = 500

    window_size = 1000
    window_step_size = 1000
    seed = 42
    rng = np.random.default_rng(seed)

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
        "p1",
        "p2",
        "p3",
        "p4",
    ]
    out_dim = 128  # mirror your power run

    # Choose SNRs for this run (mixed train + per-SNR val)
    # snr_range = [0]
    # snr_range = [3]
    # snr_range = [6]
    # snr_range = [-3]
    # snr_range = [-6]
    snr_range = [-6, -3, 0, 3, 6]
    val_snr_range = snr_range

    snr_str = "".join(str(s) for s in snr_range)
    suffix_info = f"_e{epochs}_d{out_dim}_wc{window_count}_lr{learning_rate}_snr{snr_str}"

    data_path = "data\\"
    train_data_path = data_path + "train\\"
    test_data_path  = data_path + "test\\"

    encoder_dir = "phase_encoder\\"
    ckpt_dir = "checkpoints\\" + encoder_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_file = "best_" + suffix_info + ".pt"

    csv_dir = "out\\" + encoder_dir
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = csv_dir + "metrics_" + suffix_info + ".csv"
    if os.path.exists(csv_file):
        os.remove(csv_file)

    img_cm_dir = csv_dir + "cm_" + suffix_info + "\\"

    stride = 16
    train_noise_pos_ratio = 0.5
    val_noise_pos_ratio = 0.05
    threshold = 0.02

    # ---- TRAIN: mix SNRs uniformly ----
    train_ds, train_loader = prepMultiHotStreamLoader(
        modulation_list=modulation_list,
        snr_range=snr_range,
        filepath=train_data_path,
        batch_size=batch_size,
        history_length=1,
        window_count=window_count,
        window_size=window_size,
        window_step_size=window_step_size,
        k_ratio={1: 1.0, 2: 0.0, 3: 0.0},
        seed=seed,
        rng=rng,
        split="train",
        stride=stride,
        threshold_time_frac=threshold,      # 2% overlap
        add_noise_class=True,
        noise_label="noise",
        balance_mode="global",
        noise_pos_ratio=train_noise_pos_ratio,
        resample_seed=None,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=None,
        image_mode=False,
        return_keys=False,
        pin_memory=True,
        resnet_mode=True,              # phase path builds its own specs
        drop_ambiguous=True,
        snr_levels=tuple(snr_range),   # allow override to any of these SNRs
        balance_snr=True,              # uniform SNR mix
        progress_steps=False,
        summary_prints=True,
    )

    # ---- VAL: one locked loader per SNR ----
    val_loaders_by_snr = {}
    for s in val_snr_range:
        _, vldr = prepMultiHotStreamLoader(
            modulation_list=modulation_list,
            snr_range=[s],
            filepath=test_data_path,
            batch_size=batch_size,
            history_length=1,
            window_count=window_count,
            window_size=window_size,
            window_step_size=window_step_size,
            k_ratio={1: 1.0, 2: 0.0, 3: 0.0},
            seed=seed + 1,
            rng=np.random.default_rng(seed + 1),
            split="val",
            stride=stride,
            threshold_time_frac=threshold,  # 2% overlap
            add_noise_class=True,
            noise_label="noise",
            balance_mode="global",
            noise_pos_ratio=val_noise_pos_ratio,
            resample_seed=None,
            shuffle=False,
            num_workers=4,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=None,
            image_mode=False,
            return_keys=False,
            pin_memory=True,
            resnet_mode=True,
            drop_ambiguous=True,
            snr_levels=(s,),            # lock to this SNR
            balance_snr=False,
            progress_steps=False,
            summary_prints=True,
        )
        val_loaders_by_snr[s] = vldr

    # Model + labels
    num_classes = len(modulation_list) + 1
    model = PhaseImageEncoder_ResNet18(
        out_dim=out_dim, num_classes=num_classes, pretrained=True
    )
    label_to_id = {lbl: i for i, lbl in enumerate(modulation_list + ["noise"])}

    print(f"[{datetime.now().time().strftime('%H:%M:%S')}] - Started")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    model = train(
        model=model,
        train_loader=train_loader,
        val_loaders_by_snr=val_loaders_by_snr,
        label_to_id=label_to_id,
        epochs=epochs,
        lr=learning_rate,
        wd=weight_decay,
        ckpt_dir=ckpt_dir,
        ckpt_file=ckpt_file,
        csv_file=csv_file,
        out_dir_cm=img_cm_dir,
    )

    print(f"[{datetime.now().time().strftime('%H:%M:%S')}] - Finished")
