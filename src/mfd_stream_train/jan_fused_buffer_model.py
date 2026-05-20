from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.ResNet.ResNet import MainModel as DualImageEncoder_ResNet18
from src.ResNet.ResNet_phase import PhaseImageEncoder_ResNet18
from src.ResNet.ResNet_power import PowerImageEncoder_ResNet18


@dataclass
class ModelConfig:
    num_classes: int
    d_model: int = 256
    nhead: int = 4
    num_decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_time_windows: int = 64
    add_aux_logit_token: bool = True
    aux_logit_proj_dim: int = 64
    use_frame_encoder_logits: bool = True
    modality_mode: str = "power_phase"  # power | phase | power_phase | dual
    power_embedding_dim: int = 64
    phase_embedding_dim: int = 64
    dual_embedding_dim: int = 64


class AlignmentHead(nn.Module):
    def __init__(self, in_dim: int, d_model: int, modality_id: int, num_modalities: int = 2):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.modality_embed = nn.Embedding(num_modalities, d_model)
        self.register_buffer("modality_id_buf", torch.tensor(modality_id, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        return x + self.modality_embed(self.modality_id_buf)


class CrossModalFusion(nn.Module):
    def __init__(self, d_model: int, nhead: int = 2, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(128, 2 * d_model),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

    def forward(self, step_tokens: Tensor) -> Tensor:
        x = self.encoder(step_tokens)
        fused = x.mean(dim=1)
        return self.out(fused)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)


class MemoryToQueryDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: Tensor, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        b = memory.size(0)
        tgt = self.query.expand(b, -1, -1)
        out = self.dec(tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(out.squeeze(1))


class JanFusedBufferModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        power_encoder: nn.Module | None = None,
        phase_encoder: nn.Module | None = None,
        dual_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.modality_mode = str(cfg.modality_mode).lower()
        self.power_encoder = power_encoder
        self.phase_encoder = phase_encoder
        self.dual_encoder = dual_encoder

        valid_modes = {"power", "phase", "power_phase", "dual"}
        if self.modality_mode not in valid_modes:
            raise ValueError(f"Unsupported modality_mode={cfg.modality_mode!r}. Expected one of {sorted(valid_modes)}")

        if self.modality_mode == "power_phase":
            if self.power_encoder is None or self.phase_encoder is None:
                raise ValueError("power_phase mode requires both power_encoder and phase_encoder.")
            self.align_power = AlignmentHead(cfg.power_embedding_dim, cfg.d_model, modality_id=0, num_modalities=2)
            self.align_phase = AlignmentHead(cfg.phase_embedding_dim, cfg.d_model, modality_id=1, num_modalities=2)
        elif self.modality_mode == "power":
            if self.power_encoder is None:
                raise ValueError("power mode requires power_encoder.")
            self.align_power = AlignmentHead(cfg.power_embedding_dim, cfg.d_model, modality_id=0, num_modalities=1)
            self.align_phase = None
        elif self.modality_mode == "phase":
            if self.phase_encoder is None:
                raise ValueError("phase mode requires phase_encoder.")
            self.align_power = None
            self.align_phase = AlignmentHead(cfg.phase_embedding_dim, cfg.d_model, modality_id=0, num_modalities=1)
        else:  # dual
            if self.dual_encoder is None:
                raise ValueError("dual mode requires dual_encoder.")
            self.align_dual = AlignmentHead(cfg.dual_embedding_dim, cfg.d_model, modality_id=0, num_modalities=1)
            self.align_power = None
            self.align_phase = None

        use_aux = bool(cfg.add_aux_logit_token and cfg.use_frame_encoder_logits)
        self.use_aux_logit_token = use_aux
        if use_aux:
            if self.modality_mode == "power_phase":
                self.aux_p = nn.Linear(cfg.num_classes, cfg.aux_logit_proj_dim)
                self.aux_q = nn.Linear(cfg.num_classes, cfg.aux_logit_proj_dim)
                self.aux_to_d = nn.Sequential(
                    nn.Linear(2 * cfg.aux_logit_proj_dim, cfg.d_model),
                    nn.LayerNorm(cfg.d_model),
                )
            else:
                self.aux_single = nn.Linear(cfg.num_classes, cfg.aux_logit_proj_dim)
                self.aux_to_d = nn.Sequential(
                    nn.Linear(cfg.aux_logit_proj_dim, cfg.d_model),
                    nn.LayerNorm(cfg.d_model),
                )

        self.fusion = CrossModalFusion(cfg.d_model, nhead=2, dropout=cfg.dropout)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=max(32, cfg.max_time_windows + 1))
        self.decoder = MemoryToQueryDecoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_decoder_layers,
            dim_ff=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
        self.buffer_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.num_classes),
        )
        self.last_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.num_classes),
        )

    def _encode_frames(self, frames: Tensor):
        b, t, c, h, w = frames.shape
        if c != 2:
            raise ValueError(f"Expected frames shaped [B,T,2,H,W], got {tuple(frames.shape)}")

        if self.modality_mode == "power_phase":
            power = frames[:, :, 0:1].reshape(b * t, 1, h, w)
            phase = frames[:, :, 1:2].reshape(b * t, 1, h, w)
            logits_p_bt, emb_p_bt = self.power_encoder.forward_images(power, return_embedding=True)
            logits_q_bt, emb_q_bt = self.phase_encoder.forward_images(phase, return_embedding=True)
            return {
                "power": {"logits": logits_p_bt, "emb": emb_p_bt},
                "phase": {"logits": logits_q_bt, "emb": emb_q_bt},
            }

        if self.modality_mode == "power":
            power = frames[:, :, 0:1].reshape(b * t, 1, h, w)
            logits_bt, emb_bt = self.power_encoder.forward_images(power, return_embedding=True)
            return {"power": {"logits": logits_bt, "emb": emb_bt}}

        if self.modality_mode == "phase":
            phase = frames[:, :, 1:2].reshape(b * t, 1, h, w)
            logits_bt, emb_bt = self.phase_encoder.forward_images(phase, return_embedding=True)
            return {"phase": {"logits": logits_bt, "emb": emb_bt}}

        dual = frames.reshape(b * t, 2, h, w)
        try:
            logits_bt, emb_bt = self.dual_encoder.forward_images(dual, return_head_embedding=True)
        except TypeError:
            logits_bt, _, emb_bt = self.dual_encoder.forward_images(
                dual,
                return_backbone_embedding=True,
                return_head_embedding=True,
            )
        return {"dual": {"logits": logits_bt, "emb": emb_bt}}

    def _reshape_frame_outputs(self, enc: dict[str, dict[str, Tensor]], b: int, t: int) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        frame_features_by_modality: dict[str, Tensor] = {}
        frame_logits_by_modality: dict[str, Tensor] = {}
        combined_parts: list[Tensor] = []

        for name in ("power", "phase", "dual"):
            item = enc.get(name)
            if item is None:
                continue
            emb = item["emb"].view(b, t, -1)
            logits = item["logits"].view(b, t, -1)
            frame_features_by_modality[name] = emb
            frame_logits_by_modality[name] = logits
            combined_parts.append(emb)

        if not combined_parts:
            raise RuntimeError("No frame features were produced by the encoder.")
        combined = combined_parts[0] if len(combined_parts) == 1 else torch.cat(combined_parts, dim=-1)
        return frame_features_by_modality, frame_logits_by_modality, combined

    def extract_frame_features(self, xb, *, mask: Optional[Tensor] = None, include_logits: bool = True):
        if isinstance(xb, dict):
            frames = xb.get("frames")
        else:
            frames = xb
        if frames is None:
            raise ValueError("JanFusedBufferModel requires image frames in the batch payload.")
        b, t = frames.shape[:2]
        if mask is None:
            valid_mask = torch.ones((b, t), dtype=torch.bool, device=frames.device)
        else:
            valid_mask = mask.to(device=frames.device, dtype=torch.bool)

        enc = self._encode_frames(frames)
        frame_features_by_modality, frame_logits_by_modality, combined = self._reshape_frame_outputs(enc, b, t)
        out = {
            "frame_features": combined,
            "frame_features_by_modality": frame_features_by_modality,
            "valid_mask": valid_mask,
        }
        if include_logits:
            out["frame_logits_by_modality"] = frame_logits_by_modality
        return out

    def forward_sequence_images(self, xb, *, mask: Optional[Tensor] = None, return_last: bool = True):
        if isinstance(xb, dict):
            frames = xb.get("frames")
        else:
            frames = xb
        if frames is None:
            raise ValueError("JanFusedBufferModel requires image frames in the batch payload.")
        b, t = frames.shape[:2]
        if mask is None:
            mask = torch.ones((b, t), dtype=torch.bool, device=frames.device)
        else:
            mask = mask.to(device=frames.device, dtype=torch.bool)

        enc = self._encode_frames(frames)
        frame_features_by_modality, frame_logits_by_modality, combined_frame_features = self._reshape_frame_outputs(enc, b, t)
        toks = []
        enc_logits: dict[str, Tensor | None] = {"power": None, "phase": None, "dual": None}

        if self.modality_mode == "power_phase":
            logits_p_bt = enc["power"]["logits"]
            emb_p_bt = enc["power"]["emb"]
            logits_q_bt = enc["phase"]["logits"]
            emb_q_bt = enc["phase"]["emb"]
            tok_p_bt = self.align_power(emb_p_bt)
            tok_q_bt = self.align_phase(emb_q_bt)
            toks.extend([tok_p_bt.unsqueeze(1), tok_q_bt.unsqueeze(1)])
            enc_logits["power"] = logits_p_bt.view(b, t, -1)
            enc_logits["phase"] = logits_q_bt.view(b, t, -1)
            if self.use_aux_logit_token:
                aux_p_bt = self.aux_p(logits_p_bt)
                aux_q_bt = self.aux_q(logits_q_bt)
                aux_bt = self.aux_to_d(torch.cat([aux_p_bt, aux_q_bt], dim=-1)).unsqueeze(1)
                toks.append(aux_bt)
        elif self.modality_mode == "power":
            logits_bt = enc["power"]["logits"]
            emb_bt = enc["power"]["emb"]
            toks.append(self.align_power(emb_bt).unsqueeze(1))
            enc_logits["power"] = logits_bt.view(b, t, -1)
            if self.use_aux_logit_token:
                aux_bt = self.aux_to_d(self.aux_single(logits_bt)).unsqueeze(1)
                toks.append(aux_bt)
        elif self.modality_mode == "phase":
            logits_bt = enc["phase"]["logits"]
            emb_bt = enc["phase"]["emb"]
            toks.append(self.align_phase(emb_bt).unsqueeze(1))
            enc_logits["phase"] = logits_bt.view(b, t, -1)
            if self.use_aux_logit_token:
                aux_bt = self.aux_to_d(self.aux_single(logits_bt)).unsqueeze(1)
                toks.append(aux_bt)
        else:
            logits_bt = enc["dual"]["logits"]
            emb_bt = enc["dual"]["emb"]
            toks.append(self.align_dual(emb_bt).unsqueeze(1))
            enc_logits["dual"] = logits_bt.view(b, t, -1)
            if self.use_aux_logit_token:
                aux_bt = self.aux_to_d(self.aux_single(logits_bt)).unsqueeze(1)
                toks.append(aux_bt)

        step_tokens_bt = torch.cat(toks, dim=1)
        fused_bt = self.fusion(step_tokens_bt)
        fused_seq = fused_bt.view(b, t, -1)

        memory = self.pos_enc(fused_seq)
        memory_pad_mask = ~mask
        valid = mask.float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (memory * valid.unsqueeze(-1)).sum(dim=1) / denom
        logits_buffer = self.buffer_head(pooled)

        logits_last = None
        if return_last:
            z = self.decoder(memory, memory_key_padding_mask=memory_pad_mask)
            logits_last = self.last_head(z)

        return {
            "logits": logits_buffer,
            "logits_buffer": logits_buffer,
            "logits_last": logits_last,
            "temporal_seq": memory,
            "memory_pad_mask": memory_pad_mask,
            "enc_logits": enc_logits,
            "frame_features": combined_frame_features,
            "frame_features_by_modality": frame_features_by_modality,
            "frame_logits_by_modality": frame_logits_by_modality,
        }


def _strip_module_prefix(sd: dict) -> dict:
    if sd and all(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _normalise_payload(obj, path):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj

        # checkpoints saved by common.checkpoint.save_best_checkpoint
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            meta = dict(obj)
            meta["state_dict"] = meta.pop("model_state_dict")
            return meta

        # other common aliases
        for k in ("model", "network", "weights"):
            if k in obj and isinstance(obj[k], dict):
                meta = dict(obj)
                meta["state_dict"] = meta.pop(k)
                return meta

        # raw state_dict only
        if obj and all(isinstance(k, str) for k in obj.keys()) and all(
            hasattr(v, "shape") for v in obj.values()
        ):
            return {"state_dict": obj}

        # full payload with model object
        if "model" in obj and isinstance(obj["model"], nn.Module):
            sd = obj["model"].state_dict()
            meta = {k: v for k, v in obj.items() if k != "model"}
            meta["state_dict"] = sd
            return meta

    if isinstance(obj, nn.Module):
        return {"state_dict": obj.state_dict()}

    raise ValueError(f"Unrecognised .pt format at {path!r}.")

def _load_with_fake_main(path, map_location, hint_types):
    backup_main = sys.modules.get("__main__")
    fake = types.ModuleType("__main__")
    for t in (hint_types or []):
        try:
            setattr(fake, t.__name__, t)
        except Exception:
            pass
    sys.modules["__main__"] = fake
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    finally:
        if backup_main is not None:
            sys.modules["__main__"] = backup_main
        else:
            sys.modules.pop("__main__", None)


def _load_pt_file(path, map_location="auto", hint_types=None):
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
        return _normalise_payload(obj, path)
    except Exception:
        obj = _load_with_fake_main(path, map_location, hint_types=hint_types)
        return _normalise_payload(obj, path)


def load_image_encoder_from_pt(encoder_ctor, pt_path: str, *, ctor_kwargs: dict | None = None, strict: str | bool = "auto"):
    ctor_kwargs = dict(ctor_kwargs or {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = encoder_ctor(**ctor_kwargs).to(device)
    ckpt = _load_pt_file(pt_path, map_location=device, hint_types=[encoder_ctor])
    sd = _strip_module_prefix(dict(ckpt["state_dict"]))

    current_sd = enc.state_dict()
    filtered_sd = {}
    skipped = []
    for k, v in sd.items():
        if k in current_sd and current_sd[k].shape == v.shape:
            filtered_sd[k] = v
        else:
            skipped.append(k)

    missing, unexpected = enc.load_state_dict(filtered_sd, strict=False)
    if skipped:
        print(f"[load_image_encoder_from_pt] skipped mismatched keys from {pt_path}:")
        for k in skipped:
            print(f"  - {k}")
    if missing:
        print(f"[load_image_encoder_from_pt] missing keys initialised randomly:")
        for k in missing:
            print(f"  - {k}")
    if unexpected:
        print(f"[load_image_encoder_from_pt] unexpected keys ignored:")
        for k in unexpected:
            print(f"  - {k}")

    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    meta["_source_file"] = str(pt_path)
    return enc, meta


def set_encoder_trainability(encoder: nn.Module | None, *, train_backbone: bool, train_proj: bool = True, train_classifier: bool = True):
    if encoder is None:
        return None
    for _, p in encoder.named_parameters():
        p.requires_grad = bool(train_backbone)
    if hasattr(encoder, "proj") and train_proj:
        for p in encoder.proj.parameters():
            p.requires_grad = True
    if hasattr(encoder, "classifier") and train_classifier:
        for p in encoder.classifier.parameters():
            p.requires_grad = True
    if hasattr(encoder, "net") and hasattr(encoder.net, "fc") and train_classifier:
        for p in encoder.net.fc.parameters():
            p.requires_grad = True
    return encoder


def build_model(
    model_cfg: ModelConfig,
    *,
    modality_mode: str = "power_phase",
    power_checkpoint: str | None = None,
    phase_checkpoint: str | None = None,
    dual_checkpoint: str | None = None,
):
    mode = str(modality_mode).lower()
    model_cfg.modality_mode = mode

    power_encoder = None
    phase_encoder = None
    dual_encoder = None
    meta = {}

    if mode == "power_phase":
        if not power_checkpoint or not phase_checkpoint:
            raise ValueError("power_phase mode requires both power_checkpoint and phase_checkpoint.")
        power_encoder, power_meta = load_image_encoder_from_pt(
            PowerImageEncoder_ResNet18,
            power_checkpoint,
            ctor_kwargs={
                "num_classes": model_cfg.num_classes,
                "out_dim": model_cfg.power_embedding_dim,
            },
            strict="auto",
        )
        phase_encoder, phase_meta = load_image_encoder_from_pt(
            PhaseImageEncoder_ResNet18,
            phase_checkpoint,
            ctor_kwargs={
                "num_classes": model_cfg.num_classes,
                "out_dim": model_cfg.phase_embedding_dim,
            },
            strict="auto",
        )
        meta = {"power": power_meta, "phase": phase_meta}
    elif mode == "power":
        if not power_checkpoint:
            raise ValueError("power mode requires power_checkpoint.")
        power_encoder, power_meta = load_image_encoder_from_pt(
            PowerImageEncoder_ResNet18,
            power_checkpoint,
            ctor_kwargs={
                "num_classes": model_cfg.num_classes,
                "out_dim": model_cfg.power_embedding_dim,
            },
            strict="auto",
        )
        meta = {"power": power_meta}
    elif mode == "phase":
        if not phase_checkpoint:
            raise ValueError("phase mode requires phase_checkpoint.")
        phase_encoder, phase_meta = load_image_encoder_from_pt(
            PhaseImageEncoder_ResNet18,
            phase_checkpoint,
            ctor_kwargs={
                "num_classes": model_cfg.num_classes,
                "out_dim": model_cfg.phase_embedding_dim,
            },
            strict="auto",
        )
        meta = {"phase": phase_meta}
    elif mode == "dual":
        if not dual_checkpoint:
            raise ValueError("dual mode requires dual_checkpoint.")
        dual_encoder, dual_meta = load_image_encoder_from_pt(
            DualImageEncoder_ResNet18,
            dual_checkpoint,
            ctor_kwargs={
                "num_classes": model_cfg.num_classes,
                "in_channels": 2,
                "force_3ch": False,
                "use_embedding_head": True,
                "out_dim": model_cfg.dual_embedding_dim,
            },
            strict="auto",
        )
        meta = {"dual": dual_meta}
    else:
        raise ValueError(f"Unsupported modality_mode={modality_mode!r}. Expected power, phase, power_phase, or dual.")

    model = JanFusedBufferModel(
        model_cfg,
        power_encoder=power_encoder,
        phase_encoder=phase_encoder,
        dual_encoder=dual_encoder,
    )
    return model, meta
