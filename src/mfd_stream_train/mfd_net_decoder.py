from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn


@dataclass
class ModelConfig:
    num_classes: int
    backbone_embedding_dim: int = 512
    d_model: int = 256  # retained for config compatibility; decoder now consumes encoder embedding dim directly
    use_input_projector: bool = True  # deprecated compatibility flag; ignored
    dim_feedforward: int = 512
    dropout: float = 0.1
    temporal_arch: str = "decoder"
    tcn_num_blocks: int = 0
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    mil_hidden_dim: int = 128
    use_window_state_head: bool = False
    window_state_classes: int = 3
    nhead: int = 4
    num_decoder_layers: int = 2
    max_time_windows: int = 64
    decoder_use_memory_mlp: bool = True
    use_window_modulation_head: bool = False
    window_modulation_classes: int | None = None
    use_frame_encoder_logits: bool = False


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class QueryDecoderHead(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: Tensor, *, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        batch = memory.size(0)
        tgt = self.query.expand(batch, -1, -1)
        out = self.decoder(tgt=tgt, memory=memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(out.squeeze(1))


class DualSequenceEmbeddingEncoder(nn.Module):
    def __init__(self, num_classes: int, *, checkpoint_num_classes: int, use_embedding_head: bool, out_dim: int, pretrained: bool = False):
        super().__init__()
        MainModel = _resolve_single_dual_resnet_ctor()
        self.model = MainModel(
            num_classes=checkpoint_num_classes,
            in_channels=2,
            pretrained=pretrained,
            auto_shape=True,
            force_3ch=False,
            use_embedding_head=use_embedding_head,
            out_dim=out_dim,
        )
        self.backbone_output_dim = int(getattr(self.model, "backbone_output_dim", 512))
        self.embedding_dim = int(getattr(self.model, "embedding_dim", self.backbone_output_dim))
        self.target_num_classes = int(num_classes)
        self.logit_dim = int(checkpoint_num_classes)

    def extract_sequence_embeddings(self, images: Tensor) -> Tensor:
        if images.dim() != 4:
            raise ValueError(f"Expected images as [N,2,H,W], got {tuple(images.shape)}")
        return self.model.extract_sequence_embedding(images)

    def extract_sequence_embeddings_and_logits(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if images.dim() != 4:
            raise ValueError(f"Expected images as [N,2,H,W], got {tuple(images.shape)}")
        return self.model.extract_sequence_embedding_and_logits(images)


class LegacyDecoderSequenceClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig, frame_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = frame_encoder or DualSequenceEmbeddingEncoder(
            cfg.num_classes,
            checkpoint_num_classes=cfg.num_classes,
            use_embedding_head=False,
            out_dim=cfg.backbone_embedding_dim,
        )
        self.backbone_embedding_dim = int(getattr(self.frame_encoder, "embedding_dim", cfg.backbone_embedding_dim))
        if self.backbone_embedding_dim != int(cfg.backbone_embedding_dim):
            raise ValueError(
                f"Configured backbone_embedding_dim={cfg.backbone_embedding_dim} does not match encoder embedding dim={self.backbone_embedding_dim}."
            )

        self.use_frame_encoder_logits = bool(getattr(cfg, "use_frame_encoder_logits", False))
        self.frame_encoder_logit_dim = int(getattr(self.frame_encoder, "logit_dim", cfg.num_classes)) if self.use_frame_encoder_logits else 0
        self.temporal_dim = self.backbone_embedding_dim + self.frame_encoder_logit_dim
        self.pre_memory = (
            nn.Sequential(
                nn.LayerNorm(self.temporal_dim),
                nn.Linear(self.temporal_dim, self.temporal_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(self.temporal_dim, self.temporal_dim),
            )
            if cfg.decoder_use_memory_mlp
            else nn.Identity()
        )
        self.pos_enc = SinusoidalPositionalEncoding(self.temporal_dim, max_len=max(4096, cfg.max_time_windows + 8))
        self.decoder_head = QueryDecoderHead(
            d_model=self.temporal_dim,
            nhead=cfg.nhead,
            num_layers=cfg.num_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.temporal_dim),
            nn.Linear(self.temporal_dim, cfg.dim_feedforward),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim_feedforward, cfg.num_classes),
        )
        self.window_state_head = (
            nn.Sequential(
                nn.LayerNorm(self.temporal_dim),
                nn.Linear(self.temporal_dim, self.temporal_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(self.temporal_dim, cfg.window_state_classes),
            )
            if cfg.use_window_state_head
            else None
        )
        self.window_modulation_head = (
            nn.Sequential(
                nn.LayerNorm(self.temporal_dim),
                nn.Linear(self.temporal_dim, self.temporal_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(self.temporal_dim, int(cfg.window_modulation_classes or (cfg.num_classes + 1))),
            )
            if cfg.use_window_modulation_head
            else None
        )

    @staticmethod
    def _resolve_sequence_payload(x_seq: Tensor | dict[str, Tensor]) -> tuple[str, Any, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        if torch.is_tensor(x_seq):
            if x_seq.dim() == 5:
                return "images", x_seq, None, None, None, None
            if x_seq.dim() == 3:
                return "embeddings", x_seq, None, None, None, None
            raise ValueError(f"Tensor payload must be [B,T,2,H,W] or [B,T,D], got {tuple(x_seq.shape)}")
        if isinstance(x_seq, dict):
            frames = x_seq.get("frames")
            backbone_embeddings = x_seq.get("backbone_embeddings")
            embedding_mask = x_seq.get("embedding_mask")
            frame_encoder_logits = x_seq.get("frame_encoder_logits")
            logit_mask = x_seq.get("logit_mask")
            if frames is None or backbone_embeddings is None or embedding_mask is None:
                raise ValueError("Hybrid payload dict must contain 'frames', 'backbone_embeddings', and 'embedding_mask'.")
            return "hybrid", frames, backbone_embeddings, embedding_mask, frame_encoder_logits, logit_mask
        raise TypeError(f"Unsupported sequence payload type: {type(x_seq).__name__}")

    @staticmethod
    def _build_valid_mask(payload: Tensor, mask: Optional[Tensor]) -> tuple[int, int, Tensor]:
        if payload.dim() < 3:
            raise ValueError(f"Sequence payload rank is too small: {tuple(payload.shape)}")
        b, t = int(payload.shape[0]), int(payload.shape[1])
        if mask is None:
            valid_mask = torch.ones(b, t, dtype=torch.bool, device=payload.device)
        elif tuple(mask.shape) != (b, t):
            raise ValueError(f"Mask shape must be {(b, t)}, got {tuple(mask.shape)}")
        else:
            valid_mask = mask.bool()
        return b, t, valid_mask

    def _encode_from_images(self, frames: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor | None]:
        if frames.dim() != 5:
            raise ValueError(f"Expected frames as [B,T,2,H,W], got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape
        if c != 2:
            raise ValueError(f"Expected dual-channel frames, got channel count {c}")
        frames_bt = frames.reshape(b * t, c, h, w)
        emb_bt = torch.zeros(b * t, self.backbone_embedding_dim, device=frames.device, dtype=frames.dtype)
        logits_bt = None
        if self.use_frame_encoder_logits:
            logits_bt = torch.zeros(b * t, self.frame_encoder_logit_dim, device=frames.device, dtype=frames.dtype)
        flat_valid = valid_mask.reshape(b * t)
        if torch.any(flat_valid):
            valid_frames = frames_bt[flat_valid]
            if self.use_frame_encoder_logits:
                emb_valid, logits_valid = self.frame_encoder.extract_sequence_embeddings_and_logits(valid_frames)
                emb_bt[flat_valid] = emb_valid.to(dtype=emb_bt.dtype)
                logits_bt[flat_valid] = logits_valid.to(dtype=emb_bt.dtype)
            else:
                emb_bt[flat_valid] = self.frame_encoder.extract_sequence_embeddings(valid_frames).to(dtype=emb_bt.dtype)
        emb_bt = emb_bt.view(b, t, self.backbone_embedding_dim)
        if logits_bt is not None:
            logits_bt = logits_bt.view(b, t, self.frame_encoder_logit_dim)
        return emb_bt, logits_bt

    def _encode_hybrid(
        self,
        frames: Tensor,
        backbone_embeddings: Tensor,
        embedding_mask: Tensor,
        valid_mask: Tensor,
        frame_encoder_logits: Tensor | None = None,
        logit_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if frames.dim() != 5:
            raise ValueError(f"Expected frames as [B,T,2,H,W], got {tuple(frames.shape)}")
        b, t, _, h, w = frames.shape
        if tuple(backbone_embeddings.shape[:2]) != (b, t):
            raise ValueError(f"Backbone embeddings leading shape must be {(b, t)}, got {tuple(backbone_embeddings.shape[:2])}")
        if int(backbone_embeddings.shape[-1]) != self.backbone_embedding_dim:
            raise ValueError(
                f"Expected cached backbone embeddings with dim={self.backbone_embedding_dim}, got {int(backbone_embeddings.shape[-1])}"
            )
        if tuple(embedding_mask.shape) != (b, t):
            raise ValueError(f"embedding_mask must have shape {(b, t)}, got {tuple(embedding_mask.shape)}")

        emb_bt = backbone_embeddings.reshape(b * t, self.backbone_embedding_dim).to(dtype=frames.dtype)
        frames_bt = frames.reshape(b * t, 2, h, w)
        flat_valid = valid_mask.reshape(b * t)
        flat_embedding_mask = embedding_mask.reshape(b * t).bool()

        logits_bt = None
        flat_logit_mask = None
        if self.use_frame_encoder_logits:
            logits_bt = torch.zeros(b * t, self.frame_encoder_logit_dim, device=frames.device, dtype=frames.dtype)
            flat_logit_mask = torch.zeros(b * t, dtype=torch.bool, device=frames.device)
            if frame_encoder_logits is not None:
                if tuple(frame_encoder_logits.shape[:2]) != (b, t):
                    raise ValueError(f"Cached frame encoder logits leading shape must be {(b, t)}, got {tuple(frame_encoder_logits.shape[:2])}")
                if int(frame_encoder_logits.shape[-1]) != self.frame_encoder_logit_dim:
                    raise ValueError(
                        f"Expected cached frame encoder logits with dim={self.frame_encoder_logit_dim}, got {int(frame_encoder_logits.shape[-1])}"
                    )
                cached_logits_bt = frame_encoder_logits.reshape(b * t, self.frame_encoder_logit_dim).to(dtype=frames.dtype)
                if logit_mask is None:
                    flat_logit_mask = torch.ones(b * t, dtype=torch.bool, device=frames.device)
                else:
                    if tuple(logit_mask.shape) != (b, t):
                        raise ValueError(f"logit_mask must have shape {(b, t)}, got {tuple(logit_mask.shape)}")
                    flat_logit_mask = logit_mask.reshape(b * t).bool()
                if torch.any(flat_logit_mask):
                    logits_bt[flat_logit_mask] = cached_logits_bt[flat_logit_mask]

        need_image = flat_valid & (~flat_embedding_mask)
        if self.use_frame_encoder_logits and flat_logit_mask is not None:
            need_image = need_image | (flat_valid & (~flat_logit_mask))

        if torch.any(need_image):
            emb_bt = emb_bt.clone()
            if self.use_frame_encoder_logits:
                emb_new, logits_new = self.frame_encoder.extract_sequence_embeddings_and_logits(frames_bt[need_image])
                emb_bt[need_image] = emb_new.to(dtype=emb_bt.dtype)
                logits_bt[need_image] = logits_new.to(dtype=emb_bt.dtype)
            else:
                emb_bt[need_image] = self.frame_encoder.extract_sequence_embeddings(frames_bt[need_image]).to(dtype=emb_bt.dtype)

        emb_bt = emb_bt.view(b, t, self.backbone_embedding_dim)
        if logits_bt is not None:
            logits_bt = logits_bt.view(b, t, self.frame_encoder_logit_dim)
        return emb_bt, logits_bt

    def forward_sequence_images(
        self,
        x_seq: Tensor | dict[str, Tensor],
        mask: Optional[Tensor] = None,
        return_last: bool = False,
    ) -> dict[str, Tensor | None]:
        del return_last
        kind, payload, backbone_embeddings, embedding_mask, frame_encoder_logits, logit_mask = self._resolve_sequence_payload(x_seq)
        _, _, valid_mask = self._build_valid_mask(payload, mask)
        if kind == "images":
            seq_embeddings, seq_encoder_logits = self._encode_from_images(payload, valid_mask)
            device = payload.device
        elif kind == "embeddings":
            if int(payload.shape[-1]) != self.backbone_embedding_dim:
                raise ValueError(
                    f"Expected direct embeddings with dim={self.backbone_embedding_dim}, got {int(payload.shape[-1])}"
                )
            if self.use_frame_encoder_logits:
                raise ValueError(
                    "use_frame_encoder_logits=True requires image or hybrid inputs so frame-level ResNet logits are available."
                )
            seq_embeddings = payload
            seq_encoder_logits = None
            device = payload.device
        else:
            seq_embeddings, seq_encoder_logits = self._encode_hybrid(
                payload,
                backbone_embeddings,
                embedding_mask,
                valid_mask,
                frame_encoder_logits=frame_encoder_logits,
                logit_mask=logit_mask,
            )
            device = payload.device

        seq_embeddings = seq_embeddings * valid_mask.unsqueeze(-1).to(dtype=seq_embeddings.dtype)
        temporal_inputs = seq_embeddings
        if self.use_frame_encoder_logits:
            if seq_encoder_logits is None:
                raise RuntimeError("Frame encoder logits were requested but are unavailable for this batch.")
            seq_encoder_logits = seq_encoder_logits * valid_mask.unsqueeze(-1).to(dtype=seq_embeddings.dtype)
            temporal_inputs = torch.cat([seq_embeddings, seq_encoder_logits], dim=-1)
        temporal_seq = self.pre_memory(temporal_inputs) + temporal_inputs
        temporal_seq = temporal_seq * valid_mask.unsqueeze(-1).to(dtype=temporal_seq.dtype)
        memory_tokens = self.pos_enc(temporal_seq)
        query_state = self.decoder_head(memory_tokens, memory_key_padding_mask=~valid_mask)
        logits = self.classifier(query_state)
        window_state_logits = self.window_state_head(temporal_seq) if self.window_state_head is not None else None
        window_modulation_logits = self.window_modulation_head(temporal_seq) if self.window_modulation_head is not None else None

        zero = torch.zeros((), device=device, dtype=seq_embeddings.dtype)
        return {
            "logits": logits,
            "logits_last": None,
            "contrastive_loss": zero,
            "window_state_logits": window_state_logits,
            "window_modulation_logits": window_modulation_logits,
            "temporal_seq": temporal_seq,
            "buffer_representation": query_state,
            "frame_backbone_embeddings": seq_embeddings,
            "frame_encoder_logits": seq_encoder_logits,
            "memory_tokens": memory_tokens,
            "memory_pad_mask": ~valid_mask,
        }


# ----- checkpoint loading helpers -----

def _resolve_single_dual_resnet_ctor():
    tried: list[str] = []
    for mod_name in ("src.ResNet.ResNet", "ResNet.ResNet"):
        try:
            module = __import__(mod_name, fromlist=["MainModel"])
            return getattr(module, "MainModel")
        except Exception:
            tried.append(mod_name)
    raise ImportError(
        "Could not import a dual-image ResNet MainModel. Tried: " + ", ".join(tried)
    )


def _normalise_payload(obj: Any, path: str) -> dict[str, Any]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "model_weights", "weights"):
            if key in obj and isinstance(obj[key], dict):
                meta = dict(obj)
                meta["state_dict"] = obj[key]
                return meta
        if obj and all(isinstance(k, str) for k in obj.keys()) and all(hasattr(v, "shape") for v in obj.values()):
            return {"state_dict": obj}
        if "model" in obj and isinstance(obj["model"], nn.Module):
            meta = {k: v for k, v in obj.items() if k != "model"}
            meta["state_dict"] = obj["model"].state_dict()
            return meta
    if isinstance(obj, nn.Module):
        return {"state_dict": obj.state_dict()}
    raise ValueError(f"Unrecognised .pt format at {path!r}.")


def _load_with_fake_main(path: str, map_location: str, hint_types: list[type[nn.Module]]):
    backup_main = sys.modules.get("__main__")
    fake = types.ModuleType("__main__")
    for t in hint_types:
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


def _sanitise_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\x08", r"\b")


def _load_pt_file(path: str | Path, map_location: str = "auto") -> dict[str, Any]:
    path_str = _sanitise_path(path)
    if path_str is None:
        raise ValueError("Checkpoint path is missing.")
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        obj = torch.load(path_str, map_location=map_location, weights_only=False)
        return _normalise_payload(obj, path_str)
    except Exception:
        hint_types: list[type[nn.Module]] = []
        try:
            hint_types.append(_resolve_single_dual_resnet_ctor())
        except Exception:
            pass
        obj = _load_with_fake_main(path_str, map_location, hint_types=hint_types)
        return _normalise_payload(obj, path_str)


def _infer_checkpoint_num_classes(state_dict: dict[str, Tensor], fallback: int) -> int:
    for key in ("net.fc.weight", "model.net.fc.weight", "fc.weight", "classifier.weight"):
        value = state_dict.get(key)
        if value is not None and value.ndim == 2:
            return int(value.shape[0])
    return int(fallback)


def _infer_projection_dim(state_dict: dict[str, Tensor]) -> int | None:
    for key in ("classifier.weight", "model.classifier.weight"):
        value = state_dict.get(key)
        if value is not None and value.ndim == 2:
            return int(value.shape[1])
    for key in ("proj.0.weight", "model.proj.0.weight"):
        value = state_dict.get(key)
        if value is not None and value.ndim == 2:
            return int(value.shape[0])
    return None


def _uses_projection_head(state_dict: dict[str, Tensor]) -> bool:
    return any(k.startswith("proj.") or k.startswith("classifier.") or k.startswith("model.proj.") or k.startswith("model.classifier.") for k in state_dict.keys())


def _prepare_state_dict_for_target(state_dict: dict[str, Tensor], target_model: nn.Module) -> dict[str, Tensor]:
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}

    target_sd = target_model.state_dict()
    out: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key not in target_sd:
            continue
        if tuple(value.shape) != tuple(target_sd[key].shape):
            continue
        out[key] = value
    return out


def build_embedding_sequence_model(cfg: ModelConfig, *, dual_checkpoint: str) -> tuple[LegacyDecoderSequenceClassifier, dict[str, Any]]:
    ckpt = _load_pt_file(dual_checkpoint)
    state_dict = ckpt["state_dict"]
    checkpoint_num_classes = _infer_checkpoint_num_classes(state_dict, fallback=cfg.num_classes)
    projection_dim = _infer_projection_dim(state_dict) or cfg.backbone_embedding_dim
    use_projection_head = _uses_projection_head(state_dict)
    encoder = DualSequenceEmbeddingEncoder(
        num_classes=cfg.num_classes,
        checkpoint_num_classes=checkpoint_num_classes,
        use_embedding_head=use_projection_head,
        out_dim=projection_dim,
        pretrained=False,
    )
    prepared_sd = _prepare_state_dict_for_target(state_dict, encoder.model)
    encoder.model.load_state_dict(prepared_sd, strict=False)
    model = LegacyDecoderSequenceClassifier(cfg, frame_encoder=encoder)
    meta = {
        "dual": ckpt,
        "checkpoint_type": "legacy_decoder_sequence_embeddings_only",
        "backbone_embedding_dim": encoder.backbone_output_dim,
        "sequence_embedding_dim": encoder.embedding_dim,
        "uses_projection_head": use_projection_head,
        "frame_encoder_logit_dim": encoder.logit_dim,
    }
    return model, meta


def set_encoder_trainability(model: LegacyDecoderSequenceClassifier, *, train_backbone: bool) -> LegacyDecoderSequenceClassifier:
    for param in model.frame_encoder.parameters():
        param.requires_grad = bool(train_backbone)
    return model
