import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MainModel(nn.Module):
    """
    ResNet18 image classifier (memmap-compatible).

    Important for the stream / noise-embedding cache path:
    - the *backbone embedding* is taken AFTER avgpool/flatten and BEFORE the
      classifier FC. For the standard ResNet18 path this is a 512-d vector.
    - when ``use_embedding_head=True`` the model also learns a projection head
      inside the ResNet itself and exposes that projected representation as the
      main embedding for downstream sequence models.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 3,
        pretrained: bool = True,
        auto_shape: bool = True,
        force_3ch: bool = True,
        use_embedding_head: bool = False,
        out_dim: int = 128,
        dropout: float = 0.2,
        l2norm_embedding: bool = False,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.auto_shape = bool(auto_shape)
        self.force_3ch = bool(force_3ch)
        self.use_embedding_head = bool(use_embedding_head)
        self.use_projection_head = self.use_embedding_head
        self.l2norm_embedding = bool(l2norm_embedding)

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Adapt first conv if input channels are not 3
        if self.in_channels != 3:
            old_conv = base.conv1
            new_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            with torch.no_grad():
                if pretrained:
                    # Use mean over RGB weights as a reasonable initialisation.
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)

                    if self.in_channels == 1:
                        new_conv.weight.copy_(mean_w)
                    elif self.in_channels == 2:
                        new_conv.weight[:, 0:1].copy_(mean_w)
                        new_conv.weight[:, 1:2].copy_(mean_w)
                    else:
                        repeat_factor = math.ceil(self.in_channels / 3)
                        w = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, : self.in_channels]
                        w *= 3.0 / self.in_channels
                        new_conv.weight.copy_(w)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            base.conv1 = new_conv

        if not self.use_projection_head:
            base.fc = nn.Linear(base.fc.in_features, num_classes)
            self.net = base
            self.backbone_output_dim = int(base.fc.in_features)
            self.embedding_dim = int(base.fc.in_features)
            self.projected_embedding_dim = int(base.fc.in_features)
        else:
            in_feat = int(base.fc.in_features)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.proj = nn.Sequential(
                nn.Linear(in_feat, int(out_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(int(out_dim), int(num_classes))
            self.backbone_output_dim = in_feat
            self.embedding_dim = int(out_dim)
            self.projected_embedding_dim = int(out_dim)

    def _ensure_nchw(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,W) -> (B,1,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # (B,H,W,C) -> (B,C,H,W)
        if x.ndim == 4 and x.shape[1] not in (1, 2, 3) and x.shape[-1] in (1, 2, 3):
            x = x.permute(0, 3, 1, 2).contiguous()

        # Optional fallback only when model expects 3ch.
        if self.force_3ch and self.in_channels == 3 and x.ndim == 4:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] == 2:
                x = torch.cat([x, x.mean(dim=1, keepdim=True)], dim=1)

        return x

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.auto_shape:
            x = self._ensure_nchw(x)

        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
        return x

    def _forward_backbone_embedding_from_prepared(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_projection_head:
            net = self.net
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)
            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3(x)
            x = net.layer4(x)
            x = net.avgpool(x)
            return torch.flatten(x, 1)

        return self.backbone(x).flatten(1)

    def project_backbone_embedding(self, backbone_emb: torch.Tensor) -> torch.Tensor:
        if not self.use_projection_head:
            emb = backbone_emb
        else:
            emb = self.proj(backbone_emb)
        if self.l2norm_embedding:
            emb = F.normalize(emb, dim=1)
        return emb
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for extracting the downstream image embedding used by the sequence model.
        This keeps the interface aligned with other future CNN backbones.
        """
        x = self._prepare_input(x)
        backbone_emb = self._forward_backbone_embedding_from_prepared(x)
        return self.project_backbone_embedding(backbone_emb)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        try:
            self.eval()
            return self.forward_features(x)
        finally:
            self.train(was_training)


    @torch.no_grad()
    def extract_backbone_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the representation immediately after the ResNet backbone
        global-average-pool and flatten step, i.e. BEFORE the projection/classifier.
        """
        was_training = self.training
        try:
            self.eval()
            x = self._prepare_input(x)
            return self._forward_backbone_embedding_from_prepared(x)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def extract_sequence_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the representation intended for downstream sequence models.
        This is the projected head embedding when the projection head is enabled,
        otherwise it is the raw 512-d backbone embedding.
        """
        was_training = self.training
        try:
            self.eval()
            x = self._prepare_input(x)
            backbone_emb = self._forward_backbone_embedding_from_prepared(x)
            return self.project_backbone_embedding(backbone_emb)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def extract_sequence_embedding_and_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the downstream sequence embedding together with the classifier logits
        produced by the ResNet head for the same image batch.
        """
        was_training = self.training
        try:
            self.eval()
            x = self._prepare_input(x)
            backbone_emb = self._forward_backbone_embedding_from_prepared(x)
            logits, sequence_emb = self.classify_from_backbone_embedding(
                backbone_emb,
                return_head_embedding=True,
            )
            return sequence_emb, logits
        finally:
            self.train(was_training)

    def classify_from_backbone_embedding(
        self,
        backbone_emb: torch.Tensor,
        *,
        return_head_embedding: bool = False,
    ):
        """
        Run only the cheap post-backbone part of the network from a cached
        backbone embedding.
        """
        if not self.use_projection_head:
            logits = self.net.fc(backbone_emb)
            if return_head_embedding:
                return logits, backbone_emb
            return logits

        emb = self.project_backbone_embedding(backbone_emb)
        logits = self.classifier(emb)
        if return_head_embedding:
            return logits, emb
        return logits

    def classify_from_sequence_embedding(self, sequence_emb: torch.Tensor) -> torch.Tensor:
        if sequence_emb.ndim == 1:
            sequence_emb = sequence_emb.unsqueeze(0)
        if sequence_emb.ndim != 2:
            raise ValueError(
                f"Expected sequence embeddings as [B,{self.embedding_dim}], got {tuple(sequence_emb.shape)}"
            )
        if int(sequence_emb.shape[-1]) != int(self.embedding_dim):
            raise ValueError(
                f"Expected sequence embeddings with dim={self.embedding_dim}, got {int(sequence_emb.shape[-1])}"
            )
        sequence_emb = sequence_emb.float()
        if not self.use_projection_head:
            return self.net.fc(sequence_emb)
        return self.classifier(sequence_emb)

    def forward_images(
        self,
        x: torch.Tensor,
        *,
        return_backbone_embedding: bool = False,
        return_head_embedding: bool = False,
        return_sequence_embedding: bool = False,
    ):
        x = self._prepare_input(x)
        backbone_emb = self._forward_backbone_embedding_from_prepared(x)
        logits, sequence_emb = self.classify_from_backbone_embedding(
            backbone_emb,
            return_head_embedding=True,
        )

        outputs = [logits]
        if return_backbone_embedding:
            outputs.append(backbone_emb)
        if return_head_embedding or return_sequence_embedding:
            outputs.append(sequence_emb)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def forward_backbone_embeddings(
        self,
        backbone_emb: torch.Tensor,
        *,
        return_head_embedding: bool = False,
        return_sequence_embedding: bool = False,
    ):
        if backbone_emb.ndim == 1:
            backbone_emb = backbone_emb.unsqueeze(0)
        if backbone_emb.ndim != 2:
            raise ValueError(
                f"Expected cached backbone embeddings as [B,{self.backbone_output_dim}], got {tuple(backbone_emb.shape)}"
            )
        if int(backbone_emb.shape[-1]) != int(self.backbone_output_dim):
            raise ValueError(
                f"Expected cached backbone embeddings with dim={self.backbone_output_dim}, got {int(backbone_emb.shape[-1])}"
            )
        backbone_emb = backbone_emb.float()
        return self.classify_from_backbone_embedding(
            backbone_emb,
            return_head_embedding=(return_head_embedding or return_sequence_embedding),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_backbone_embedding: bool = False,
        return_head_embedding: bool = False,
        return_sequence_embedding: bool = False,
    ):
        return self.forward_images(
            x,
            return_backbone_embedding=return_backbone_embedding,
            return_head_embedding=return_head_embedding,
            return_sequence_embedding=return_sequence_embedding,
        )
