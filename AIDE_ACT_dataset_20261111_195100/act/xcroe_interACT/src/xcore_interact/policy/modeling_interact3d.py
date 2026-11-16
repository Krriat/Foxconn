from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .configuration_interact3d import InterACT3DConfig


def _transformer_encoder_layer(config: InterACT3DConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.n_heads,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


def _transformer_decoder(config: InterACT3DConfig, layers: int) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=config.d_model,
        nhead=config.n_heads,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=layers, norm=nn.LayerNorm(config.d_model))


class SharedRgbBackbone(nn.Module):
    def __init__(self, config: InterACT3DConfig):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError("torchvision is required by the InterACT RGB branch") from exc
        weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained_backbone else None
        network = resnet18(weights=weights)
        self.features = nn.Sequential(*list(network.children())[:-2])
        self.projection = nn.Conv2d(512, config.d_model, kernel_size=1)
        self.view_embedding = nn.Parameter(torch.empty(config.camera_count, 1, config.d_model))
        self.image_pos = nn.Parameter(torch.empty(config.camera_count, 49, config.d_model))
        nn.init.normal_(self.view_embedding, std=0.02)
        nn.init.normal_(self.image_pos, std=0.02)
        if config.freeze_backbone:
            for parameter in self.features.parameters():
                parameter.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: Tensor) -> Tensor:
        # images: B,V,3,H,W in RGB [0,1]
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError(f"images must be [B,3,3,H,W], got {tuple(images.shape)}")
        batch, views = images.shape[:2]
        flat = images.reshape(batch * views, 3, images.shape[-2], images.shape[-1])
        flat = (flat - self.mean) / self.std
        feature = self.projection(self.features(flat))
        feature = feature.flatten(2).transpose(1, 2)
        tokens = feature.shape[1]
        if tokens != 49:
            positional = F.interpolate(
                self.image_pos.transpose(1, 2), size=tokens, mode="linear", align_corners=False
            ).transpose(1, 2)
        else:
            positional = self.image_pos
        feature = feature.reshape(batch, views, tokens, -1)
        feature = feature + self.view_embedding.unsqueeze(0) + positional.unsqueeze(0)
        return feature.flatten(1, 2)


class PointCloudEncoder(nn.Module):
    """PointNet-style per-point encoder retaining geometry tokens."""

    def __init__(self, config: InterACT3DConfig):
        super().__init__()
        self.point_tokens = config.point_tokens
        self.features = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, config.d_model, 1),
        )
        self.position = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, config.d_model)
        )

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        if points.ndim != 3 or points.shape[-1] != 6:
            raise ValueError(f"point cloud must be [B,N,6], got {tuple(points.shape)}")
        if points.shape[1] < self.point_tokens:
            raise ValueError("point cloud has fewer rows than configured point_tokens")
        # Recorded points are randomly permuted, so an even fixed subsample has
        # broad spatial coverage and avoids an expensive FPS kernel on Orin.
        indices = torch.linspace(
            0, points.shape[1] - 1, self.point_tokens, device=points.device
        ).long()
        points = points.index_select(1, indices)
        mask = (~torch.isfinite(points).all(dim=-1)) | (points[..., 2] <= 0)
        clean = torch.nan_to_num(points)
        clean[..., 3:] = clean[..., 3:] / 255.0
        tokens = self.features(clean.transpose(1, 2)).transpose(1, 2)
        tokens = tokens + self.position(clean[..., :3])
        tokens = tokens.masked_fill(mask.unsqueeze(-1), 0)
        return tokens, mask


class SegmentedInteractionBlock(nn.Module):
    """InterACT segment-local attention followed by cross-segment CLS attention."""

    def __init__(self, config: InterACT3DConfig):
        super().__init__()
        self.local = _transformer_encoder_layer(config)
        self.cross = _transformer_encoder_layer(config)

    def forward(
        self,
        tokens: Tensor,
        segment_ids: Tensor,
        cls_indices: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        local_mask = segment_ids[:, None] != segment_ids[None, :]
        tokens = self.local(tokens, src_mask=local_mask, src_key_padding_mask=padding_mask)
        cls = tokens.index_select(1, cls_indices)
        cls = self.cross(cls)
        result = tokens.clone()
        result[:, cls_indices] = cls
        return result


class InterACTEncoder(nn.Module):
    def __init__(self, config: InterACT3DConfig):
        super().__init__()
        self.use_point_cloud = config.use_point_cloud
        self.state_layout = config.state_layout
        self.left_state = nn.Linear(1, config.d_model)
        self.right_state = nn.Linear(1, config.d_model)
        self.state_slot = nn.Parameter(torch.empty(2, 8, config.d_model))
        self.cls = nn.Parameter(torch.empty(4 if config.use_point_cloud else 3, 1, config.d_model))
        self.blocks = nn.ModuleList(
            SegmentedInteractionBlock(config) for _ in range(config.encoder_blocks)
        )
        self.norm = nn.LayerNorm(config.d_model)
        nn.init.normal_(self.state_slot, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(
        self, state: Tensor, rgb_tokens: Tensor,
        geometry_tokens: Tensor | None = None, geometry_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, list[slice]]:
        if state.ndim != 2 or state.shape[-1] != 16:
            raise ValueError(f"state must be [B,16], got {tuple(state.shape)}")
        batch = state.shape[0]
        if self.state_layout == "dual_arm_suction":
            left_state = torch.cat((state[:, :7], state[:, 14:15]), dim=-1)
            right_state = torch.cat((state[:, 7:14], state[:, 15:16]), dim=-1)
        else:
            left_state, right_state = state[:, :8], state[:, 8:]
        left = self.left_state(left_state.unsqueeze(-1)) + self.state_slot[0]
        right = self.right_state(right_state.unsqueeze(-1)) + self.state_slot[1]
        segments = [left, right, rgb_tokens]
        if self.use_point_cloud:
            if geometry_tokens is None or geometry_mask is None:
                raise ValueError("this checkpoint requires head point-cloud tokens")
            segments.append(geometry_tokens)
        pieces: list[Tensor] = []
        masks: list[Tensor] = []
        ids: list[Tensor] = []
        slices: list[slice] = []
        cls_indices: list[int] = []
        offset = 0
        for index, segment in enumerate(segments):
            cls_indices.append(offset)
            piece = torch.cat((self.cls[index].expand(batch, -1, -1), segment), dim=1)
            pieces.append(piece)
            if index == 3:
                mask = torch.cat(
                    (torch.zeros(batch, 1, dtype=torch.bool, device=state.device), geometry_mask),
                    dim=1,
                )
            else:
                mask = torch.zeros(batch, piece.shape[1], dtype=torch.bool, device=state.device)
            masks.append(mask)
            ids.append(torch.full((piece.shape[1],), index, device=state.device, dtype=torch.long))
            slices.append(slice(offset, offset + piece.shape[1]))
            offset += piece.shape[1]
        tokens = torch.cat(pieces, dim=1)
        padding_mask = torch.cat(masks, dim=1)
        segment_ids = torch.cat(ids)
        cls_tensor = torch.tensor(cls_indices, dtype=torch.long, device=state.device)
        for block in self.blocks:
            tokens = block(tokens, segment_ids, cls_tensor, padding_mask)
        return self.norm(tokens), padding_mask, slices


class MultiArmDecoder(nn.Module):
    def __init__(self, config: InterACT3DConfig):
        super().__init__()
        self.chunk_size = config.chunk_size
        self.query = nn.Parameter(torch.empty(2, config.chunk_size, config.d_model))
        nn.init.normal_(self.query, std=0.02)
        self.left_pre = _transformer_decoder(config, config.pre_decoder_layers)
        self.right_pre = _transformer_decoder(config, config.pre_decoder_layers)
        self.sync = nn.MultiheadAttention(
            config.d_model, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.sync_norm = nn.LayerNorm(config.d_model)
        self.left_post = _transformer_decoder(config, config.post_decoder_layers)
        self.right_post = _transformer_decoder(config, config.post_decoder_layers)
        self.left_head = nn.Linear(config.d_model, config.arm_action_dim)
        self.right_head = nn.Linear(config.d_model, config.arm_action_dim)

    def forward(self, memory: Tensor, memory_mask: Tensor, slices: list[slice]) -> Tensor:
        batch = memory.shape[0]
        left_query = self.query[0].expand(batch, -1, -1)
        right_query = self.query[1].expand(batch, -1, -1)
        # Each arm sees its own state and the visual segments present in this model.
        visual_memory = [memory[:, part] for part in slices[2:]]
        visual_masks = [memory_mask[:, part] for part in slices[2:]]
        left_memory = torch.cat([memory[:, slices[0]], *visual_memory], 1)
        right_memory = torch.cat([memory[:, slices[1]], *visual_memory], 1)
        left_mask = torch.cat([memory_mask[:, slices[0]], *visual_masks], 1)
        right_mask = torch.cat([memory_mask[:, slices[1]], *visual_masks], 1)
        left = self.left_pre(left_query, left_memory, memory_key_padding_mask=left_mask)
        right = self.right_pre(right_query, right_memory, memory_key_padding_mask=right_mask)
        paired = torch.cat((left, right), dim=1)
        synchronized, _ = self.sync(paired, paired, paired, need_weights=False)
        paired = self.sync_norm(paired + synchronized)
        left, right = paired[:, : self.chunk_size], paired[:, self.chunk_size :]
        left = self.left_post(left, memory, memory_key_padding_mask=memory_mask)
        right = self.right_post(right, memory, memory_key_padding_mask=memory_mask)
        return torch.cat((self.left_head(left), self.right_head(right)), dim=-1)


class InterACT3DPolicy(nn.Module):
    """Bimanual RGB InterACT, optionally augmented with a PointNet segment."""

    def __init__(self, config: InterACT3DConfig):
        super().__init__()
        self.config = config
        self.rgb = SharedRgbBackbone(config)
        self.geometry = PointCloudEncoder(config) if config.use_point_cloud else None
        self.encoder = InterACTEncoder(config)
        self.decoder = MultiArmDecoder(config)

    def predict_action_chunk(
        self, images: Tensor, points: Tensor | None = None, state: Tensor | None = None,
    ) -> Tensor:
        if state is None:
            raise ValueError("state is required")
        rgb_tokens = self.rgb(images)
        geometry_tokens = geometry_mask = None
        if self.geometry is not None:
            if points is None:
                raise ValueError("this checkpoint requires head point-cloud input")
            geometry_tokens, geometry_mask = self.geometry(points)
        memory, memory_mask, slices = self.encoder(
            state, rgb_tokens, geometry_tokens, geometry_mask
        )
        prediction = self.decoder(memory, memory_mask, slices)
        if self.config.state_layout == "dual_arm_suction":
            # Decoder heads are left_7j+suction, right_7j+suction. Return the
            # original dataset order so normalization and loss retain their meaning.
            prediction = torch.cat((prediction[..., :7], prediction[..., 8:15],
                                    prediction[..., 7:8], prediction[..., 15:16]), dim=-1)
        return prediction

    def forward(
        self,
        images: Tensor,
        points: Tensor | None = None,
        state: Tensor | None = None,
        actions: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> dict[str, Tensor]:
        prediction = self.predict_action_chunk(images, points, state)
        output: dict[str, Tensor] = {"action": prediction}
        if actions is not None:
            if actions.shape != prediction.shape:
                raise ValueError(
                    f"actions shape {tuple(actions.shape)} != prediction {tuple(prediction.shape)}"
                )
            absolute_error = (prediction - actions).abs()
            if action_is_pad is not None:
                if action_is_pad.shape != actions.shape[:2]:
                    raise ValueError("action_is_pad must be [B,chunk_size]")
                valid = (~action_is_pad).unsqueeze(-1).to(absolute_error.dtype)
                loss = (absolute_error * valid).sum() / (valid.sum() * actions.shape[-1]).clamp_min(1)
            else:
                loss = absolute_error.mean()
            output["loss"] = loss
            output["l1_loss"] = loss.detach()
        return output

    def save_pretrained(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(target)
        state = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        try:
            from safetensors.torch import save_file

            save_file(state, target / "model.safetensors")
        except ImportError:
            torch.save(state, target / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> "InterACT3DPolicy":
        directory = Path(directory)
        config = InterACT3DConfig.from_pretrained(directory)
        # Checkpoints already contain the complete backbone. Do not trigger an
        # ImageNet download merely to construct the module for loading.
        used_pretrained_initialization = config.pretrained_backbone
        config.pretrained_backbone = False
        model = cls(config)
        model.config.pretrained_backbone = used_pretrained_initialization
        safe_path = directory / "model.safetensors"
        if safe_path.is_file():
            from safetensors.torch import load_file

            state = load_file(safe_path, device=str(map_location))
        else:
            state = torch.load(
                directory / "pytorch_model.bin", map_location=map_location, weights_only=True
            )
        model.load_state_dict(state)
        return model
