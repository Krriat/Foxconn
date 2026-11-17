from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import roi_align


@dataclass(frozen=True)
class ROIModelConfig:
    num_cameras: int = 2
    state_dim: int = 8
    action_dim: int = 8
    chunk_size: int = 100
    image_size: int = 224
    global_grid: tuple[int, int] = (4, 4)
    roi_grid: tuple[int, int] = (6, 6)
    roi_expand: float = 1.6
    hidden_dim: int = 256
    backbone_width: int = 64
    nheads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    posterior_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    latent_dim: int = 32
    use_vae: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ROIModelConfig":
        data = dict(value)
        for key in ("global_grid", "roi_grid"):
            if key in data:
                data[key] = tuple(int(item) for item in data[key])
        return cls(**data)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_group_count(output_channels), output_channels),
            )
            if stride != 1 or input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.skip(inputs)
        values = F.relu(self.norm1(self.conv1(inputs)), inplace=True)
        values = self.norm2(self.conv2(values))
        return F.relu(values + residual, inplace=True)


class CompactResNet(nn.Module):
    """ResNet-18-shaped backbone used by the exported ROI-ACT checkpoint."""

    def __init__(self, width: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(_group_count(width), width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(width, width, stride=1)
        self.layer2 = self._make_layer(width, width * 2, stride=2)
        self.layer3 = self._make_layer(width * 2, width * 4, stride=2)
        self.layer4 = self._make_layer(width * 4, width * 8, stride=2)

    @staticmethod
    def _make_layer(
        input_channels: int, output_channels: int, stride: int
    ) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(input_channels, output_channels, stride=stride),
            ResidualBlock(output_channels, output_channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        values = self.stem(inputs)
        values = self.layer1(values)
        values = self.layer2(values)
        values = self.layer3(values)
        return self.layer4(values)


def _expand_boxes(boxes: Tensor, factor: float) -> Tensor:
    coordinates = boxes[..., :4].clamp(0.0, 1.0)
    x1, y1, x2, y2 = coordinates.unbind(dim=-1)
    centers_x = (x1 + x2) * 0.5
    centers_y = (y1 + y2) * 0.5
    widths = (x2 - x1).clamp_min(1e-4) * factor
    heights = (y2 - y1).clamp_min(1e-4) * factor
    return torch.stack(
        (
            (centers_x - widths * 0.5).clamp(0.0, 1.0),
            (centers_y - heights * 0.5).clamp(0.0, 1.0),
            (centers_x + widths * 0.5).clamp(0.0, 1.0),
            (centers_y + heights * 0.5).clamp(0.0, 1.0),
        ),
        dim=-1,
    )


class ROIVisualEncoder(nn.Module):
    def __init__(self, config: ROIModelConfig):
        super().__init__()
        hidden = config.hidden_dim
        self.num_cameras = config.num_cameras
        self.global_grid = config.global_grid
        self.roi_grid = config.roi_grid
        self.roi_expand = config.roi_expand
        self.backbone = CompactResNet(config.backbone_width)
        self.feature_projection = nn.Conv2d(
            config.backbone_width * 8, hidden, kernel_size=1
        )
        self.box_mlp = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.global_pos = nn.Parameter(
            torch.empty(1, 1, config.global_grid[0] * config.global_grid[1], hidden)
        )
        self.roi_pos = nn.Parameter(
            torch.empty(1, 1, config.roi_grid[0] * config.roi_grid[1], hidden)
        )
        self.camera_embed = nn.Parameter(
            torch.empty(1, config.num_cameras, 1, hidden)
        )
        self.type_embed = nn.Parameter(torch.empty(3, hidden))

    def _roi_features(self, features: Tensor, boxes: Tensor) -> Tensor:
        batch_cameras, _, height, width = features.shape
        expanded = _expand_boxes(boxes.reshape(batch_cameras, 5), self.roi_expand)
        valid = boxes.reshape(batch_cameras, 5)[:, 4] > 0.5
        full_frame = torch.tensor(
            [0.0, 0.0, 1.0, 1.0], device=features.device, dtype=features.dtype
        ).expand_as(expanded)
        expanded = torch.where(valid[:, None], expanded, full_frame)
        scaled = expanded * torch.tensor(
            [width - 1, height - 1, width - 1, height - 1],
            device=features.device,
            dtype=features.dtype,
        )
        indices = torch.arange(
            batch_cameras, device=features.device, dtype=features.dtype
        )[:, None]
        regions = torch.cat((indices, scaled), dim=1)
        return roi_align(
            features,
            regions,
            output_size=self.roi_grid,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )

    def forward(self, images: Tensor, boxes: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"images must have shape [B,C,3,H,W], got {images.shape}")
        batch, cameras, channels, height, width = images.shape
        if cameras != self.num_cameras or channels != 3:
            raise ValueError(
                f"expected {self.num_cameras} RGB cameras, got {images.shape}"
            )
        if boxes.shape != (batch, cameras, 5):
            raise ValueError(
                f"boxes must have shape {(batch, cameras, 5)}, got {boxes.shape}"
            )

        features = self.feature_projection(
            self.backbone(images.reshape(batch * cameras, channels, height, width))
        )
        global_features = F.adaptive_avg_pool2d(features, self.global_grid)
        roi_features = self._roi_features(features, boxes)

        def tokens(values: Tensor) -> Tensor:
            return values.flatten(2).transpose(1, 2).reshape(batch, cameras, -1, values.shape[1])

        global_tokens = tokens(global_features)
        roi_tokens = tokens(roi_features)
        camera = self.camera_embed
        global_tokens = (
            global_tokens + self.global_pos + camera + self.type_embed[0]
        )
        roi_tokens = roi_tokens + self.roi_pos + camera + self.type_embed[1]
        box_tokens = (
            self.box_mlp(boxes).unsqueeze(2) + camera + self.type_embed[2]
        )
        return torch.cat((global_tokens, roi_tokens, box_tokens), dim=2).flatten(1, 2)


class ROIConditionedACT(nn.Module):
    """ROI-conditioned ACT architecture matching ``green_box_v1/best.pt``."""

    def __init__(self, config: ROIModelConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.action_queries = nn.Parameter(torch.empty(1, config.chunk_size, hidden))
        self.posterior_cls = nn.Parameter(torch.empty(1, 1, hidden))
        self.posterior_pos = nn.Parameter(
            torch.empty(1, config.chunk_size + 2, hidden)
        )
        self.visual = ROIVisualEncoder(config)
        self.qpos_projection = nn.Linear(config.state_dim, hidden)
        self.latent_projection = nn.Linear(config.latent_dim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers
        )
        self.action_head = nn.Linear(hidden, config.action_dim)
        self.pad_head = nn.Linear(hidden, 1)

        posterior_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.posterior = nn.TransformerEncoder(
            posterior_layer, num_layers=config.posterior_layers
        )
        self.posterior_action_projection = nn.Linear(config.action_dim, hidden)
        self.posterior_stats = nn.Linear(hidden, config.latent_dim * 2)

    def forward(
        self,
        qpos: Tensor,
        images: Tensor,
        boxes: Tensor,
        actions: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        batch = qpos.shape[0]
        posterior_stats: tuple[Tensor, Tensor] | None = None
        if self.config.use_vae and actions is not None:
            posterior_tokens = torch.cat(
                (
                    self.posterior_cls.expand(batch, -1, -1),
                    self.qpos_projection(qpos).unsqueeze(1),
                    self.posterior_action_projection(actions),
                ),
                dim=1,
            )
            posterior_mask = None
            if action_is_pad is not None:
                prefix = torch.zeros(
                    (batch, 2), dtype=torch.bool, device=action_is_pad.device
                )
                posterior_mask = torch.cat((prefix, action_is_pad.bool()), dim=1)
            encoded = self.posterior(
                posterior_tokens + self.posterior_pos[:, : posterior_tokens.shape[1]],
                src_key_padding_mask=posterior_mask,
            )
            mean, log_variance = self.posterior_stats(encoded[:, 0]).chunk(2, dim=-1)
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
            posterior_stats = (mean, log_variance)
        else:
            latent = torch.zeros(
                (batch, self.config.latent_dim),
                device=qpos.device,
                dtype=qpos.dtype,
            )

        memory = torch.cat(
            (
                self.qpos_projection(qpos).unsqueeze(1),
                self.latent_projection(latent).unsqueeze(1),
                self.visual(images, boxes),
            ),
            dim=1,
        )
        memory = self.encoder(memory)
        queries = self.action_queries.expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded), self.pad_head(decoded).squeeze(-1), posterior_stats

    @torch.inference_mode()
    def predict_action_chunk(self, qpos: Tensor, images: Tensor, boxes: Tensor) -> Tensor:
        actions, _padding, _posterior = self(qpos, images, boxes)
        return actions


def load_roi_act_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ROIConditionedACT, dict[str, Any]]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint root must be a mapping: {path}")
    state_dict = checkpoint.get("model")
    model_config = checkpoint.get("model_config")
    if not isinstance(state_dict, dict) or not isinstance(model_config, dict):
        raise ValueError("checkpoint must contain model and model_config mappings")
    config = ROIModelConfig.from_dict(model_config)
    model = ROIConditionedACT(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model, checkpoint
