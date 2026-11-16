from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class InterACT3DConfig:
    """Bimanual RGB InterACT; point-cloud input is an explicit opt-in."""

    use_point_cloud: bool = False
    state_layout: str = "bimanual_gripper"
    state_dim: int = 16
    action_dim: int = 16
    arm_action_dim: int = 8
    chunk_size: int = 50
    image_size: int = 224
    camera_count: int = 3
    point_count: int = 1024
    point_tokens: int = 128
    d_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 1024
    encoder_blocks: int = 3
    pre_decoder_layers: int = 2
    post_decoder_layers: int = 2
    dropout: float = 0.1
    pretrained_backbone: bool = True
    freeze_backbone: bool = False
    left_gripper_enabled: bool = True
    right_gripper_enabled: bool = True
    left_gripper_open: float = 4000.0
    left_gripper_closed: float = 3600.0
    right_gripper_open: float = 1500.0
    right_gripper_closed: float = 900.0
    model_type: str = "interact_rgb"
    architecture_source: str = "Soltanilara/InterACT-LeRobot"

    def __post_init__(self) -> None:
        self.model_type = "interact3d" if self.use_point_cloud else "interact_rgb"
        if self.state_layout not in {"bimanual_gripper", "dual_arm_suction"}:
            raise ValueError("unsupported state_layout")
        if self.state_dim != 16 or self.action_dim != 16 or self.arm_action_dim != 8:
            raise ValueError("this xCore bimanual schema is fixed at state/action=16 and arm_action=8")
        if self.camera_count != 3:
            raise ValueError("the current rig requires head + left_wrist + right_wrist")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        for name in (
            "chunk_size",
            "image_size",
            "point_count",
            "point_tokens",
            "d_model",
            "encoder_blocks",
            "pre_decoder_layers",
            "post_decoder_layers",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.point_tokens > self.point_count:
            raise ValueError("point_tokens cannot exceed point_count")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        for side in ("left", "right"):
            opened = float(getattr(self, f"{side}_gripper_open"))
            closed = float(getattr(self, f"{side}_gripper_closed"))
            if not (0 <= opened <= 4095 and 0 <= closed <= 4095 and opened != closed):
                raise ValueError(f"{side} gripper endpoints must be distinct values in [0,4095]")

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n"
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "InterACT3DConfig":
        payload = json.loads((Path(directory) / "config.json").read_text())
        # Old checkpoints had four segments and no modality flag.
        payload.setdefault("use_point_cloud", payload.get("model_type", "interact3d") == "interact3d")
        return cls(**payload)
