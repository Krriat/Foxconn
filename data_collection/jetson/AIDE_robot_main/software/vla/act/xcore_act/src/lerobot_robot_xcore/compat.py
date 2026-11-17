from __future__ import annotations

import platform
import re


def patch_accelerate_custom_kernel_check() -> None:
    """Handle Jetson kernels such as ``5.15-niic-tegra``.

    Accelerate 1.14 assumes Linux releases always contain an x.y.z triplet and
    raises before training on this machine's valid 5.15 custom kernel. The
    minimum supported version is 5.5, so skipping only that broken parser is
    safe when the parsed major/minor version is already new enough.
    """

    release = platform.uname().release
    if re.search(r"\d+\.\d+\.\d+", release):
        return
    version = re.search(r"(\d+)\.(\d+)", release)
    if version is None or (int(version.group(1)), int(version.group(2))) < (5, 5):
        return

    import accelerate.accelerator as accelerator_module

    accelerator_module.check_os_kernel = lambda: None


def patch_accelerate_without_torch_distributed() -> None:
    """Let single-GPU Accelerate clean up a Jetson torch build without DDP."""

    import torch

    distributed = getattr(torch, "distributed", None)
    if distributed is not None and not hasattr(distributed, "is_initialized"):
        distributed.is_initialized = lambda: False
