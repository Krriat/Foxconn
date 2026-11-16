from __future__ import annotations

import numpy as np


def depth_rgb_to_xyzrgb(
    depth_raw: np.ndarray,
    rgb: np.ndarray,
    *,
    fx: float,
    fy: float,
    ppx: float,
    ppy: float,
    depth_scale: float,
    point_count: int,
    min_depth_m: float = 0.15,
    max_depth_m: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Convert aligned RealSense color/depth to a fixed XYZRGB point set.

    Invalid/padded rows are all zero. RGB is stored as 0..255 float32 so the
    dataset remains lossless; the model scales it to 0..1.
    """
    if depth_raw.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_raw.shape}")
    if rgb.shape != (*depth_raw.shape, 3):
        raise ValueError(f"aligned RGB shape {rgb.shape} does not match depth")
    if point_count <= 0 or depth_scale <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("point_count, depth_scale, fx and fy must be positive")

    depth_m = depth_raw.astype(np.float32, copy=False) * np.float32(depth_scale)
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= np.float32(min_depth_m))
        & (depth_m <= np.float32(max_depth_m))
    )
    rows, cols = np.nonzero(valid)
    output = np.zeros((point_count, 6), dtype=np.float32)
    if rows.size == 0:
        return output

    rng = np.random.default_rng(seed)
    if rows.size >= point_count:
        selected = rng.choice(rows.size, size=point_count, replace=False)
    else:
        # Keep each real point once; zero padding remains an explicit mask.
        selected = rng.permutation(rows.size)
    rows = rows[selected]
    cols = cols[selected]
    z = depth_m[rows, cols]
    count = z.size
    output[:count, 0] = (cols.astype(np.float32) - np.float32(ppx)) * z / np.float32(fx)
    output[:count, 1] = (rows.astype(np.float32) - np.float32(ppy)) * z / np.float32(fy)
    output[:count, 2] = z
    output[:count, 3:] = rgb[rows, cols].astype(np.float32, copy=False)
    return output


def valid_point_mask(points: np.ndarray) -> np.ndarray:
    if points.ndim < 2 or points.shape[-1] != 6:
        raise ValueError(f"points must end in [N,6], got {points.shape}")
    return np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)

