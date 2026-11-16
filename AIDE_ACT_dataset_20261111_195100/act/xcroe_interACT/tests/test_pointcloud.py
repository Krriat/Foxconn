import numpy as np

from xcore_interact.pointcloud import depth_rgb_to_xyzrgb, valid_point_mask


def test_depth_deprojection_and_padding():
    depth = np.array([[1000, 0], [2000, 1000]], dtype=np.uint16)
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    points = depth_rgb_to_xyzrgb(
        depth,
        rgb,
        fx=100.0,
        fy=100.0,
        ppx=0.0,
        ppy=0.0,
        depth_scale=0.001,
        point_count=5,
        min_depth_m=0.1,
        max_depth_m=1.5,
        seed=0,
    )
    assert points.shape == (5, 6)
    assert valid_point_mask(points).sum() == 2
    assert np.all(points[2:] == 0)


def test_rejects_unaligned_color():
    depth = np.ones((2, 2), dtype=np.uint16)
    rgb = np.ones((3, 2, 3), dtype=np.uint8)
    try:
        depth_rgb_to_xyzrgb(
            depth,
            rgb,
            fx=1,
            fy=1,
            ppx=0,
            ppy=0,
            depth_scale=0.001,
            point_count=2,
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("unaligned RGB was accepted")

