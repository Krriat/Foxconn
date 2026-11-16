from types import SimpleNamespace

import numpy as np

from xcore_interact.cameras import _decode_color, _decode_depth


def _image(array: np.ndarray, encoding: str, *, step: int | None = None):
    return SimpleNamespace(
        encoding=encoding,
        width=array.shape[1],
        height=array.shape[0],
        step=step or array.strides[0],
        is_bigendian=False,
        data=array.tobytes(),
    )


def test_rgb_message_is_returned_as_bgr():
    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    decoded = _decode_color(_image(rgb, "rgb8"), width=2, height=1, topic="/rgb")
    np.testing.assert_array_equal(decoded, rgb[..., ::-1])
    assert decoded.flags.c_contiguous


def test_ros_16uc1_depth_is_millimetres():
    depth = np.array([[0, 1000], [1500, 2000]], dtype=np.uint16)
    decoded, scale = _decode_depth(
        _image(depth, "16UC1"), width=2, height=2, topic="/depth"
    )
    np.testing.assert_array_equal(decoded, depth)
    assert scale == 0.001
