import numpy as np
import pytest

from securemr.py2smr import ops, trace
from .conftest import run_op_test


@trace(inputs=["uv", "timestamp", "camera_matrix", "left_image", "right_image"], outputs=["points"])
def traced_uv_to_3d(uv, timestamp, camera_matrix, left_image, right_image):
    return ops.uv_to_3d_in_cam_space(uv, timestamp, camera_matrix, left_image, right_image, output_name="points")


def test_uv_to_3d_in_cam_space_host():
    uv = np.array([[10, 12], [20, 30]], dtype=np.int32)
    timestamp = np.zeros((1, 4), dtype=np.int32)
    camera_matrix = np.eye(3, dtype=np.float32)
    left_image = np.zeros((2, 2, 3), dtype=np.uint8)
    right_image = np.zeros((2, 2, 3), dtype=np.uint8)
    _, verification = run_op_test(
        traced_uv_to_3d,
        {
            "uv": uv,
            "timestamp": timestamp,
            "camera_matrix": camera_matrix,
            "left_image": left_image,
            "right_image": right_image,
        },
        expected_output_name="points",
        test_device=False,
    )
    assert verification.success


def test_uv_to_3d_in_cam_space_device_skip():
    pytest.skip("uv_to_3d_in_cam_space relies on depth sensor data; host stub cannot match device output")
