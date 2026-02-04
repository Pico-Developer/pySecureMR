import numpy as np

from securemr.py2smr import ops, trace
from .conftest import run_op_test, skip_if_no_device


@trace(inputs=["rotation", "translation", "scale"], outputs=["transform"])
def traced_get_transform_mat(rotation, translation, scale):
    return ops.get_transform_mat(rotation, translation, scale, output_name="transform")


def test_get_transform_mat_host():
    rotation = np.array([0.2, -0.1, 0.05], dtype=np.float32)
    translation = np.array([0.3, -0.2, 0.5], dtype=np.float32)
    scale = np.array([1.2, 0.9, 0.5], dtype=np.float32)
    result, verification = run_op_test(
        traced_get_transform_mat,
        {"rotation": rotation, "translation": translation, "scale": scale},
        expected_output_name="transform",
        test_device=False,
    )
    assert verification.success
    assert result.shape == (4, 4)


@skip_if_no_device
def test_get_transform_mat_device():
    rotation = np.array([0.2, -0.1, 0.05], dtype=np.float32)
    translation = np.array([0.3, -0.2, 0.5], dtype=np.float32)
    scale = np.array([1.2, 0.9, 0.5], dtype=np.float32)
    _, verification = run_op_test(
        traced_get_transform_mat,
        {"rotation": rotation, "translation": translation, "scale": scale},
        expected_output_name="transform",
        test_device=True,
    )
    assert verification.success
