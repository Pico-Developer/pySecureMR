import numpy as np
import pytest

from securemr.py2smr import ops, trace
from .conftest import run_op_test


@trace(inputs=["gltf", "texture"], outputs=["texture_id"])
def traced_load_texture(gltf, texture):
    return ops.load_texture(gltf, texture, output_name="texture_id")


def test_load_texture_host():
    gltf = np.zeros((1,), dtype=np.uint8)
    texture = np.zeros((2, 2, 3), dtype=np.uint8)
    _, verification = run_op_test(
        traced_load_texture,
        {"gltf": gltf, "texture": texture},
        expected_output_name="texture_id",
        test_device=False,
    )
    assert verification.success


def test_load_texture_device_skip():
    pytest.skip("load_texture requires glTF placeholder; host-only stub")
