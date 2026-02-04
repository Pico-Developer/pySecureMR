import numpy as np
import pytest

from securemr.py2smr import ops, trace
from .conftest import run_op_test


@trace(inputs=["gltf", "texture", "texture_ids"], outputs=["out"])
def traced_update_gltf(gltf, texture, texture_ids):
    ops.update_gltf(gltf, update_type="texture", values=texture, ids=texture_ids)
    dummy = np.array([1], dtype=np.int32)
    return ops.assignment(dummy, np.array([0], dtype=np.int32), output_name="out")


def test_update_gltf_host():
    gltf = np.zeros((1,), dtype=np.uint8)
    texture = np.zeros((2, 2, 3), dtype=np.uint8)
    texture_ids = np.array([0], dtype=np.uint16)
    _, verification = run_op_test(
        traced_update_gltf,
        {"gltf": gltf, "texture": texture, "texture_ids": texture_ids},
        expected_output_name="out",
        test_device=False,
    )
    assert verification.success


def test_update_gltf_device_skip():
    pytest.skip("update_gltf requires glTF placeholder; host-only stub")
