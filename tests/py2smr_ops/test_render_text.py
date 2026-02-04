import numpy as np
import pytest

from securemr.py2smr import ops, trace
from .conftest import run_op_test


@trace(inputs=["gltf"], outputs=["out"])
def traced_render_text(gltf):
    ops.render_text(
        gltf,
        text="hello",
        language_and_locale="en-us",
        canvas_width=256,
        canvas_height=64,
        typeface="bold",
    )
    dummy = np.array([1], dtype=np.int32)
    return ops.assignment(dummy, np.array([0], dtype=np.int32), output_name="out")


def test_render_text_host():
    gltf = np.zeros((1,), dtype=np.uint8)
    _, verification = run_op_test(
        traced_render_text,
        {"gltf": gltf},
        expected_output_name="out",
        test_device=False,
    )
    assert verification.success


def test_render_text_device_skip():
    pytest.skip("render_text requires glTF placeholder; host-only stub")
