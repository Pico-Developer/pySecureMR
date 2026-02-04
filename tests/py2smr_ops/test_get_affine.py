import numpy as np

from securemr.py2smr import ops, trace
from .conftest import run_op_test, skip_if_no_device


@trace(inputs=["src", "dst"], outputs=["affine"])
def traced_get_affine(src, dst):
    return ops.get_affine(src, dst, output_name="affine")


def test_get_affine_host():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dst = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    result, verification = run_op_test(
        traced_get_affine,
        {"src": src, "dst": dst},
        expected_output_name="affine",
        test_device=False,
    )
    assert verification.success
    assert result.shape == (2, 3)


@skip_if_no_device
def test_get_affine_device():
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dst = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    _, verification = run_op_test(
        traced_get_affine,
        {"src": src, "dst": dst},
        expected_output_name="affine",
        test_device=True,
    )
    assert verification.success
