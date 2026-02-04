import numpy as np

from securemr.py2smr import ops, trace
from .conftest import run_op_test, skip_if_no_device


@trace(inputs=["mat"], outputs=["inv"])
def traced_inversion(mat):
    return ops.inversion(mat, output_name="inv")


def test_inversion_host():
    mat = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    result, verification = run_op_test(
        traced_inversion,
        {"mat": mat},
        expected_output_name="inv",
        test_device=False,
    )
    assert verification.success
    assert result.shape == (2, 2)


@skip_if_no_device
def test_inversion_device():
    mat = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    _, verification = run_op_test(
        traced_inversion,
        {"mat": mat},
        expected_output_name="inv",
        test_device=True,
    )
    assert verification.success
