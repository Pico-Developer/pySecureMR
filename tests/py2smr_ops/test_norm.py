import numpy as np

from securemr.py2smr import ops, trace
from .conftest import run_op_test, skip_if_no_device


@trace(inputs=["tensor"], outputs=["norm"])
def traced_norm(tensor):
    return ops.norm(tensor, norm_type="L2", output_name="norm")


def test_norm_host():
    tensor = np.array([[3.0, 4.0]], dtype=np.float32)
    result, verification = run_op_test(
        traced_norm,
        {"tensor": tensor},
        expected_output_name="norm",
        test_device=False,
    )
    assert verification.success
    assert result.shape == (1,)


@skip_if_no_device
def test_norm_device():
    tensor = np.array([[3.0, 4.0]], dtype=np.float32)
    _, verification = run_op_test(
        traced_norm,
        {"tensor": tensor},
        expected_output_name="norm",
        test_device=True,
    )
    assert verification.success
