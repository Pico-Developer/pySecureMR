import numpy as np

from securemr.py2smr import ops, trace
from .conftest import run_op_test, skip_if_no_device


@trace(inputs=["tensor"], outputs=["swapped"])
def traced_swap(tensor):
    return ops.swap_hwc_chw(tensor, output_name="swapped")


def test_swap_hwc_chw_host():
    tensor = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    result, verification = run_op_test(
        traced_swap,
        {"tensor": tensor},
        expected_output_name="swapped",
        test_device=False,
    )
    assert verification.success
    assert result.shape == (4, 2, 3)


@skip_if_no_device
def test_swap_hwc_chw_device():
    tensor = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    _, verification = run_op_test(
        traced_swap,
        {"tensor": tensor},
        expected_output_name="swapped",
        test_device=True,
    )
    assert verification.success
