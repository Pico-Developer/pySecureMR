import numpy as np
import pytest

from securemr.py2smr import ops, trace
from .conftest import run_op_test


@trace(inputs=["tensor"], outputs=["out"])
def traced_unknown(tensor):
    return ops.unknown(tensor, output_name="out")


def test_unknown_host():
    tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result, verification = run_op_test(
        traced_unknown,
        {"tensor": tensor},
        expected_output_name="out",
        test_device=False,
    )
    assert verification.success
    assert np.allclose(result, tensor)


def test_unknown_device_skip():
    pytest.skip("unknown operator type not supported on device")
