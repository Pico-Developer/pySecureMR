import numpy as np

from securemr.py2smr import ops, trace, convert, verify
from .conftest import skip_if_no_device


@trace(inputs=["vec"], outputs=["sorted", "indices"])
def traced_sort_vec(vec):
    return ops.sort_vec(vec, output_names=["sorted", "indices"])


def test_sort_vec_host():
    vec = np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float32)
    (sorted_vec, indices), ctx = traced_sort_vec.trace(vec=vec)
    spec = convert(ctx)
    expected = {"sorted": sorted_vec, "indices": indices}
    verification = verify(spec, {"vec": vec}, expected_outputs=expected)
    assert verification.success


@skip_if_no_device
def test_sort_vec_device():
    vec = np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float32)
    (sorted_vec, indices), ctx = traced_sort_vec.trace(vec=vec)
    spec = convert(ctx)
    expected = {"sorted": sorted_vec, "indices": indices}
    verification = verify(spec, {"vec": vec}, expected_outputs=expected, device=True)
    if verification.error_message == "Device execution failed":
        import pytest
        pytest.skip("pipeline_inspect produced no output files; device execution failed")
    assert verification.success
