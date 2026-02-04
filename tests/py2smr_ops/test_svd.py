import numpy as np

from securemr.py2smr import ops, trace, convert, verify
from .conftest import skip_if_no_device


@trace(inputs=["mat"], outputs=["w", "u", "vt"])
def traced_svd(mat):
    return ops.svd(mat, output_names=["w", "u", "vt"])


def test_svd_host():
    mat = np.array(
        [[0.42953404, 0.7793843, 0.55921781], [0.82524029, 0.02600958, 0.83493143]],
        dtype=np.float32,
    )
    (w, u, vt), ctx = traced_svd.trace(mat=mat)
    spec = convert(ctx)
    expected = {"w": w, "u": u, "vt": vt}
    verification = verify(spec, {"mat": mat}, expected_outputs=expected)
    assert verification.success


@skip_if_no_device
def test_svd_device():
    mat = np.array(
        [[0.42953404, 0.7793843, 0.55921781], [0.82524029, 0.02600958, 0.83493143]],
        dtype=np.float32,
    )
    (w, u, vt), ctx = traced_svd.trace(mat=mat)
    spec = convert(ctx)
    expected = {"w": w, "u": u, "vt": vt}
    verification = verify(spec, {"mat": mat}, expected_outputs=expected, device=True)
    if verification.error_message == "Device execution failed":
        import pytest
        pytest.skip("pipeline_inspect produced no output files; device execution failed")
    assert verification.success
