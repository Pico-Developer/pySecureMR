import numpy as np
import pytest

from securemr.py2smr import ops, trace, convert, verify


@trace(inputs=[], outputs=["right", "left", "timestamp", "cam_matrix"])
def traced_rectified_vst_access():
    return ops.rectified_vst_access(output_names=["right", "left", "timestamp", "cam_matrix"])


def test_rectified_vst_access_host():
    (right, left, timestamp, cam_matrix), ctx = traced_rectified_vst_access.trace()
    spec = convert(ctx)
    expected = {
        "right": right,
        "left": left,
        "timestamp": timestamp,
        "cam_matrix": cam_matrix,
    }
    verification = verify(spec, {}, expected_outputs=expected)
    assert verification.success


def test_rectified_vst_access_device_skip():
    pytest.skip("rectified_vst_access depends on device camera stream; host stub cannot match device output")
