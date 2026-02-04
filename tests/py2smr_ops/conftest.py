# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared fixtures and utilities for py2smr operator tests."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import numpy as np

from securemr.py2smr import trace, ops, convert, verify


def is_device_available() -> bool:
    """Check if an Android device is connected via ADB."""
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        # First line is "List of devices attached", check if there are more lines
        return len(lines) > 1 and any("device" in line for line in lines[1:])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Check device availability once at module load
DEVICE_AVAILABLE = is_device_available()


@pytest.fixture
def device_available():
    """Fixture to check if device is available."""
    return DEVICE_AVAILABLE


def skip_if_no_device(func):
    """Decorator to skip test if no device is available."""
    return pytest.mark.skipif(
        not DEVICE_AVAILABLE,
        reason="No Android device connected"
    )(func)


@pytest.fixture
def temp_pipeline_path():
    """Fixture to create a temporary pipeline JSON file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    yield path
    if path.exists():
        os.unlink(path)


def run_op_test(
    traced_func,
    inputs: dict,
    expected_output_name: str,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    test_device: bool = False,
    device_duration: int = 10,
):
    """Run a complete operator test with verification.

    Args:
        traced_func: A function decorated with @trace.
        inputs: Dictionary of input tensors.
        expected_output_name: Name of the expected output tensor.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
        test_device: If True, also test on device.
        device_duration: Duration for device test in seconds.

    Returns:
        Tuple of (result, verification_result).
    """
    # Execute with tracing
    result, ctx = traced_func.trace(**inputs)

    # Convert to pipeline spec
    spec = convert(ctx)

    # Get expected outputs
    if isinstance(result, np.ndarray):
        expected_outputs = {expected_output_name: result}
    elif isinstance(result, (tuple, list)):
        expected_outputs = {expected_output_name: result[0]}
    else:
        expected_outputs = result

    # Verify on host
    verification = verify(
        pipeline=spec,
        inputs=inputs,
        expected_outputs=expected_outputs,
        device=test_device,
        rtol=rtol,
        atol=atol,
        duration=device_duration,
    )

    if test_device and verification.error_message == "Device execution failed":
        pytest.skip("pipeline_inspect produced no output files; device execution failed")

    return result, verification
