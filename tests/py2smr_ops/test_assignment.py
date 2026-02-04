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
"""Tests for the assignment operator."""

import numpy as np
import pytest

from securemr.py2smr import trace, ops
from .conftest import run_op_test, skip_if_no_device


class TestAssignment:
    """Tests for assignment operation."""

    def test_full_copy(self):
        """Test full tensor copy."""
        @trace(inputs=["src", "dst"], outputs=["output"])
        def full_copy(src, dst):
            return ops.assignment(src, dst, output_name="output")

        src = np.random.rand(4, 4).astype(np.float32)
        dst = np.zeros((4, 4), dtype=np.float32)

        result, verification = run_op_test(
            full_copy,
            {"src": src, "dst": dst},
            "output",
        )

        assert verification.success, verification.error_message
        np.testing.assert_allclose(result, src, rtol=1e-5)

    def test_slice_assignment(self):
        """Test assignment with slicing."""
        @trace(inputs=["src", "dst"], outputs=["output"])
        def slice_assign(src, dst):
            return ops.assignment(
                src, dst,
                src_slices=[[0, 2], [0, 2]],
                dst_slices=[[1, 3], [1, 3]],
                output_name="output"
            )

        src = np.random.rand(4, 4).astype(np.float32)
        dst = np.zeros((4, 4), dtype=np.float32)

        result, verification = run_op_test(
            slice_assign,
            {"src": src, "dst": dst},
            "output",
        )

        assert verification.success, verification.error_message
        # Check that the slice was correctly assigned
        np.testing.assert_allclose(result[1:3, 1:3], src[0:2, 0:2], rtol=1e-5)
        # Check that other parts remain zero
        np.testing.assert_allclose(result[0, :], 0, rtol=1e-5)

    def test_src_slice_only(self):
        """Test assignment with only source slicing."""
        @trace(inputs=["src", "dst"], outputs=["output"])
        def src_slice(src, dst):
            return ops.assignment(
                src, dst,
                src_slices=[[1, 3], [1, 3]],
                output_name="output"
            )

        src = np.random.rand(4, 4).astype(np.float32)
        dst = np.zeros((2, 2), dtype=np.float32)

        result, verification = run_op_test(
            src_slice,
            {"src": src, "dst": dst},
            "output",
        )

        assert verification.success, verification.error_message
        np.testing.assert_allclose(result, src[1:3, 1:3], rtol=1e-5)

    def test_large_tensor(self):
        """Test assignment with larger tensors."""
        @trace(inputs=["src", "dst"], outputs=["output"])
        def large_assign(src, dst):
            return ops.assignment(src, dst, output_name="output")

        src = np.random.rand(64, 64).astype(np.float32)
        dst = np.zeros((64, 64), dtype=np.float32)

        result, verification = run_op_test(
            large_assign,
            {"src": src, "dst": dst},
            "output",
        )

        assert verification.success, verification.error_message

    @skip_if_no_device
    def test_assignment_on_device(self):
        """Test assignment operation on device."""
        @trace(inputs=["src", "dst"], outputs=["output"])
        def device_assign(src, dst):
            return ops.assignment(src, dst, output_name="output")

        src = np.random.rand(8, 8).astype(np.float32)
        dst = np.zeros((8, 8), dtype=np.float32)

        result, verification = run_op_test(
            device_assign,
            {"src": src, "dst": dst},
            "output",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
