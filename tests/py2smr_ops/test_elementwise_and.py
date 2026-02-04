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
"""Tests for ELEMENTWISE_AND operator."""

import numpy as np

from securemr.py2smr import trace, ops
from .conftest import run_op_test, skip_if_no_device


class TestElementwiseAnd:
    def test_basic_and(self):
        @trace(inputs=["a", "b"], outputs=["output"])
        def elementwise_and(a, b):
            return ops.elementwise_and(a, b, output_name="output")

        a = np.array([0, 1, 0, 1], dtype=np.int32)
        b = np.array([0, 0, 1, 1], dtype=np.int32)

        result, verification = run_op_test(
            elementwise_and,
            {"a": a, "b": b},
            "output",
        )

        assert verification.success, verification.error_message
        np.testing.assert_array_equal(result, np.array([0, 0, 0, 1], dtype=np.int32))

    @skip_if_no_device
    def test_elementwise_and_on_device(self):
        @trace(inputs=["a", "b"], outputs=["output"])
        def elementwise_and(a, b):
            return ops.elementwise_and(a, b, output_name="output")

        a = np.array([1, 1, 0, 0], dtype=np.int32)
        b = np.array([1, 0, 1, 0], dtype=np.int32)

        result, verification = run_op_test(
            elementwise_and,
            {"a": a, "b": b},
            "output",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
        np.testing.assert_array_equal(result, np.array([1, 0, 0, 0], dtype=np.int32))
