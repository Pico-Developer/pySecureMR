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
"""Tests for the APPLY_AFFINE_POINT operator."""

import numpy as np

from securemr.py2smr import trace, ops
from .conftest import run_op_test, skip_if_no_device


class TestApplyAffinePoint:
    def test_identity(self):
        @trace(inputs=["affine", "points"], outputs=["output"])
        def apply_affine_point(affine, points):
            return ops.apply_affine_point(affine, points, output_name="output")

        affine = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        points = np.array([
            [[0.0, 0.0]],
            [[1.0, 2.0]],
            [[-3.0, 4.0]],
        ], dtype=np.float32)

        result, verification = run_op_test(
            apply_affine_point,
            {"affine": affine, "points": points},
            "output",
        )

        assert verification.success, verification.error_message
        np.testing.assert_allclose(result, points, rtol=1e-5, atol=1e-5)

    def test_scale_translate(self):
        @trace(inputs=["affine", "points"], outputs=["output"])
        def apply_affine_point(affine, points):
            return ops.apply_affine_point(affine, points, output_name="output")

        affine = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -2.0]], dtype=np.float32)
        points = np.array([
            [[0.5, 1.0]],
            [[-1.0, 2.0]],
        ], dtype=np.float32)
        expected = np.array([
            [[2.0, 1.0]],
            [[-1.0, 4.0]],
        ], dtype=np.float32)

        result, verification = run_op_test(
            apply_affine_point,
            {"affine": affine, "points": points},
            "output",
        )

        assert verification.success, verification.error_message
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    @skip_if_no_device
    def test_apply_affine_point_on_device(self):
        @trace(inputs=["affine", "points"], outputs=["output"])
        def apply_affine_point(affine, points):
            return ops.apply_affine_point(affine, points, output_name="output")

        affine = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], dtype=np.float32)
        points = np.array([
            [[0.0, 0.0]],
            [[3.0, 4.0]],
        ], dtype=np.float32)

        result, verification = run_op_test(
            apply_affine_point,
            {"affine": affine, "points": points},
            "output",
            test_device=True,
            device_duration=10,
        )

        assert verification.success, verification.error_message
        np.testing.assert_allclose(result, np.array([[[2.0, -1.0]], [[5.0, 3.0]]], dtype=np.float32))
