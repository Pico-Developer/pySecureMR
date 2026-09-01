#!/usr/bin/env python3
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
"""Unit tests for py2smr module."""

import json
import os
import tempfile

import numpy as np
import pytest

from securemr.py2smr import trace, ops, convert, verify
from securemr.py2smr.tracer import TraceContext, TracedOp, TensorInfo, get_current_trace
from securemr.core.utils import convert_to_dtype
from securemr.py2smr.converter import trace_to_pipeline_spec
from securemr.py2smr.verifier import (
    compare_outputs,
    VerificationResult,
    run_pipeline_python,
    validate_pipeline_spec,
)
from securemr.core.types import EOperatorType


def test_convert_to_dtype_accepts_schema_v2_string_aliases():
    assert convert_to_dtype("int32", target="numpy") == np.int32
    assert convert_to_dtype("float32", target="numpy") == np.float32


def test_type_convert_alias_converts_host_dtype():
    pipeline = {
        "tensors": {
            "input": {
                "dimensions": [2, 1],
                "channels": 1,
                "data_type": 6,
                "usage": 6,
                "is_placeholder": True,
            },
            "output": {
                "dimensions": [2, 1],
                "channels": 1,
                "data_type": 5,
                "usage": 6,
                "is_placeholder": True,
            },
        },
        "operators": [{
            "type": "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO",
            "inputs": ["input"],
            "outputs": ["output"],
        }],
        "inputs": ["input"],
        "outputs": ["output"],
    }

    outputs = run_pipeline_python(
        pipeline,
        {"input": np.array([[1.5], [2.5]], dtype=np.float32)},
    )

    assert outputs["output"].dtype == np.int32
    np.testing.assert_array_equal(outputs["output"], [[1], [2]])


class TestTracer:
    """Tests for the tracer module."""

    def test_trace_decorator_basic(self):
        """Test basic trace decorator functionality."""
        @trace(inputs=["x"], outputs=["y"])
        def simple_func(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result, ctx = simple_func.trace(x=input_arr)

        assert isinstance(ctx, TraceContext)
        assert len(ctx.operations) == 1
        assert ctx.operations[0].op_type == EOperatorType.ARITHMETIC_COMPOSE
        assert "x" in ctx.tensors
        assert ctx.tensors["x"].is_input

    def test_trace_multiple_ops(self):
        """Test tracing multiple operations."""
        @trace(inputs=["image"], outputs=["result"])
        def preprocess(image):
            normalized = ops.arithmetic(image, "{0} / 255.0")
            scaled = ops.arithmetic(normalized, "{0} * 2.0 - 1.0")
            return scaled

        input_arr = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        result, ctx = preprocess.trace(image=input_arr)

        assert len(ctx.operations) == 2
        assert ctx.operations[0].attrs == ["{0} / 255.0"]
        assert ctx.operations[1].attrs == ["{0} * 2.0 - 1.0"]

    def test_trace_multiple_outputs(self):
        """Test tracing function with multiple outputs."""
        @trace(inputs=["x"], outputs=["min_val", "max_val"])
        def minmax(x):
            min_val = ops.arithmetic(x, "{0} - 1.0")
            max_val = ops.arithmetic(x, "{0} + 1.0")
            return min_val, max_val

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (min_result, max_result), ctx = minmax.trace(x=input_arr)

        assert "min_val" in ctx.tensors
        assert "max_val" in ctx.tensors
        assert ctx.tensors["min_val"].is_output
        assert ctx.tensors["max_val"].is_output

    def test_trace_context_not_active_outside(self):
        """Test that trace context is not active outside traced function."""
        assert get_current_trace() is None

    def test_missing_input_raises_error(self):
        """Test that missing input raises ValueError."""
        @trace(inputs=["x", "y"], outputs=["z"])
        def add_func(x, y):
            return ops.arithmetic(x, "{0} + 1.0")

        with pytest.raises(ValueError, match="Missing required input"):
            add_func.trace(x=np.array([1.0]))

    def test_non_array_input_raises_error(self):
        """Test that non-array input raises TypeError."""
        @trace(inputs=["x"], outputs=["y"])
        def simple_func(x):
            return ops.arithmetic(x, "{0} * 2.0")

        with pytest.raises(TypeError, match="must be a numpy array"):
            simple_func.trace(x=[1.0, 2.0])


class TestOps:
    """Tests for the ops module."""

    def test_arithmetic_basic(self):
        """Test basic arithmetic operation."""
        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = ops.arithmetic(input_arr, "{0} * 2.0")

        expected = input_arr * 2.0
        np.testing.assert_allclose(result, expected)

    def test_arithmetic_complex_expression(self):
        """Test arithmetic with complex expression."""
        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = ops.arithmetic(input_arr, "{0} / 255.0 * 2.0 - 1.0")

        expected = input_arr / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(result, expected)

    def test_elementwise_min(self):
        """Test elementwise minimum."""
        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=np.float32)
        result = ops.elementwise_min(a, b)

        expected = np.minimum(a, b)
        np.testing.assert_allclose(result, expected)

    def test_elementwise_max(self):
        """Test elementwise maximum."""
        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=np.float32)
        result = ops.elementwise_max(a, b)

        expected = np.maximum(a, b)
        np.testing.assert_allclose(result, expected)

    def test_elementwise_multiply(self):
        """Test elementwise multiplication."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
        result = ops.elementwise_multiply(a, b)

        expected = a * b
        np.testing.assert_allclose(result, expected)

    def test_normalize(self):
        """Test L2 normalization."""
        input_arr = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        result = ops.normalize(input_arr)

        expected = input_arr / np.linalg.norm(input_arr)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_argmax(self):
        """Test argmax operation."""
        input_arr = np.array([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=np.float32)
        result = ops.argmax(input_arr, axis=-1)

        expected = np.array([1, 0], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_nms_basic(self):
        """Test basic NMS operation."""
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],  # High overlap with first
            [50, 50, 60, 60],  # No overlap
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        result = ops.nms(scores, boxes, threshold=0.5)

        # Should keep first and third (second overlaps too much with first)
        assert 0 in result
        assert 2 in result

    def test_nms_empty(self):
        """Test NMS with empty input."""
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)

        result = ops.nms(scores, boxes, threshold=0.5)
        assert len(result) == 0

    def test_ops_record_to_trace(self):
        """Test that ops record to trace context."""
        @trace(inputs=["x"], outputs=["y"])
        def traced_func(x):
            return ops.arithmetic(x, "{0} + 1.0")

        input_arr = np.array([[1.0, 2.0]], dtype=np.float32)
        _, ctx = traced_func.trace(x=input_arr)

        assert len(ctx.operations) == 1
        op = ctx.operations[0]
        assert op.op_type == EOperatorType.ARITHMETIC_COMPOSE
        assert op.attrs == ["{0} + 1.0"]


class TestConverter:
    """Tests for the converter module."""

    def test_convert_basic(self):
        """Test basic conversion to pipeline spec."""
        @trace(inputs=["input"], outputs=["output"])
        def simple_func(input):
            return ops.arithmetic(input, "{0} + 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        _, ctx = simple_func.trace(input=input_arr)

        spec = trace_to_pipeline_spec(ctx)

        assert "tensors" in spec
        assert "operators" in spec
        assert "inputs" in spec
        assert "outputs" in spec

        assert "input" in spec["inputs"]
        assert "output" in spec["outputs"]
        assert len(spec["operators"]) == 1

    def test_convert_saves_to_file(self):
        """Test that convert saves to file."""
        @trace(inputs=["x"], outputs=["y"])
        def simple_func(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0]], dtype=np.float32)
        _, ctx = simple_func.trace(x=input_arr)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            spec = convert(ctx, output=output_path)

            assert os.path.exists(output_path)
            with open(output_path, "r") as f:
                loaded_spec = json.load(f)

            assert loaded_spec == spec
        finally:
            os.unlink(output_path)

    def test_convert_tensor_shapes(self):
        """Test that tensor shapes are correctly converted."""
        @trace(inputs=["image"], outputs=["result"])
        def process_image(image):
            return ops.arithmetic(image, "{0} / 255.0")

        # 3D tensor (H, W, C)
        input_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        _, ctx = process_image.trace(image=input_arr)

        spec = trace_to_pipeline_spec(ctx)

        # Check input tensor spec
        input_spec = spec["tensors"]["image"]
        assert input_spec["dimensions"] == [224, 224]  # W, H
        assert input_spec["channels"] == 3

    def test_convert_operator_attrs(self):
        """Test that operator attributes are correctly converted."""
        @trace(inputs=["x"], outputs=["y"])
        def func(x):
            return ops.arithmetic(x, "{0} / 255.0 * 2.0 - 1.0")

        input_arr = np.array([[1.0]], dtype=np.float32)
        _, ctx = func.trace(x=input_arr)

        spec = trace_to_pipeline_spec(ctx)

        op_spec = spec["operators"][0]
        assert op_spec["expression"] == "{0} / 255.0 * 2.0 - 1.0"

    def test_convert_spatial_only_ops_use_sdk_json_fields(self):
        """Test that Spatial-only operators emit fields accepted by the SDK loader."""
        @trace(inputs=["scene", "scale"], outputs=["visible"])
        def func(scene, scale):
            visible = ops.scenegraph_visibility(scene, visible=False)
            ops.update_component(scene, scale, entity_path="/target", property="Transform.Scale")
            return visible

        _, ctx = func.trace(
            scene=np.array([[1]], dtype=np.uint8),
            scale=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        )
        spec = trace_to_pipeline_spec(ctx)

        assert spec["operators"][0] == {
            "type": "XR_SECURE_MR_OPERATOR_TYPE_SCENEGRAPH_VISIBILITY_PICO",
            "inputs": ["scene"],
            "outputs": [],
            "scenegraph": "scene",
            "visible": False,
        }
        assert spec["operators"][1] == {
            "type": "XR_SECURE_MR_OPERATOR_TYPE_UPDATE_COMPONENT_PICO",
            "inputs": ["scene", "scale"],
            "outputs": [],
            "scenegraph": "scene",
            "data": "scale",
            "entity_path": "/target",
            "property": "Transform.Scale",
        }

    def test_convert_xr_gltf_ops_emits_native_named_fields(self):
        @trace(inputs=["gltf", "pose", "texture", "texture_ids"], outputs=["texture_id"])
        def func(gltf, pose, texture, texture_ids):
            ops.switch_gltf_render_status(gltf, pose=pose, view_locked=False, visible=True)
            ops.update_gltf(gltf, update_type="texture", values=texture, ids=texture_ids)
            return ops.load_texture(gltf, texture, output_name="texture_id")

        _, ctx = func.trace(
            gltf=np.zeros((1,), dtype=np.uint8),
            pose=np.eye(4, dtype=np.float32),
            texture=np.zeros((2, 2, 3), dtype=np.uint8),
            texture_ids=np.array([0], dtype=np.uint16),
        )
        spec = trace_to_pipeline_spec(ctx)

        render, update, load = spec["operators"]
        assert render["gltf"] == "gltf"
        assert render["pose"] == "pose"
        assert render["view_locked"] is False
        assert render["visible"] is True
        assert update["update_type"] == "texture"
        assert update["gltf"] == "gltf"
        assert update["texture_src"] == "texture"
        assert update["texture_id"] == "texture_ids"
        assert load["gltf"] == "gltf"
        assert load["rgb_image"] == "texture"
        assert spec["outputs"] == ["texture_id"]

    @pytest.mark.parametrize(
        "dynamic_fields",
        [
            ("view_locked",),
            ("visible",),
            ("view_locked", "visible"),
            ("pose", "visible"),
        ],
    )
    def test_convert_xr_gltf_status_preserves_dynamic_tensor_roles(self, dynamic_fields):
        @trace(inputs=["gltf", *dynamic_fields], outputs=["gltf"])
        def func(gltf, **values):
            ops.switch_gltf_render_status(
                gltf,
                pose=values.get("pose"),
                view_locked=values.get("view_locked"),
                visible=values.get("visible"),
            )
            return gltf

        inputs = {"gltf": np.zeros((1,), dtype=np.uint8)}
        inputs.update({
            name: np.array([index + 1], dtype=np.int32)
            for index, name in enumerate(dynamic_fields)
        })
        _, ctx = func.trace(**inputs)
        render = trace_to_pipeline_spec(ctx)["operators"][0]

        assert render["inputs"] == ["gltf", *dynamic_fields]
        assert render["gltf"] == "gltf"
        for name in dynamic_fields:
            assert render[name] == name
        for name in {"pose", "view_locked", "visible"} - set(dynamic_fields):
            assert name not in render

    def test_host_verifier_resolves_dynamic_gltf_status_tensors(self, monkeypatch):
        calls = []

        def capture(gltf, pose=None, view_locked=None, visible=None):
            calls.append((gltf, pose, view_locked, visible))

        monkeypatch.setattr(ops, "switch_gltf_render_status", capture)
        spec = {
            "tensors": {
                "gltf": {"dimensions": [1, 1], "channels": 1, "data_type": 1},
                "visible": {"dimensions": [1, 1], "channels": 1, "data_type": 5},
            },
            "operators": [{
                "type": "switch_gltf_render_status",
                "inputs": ["gltf", "visible"],
                "outputs": [],
                "gltf": "gltf",
                "visible": "visible",
            }],
            "inputs": ["gltf", "visible"],
            "outputs": [],
        }
        gltf = np.zeros((1,), dtype=np.uint8)
        visible = np.array([1], dtype=np.int32)
        run_pipeline_python(spec, {"gltf": gltf, "visible": visible})

        assert len(calls) == 1
        assert calls[0][0] is gltf
        assert calls[0][1] is None
        assert calls[0][2] is None
        assert calls[0][3] is visible


class TestVerifier:
    """Tests for the verifier module."""

    def test_validate_pipeline_spec_accepts_2d_matrix_tensor(self):
        """Test that MAT tensors can be declared as row/column matrices."""
        spec = {
            "tensors": {
                "row_vec": {
                    "dimensions": [1, 4],
                    "channels": 1,
                    "usage": 6,
                },
                "named_matrix": {
                    "dimensions": [4, 1],
                    "channels": 1,
                    "tensor_type": "matrix",
                },
            }
        }

        validate_pipeline_spec(spec)

    @pytest.mark.parametrize("dimensions", [[], [4]])
    def test_validate_pipeline_spec_rejects_1d_matrix_tensor(self, dimensions):
        """Test that matrix/MAT tensors require at least two dimensions."""
        spec = {
            "tensors": {
                "bad_vec": {
                    "dimensions": dimensions,
                    "channels": 1,
                    "usage": 6,
                }
            }
        }

        with pytest.raises(ValueError, match="matrix tensors must have at least 2 dimensions"):
            validate_pipeline_spec(spec)

    def test_run_pipeline_python_rejects_invalid_matrix_tensor(self):
        """Test host verification catches invalid JSON before native runtime."""
        spec = {
            "tensors": {
                "bad_vec": {
                    "dimensions": [4],
                    "channels": 1,
                    "usage": 6,
                }
            },
            "operators": [],
            "outputs": [],
        }

        with pytest.raises(ValueError, match="Tensor 'bad_vec'.*matrix tensors"):
            run_pipeline_python(spec, {})

    def test_validate_pipeline_spec_rejects_bad_swap_hwc_chw_shape(self):
        """Test swap_hwc_chw rejects 4D CHW tensor declarations."""
        spec = {
            "tensors": {
                "image_hwc": {
                    "dimensions": [1024, 768],
                    "channels": 3,
                    "usage": 6,
                },
                "bad_chw": {
                    "dimensions": [1, 3, 1024, 768],
                    "channels": 1,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "swap_hwc_chw",
                    "inputs": ["image_hwc"],
                    "outputs": ["bad_chw"],
                }
            ],
        }

        with pytest.raises(ValueError, match="swap_hwc_chw output 'bad_chw'"):
            validate_pipeline_spec(spec)

    def test_validate_pipeline_spec_accepts_swap_hwc_chw_shape(self):
        """Test swap_hwc_chw accepts SecureMR channelized CHW tensor declarations."""
        spec = {
            "tensors": {
                "image_hwc": {
                    "dimensions": [1024, 768],
                    "channels": 3,
                    "usage": 6,
                },
                "image_chw": {
                    "dimensions": [3, 1024],
                    "channels": 768,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "swap_hwc_chw",
                    "inputs": ["image_hwc"],
                    "outputs": ["image_chw"],
                }
            ],
        }

        validate_pipeline_spec(spec)

    def test_compare_outputs_success(self):
        """Test successful output comparison."""
        expected = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        actual = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

        result = compare_outputs(expected, actual)

        assert result.success
        assert result.error_message is None

    def test_compare_outputs_within_tolerance(self):
        """Test comparison within tolerance."""
        expected = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        actual = {"out": np.array([1.0001, 2.0001, 3.0001], dtype=np.float32)}

        result = compare_outputs(expected, actual, rtol=1e-3, atol=1e-3)

        assert result.success

    def test_compare_outputs_failure(self):
        """Test failed output comparison."""
        expected = {"out": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        actual = {"out": np.array([1.0, 2.0, 5.0], dtype=np.float32)}

        result = compare_outputs(expected, actual, rtol=1e-4, atol=1e-4)

        assert not result.success
        assert result.error_message is not None
        assert "out" in result.error_message

    def test_compare_outputs_missing_key(self):
        """Test comparison with missing output key."""
        expected = {"out1": np.array([1.0]), "out2": np.array([2.0])}
        actual = {"out1": np.array([1.0])}

        result = compare_outputs(expected, actual)

        assert not result.success
        assert "Missing output" in result.error_message

    def test_compare_outputs_shape_mismatch(self):
        """Test comparison with shape mismatch."""
        expected = {"out": np.array([1.0, 2.0, 3.0])}
        actual = {"out": np.array([1.0, 2.0])}

        result = compare_outputs(expected, actual)

        assert not result.success
        assert "Shape mismatch" in result.error_message


class TestIntegration:
    """Integration tests for the full py2smr workflow."""

    def test_full_workflow_arithmetic(self):
        """Test full workflow with arithmetic operations."""
        @trace(inputs=["input"], outputs=["output"])
        def normalize_image(input):
            return ops.arithmetic(input, "{0} / 255.0")

        # Create test input
        input_arr = np.array([[100, 200], [50, 150]], dtype=np.uint8)

        # Trace execution
        result, ctx = normalize_image.trace(input=input_arr)

        # Convert to pipeline spec
        spec = convert(ctx)

        # Verify spec structure
        assert spec["inputs"] == ["input"]
        assert spec["outputs"] == ["output"]
        assert len(spec["operators"]) == 1
        assert spec["operators"][0]["expression"] == "{0} / 255.0"

        # Verify result
        expected = input_arr.astype(np.float32) / 255.0
        np.testing.assert_allclose(result, expected)

    def test_full_workflow_multiple_ops(self):
        """Test full workflow with multiple operations."""
        @trace(inputs=["image"], outputs=["processed"])
        def preprocess(image):
            # Normalize to [0, 1]
            normalized = ops.arithmetic(image, "{0} / 255.0")
            # Scale to [-1, 1]
            scaled = ops.arithmetic(normalized, "{0} * 2.0 - 1.0")
            return scaled

        input_arr = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        result, ctx = preprocess.trace(image=input_arr)

        spec = convert(ctx)

        assert len(spec["operators"]) == 2
        assert spec["operators"][0]["expression"] == "{0} / 255.0"
        assert spec["operators"][1]["expression"] == "{0} * 2.0 - 1.0"

        # Verify result
        expected = input_arr.astype(np.float32) / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(result, expected)

    def test_full_workflow_elementwise_ops(self):
        """Test full workflow with elementwise operations."""
        @trace(inputs=["a", "b"], outputs=["result"])
        def clamp(a, b):
            min_val = ops.elementwise_min(a, b)
            max_val = ops.elementwise_max(a, b)
            return ops.elementwise_multiply(min_val, max_val)

        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=np.float32)

        result, ctx = clamp.trace(a=a, b=b)

        spec = convert(ctx)

        assert len(spec["operators"]) == 3
        assert "a" in spec["inputs"]
        assert "b" in spec["inputs"]

    def test_save_and_load_pipeline(self):
        """Test saving and loading pipeline JSON."""
        @trace(inputs=["x"], outputs=["y"])
        def double(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        _, ctx = double.trace(x=input_arr)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            convert(ctx, output=output_path)

            with open(output_path, "r") as f:
                loaded_spec = json.load(f)

            # Verify loaded spec is valid
            assert "tensors" in loaded_spec
            assert "operators" in loaded_spec
            assert loaded_spec["inputs"] == ["x"]
            assert loaded_spec["outputs"] == ["y"]
        finally:
            os.unlink(output_path)


class TestPythonExecutor:
    """Tests for the pure Python pipeline executor."""

    def test_run_pipeline_python_preserves_non_square_h_w_output_shape(self):
        """Host shape reconstruction must preserve schema [H, W] order."""
        spec = {
            "tensors": {
                "output": {"dimensions": [3, 5], "channels": 1, "data_type": 6},
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_MICROPHONE_PICO",
                    "inputs": [],
                    "outputs": ["output"],
                }
            ],
            "inputs": [],
            "outputs": ["output"],
        }

        outputs = run_pipeline_python(spec, {})

        assert outputs["output"].shape == (3, 5)

    def test_run_pipeline_python_basic(self):
        """Test basic pipeline execution with pure Python."""
        from securemr.py2smr.verifier import run_pipeline_python

        spec = {
            "tensors": {
                "input": {"dimensions": [2, 2], "channels": 1, "data_type": 6},
                "output": {"dimensions": [2, 2], "channels": 1, "data_type": 6},
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": ["input"],
                    "outputs": ["output"],
                    "expression": "{0} * 2.0",
                }
            ],
            "inputs": ["input"],
            "outputs": ["output"],
        }

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        outputs = run_pipeline_python(spec, {"input": input_arr})

        expected = input_arr * 2.0
        np.testing.assert_allclose(outputs["output"], expected)

    def test_run_pipeline_python_multiple_ops(self):
        """Test pipeline with multiple operators."""
        from securemr.py2smr.verifier import run_pipeline_python

        spec = {
            "tensors": {
                "input": {"dimensions": [2, 2], "channels": 1, "data_type": 6},
                "temp": {"dimensions": [2, 2], "channels": 1, "data_type": 6},
                "output": {"dimensions": [2, 2], "channels": 1, "data_type": 6},
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": ["input"],
                    "outputs": ["temp"],
                    "expression": "{0} / 255.0",
                },
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO",
                    "inputs": ["temp"],
                    "outputs": ["output"],
                    "expression": "{0} * 2.0 - 1.0",
                },
            ],
            "inputs": ["input"],
            "outputs": ["output"],
        }

        input_arr = np.array([[100.0, 200.0], [50.0, 150.0]], dtype=np.float32)
        outputs = run_pipeline_python(spec, {"input": input_arr})

        expected = input_arr / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(outputs["output"], expected)

    def test_run_pipeline_python_preserves_supplied_rectified_vst_outputs(self):
        """User-provided VST tensors should not be overwritten by host stubs."""
        from securemr.py2smr.verifier import run_pipeline_python

        spec = {
            "tensors": {
                "vst_right_image": {
                    "dimensions": [2, 2],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "vst_left_image": {
                    "dimensions": [2, 2],
                    "channels": 3,
                    "data_type": 1,
                    "is_placeholder": True,
                    "usage": 6,
                },
                "vst_timestamp": {"tensor_type": "timestamp", "is_placeholder": True},
                "vst_camera_matrix": {
                    "dimensions": [3, 3],
                    "channels": 1,
                    "data_type": 6,
                    "is_placeholder": True,
                    "usage": 6,
                },
            },
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO",
                    "inputs": [],
                    "outputs": [
                        "vst_right_image",
                        "vst_left_image",
                        "vst_timestamp",
                        "vst_camera_matrix",
                    ],
                }
            ],
            "inputs": [],
            "outputs": ["vst_right_image", "vst_left_image", "vst_timestamp", "vst_camera_matrix"],
        }
        right = np.ones((2, 2, 3), dtype=np.uint8) * 9
        left = np.ones((2, 2, 3), dtype=np.uint8) * 7

        outputs = run_pipeline_python(
            spec,
            {"vst_right_image": right, "vst_left_image": left},
        )

        np.testing.assert_array_equal(outputs["vst_right_image"], right)
        np.testing.assert_array_equal(outputs["vst_left_image"], left)

    def test_run_pipeline_python_decodes_mediapipe_face_postprocess(self):
        """Known face detector JavaScript postprocess should produce post_det."""
        from securemr.py2smr.verifier import run_pipeline_python

        coords_1 = np.zeros((512, 16), dtype=np.float32)
        coords_2 = np.zeros((384, 16), dtype=np.float32)
        scores_1 = np.full((512, 1), -10.0, dtype=np.float32)
        scores_2 = np.full((384, 1), -10.0, dtype=np.float32)
        scores_1[0, 0] = 10.0
        coords_1[0, :14] = [
            0.0,
            0.0,
            20.0,
            30.0,
            -5.0,
            -5.0,
            5.0,
            -5.0,
            0.0,
            0.0,
            -4.0,
            6.0,
            4.0,
            6.0,
        ]
        spec = {
            "tensors": {},
            "operators": [
                {
                    "type": "XR_SECURE_MR_OPERATOR_TYPE_JAVASCRIPT_PICO",
                    "inputs": [
                        {"name": "box_coords_1", "tensor": "box_coords_1"},
                        {"name": "box_coords_2", "tensor": "box_coords_2"},
                        {"name": "box_scores_1", "tensor": "box_scores_1"},
                        {"name": "box_scores_2", "tensor": "box_scores_2"},
                        {"name": "post_det_template", "tensor": "post_det_template"},
                    ],
                    "outputs": [{"name": "post_det", "tensor": "post_det"}],
                    "script": "function anchorFor(){} function decodeDetection(){}",
                }
            ],
            "inputs": [],
            "outputs": ["post_det"],
        }

        outputs = run_pipeline_python(
            spec,
            {
                "box_coords_1": coords_1,
                "box_coords_2": coords_2,
                "box_scores_1": scores_1,
                "box_scores_2": scores_2,
                "post_det_template": np.zeros((1, 21), dtype=np.float32),
            },
        )

        assert outputs["post_det"].shape == (1, 21)
        assert outputs["post_det"][0, 4] > 0.99

    def test_verify_with_python_executor(self):
        """Test verify function uses pure Python executor."""
        @trace(inputs=["x"], outputs=["y"])
        def double(x):
            return ops.arithmetic(x, "{0} * 2.0")

        input_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result, ctx = double.trace(x=input_arr)

        spec = convert(ctx)

        # Verify should work without native bindings
        verification = verify(
            pipeline=spec,
            inputs={"x": input_arr},
            expected_outputs={"y": result},
        )

        assert verification.success
