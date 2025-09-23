#!/usr/bin/env python3
"""
Test script for JSON serialization functionality.
"""

import json
import pathlib
import sys
import os
import tempfile
import numpy as np

import securemr as smr
from securemr.serialization import DeserializedPipeline, JSONSerialization


def test_serialization():
    """Test the JSON serialization functionality."""
    print("Testing JSON serialization...")

    # Create a simple pipeline
    pipeline = smr.Pipeline()

    # Create some placeholder tensors (similar to mnistwild example)
    ph_affine = pipeline.allocate_placeholder([2, 3], 280577)  # FLOAT32 matrix
    ph_image = pipeline.allocate_placeholder([2464, 3248], 71683)  # UINT8 BGR image
    ph_output = pipeline.allocate_placeholder([224, 224], 280577)  # FLOAT32 output

    # Serialize the pipeline
    spec = JSONSerialization.serialize_pipeline(pipeline)

    # Verify the structure
    assert "metadata" in spec
    assert "tensors" in spec
    assert "operators" in spec

    # Check metadata
    assert spec["metadata"]["version"] == 1

    # Check tensors
    expected_tensors = [
        "right_eye_uint8", "left_eye_uint8", "timestamp_tensor", "camera_matrix_tensor",
        "affine_tensor", "crop_rgb_tensor", "cropped_image", "crop_gray_tensor",
        "crop_float_tensor", "normalized_input_tensor", "predicted_score", "predicted_class"
    ]

    for tensor_name in expected_tensors:
        assert tensor_name in spec["tensors"], f"Missing tensor: {tensor_name}"

    # Check operators
    operator_types = [op["type"] for op in spec["operators"]]
    expected_operators = [
        "camera_access", "get_affine", "apply_affine", "assignment",
        "cvt_color", "type_convert", "arithmetic"
    ]

    for op_type in expected_operators:
        assert op_type in operator_types, f"Missing operator: {op_type}"

    # Test file serialization
    test_file = "/tmp/test_pipeline.json"
    JSONSerialization.serialize_to_file(pipeline, test_file)

    # Verify file exists and is valid JSON
    assert os.path.exists(test_file), "Serialization file not created"

    with open(test_file, 'r') as f:
        file_spec = json.load(f)

    assert file_spec == spec, "File content doesn't match in-memory spec"

    # Test deserialization
    deserialized = JSONSerialization.deserialize_from_file(test_file)
    assert deserialized == spec, "Deserialized content doesn't match original"

    print("✓ All tests passed!")
    print(f"Generated JSON file: {test_file}")
    print("File kept for inspection (not cleaned up)")


def test_with_model():
    """Test serialization with QNN model."""
    print("Testing serialization with QNN model...")

    pipeline = smr.Pipeline()

    # Create a mock QNN model (we'll just use a dummy object)
    class MockQnnModel:
        def __init__(self):
            self.name = "test_model"

    model = MockQnnModel()

    # Serialize with model
    spec = JSONSerialization.serialize_pipeline(
        pipeline,
        model=model,
        model_asset="test_model.serialized.bin",
        model_name="test_model"
    )

    # Check that run_algorithm operator was added
    operator_types = [op["type"] for op in spec["operators"]]
    assert "run_algorithm" in operator_types, "run_algorithm operator not found"

    # Find the run_algorithm operator
    run_algo_op = None
    for op in spec["operators"]:
        if op["type"] == "run_algorithm":
            run_algo_op = op
            break

    assert run_algo_op is not None, "run_algorithm operator not found"
    assert run_algo_op["model_asset"] == "test_model.serialized.bin"
    assert run_algo_op["model_name"] == "test_model"

    print("✓ Model serialization test passed!")


def test_deserialized_pipeline_basic():
    """Test basic DeserializedPipeline functionality."""

    # Create a simple pipeline first with only supported operators
    p = smr.Pipeline()

    # Create placeholder for input and output
    input_shape = [4, 4]
    flag = int(smr.EDataType.FLOAT32) | int(smr.BaseType.MAT) | (int(smr.BaseType.CHANNEL_MASK) & 1)
    print(f"Flag value: {flag}")

    ph_in = p.allocate_placeholder(input_shape, flag)
    ph_out = p.allocate_placeholder(input_shape, flag)

    # Check what flag the pipeline actually created
    t_in = p.query_local_tensor(ph_in)
    t_out = p.query_local_tensor(ph_out)
    print(f"Input tensor flag: {t_in.get_type_flag()}")
    print(f"Output tensor flag: {t_out.get_type_flag()}")

    # Create an arithmetic operator (this should be supported)
    op_id = p.allocate_operator(smr.EOperatorType.ARITHMETIC_COMPOSE, ["{0} + 2.0"])

    op = p.query_operator(op_id)
    t_in = p.query_local_tensor(ph_in)
    t_out = p.query_local_tensor(ph_out)

    # Connect operator
    op.data_as_operand(t_in, 0)
    op.connect_result_to_data_array(0, t_out)

    # Create a simple JSON spec manually to avoid unsupported operators
    pipeline_spec = {
        "metadata": {"version": 1},
        "tensors": {
            "input_tensor": {
                "channels": 1,
                "data_type": 6,  # FLOAT32
                "dimensions": [4, 4],
                "is_placeholder": True,
                "usage": 6
            },
            "output_tensor": {
                "channels": 1,
                "data_type": 6,  # FLOAT32
                "dimensions": [4, 4],
                "is_placeholder": True,
                "usage": 6
            }
        },
        "operators": [
            {
                "type": "arithmetic",
                "expression": "{0} + 2.0",
                "inputs": ["input_tensor"],
                "outputs": ["output_tensor"]
            }
        ]
    }

    # Serialize to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(pipeline_spec, f, indent=2)
        temp_file = f.name

    try:
        # Test DeserializedPipeline
        pipeline = DeserializedPipeline(temp_file)

        # Create test input
        input_data = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2, 4.2],
            [1.3, 2.3, 3.3, 4.3],
        ], dtype=np.float32)

        # Execute pipeline
        output_tensor = pipeline(input_data)

        # Convert output to numpy
        output_data = np.frombuffer(output_tensor.to_bytes(), dtype=np.float32).reshape(4, 4)

        # Verify result
        expected = input_data + 2.0
        assert np.allclose(output_data, expected, rtol=1e-4, atol=1e-4), f"Expected {expected}, got {output_data}"

        print("✓ Basic DeserializedPipeline test passed")

    finally:
        # Clean up
        os.unlink(temp_file)
