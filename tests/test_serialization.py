#!/usr/bin/env python3
"""
Test script for JSON serialization functionality.
"""

import json
import os
import tempfile
import numpy as np

import securemr as smr
from securemr.serialization import DeserializedPipeline


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
        "inputs": ["input_tensor"],
        "outputs": ["output_tensor"],
        "tensors": {
            "input_tensor": {
                "channels": 1,
                "data_type": int(smr.EDataType.FLOAT32),
                "dimensions": [4, 4],
                "is_placeholder": True,
                "usage": 6
            },
            "output_tensor": {
                "channels": 1,
                "data_type": int(smr.EDataType.FLOAT32),
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
        output_view = output_tensor.numpy()
        output_data = output_view[:, :, 0] if output_view.ndim == 3 else output_view

        # Verify result
        expected = input_data + 2.0
        assert np.allclose(output_data, expected, rtol=1e-4, atol=1e-4), f"Expected {expected}, got {output_data}"

        print("✓ Basic DeserializedPipeline test passed")

    finally:
        # Clean up
        os.unlink(temp_file)
