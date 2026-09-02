import numpy as np
import pytest

from securemr.py2smr import convert, ops, trace
from .conftest import run_op_test


@trace(inputs=["gltf"], outputs=["out"])
def traced_render_text(gltf):
    ops.render_text(
        gltf,
        text="hello",
        language_and_locale="en-us",
        canvas_width=256,
        canvas_height=64,
        typeface="bold",
    )
    dummy = np.array([1], dtype=np.int32)
    return ops.assignment(dummy, np.array([0], dtype=np.int32), output_name="out")


def test_render_text_host():
    gltf = np.zeros((1,), dtype=np.uint8)
    _, verification = run_op_test(
        traced_render_text,
        {"gltf": gltf},
        expected_output_name="out",
        test_device=False,
    )
    assert verification.success


@pytest.mark.parametrize(
    "optional_names",
    [
        ("colors",),
        ("texture_id",),
        ("font_size",),
        ("colors", "font_size"),
        ("start", "colors", "texture_id", "font_size"),
    ],
)
def test_render_text_serializes_compacted_optional_input_indices(optional_names):
    """Optional tensors must retain their names after compacting inputs."""
    @trace(inputs=["gltf", *optional_names], outputs=["out"])
    def traced(gltf, **optional_inputs):
        ops.render_text(
            gltf,
            text="hello",
            language_and_locale="en-us",
            canvas_width=256,
            canvas_height=64,
            start_position=optional_inputs.get("start"),
            colors=optional_inputs.get("colors"),
            texture_id=optional_inputs.get("texture_id"),
            font_size=optional_inputs.get("font_size"),
        )
        return ops.assignment(
            np.array([1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            output_name="out",
        )

    inputs = {"gltf": np.zeros((1,), dtype=np.uint8)}
    inputs.update({
        name: np.array([index + 1], dtype=np.float32)
        for index, name in enumerate(optional_names)
    })

    _, ctx = traced.trace(**inputs)
    render_spec = next(
        operator for operator in convert(ctx)["operators"]
        if operator["type"].endswith("RENDER_TEXT_PICO")
    )

    assert render_spec["gltf"] == "gltf"
    assert render_spec["inputs"] == ["gltf", *optional_names]
    assert render_spec["start"] == ("start" if "start" in optional_names else [0.0, 0.0])
    assert render_spec["colors"] == ("colors" if "colors" in optional_names else [[255, 255, 255, 255], [0, 0, 0, 0]])
    assert render_spec["texture_id"] == ("texture_id" if "texture_id" in optional_names else 0)
    assert render_spec["font_size"] == ("font_size" if "font_size" in optional_names else 16.0)


def test_render_text_device_skip():
    pytest.skip("render_text requires glTF placeholder; host-only stub")
