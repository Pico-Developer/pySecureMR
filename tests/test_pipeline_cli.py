import json

import numpy as np
import pytest

from pyspatialml import pipeline_cli


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_init_pipeline_refuses_existing_file_without_force(tmp_path):
    pipeline = tmp_path / "pipeline.json"

    assert pipeline_cli.init_pipeline(pipeline) == 0
    with pytest.raises(pipeline_cli.PipelineCliError, match="already exists"):
        pipeline_cli.init_pipeline(pipeline)

    assert pipeline_cli.init_pipeline(pipeline, force=True) == 0
    assert _read_json(pipeline) == {"tensors": {}, "operators": [], "inputs": [], "outputs": []}


def test_add_tensor_writes_matrix_descriptor_and_boundaries(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    pipeline_cli.add_tensor(
        pipeline,
        "image",
        shape="128,64,3",
        dtype="uint8",
        usage="matrix",
        is_input=True,
        is_output=True,
    )

    spec = _read_json(pipeline)
    tensor = spec["tensors"]["image"]
    assert tensor["dimensions"] == [128, 64]
    assert tensor["channels"] == 3
    assert tensor["data_type"] == 1
    assert tensor["usage"] == 6
    assert tensor["is_placeholder"] is True
    assert "flag" in tensor
    assert spec["inputs"] == ["image"]
    assert spec["outputs"] == ["image"]


def test_add_tensor_supports_scalar_values(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    pipeline_cli.add_tensor(
        pipeline,
        "threshold",
        shape="1",
        dtype="float32",
        usage="scalar",
        value="0.5",
    )

    tensor = _read_json(pipeline)["tensors"]["threshold"]
    assert tensor["dimensions"] == [1, 1]
    assert tensor["channels"] == 1
    assert tensor["usage"] == 2
    assert tensor["value"] == [0.5]
    assert "flag" not in tensor


def test_add_tensor_rejects_duplicate_and_invalid_dtype(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="already exists"):
        pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unsupported dtype"):
        pipeline_cli.add_tensor(pipeline, "y", shape="1,1", dtype="bad_dtype")


def test_add_op_writes_common_operator_fields(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)

    pipeline_cli.add_op(
        pipeline,
        "arithmetic",
        inputs=["x"],
        outputs=["y"],
        expression="{0} + 1.0",
        attrs=["unused"],
    )

    op = _read_json(pipeline)["operators"][0]
    assert op["type"] == "XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO"
    assert op["inputs"] == ["x"]
    assert op["outputs"] == ["y"]
    assert op["attrs"] == ["unused"]
    assert op["expression"] == "{0} + 1.0"


def test_add_op_arithmetic_requires_expression(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)

    with pytest.raises(pipeline_cli.PipelineCliError, match="Arithmetic operators require --expression"):
        pipeline_cli.add_op(pipeline, "arithmetic", inputs=["x"], outputs=["y"])


@pytest.mark.parametrize(
    ("op_type", "inputs", "outputs", "message"),
    [
        ("convert_color", ["x"], ["y"], "convert_color operators require --flag"),
        ("customized_compare", ["x", "y"], ["y"], "customized_compare operators require --attr"),
        ("javascript", ["x"], ["y"], "javascript operators require --attr"),
        ("render_text", ["gltf"], [], "render_text operators require --attr config and --attr text"),
        ("update_gltf", ["gltf"], [], "update_gltf operators require --attr"),
        ("run_model_inference", ["x"], ["y"], "run_model_inference operators require --model"),
    ],
)
def test_add_op_requires_operator_metadata(tmp_path, op_type, inputs, outputs, message):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)
    pipeline_cli.add_tensor(pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")

    with pytest.raises(pipeline_cli.PipelineCliError, match=message):
        pipeline_cli.add_op(pipeline, op_type, inputs=inputs, outputs=outputs)


@pytest.mark.parametrize(
    ("op_type", "inputs", "outputs", "message"),
    [
        ("convert_color", [], ["y"], "convert_color operators require exactly 1 input"),
        ("elementwise_min", ["x"], ["y"], "elementwise_min operators require exactly 2 input"),
        ("solve_p_n_p", ["x", "y"], ["y", "z"], "solve_p_n_p operators require exactly 3 input"),
        ("rectified_vst_access", ["x"], ["y"], "rectified_vst_access operators require exactly 0 input"),
        ("svd", ["x"], ["y"], "svd operators require exactly 3 output"),
        ("javascript", ["x"], [], "javascript operators require at least 1 output"),
    ],
)
def test_add_op_rejects_bad_operator_arity(tmp_path, op_type, inputs, outputs, message):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    for name in {"x", "y", "z", *inputs, *outputs}:
        pipeline_cli.add_tensor(pipeline, name, shape="2,2", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match=message):
        pipeline_cli.add_op(pipeline, op_type, inputs=inputs, outputs=outputs)


def test_add_op_accepts_required_operator_metadata(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    model = tmp_path / "model.tflite"
    model.write_bytes(b"model")
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="2,2", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="2,2", dtype="float32", is_output=True)
    pipeline_cli.add_tensor(pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")

    pipeline_cli.add_op(pipeline, "convert_color", inputs=["x"], outputs=["y"], flag="4")
    pipeline_cli.add_op(pipeline, "customized_compare", inputs=["x", "y"], outputs=["y"], attrs=[">="])
    pipeline_cli.add_op(pipeline, "javascript", inputs=["x"], outputs=["y"], attrs=["out = in;"])
    pipeline_cli.add_op(pipeline, "render_text", inputs=["gltf"], outputs=[], attrs=["bold#en-us#512#64", "hello"])
    pipeline_cli.add_op(pipeline, "update_gltf", inputs=["gltf"], outputs=[], attrs=["texture"])
    pipeline_cli.add_op(pipeline, "run_model_inference", inputs=["x"], outputs=["y"], model="model.tflite")

    operators = _read_json(pipeline)["operators"]
    assert operators[0]["flag"] == 4
    assert operators[1]["attrs"] == [">="]
    assert operators[2]["attrs"] == ["out = in;"]
    assert operators[3]["attrs"] == ["bold#en-us#512#64", "hello"]
    assert operators[4]["attrs"] == ["texture"]
    assert operators[5]["model"]["bin_path"] == "model.tflite"


def test_add_op_rejects_unknown_tensor_reference(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unknown tensor"):
        pipeline_cli.add_op(pipeline, "assignment", inputs=["x"], outputs=["missing"])


def test_remove_tensor_removes_tensor_and_boundaries(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32", is_input=True, is_output=True)

    assert pipeline_cli.remove_tensor(pipeline, "x") == 0

    spec = _read_json(pipeline)
    assert "x" not in spec["tensors"]
    assert spec["inputs"] == []
    assert spec["outputs"] == []


def test_remove_tensor_rejects_operator_references_without_force(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="1,1", dtype="float32", is_output=True)
    pipeline_cli.add_op(pipeline, "assignment", inputs=["x"], outputs=["y"])

    with pytest.raises(pipeline_cli.PipelineCliError) as exc_info:
        pipeline_cli.remove_tensor(pipeline, "x")

    message = str(exc_info.value)
    assert "Tensor 'x' is referenced by operator(s): #0 XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO" in message
    assert "--force" in message
    assert "x" in _read_json(pipeline)["tensors"]


def test_remove_tensor_force_allows_dangling_operator_reference(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "y", shape="1,1", dtype="float32", is_output=True)
    pipeline_cli.add_op(pipeline, "assignment", inputs=["x"], outputs=["y"])

    assert pipeline_cli.remove_tensor(pipeline, "x", force=True) == 0

    spec = _read_json(pipeline)
    assert "x" not in spec["tensors"]
    assert spec["inputs"] == []
    assert spec["operators"][0]["inputs"] == ["x"]


def test_remove_tensor_rejects_missing_tensor(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    with pytest.raises(pipeline_cli.PipelineCliError, match="Tensor not found"):
        pipeline_cli.remove_tensor(pipeline, "missing")


def test_add_op_rejects_xr_only_operator_when_manifest_supports_spatial(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "display.json"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["spatial"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")

    with pytest.raises(pipeline_cli.PipelineCliError, match="XR-only operator.*only includes spatial"):
        pipeline_cli.add_op(
            pipeline,
            "render_text",
            inputs=["gltf"],
            outputs=[],
            attrs=["bold#en-us#512#64", "hello"],
        )


def test_add_op_rejects_spatial_only_operator_when_manifest_supports_xr(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "scene.json"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "scene", "path": "pipeline/scene.json"}],
            "runtime": {"supported_modes": ["xr"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "component", shape="1,1", dtype="uint8")
    pipeline_cli.add_tensor(pipeline, "out", shape="1,1", dtype="uint8")

    with pytest.raises(pipeline_cli.PipelineCliError, match="Spatial-only operator.*only includes xr"):
        pipeline_cli.add_op(
            pipeline,
            "update_component",
            inputs=["component"],
            outputs=["out"],
            attrs=["visibility"],
        )


def test_add_op_with_xr_only_operator_narrows_both_mode_manifest(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "display.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")

    pipeline_cli.add_op(
        pipeline,
        "render_text",
        inputs=["gltf"],
        outputs=[],
        attrs=["bold#en-us#512#64", "hello"],
    )

    assert _read_json(manifest)["runtime"]["supported_modes"] == ["xr"]


def test_add_op_with_spatial_only_operator_narrows_both_mode_manifest(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "scene.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "scene", "path": "pipeline/scene.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "component", shape="1,1", dtype="uint8")
    pipeline_cli.add_tensor(pipeline, "out", shape="1,1", dtype="uint8")

    pipeline_cli.add_op(
        pipeline,
        "update_component",
        inputs=["component"],
        outputs=["out"],
        attrs=["visibility"],
    )

    assert _read_json(manifest)["runtime"]["supported_modes"] == ["spatial"]


def test_remove_op_widens_manifest_when_no_exclusive_operators_remain(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "display.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["xr", "spatial"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")
    pipeline_cli.add_op(
        pipeline,
        "render_text",
        inputs=["gltf"],
        outputs=[],
        attrs=["bold#en-us#512#64", "hello"],
    )
    assert _read_json(manifest)["runtime"]["supported_modes"] == ["xr"]

    pipeline_cli.remove_op(pipeline, 0)

    assert _read_json(manifest)["runtime"]["supported_modes"] == ["xr", "spatial"]
    assert _read_json(pipeline)["operators"] == []


def test_add_op_keeps_manifest_narrowed_by_mode_specific_sibling(tmp_path):
    package = tmp_path / "pkg"
    neutral_pipeline = package / "pipeline" / "neutral.json"
    display_pipeline = package / "pipeline" / "display.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [
                {"id": "neutral", "path": "pipeline/neutral.json"},
                {"id": "display", "path": "pipeline/display.json"},
            ],
            "runtime": {"supported_modes": ["xr"]},
        },
    )
    pipeline_cli.init_pipeline(neutral_pipeline)
    pipeline_cli.add_tensor(neutral_pipeline, "x", shape="1,1", dtype="float32")
    pipeline_cli.add_tensor(neutral_pipeline, "y", shape="1,1", dtype="float32")
    pipeline_cli.init_pipeline(display_pipeline)
    pipeline_cli.add_tensor(display_pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")
    pipeline_cli.add_op(
        display_pipeline,
        "render_text",
        inputs=["gltf"],
        outputs=[],
        attrs=["bold#en-us#512#64", "hello"],
    )

    pipeline_cli.add_op(neutral_pipeline, "assignment", inputs=["x"], outputs=["y"])

    assert _read_json(manifest)["runtime"]["supported_modes"] == ["xr"]


def test_remove_op_validation_failure_leaves_pipeline_and_manifest_unchanged(tmp_path):
    package = tmp_path / "pkg"
    pipeline = package / "pipeline" / "display.json"
    manifest = package / "manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "2",
            "id": "demo",
            "pipelines": [{"id": "display", "path": "pipeline/display.json"}],
            "runtime": {"supported_modes": ["xr"]},
        },
    )
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")
    pipeline_cli.add_op(
        pipeline,
        "render_text",
        inputs=["gltf"],
        outputs=[],
        attrs=["bold#en-us#512#64", "hello"],
    )
    original_manifest = manifest.read_text(encoding="utf-8")
    spec = _read_json(pipeline)
    spec["tensors"]["x"]["dimensions"] = [1]
    _write_json(pipeline, spec)
    original_pipeline = pipeline.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="matrix tensors must have at least 2 dimensions"):
        pipeline_cli.remove_op(pipeline, 0)

    assert manifest.read_text(encoding="utf-8") == original_manifest
    assert pipeline.read_text(encoding="utf-8") == original_pipeline


def test_remove_op_rejects_out_of_range_index(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)

    with pytest.raises(pipeline_cli.PipelineCliError, match="Operator index out of range"):
        pipeline_cli.remove_op(pipeline, 0)


def test_add_op_rejects_mixing_xr_and_spatial_only_operators_without_manifest(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "gltf", shape="1,1", dtype="uint8", usage="gltf")
    pipeline_cli.add_tensor(pipeline, "component", shape="1,1", dtype="uint8")
    pipeline_cli.add_tensor(pipeline, "out", shape="1,1", dtype="uint8")
    pipeline_cli.add_op(
        pipeline,
        "render_text",
        inputs=["gltf"],
        outputs=[],
        attrs=["bold#en-us#512#64", "hello"],
    )

    with pytest.raises(pipeline_cli.PipelineCliError, match="mix XR-only and Spatial-only"):
        pipeline_cli.add_op(
            pipeline,
            "update_component",
            inputs=["component"],
            outputs=["out"],
            attrs=["visibility"],
        )


def test_add_op_model_requires_tflite_and_writes_inline_metadata(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "input", shape="1,4", dtype="float32", is_input=True)
    pipeline_cli.add_tensor(pipeline, "output", shape="1,2", dtype="float32", is_output=True)

    with pytest.raises(pipeline_cli.PipelineCliError, match=".tflite"):
        pipeline_cli.add_op(
            pipeline,
            "run_model_inference",
            inputs=["input"],
            outputs=["output"],
            model="model/demo.bin",
        )

    pipeline_cli.add_op(
        pipeline,
        "run_model_inference",
        inputs=["input"],
        outputs=["output"],
        model="model/demo.tflite",
        model_name="demo",
        model_target="cpu",
        cpu_target_num_threads=4,
    )

    op = _read_json(pipeline)["operators"][0]
    assert op["model_type"] == "tflite"
    assert op["model_target"] == "cpu"
    assert op["cpu_target_num_threads"] == 4
    assert op["model"]["bin_path"] == "model/demo.tflite"
    assert op["model"]["model_name"] == "demo"
    assert "model_file" not in op
    assert "model_asset" not in op
    assert "model_id" not in op


def test_set_input_and_set_output_mark_placeholders(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")
    pipeline_cli.add_tensor(pipeline, "y", shape="1,1", dtype="float32")

    pipeline_cli.set_input(pipeline, ["x"])
    pipeline_cli.set_output(pipeline, ["y"])

    spec = _read_json(pipeline)
    assert spec["inputs"] == ["x"]
    assert spec["outputs"] == ["y"]
    assert spec["tensors"]["x"]["is_placeholder"] is True
    assert spec["tensors"]["y"]["is_placeholder"] is True

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unknown tensor"):
        pipeline_cli.set_input(pipeline, ["missing"])


def test_validate_pipeline_reports_bad_references(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        json.dumps({"tensors": {}, "operators": [], "inputs": ["missing"], "outputs": []}),
        encoding="utf-8",
    )

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unknown inputs tensor"):
        pipeline_cli.validate_pipeline(pipeline)


def test_inspect_pipeline_prints_summary(capsys, tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32", is_input=True)

    assert pipeline_cli.inspect_pipeline(pipeline) == 0

    captured = capsys.readouterr()
    assert "Tensors: 1" in captured.out
    assert "Operators: 0" in captured.out
    assert "Inputs: x" in captured.out


def test_trace_pipeline_writes_converted_spec(tmp_path):
    source = tmp_path / "source.py"
    sample = tmp_path / "sample.npy"
    output = tmp_path / "pipeline.json"
    source.write_text(
        "\n".join(
            [
                "from securemr.py2smr import trace, ops",
                "@trace(inputs=['x'], outputs=['y'])",
                "def build(x):",
                "    return ops.arithmetic(x, '{0} * 2.0', output_name='y')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    np.save(sample, np.ones((2, 2), dtype=np.float32))

    assert pipeline_cli.trace_pipeline(
        source,
        function_name="build",
        output=output,
        inputs=[f"x={sample}"],
    ) == 0

    spec = _read_json(output)
    assert spec["inputs"] == ["x"]
    assert spec["outputs"] == ["y"]
    assert spec["operators"][0]["expression"] == "{0} * 2.0"


def test_trace_pipeline_requires_trace_decorator_and_input_files(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("def build(x):\n    return x\n", encoding="utf-8")

    with pytest.raises(pipeline_cli.PipelineCliError, match="not traceable"):
        pipeline_cli.trace_pipeline(
            source,
            function_name="build",
            output=tmp_path / "pipeline.json",
            inputs=[],
        )

    with pytest.raises(pipeline_cli.PipelineCliError, match="file not found"):
        pipeline_cli.trace_pipeline(
            tmp_path / "missing.py",
            function_name="build",
            output=tmp_path / "pipeline.json",
            inputs=[],
        )
