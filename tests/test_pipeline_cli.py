import json

import numpy as np
import pytest

from pyspatialml import pipeline_cli


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_add_op_rejects_unknown_tensor_reference(tmp_path):
    pipeline = tmp_path / "pipeline.json"
    pipeline_cli.init_pipeline(pipeline)
    pipeline_cli.add_tensor(pipeline, "x", shape="1,1", dtype="float32")

    with pytest.raises(pipeline_cli.PipelineCliError, match="Unknown tensor"):
        pipeline_cli.add_op(pipeline, "assignment", inputs=["x"], outputs=["missing"])


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
