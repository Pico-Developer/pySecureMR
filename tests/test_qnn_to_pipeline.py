import os
import shutil
from pathlib import Path

import pytest

import securemr as smr
from securemr.qnn_to_pipeline import build_pipeline_spec, save_pipeline
from securemr.serialization import DeserializedPipeline

pytestmark = pytest.mark.skipif(
    not os.getenv("QNN_SDK_ROOT"),
    reason="QNN_SDK_ROOT is required for QNN model inference tests.",
)


def _prepare_context_binary(tmp_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    bin_src = project_root / "examples" / "mnistwild" / "mnist.serialized.bin"
    json_src = project_root.parent / "SecureMR_Samples" / "assets" / "mnistwild" / "mnist.serialized.json"
    if not json_src.exists():
        raise FileNotFoundError(f"Missing sample QNN JSON: {json_src}")
    dest_bin = tmp_path / "mnist.serialized.bin"
    shutil.copy(bin_src, dest_bin)
    shutil.copy(json_src, tmp_path / "mnist.serialized.bin.json")
    return dest_bin


def test_qnn_pipeline_execution(tmp_path):
    context_binary = _prepare_context_binary(tmp_path)
    spec = build_pipeline_spec(str(context_binary))

    assert len(spec["operators"]) == 2
    assert spec["operators"][0]["type"] == "js_scripting"
    assert spec["operators"][1]["type"] == "run_algorithm"

    pipeline_path = tmp_path / "pipeline.json"
    save_pipeline(spec, pipeline_path)

    pipeline = DeserializedPipeline(str(pipeline_path))
    try:
        outputs = pipeline()
        if isinstance(outputs, smr.Tensor):
            outputs = [outputs]
        assert outputs, "Expected at least one output tensor."
        for tensor in outputs:
            assert isinstance(tensor, smr.Tensor)
            assert tensor.get_type_flag() != 0
    finally:
        pipeline.close()
