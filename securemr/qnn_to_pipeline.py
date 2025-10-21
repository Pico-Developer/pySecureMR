"""Utilities to build a SecureMR pipeline JSON for QNN context binaries.

The generated pipeline contains two operators:
  1. A JS scripting operator that emits random tensors matching the QNN inputs.
  2. A model inference operator represented in the serialization spec.

The resulting ``pipeline.json`` can be loaded with :class:`DeserializedPipeline`,
which swaps the serialized model inference operator for
``ModelInferenceOperator`` at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import securemr as smr

from .serialization import type_to_name
from .utils import convert_from_dtype, ensure_tensor_dimensions, mat_flag, qnn_dtype_to_smr


@dataclass
class TensorInfo:
    """Represents tensor metadata resolved from the QNN context binary."""

    qnn_name: str
    tensor_name: str
    spatial_dims: List[int]
    channels: int
    dtype: smr.EDataType
    total_size: int


def _resolve_qnn_tool() -> str | None:
    """Return the absolute path to ``qnn-context-binary-utility`` if available."""
    if (env_root := os.getenv("QNN_SDK_ROOT")):
        candidate = (
            Path(env_root)
            / "bin"
            / "x86_64-linux-clang"
            / "qnn-context-binary-utility"
        )
        if candidate.exists():
            return str(candidate)
    return shutil.which("qnn-context-binary-utility")


def _load_context_info(context_binary: str) -> Dict:
    """Load JSON metadata for ``context_binary`` using the QNN utility."""
    tool_path = _resolve_qnn_tool()
    json_path = f"{context_binary}.json"

    if tool_path:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run(
                [
                    tool_path,
                    "--context_binary",
                    context_binary,
                    "--json_file",
                    tmp_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with open(tmp_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "qnn-context-binary-utility not found. Please ensure QNN_SDK_ROOT "
                "is set or the utility is available on PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to invoke qnn-context-binary-utility: {exc.stderr.decode()}"
            ) from exc
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    raise RuntimeError(
        "Unable to obtain context metadata. Ensure qnn-context-binary-utility is "
        "available or provide a sidecar JSON file alongside the binary."
    )


def _sanitize_tensor_name(name: str, suffix: str = "") -> str:
    base = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if suffix:
        base = f"{base}_{suffix}"
    base = base.strip("_") or "tensor"
    return base


def _infer_spatial_dims(dimensions: Sequence[int]) -> tuple[List[int], int]:
    dims = [int(d) for d in dimensions]
    if not dims:
        return [1], 1

    if len(dims) == 1:
        return [dims[0]], 1

    if len(dims) >= 3:
        channels = max(int(dims[-1]), 1)
        spatial = [max(int(v), 1) for v in dims[1:-1]]
        if not spatial:
            spatial = [channels]
            channels = 1
        return spatial, channels

    # len == 2 -> treat as [batch, features]
    return [max(int(dims[1]), 1)], 1


def _tensor_info_from_entry(entry: Dict, suffix: str = "") -> TensorInfo:
    info = entry.get("info", {})
    qnn_name = str(info.get("name", "tensor"))
    tensor_name = _sanitize_tensor_name(qnn_name, suffix)
    spatial, channels = _infer_spatial_dims(info.get("dimensions", []))
    dtype = qnn_dtype_to_smr(info.get("dataType", ""))
    total_size = channels
    for d in spatial:
        total_size *= max(int(d), 1)
    return TensorInfo(
        qnn_name=qnn_name,
        tensor_name=tensor_name,
        spatial_dims=[int(d) for d in spatial],
        channels=int(max(channels, 1)),
        dtype=dtype,
        total_size=int(max(total_size, 1)),
    )


def _build_js_script(inputs: Sequence[TensorInfo]) -> str:
    lines: List[str] = [
        "function randomUniform() {",
        "    return Math.random();",
        "}",
    ]
    for idx, info in enumerate(inputs):
        var_name = f"out_{info.tensor_name}"
        lines.append(f"var {var_name} = [];")
        lines.append(f"for (var i = 0; i < {info.total_size}; ++i) {{")
        lines.append(f"    {var_name}[i] = randomUniform();")
        lines.append("}")
    return "\n".join(lines)


def _make_tensor_entry(info: TensorInfo, *, is_placeholder: bool) -> Dict:
    flag = mat_flag(info.dtype, info.channels)
    dims = ensure_tensor_dimensions(info.spatial_dims)
    return {
        "dimensions": dims,
        "channels": info.channels,
        "data_type": convert_from_dtype(info.dtype, source="smr"),
        "is_placeholder": bool(is_placeholder),
        "usage": 2 if is_placeholder else 6,
        "flag": flag,
    }

def _make_dummy_tensor(is_placeholder: bool) -> Dict:
    flag = mat_flag(smr.EDataType.FLOAT32, 1)
    return {
        "dimensions": [1, 1],
        "channels": 1,
        "data_type": convert_from_dtype(smr.EDataType.FLOAT32, source="smr"),
        "is_placeholder": bool(is_placeholder),
        "usage": 2 if is_placeholder else 6,
        "flag": flag,
    }


def build_pipeline_spec(
    context_binary: str,
    *,
    device: str = "host",
) -> Dict:
    """Return a pipeline spec for ``context_binary``."""
    context_binary = os.path.abspath(context_binary)
    metadata = _load_context_info(context_binary)
    graphs = metadata.get("info", {}).get("graphs", [])
    if not graphs:
        raise RuntimeError("No graphs found in context binary metadata.")

    graph_info = graphs[0].get("info", {})
    raw_inputs = graph_info.get("graphInputs", [])
    raw_outputs = graph_info.get("graphOutputs", [])

    if not raw_inputs:
        raise RuntimeError("Model must expose at least one input tensor.")

    input_infos = [
        _tensor_info_from_entry(entry, suffix="input") for entry in raw_inputs
    ]
    output_infos = [
        _tensor_info_from_entry(entry, suffix="output") for entry in raw_outputs
    ]

    tensors: Dict[str, Dict] = {}
    for info in input_infos:
        tensors[info.tensor_name] = _make_tensor_entry(info, is_placeholder=False)
    for info in output_infos:
        tensors[info.tensor_name] = _make_tensor_entry(info, is_placeholder=True)

    tensors["dummy_input"] = _make_dummy_tensor(True)

    js_script = _build_js_script(input_infos)

    js_operator = {
        "type": type_to_name(smr.EOperatorType.JS_SCRIPTING),
        "attrs": [js_script],
        "inputs": ["dummy_input"],
        "outputs": [info.tensor_name for info in input_infos],
    }

    model_operator = {
        "type": type_to_name(smr.EOperatorType.RUN_MODEL_INFERENCE),
        "inputs": [
            {"name": info.qnn_name, "tensor": info.tensor_name} for info in input_infos
        ],
        "outputs": [
            {"name": info.qnn_name, "tensor": info.tensor_name} for info in output_infos
        ],
        "model_name": os.path.basename(context_binary).split(".")[0],
        "model": os.path.basename(context_binary),
        "model_dir": os.path.dirname(context_binary)
    }

    spec = {
        "metadata": {"version": 1},
        "tensors": tensors,
        "operators": [js_operator, model_operator],
        "inputs": [],
        "outputs": [info.tensor_name for info in output_infos],
    }
    return spec


def save_pipeline(spec: Dict, output_path: str | os.PathLike) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SecureMR pipeline for QNN context binary.")
    parser.add_argument("context_binary", help="Path to the QNN .bin context binary")
    parser.add_argument(
        "--output",
        default="pipeline.json",
        help="Output path for the generated pipeline JSON (default: pipeline.json)",
    )
    parser.add_argument(
        "--device",
        default="host",
        help="Execution device for ModelInferenceOperator (default: host)",
    )
    args = parser.parse_args(argv)

    spec = build_pipeline_spec(args.context_binary, device=args.device)
    save_pipeline(spec, args.output)
    print(f"Pipeline saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
