# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helpers for authoring SpatialML packages."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]
SUPPORTED_EXECUTION_MODES = {"xr", "spatial"}


@dataclass(frozen=True)
class PipelinePackageEntry:
    """Manifest entry for one pipeline JSON file in a SpatialML package."""

    id: str
    path: str

    def to_dict(self) -> JsonDict:
        """Return the pipeline zoo representation."""
        return {"id": self.id, "path": _normalize_package_path(self.path)}


@dataclass(frozen=True)
class PipelineZooPackageSpec:
    """Top-level SpatialML manifest schema."""

    package_id: str
    pipelines: Sequence[PipelinePackageEntry]
    supported_modes: Sequence[str] = field(default_factory=tuple)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "2"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> JsonDict:
        """Return a manifest dictionary matching the SpatialML package schema."""
        manifest: JsonDict = {"schema_version": self.schema_version, "id": self.package_id}
        manifest["pipelines"] = [entry.to_dict() for entry in self.pipelines]
        runtime = dict(self.runtime)
        if self.supported_modes:
            runtime["supported_modes"] = _normalize_supported_modes(self.supported_modes)
        if runtime:
            manifest["runtime"] = runtime
        if self.metadata:
            manifest["metadata"] = dict(self.metadata)
        return manifest


def create_litert_model_spec(
    model_path: str,
    model_name: str,
    *,
    model_target: str = "npu",
    input_tensors: Optional[Sequence[Mapping[str, Any]]] = None,
    output_tensors: Optional[Sequence[Mapping[str, Any]]] = None,
    cpu_target_num_threads: int = 1,
) -> JsonDict:
    """Create an inline LiteRT/TFLite model spec for a model inference operator.

    Args:
        model_path: Package-relative path to the ``.tflite`` model file.
        model_name: Logical model name referenced by pipeline inference operators.
        model_target: Runtime target requested by the package (``npu`` by default).
        input_tensors: Optional model input tensor metadata.
        output_tensors: Optional model output tensor metadata.
        cpu_target_num_threads: Number of CPU threads when ``model_target`` selects CPU.

    Returns:
        A dictionary ready to place under a ``run_algorithm`` operator's ``model`` key.
    """
    model_spec: JsonDict = {
        "bin_path": _normalize_package_path(model_path),
        "model_name": model_name,
        "model_type": "tflite",
        "model_target": model_target,
        "cpu_target_num_threads": int(cpu_target_num_threads),
    }
    if input_tensors is not None:
        model_spec["input"] = [dict(tensor) for tensor in input_tensors]
    if output_tensors is not None:
        model_spec["output"] = [dict(tensor) for tensor in output_tensors]
    return model_spec


def configure_litert_inference_operator(
    operator_spec: JsonDict,
    *,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    model: Optional[Mapping[str, Any]] = None,
    model_target: str = "npu",
    cpu_target_num_threads: int = 1,
    input_tensors: Optional[Sequence[Mapping[str, Any]]] = None,
    output_tensors: Optional[Sequence[Mapping[str, Any]]] = None,
) -> JsonDict:
    """Return an inference operator spec with inline LiteRT/TFLite model metadata."""
    result = dict(operator_spec)
    if model is not None and model_path is not None:
        raise ValueError("Specify either model or model_path for an inference operator, not both")
    if model is not None:
        result["model"] = dict(model)
        if not result["model"].get("bin_path"):
            raise ValueError("Inline model metadata requires bin_path")
        result["model"]["bin_path"] = _normalize_package_path(str(result["model"]["bin_path"]))
        result["model"].setdefault("model_type", "tflite")
        result["model"].setdefault("model_target", model_target)
        result["model"].setdefault("cpu_target_num_threads", int(cpu_target_num_threads))
    elif model_path is not None:
        result["model"] = create_litert_model_spec(
            model_path,
            model_name or "main",
            model_target=model_target,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            cpu_target_num_threads=cpu_target_num_threads,
        )
    else:
        raise ValueError("A run_algorithm operator requires inline model metadata")
    result["model_type"] = "tflite"
    result["model_target"] = model_target
    result["cpu_target_num_threads"] = int(cpu_target_num_threads)
    if model_name is not None:
        result["model_name"] = model_name
        result["model"].setdefault("model_name", model_name)
    result.pop("model_asset", None)
    result.pop("model_file", None)
    result.pop("model_id", None)
    result.pop("bin_path", None)
    return result


def write_pipeline_zoo_package(
    output_dir: PathLike,
    package: PipelineZooPackageSpec,
    *,
    pipelines: Mapping[str, Union[JsonDict, PathLike]],
    assets: Optional[Mapping[str, PathLike]] = None,
    indent: int = 2,
) -> JsonDict:
    """Write a SpatialML package directory.

    The generated layout follows the package schema: ``manifest.json`` at the
    package root, package-relative pipeline paths, and binary assets copied
    without rewriting their package paths.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest = package.to_manifest_dict()
    _write_json(root / "manifest.json", manifest, indent=indent)

    for entry in package.pipelines:
        source = pipelines.get(entry.id)
        if source is None:
            source = pipelines.get(entry.path)
        if source is None:
            raise KeyError(f"Missing pipeline content for '{entry.id}' ({entry.path})")
        _write_json_or_copy(root / entry.path, source, indent=indent)

    for package_path, source_path in (assets or {}).items():
        destination = _package_destination(root, package_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)

    return manifest


def load_pipeline_zoo_manifest(path: PathLike) -> JsonDict:
    """Load and minimally validate a SpatialML ``manifest.json`` file."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    validate_pipeline_zoo_manifest(manifest)
    return manifest


def validate_pipeline_zoo_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the manifest fields required by the SpatialML package schema."""
    required = ["id", "pipelines"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"SpatialML manifest missing required fields: {', '.join(missing)}")
    if str(manifest.get("schema_version", "")) != "2":
        raise ValueError("SpatialML manifest schema_version must be 2")
    if not isinstance(manifest["pipelines"], list) or not manifest["pipelines"]:
        raise ValueError("SpatialML manifest requires a non-empty 'pipelines' list")
    for index, pipeline in enumerate(manifest["pipelines"]):
        if not isinstance(pipeline, Mapping) or not pipeline.get("id") or not pipeline.get("path"):
            raise ValueError(f"SpatialML manifest pipeline #{index} requires 'id' and 'path'")
        _normalize_package_path(str(pipeline["path"]))
    runtime = manifest.get("runtime", {})
    if runtime and not isinstance(runtime, Mapping):
        raise ValueError("SpatialML manifest 'runtime' must be an object when present")
    supported_modes = runtime.get("supported_modes", []) if runtime else []
    if supported_modes:
        _normalize_supported_modes(supported_modes)


def _write_json(path: Path, payload: Mapping[str, Any], *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent, ensure_ascii=False)
        file.write("\n")


def _write_json_or_copy(path: Path, source: Union[JsonDict, PathLike], *, indent: int) -> None:
    if isinstance(source, Mapping):
        _write_json(path, source, indent=indent)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)


def _package_destination(root: Path, package_path: PathLike) -> Path:
    normalized = _normalize_package_path(str(package_path))
    destination = root / normalized
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Package path escapes output directory: {package_path}") from exc
    return destination


def _normalize_package_path(package_path: str) -> str:
    normalized = package_path.replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("Package paths must not be empty")
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid package-relative path: {package_path}")
    return str(PurePosixPath(*parts))


def _normalize_supported_modes(supported_modes: Sequence[str]) -> List[str]:
    if isinstance(supported_modes, str):
        raise ValueError("supported_modes must be a list containing 'xr', 'spatial', or both")
    normalized_modes = []
    for mode in supported_modes:
        normalized = str(mode).strip().lower()
        if normalized not in SUPPORTED_EXECUTION_MODES:
            allowed = ", ".join(sorted(SUPPORTED_EXECUTION_MODES))
            raise ValueError(f"Unsupported execution mode '{mode}'. Expected one of: {allowed}")
        if normalized not in normalized_modes:
            normalized_modes.append(normalized)
    if not normalized_modes:
        raise ValueError("supported_modes must contain at least one execution mode")
    return normalized_modes


__all__ = [
    "PipelinePackageEntry",
    "PipelineZooPackageSpec",
    "SUPPORTED_EXECUTION_MODES",
    "configure_litert_inference_operator",
    "create_litert_model_spec",
    "load_pipeline_zoo_manifest",
    "validate_pipeline_zoo_manifest",
    "write_pipeline_zoo_package",
]
