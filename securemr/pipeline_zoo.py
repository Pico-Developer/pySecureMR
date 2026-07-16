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
"""Helpers for authoring SpatialML Pipeline Zoo packages."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Union


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]
SUPPORTED_EXECUTION_MODES = {"xr", "spatial"}


@dataclass(frozen=True)
class PipelinePackageEntry:
    """Manifest entry for one pipeline JSON file in a Pipeline Zoo package."""

    id: str
    path: str

    def to_dict(self) -> JsonDict:
        """Return the package manifest representation."""
        return {"id": self.id, "path": _normalize_package_path(self.path)}


@dataclass(frozen=True)
class ModelPackageSpec:
    """Manifest model block used by the Pipeline Zoo package schema."""

    bin_path: str
    json_path: str = "model/model.json"
    extra_json_path: Optional[str] = None
    model_id: Optional[str] = None

    def to_dict(self) -> JsonDict:
        """Return the package manifest representation."""
        result = {
            "bin_path": _normalize_package_path(self.bin_path),
            "json_path": _normalize_package_path(self.json_path),
        }
        if self.model_id:
            result["id"] = self.model_id
        if self.extra_json_path:
            result["extra_json_path"] = _normalize_package_path(self.extra_json_path)
        return result


@dataclass(frozen=True)
class PipelineZooPackageSpec:
    """Top-level Pipeline Zoo manifest schema."""

    package_id: str
    pipelines: Sequence[PipelinePackageEntry]
    model: ModelPackageSpec
    models: Sequence[ModelPackageSpec] = field(default_factory=tuple)
    display_name: Optional[str] = None
    task: Optional[str] = None
    supported_modes: Sequence[str] = field(default_factory=tuple)
    labels: Sequence[str] = field(default_factory=list)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    format_version: int = 1
    schema_version: str = "1.0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> JsonDict:
        """Return a manifest dictionary matching the Pipeline Zoo package schema."""
        manifest: JsonDict = {
            "format_version": self.format_version,
            "schema_version": self.schema_version,
            "id": self.package_id,
        }
        if self.task:
            manifest["task"] = self.task
        if self.display_name:
            manifest["display_name"] = self.display_name
        manifest["pipelines"] = [entry.to_dict() for entry in self.pipelines]
        manifest["model"] = self.model.to_dict()
        if self.models:
            manifest["models"] = [model.to_dict() for model in self.models]
        if self.labels:
            manifest["labels"] = list(self.labels)
        runtime = dict(self.runtime)
        if self.supported_modes:
            runtime["supported_modes"] = _normalize_supported_modes(self.supported_modes)
        if runtime:
            manifest["runtime"] = runtime
        if self.metadata:
            manifest["metadata"] = dict(self.metadata)
        return manifest


def create_litert_model_json(
    model_path: str,
    model_name: str,
    *,
    model_target: str = "npu",
    input_tensors: Optional[Sequence[Mapping[str, Any]]] = None,
    output_tensors: Optional[Sequence[Mapping[str, Any]]] = None,
    cpu_target_num_threads: int = 1,
    extra_config: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    """Create a LiteRT model JSON payload compatible with Pipeline Zoo packages.

    Args:
        model_path: Package-relative path to the ``.tflite`` model file.
        model_name: Logical model name referenced by pipeline inference operators.
        model_target: Runtime target requested by the package (``npu`` by default).
        input_tensors: Optional model input tensor metadata.
        output_tensors: Optional model output tensor metadata.
        cpu_target_num_threads: Number of CPU threads when ``model_target`` selects CPU.
        extra_config: Additional entries to merge into ``specific_config``.

    Returns:
        A dictionary ready to serialize as ``model/model.json``.
    """
    specific_config: JsonDict = {
        "model_target": model_target,
        "cpu_target_num_threads": int(cpu_target_num_threads),
    }
    if extra_config:
        specific_config.update(dict(extra_config))

    model_json: JsonDict = {
        "model_name": model_name,
        "path_to_zoo": _normalize_package_path(model_path),
        "engine_type": "litert",
        "model_target": model_target,
        "specific_config": specific_config,
    }
    if input_tensors is not None:
        model_json["input"] = [dict(tensor) for tensor in input_tensors]
    if output_tensors is not None:
        model_json["output"] = [dict(tensor) for tensor in output_tensors]
    return model_json


def configure_litert_inference_operator(
    operator_spec: JsonDict,
    *,
    model_name: Optional[str] = None,
    model: Optional[Union[str, Mapping[str, Any]]] = None,
    model_id: Optional[str] = None,
    model_target: str = "npu",
    cpu_target_num_threads: int = 1,
) -> JsonDict:
    """Return an inference operator spec configured for LiteRT execution."""
    result = dict(operator_spec)
    if model is not None and model_id is not None:
        raise ValueError("Specify either model or model_id for an inference operator, not both")
    if isinstance(model, Mapping):
        result["model"] = dict(model)
    elif isinstance(model, str):
        result["model"] = model
    if model_id is not None:
        result["model_id"] = model_id
    result["model_type"] = "litert"
    result["model_target"] = model_target
    result["cpu_target_num_threads"] = int(cpu_target_num_threads)
    if model_name is not None:
        result["model_name"] = model_name
    result.pop("model_asset", None)
    result.pop("model_file", None)
    return result


def write_pipeline_zoo_package(
    output_dir: PathLike,
    package: PipelineZooPackageSpec,
    *,
    pipelines: Mapping[str, Union[JsonDict, PathLike]],
    model_json: Optional[Union[JsonDict, PathLike]] = None,
    assets: Optional[Mapping[str, PathLike]] = None,
    indent: int = 2,
) -> JsonDict:
    """Write a SpatialML Pipeline Zoo package directory.

    The generated layout follows the package schema: ``manifest.json`` at the
    package root, package-relative pipeline paths, package-relative model
    metadata, and binary assets copied without rewriting their manifest paths.
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

    if model_json is not None:
        _write_json_or_copy(root / package.model.json_path, model_json, indent=indent)

    for package_path, source_path in (assets or {}).items():
        destination = _package_destination(root, package_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)

    return manifest


def load_pipeline_zoo_manifest(path: PathLike) -> JsonDict:
    """Load and minimally validate a Pipeline Zoo ``manifest.json`` file."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    validate_pipeline_zoo_manifest(manifest)
    return manifest


def validate_pipeline_zoo_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the manifest fields required by the Pipeline Zoo package schema."""
    required = ["format_version", "id", "pipelines", "model"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"Pipeline Zoo manifest missing required fields: {', '.join(missing)}")
    if manifest["format_version"] != 1:
        raise ValueError("Pipeline Zoo manifest format_version must be 1")
    if not isinstance(manifest["pipelines"], list) or not manifest["pipelines"]:
        raise ValueError("Pipeline Zoo manifest requires a non-empty 'pipelines' list")
    for index, pipeline in enumerate(manifest["pipelines"]):
        if not isinstance(pipeline, Mapping) or not pipeline.get("id") or not pipeline.get("path"):
            raise ValueError(f"Pipeline Zoo manifest pipeline #{index} requires 'id' and 'path'")
        _normalize_package_path(str(pipeline["path"]))
    model = manifest["model"]
    if not isinstance(model, Mapping):
        raise ValueError("Pipeline Zoo manifest 'model' must be an object")
    _validate_model_manifest_entry(model, "model")
    models = manifest.get("models", [])
    if models:
        if isinstance(models, Mapping):
            iterable_models = models.values()
        elif isinstance(models, list):
            iterable_models = models
        else:
            raise ValueError("Pipeline Zoo manifest 'models' must be a list or object when present")
        for index, model_entry in enumerate(iterable_models):
            if not isinstance(model_entry, Mapping):
                raise ValueError(f"Pipeline Zoo manifest models entry #{index} must be an object")
            _validate_model_manifest_entry(model_entry, f"models[{index}]")
    runtime = manifest.get("runtime", {})
    if runtime and not isinstance(runtime, Mapping):
        raise ValueError("Pipeline Zoo manifest 'runtime' must be an object when present")
    supported_modes = runtime.get("supported_modes", []) if runtime else []
    if supported_modes:
        _normalize_supported_modes(supported_modes)


def _write_json(path: Path, payload: Mapping[str, Any], *, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent, ensure_ascii=False)
        file.write("\n")


def _validate_model_manifest_entry(model: Mapping[str, Any], label: str) -> None:
    if not model.get("bin_path") or not model.get("json_path"):
        raise ValueError(f"Pipeline Zoo manifest {label} requires 'bin_path' and 'json_path'")
    _normalize_package_path(str(model["bin_path"]))
    _normalize_package_path(str(model["json_path"]))


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
    "ModelPackageSpec",
    "PipelinePackageEntry",
    "PipelineZooPackageSpec",
    "SUPPORTED_EXECUTION_MODES",
    "configure_litert_inference_operator",
    "create_litert_model_json",
    "load_pipeline_zoo_manifest",
    "validate_pipeline_zoo_manifest",
    "write_pipeline_zoo_package",
]
