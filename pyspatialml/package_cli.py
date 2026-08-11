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

"""SpatialML package authoring commands."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence

from securemr.pipeline_zoo import (
    PipelinePackageEntry,
    PipelineZooPackageSpec,
    load_pipeline_zoo_manifest,
)
from securemr.py2smr.verifier import validate_pipeline_spec
from pyspatialml.zip_utils import ZipSafetyError, safe_extract_zip


class PackageCliError(RuntimeError):
    """Raised when package creation or inspection fails."""


@dataclass(frozen=True)
class PipelineInput:
    """One source pipeline and its pipeline zoo id."""

    id: str
    source: Path


@dataclass(frozen=True)
class AssetCopy:
    """Resolved package asset copy operation."""

    source: Path
    package_path: str


_GLTF_EXTENSIONS = {".gltf", ".glb"}
_GLTF_KEYS = {
    "asset",
    "asset_path",
    "gltf",
    "gltf_path",
    "gltf_asset",
    "scene",
    "scene_path",
}


def create_package(
    *,
    package_id: str,
    pipelines: Sequence[str],
    output: Path,
    supported_modes: Sequence[str] = (),
    asset_roots: Sequence[Path] = (),
    force: bool = False,
    zip_output: bool = False,
    assume_yes: bool = False,
) -> int:
    """Create a schema-v2 SpatialML package directory or zip."""
    if not package_id:
        raise PackageCliError("Package id is required")
    pipeline_inputs = [_parse_pipeline_arg(item) for item in pipelines]
    if not pipeline_inputs:
        raise PackageCliError("At least one --pipeline is required")
    _ensure_unique([item.id for item in pipeline_inputs], "pipeline id")

    package_root = output
    archive_path: Optional[Path] = None
    if zip_output or output.suffix == ".zip":
        archive_path = output
        package_root = output.with_suffix("")

    if package_root.exists():
        if not force and not _confirm_overwrite(package_root, assume_yes=assume_yes):
            raise PackageCliError(f"Output package already exists: {package_root}")
        if package_root.is_dir():
            shutil.rmtree(package_root)
        else:
            package_root.unlink()
    if archive_path and archive_path.exists():
        if not force and not _confirm_overwrite(archive_path, assume_yes=assume_yes):
            raise PackageCliError(f"Output archive already exists: {archive_path}")
        archive_path.unlink()

    package_root.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[PipelinePackageEntry] = []
    copied_assets: dict[str, Path] = {}

    for item in pipeline_inputs:
        source_pipeline = item.source
        if not source_pipeline.is_file():
            raise PackageCliError(f"Pipeline file not found: {source_pipeline}")
        pipeline_spec = _read_json(source_pipeline)
        validate_pipeline_spec(pipeline_spec)

        package_pipeline_path = f"pipeline/{item.id}.json"
        normalized_spec = _normalize_pipeline_assets(
            pipeline_spec,
            source_pipeline=source_pipeline,
            asset_roots=asset_roots,
            copied_assets=copied_assets,
        )
        _write_json(package_root / package_pipeline_path, normalized_spec)
        manifest_entries.append(PipelinePackageEntry(item.id, package_pipeline_path))

    package = PipelineZooPackageSpec(
        package_id=package_id,
        supported_modes=supported_modes,
        pipelines=manifest_entries,
    )
    _write_json(package_root / "manifest.json", package.to_manifest_dict())

    for package_path, source_path in copied_assets.items():
        destination = _package_destination(package_root, package_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)

    if archive_path:
        _zip_directory(package_root, archive_path)
        print(f"Created package archive: {archive_path}")
    else:
        print(f"Created package: {package_root}")
    return 0


def validate_package(path: Path) -> int:
    """Validate a package directory or zip archive."""
    root = _materialize_package(path)
    manifest = _load_validated_manifest(root)
    for pipeline in manifest["pipelines"]:
        pipeline_path = root / pipeline["path"]
        if not pipeline_path.is_file():
            raise PackageCliError(f"Manifest references missing pipeline: {pipeline['path']}")
        spec = _read_json(pipeline_path)
        validate_pipeline_spec(spec)
        for asset in _iter_pipeline_asset_refs(spec):
            if not (root / asset).is_file():
                raise PackageCliError(f"Pipeline references missing package asset: {asset}")
    print(f"Package is valid: {path}")
    return 0


def inspect_package(path: Path) -> int:
    """Print a compact package summary."""
    root = _materialize_package(path)
    manifest = _load_validated_manifest(root)
    print(f"Package: {manifest['id']}")
    print(f"Schema: {manifest.get('schema_version')}")
    pipelines = manifest.get("pipelines", [])
    print(f"Pipelines: {len(pipelines)}")
    for pipeline in pipelines:
        print(f"  {pipeline['id']} -> {pipeline['path']}")
    runtime = manifest.get("runtime", {})
    supported_modes = runtime.get("supported_modes", [])
    if supported_modes:
        print(f"Supported modes: {', '.join(supported_modes)}")
    assets = sorted(_package_assets(root))
    if assets:
        print("Assets:")
        for asset in assets:
            print(f"  {asset}")
    return 0


def print_package_error(exc: Exception) -> None:
    """Print a concise package command error."""
    print(f"Error [PSM_PACKAGE]: {exc}", file=sys.stderr)


def _load_validated_manifest(root: Path) -> dict[str, Any]:
    try:
        return load_pipeline_zoo_manifest(root)
    except ValueError as exc:
        raise PackageCliError(str(exc)) from exc


def _confirm_overwrite(path: Path, *, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        answer = input(f"Overwrite existing output {path}? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _parse_pipeline_arg(value: str) -> PipelineInput:
    if "=" not in value:
        raise PackageCliError("--pipeline must use id=path format")
    pipeline_id, source = value.split("=", 1)
    pipeline_id = pipeline_id.strip()
    if not pipeline_id:
        raise PackageCliError("Pipeline id cannot be empty")
    _normalize_package_path(pipeline_id)
    return PipelineInput(id=pipeline_id, source=Path(source))


def _normalize_pipeline_assets(
    spec: Mapping[str, Any],
    *,
    source_pipeline: Path,
    asset_roots: Sequence[Path],
    copied_assets: dict[str, Path],
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(spec))
    for op in normalized.get("operators", []):
        if not isinstance(op, dict):
            continue
        model = op.get("model")
        if isinstance(model, dict):
            bin_path = model.get("bin_path")
            if bin_path:
                package_path = _register_asset(
                    str(bin_path),
                    target_dir="model",
                    allowed_extensions={".tflite"},
                    source_pipeline=source_pipeline,
                    asset_roots=asset_roots,
                    copied_assets=copied_assets,
                )
                model["bin_path"] = package_path
                # Current XR deserializers still read these fields at operator
                # level even though model metadata is also nested under model.
                op["model_type"] = "tflite"
                op.pop("model_file", None)
                op.pop("model_asset", None)
                op.pop("model_id", None)
        for key in list(op.keys()):
            if key.lower() in _GLTF_KEYS and isinstance(op[key], str):
                op[key] = _register_asset(
                    op[key],
                    target_dir="gltf",
                    allowed_extensions=_GLTF_EXTENSIONS,
                    source_pipeline=source_pipeline,
                    asset_roots=asset_roots,
                    copied_assets=copied_assets,
                )

    tensors = normalized.get("tensors", {})
    if isinstance(tensors, dict):
        for tensor in tensors.values():
            if not isinstance(tensor, dict):
                continue
            if str(tensor.get("tensor_type") or tensor.get("type") or "").lower() == "gltf" and tensor.get("asset"):
                tensor["asset"] = _register_asset(
                    str(tensor["asset"]),
                    target_dir="gltf",
                    allowed_extensions=_GLTF_EXTENSIONS,
                    source_pipeline=source_pipeline,
                    asset_roots=asset_roots,
                    copied_assets=copied_assets,
                )
    return normalized


def _register_asset(
    reference: str,
    *,
    target_dir: str,
    allowed_extensions: set[str],
    source_pipeline: Path,
    asset_roots: Sequence[Path],
    copied_assets: dict[str, Path],
) -> str:
    source = _resolve_asset_source(reference, source_pipeline=source_pipeline, asset_roots=asset_roots)
    if source.suffix.lower() not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise PackageCliError(f"Asset '{reference}' must use one of: {allowed}")
    package_path = _normalize_package_path(f"{target_dir}/{source.name}")
    existing = copied_assets.get(package_path)
    if existing is not None and existing.resolve() != source.resolve():
        raise PackageCliError(
            f"Asset name collision for {package_path}: {existing} and {source}"
        )
    copied_assets[package_path] = source
    return package_path


def _resolve_asset_source(
    reference: str,
    *,
    source_pipeline: Path,
    asset_roots: Sequence[Path],
) -> Path:
    ref_path = Path(reference)
    searched: list[Path] = []
    if ref_path.is_absolute():
        searched.append(ref_path)
    else:
        searched.extend(
            [
                source_pipeline.parent / ref_path,
                Path.cwd() / ref_path,
                *[root / ref_path for root in asset_roots],
            ]
        )
    for candidate in searched:
        if candidate.is_file():
            return candidate.resolve()
    searched_text = "\n  ".join(str(path) for path in searched)
    raise PackageCliError(
        f"Referenced asset not found: {reference}\nSearched:\n  {searched_text}"
    )


def _iter_pipeline_asset_refs(spec: Mapping[str, Any]) -> Iterable[str]:
    for op in spec.get("operators", []):
        if not isinstance(op, Mapping):
            continue
        model = op.get("model")
        if isinstance(model, Mapping) and model.get("bin_path"):
            yield _normalize_package_path(str(model["bin_path"]))
        for key, value in op.items():
            if key.lower() in _GLTF_KEYS and isinstance(value, str):
                yield _normalize_package_path(value)
    tensors = spec.get("tensors", {})
    if isinstance(tensors, Mapping):
        for tensor in tensors.values():
            if not isinstance(tensor, Mapping):
                continue
            if str(tensor.get("tensor_type") or tensor.get("type") or "").lower() == "gltf" and tensor.get("asset"):
                yield _normalize_package_path(str(tensor["asset"]))


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise PackageCliError(f"JSON file must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _package_destination(root: Path, package_path: str) -> Path:
    normalized = _normalize_package_path(package_path)
    destination = root / normalized
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PackageCliError(f"Package path escapes output directory: {package_path}") from exc
    return destination


def _normalize_package_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        raise PackageCliError("Package paths must not be empty")
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise PackageCliError(f"Invalid package-relative path: {path}")
    return str(PurePosixPath(*parts))


def _ensure_unique(values: Sequence[str], label: str) -> None:
    seen = set()
    duplicates = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise PackageCliError(f"Duplicate {label}: {', '.join(duplicates)}")


def _zip_directory(root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))


def _materialize_package(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.is_file() and path.suffix == ".zip":
        extract_root = path.with_suffix("")
        if extract_root.exists():
            shutil.rmtree(extract_root)
        with zipfile.ZipFile(path) as archive:
            try:
                safe_extract_zip(archive, extract_root)
            except ZipSafetyError as exc:
                raise PackageCliError(str(exc)) from exc
        return _find_package_root(extract_root)
    raise PackageCliError(f"Package not found: {path}")


def _package_assets(root: Path) -> list[str]:
    assets = []
    for directory in ("model", "gltf"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                assets.append(str(path.relative_to(root)).replace("\\", "/"))
    return assets


def _find_package_root(root: Path) -> Path:
    if (root / "manifest.json").is_file():
        return root
    candidates = [path for path in root.iterdir() if path.is_dir() and not path.name.startswith("__MACOSX")]
    manifest_dirs = [path for path in candidates if (path / "manifest.json").is_file()]
    if len(manifest_dirs) == 1:
        return manifest_dirs[0]
    return root
