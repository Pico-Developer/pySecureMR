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
from tempfile import TemporaryDirectory
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
_XR_ONLY_OPERATORS = {
    "LOAD_TEXTURE",
    "RENDER_TEXT",
    "SWITCH_GLTF_RENDER_STATUS",
    "UPDATE_GLTF",
}
_SPATIAL_ONLY_OPERATORS = {
    "SCENEGRAPH_VISIBILITY",
    "UPDATE_COMPONENT",
}


def create_package(
    *,
    package_id: str,
    pipelines: Sequence[str],
    output: Path,
    source: Optional[Path] = None,
    supported_modes: Sequence[str] = (),
    asset_roots: Sequence[Path] = (),
    force: bool = False,
    zip_output: bool = False,
    assume_yes: bool = False,
) -> int:
    """Create a schema-v2 SpatialML package directory or zip."""
    if source is not None and _looks_like_existing_package(source):
        return _copy_existing_package(
            source=source,
            output=output,
            force=force,
            zip_output=zip_output,
            assume_yes=assume_yes,
        )
    if source is not None:
        asset_roots = [source, *asset_roots]
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
    packaged_pipeline_specs: list[tuple[str, Mapping[str, Any]]] = []

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
        packaged_pipeline_specs.append((item.id, normalized_spec))

    package = PipelineZooPackageSpec(
        package_id=package_id,
        supported_modes=supported_modes,
        pipelines=manifest_entries,
    )
    manifest = package.to_manifest_dict()
    inferred_modes = _validate_pipeline_modes(manifest, packaged_pipeline_specs)
    if not supported_modes:
        manifest.setdefault("runtime", {})["supported_modes"] = inferred_modes
    _write_json(package_root / "manifest.json", manifest)

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


def _copy_existing_package(
    *,
    source: Path,
    output: Path,
    force: bool,
    zip_output: bool,
    assume_yes: bool,
) -> int:
    source_root = _materialize_package(source)
    if not (source_root / "manifest.json").is_file():
        raise PackageCliError(f"Package manifest not found: {source_root / 'manifest.json'}")
    manifest = _load_validated_manifest(source_root)
    try:
        _validate_package_root(source_root, manifest)
    except PackageCliError:
        if not force:
            raise
        needs_reconcile = True
    else:
        needs_reconcile = False

    package_root = output
    archive_path: Optional[Path] = None
    if zip_output or output.suffix == ".zip":
        archive_path = output
        package_root = output.with_suffix("")

    same_package_root = source_root.resolve() == package_root.resolve()
    if needs_reconcile:
        if package_root.exists() and same_package_root and archive_path is None:
            raise PackageCliError("Source package and output package directory are the same")
        if package_root.exists() and not same_package_root and not force and not _confirm_overwrite(
            package_root, assume_yes=assume_yes
        ):
            raise PackageCliError(f"Output package already exists: {package_root}")
        if archive_path and archive_path.exists() and not force and not _confirm_overwrite(
            archive_path, assume_yes=assume_yes
        ):
            raise PackageCliError(f"Output archive already exists: {archive_path}")
        return _copy_existing_package_with_reconcile(
            source_root=source_root,
            package_root=package_root,
            archive_path=archive_path,
            same_package_root=same_package_root,
        )

    if package_root.exists():
        if same_package_root:
            if archive_path is None:
                raise PackageCliError("Source package and output package directory are the same")
        elif not force and not _confirm_overwrite(package_root, assume_yes=assume_yes):
            raise PackageCliError(f"Output package already exists: {package_root}")
        elif package_root.is_dir():
            shutil.rmtree(package_root)
        else:
            package_root.unlink()
    if archive_path and archive_path.exists():
        if not force and not _confirm_overwrite(archive_path, assume_yes=assume_yes):
            raise PackageCliError(f"Output archive already exists: {archive_path}")
        archive_path.unlink()

    if not same_package_root:
        shutil.copytree(source_root, package_root)

    if archive_path:
        _zip_directory(package_root, archive_path)
        print(f"Created package archive: {archive_path}")
    else:
        print(f"Created package: {package_root}")
    return 0


def _copy_existing_package_with_reconcile(
    *,
    source_root: Path,
    package_root: Path,
    archive_path: Optional[Path],
    same_package_root: bool,
) -> int:
    with TemporaryDirectory(prefix="pyspatialml-package-") as tmp:
        staged_root = Path(tmp) / "package"
        shutil.copytree(source_root, staged_root)
        _reconcile_existing_package_root(staged_root)

        if not same_package_root:
            if package_root.exists():
                if package_root.is_dir():
                    shutil.rmtree(package_root)
                else:
                    package_root.unlink()
            shutil.copytree(staged_root, package_root)
            archive_root = package_root
        else:
            archive_root = staged_root

        if archive_path:
            if archive_path.exists():
                archive_path.unlink()
            _zip_directory(archive_root, archive_path)
            print(f"Created package archive: {archive_path}")
        else:
            print(f"Created package: {package_root}")
    return 0


def _reconcile_existing_package_root(root: Path) -> None:
    manifest = _load_validated_manifest(root)
    _reconcile_manifest_modes(root, manifest)
    _write_json(root / "manifest.json", manifest)
    _validate_package_root(root, manifest)


def _looks_like_existing_package(path: Path) -> bool:
    if path.is_file() and path.suffix == ".zip":
        return True
    if path.is_dir() and (path / "manifest.json").is_file():
        return True
    return False


def validate_package(path: Path) -> int:
    """Validate a package directory or zip archive."""
    root = _materialize_package(path)
    manifest = _load_validated_manifest(root)
    _validate_package_root(root, manifest)
    print(f"Package is valid: {path}")
    return 0


def _validate_package_root(root: Path, manifest: Mapping[str, Any]) -> None:
    pipeline_specs: list[tuple[str, Mapping[str, Any]]] = []
    _collect_package_pipeline_specs(root, manifest, pipeline_specs)
    _validate_pipeline_modes(manifest, pipeline_specs)


def _collect_package_pipeline_specs(
    root: Path,
    manifest: Mapping[str, Any],
    pipeline_specs: list[tuple[str, Mapping[str, Any]]],
) -> None:
    for pipeline in manifest["pipelines"]:
        pipeline_path = _resolve_package_file(root, str(pipeline["path"]), label="pipeline")
        if not pipeline_path.is_file():
            raise PackageCliError(f"Manifest references missing pipeline: {pipeline['path']}")
        spec = _read_json(pipeline_path)
        validate_pipeline_spec(spec)
        pipeline_specs.append((str(pipeline["id"]), spec))
        for asset in _iter_pipeline_asset_refs(spec):
            asset_path = _resolve_package_file(root, asset, label="asset")
            if not asset_path.is_file():
                raise PackageCliError(f"Pipeline references missing package asset: {asset}")


def _reconcile_manifest_modes(root: Path, manifest: dict[str, Any]) -> None:
    pipeline_specs: list[tuple[str, Mapping[str, Any]]] = []
    _collect_package_pipeline_specs(root, manifest, pipeline_specs)
    manifest.setdefault("runtime", {})["supported_modes"] = _infer_pipeline_modes(pipeline_specs)


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
    except (OSError, ValueError) as exc:
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


def _validate_pipeline_modes(
    manifest: Mapping[str, Any],
    pipeline_specs: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[str]:
    runtime = manifest.get("runtime", {})
    supported_modes = runtime.get("supported_modes", []) if isinstance(runtime, Mapping) else []
    modes = {str(mode).strip().lower() for mode in supported_modes}

    xr_only = []
    spatial_only = []
    for pipeline_id, spec in pipeline_specs:
        for op_type in _pipeline_operator_type_names(spec):
            if op_type in _XR_ONLY_OPERATORS:
                xr_only.append((pipeline_id, op_type))
            if op_type in _SPATIAL_ONLY_OPERATORS:
                spatial_only.append((pipeline_id, op_type))

    if xr_only and spatial_only:
        xr_details = _format_mode_operator_details(xr_only)
        spatial_details = _format_mode_operator_details(spatial_only)
        raise PackageCliError(
            "Package pipelines mix XR-only and Spatial-only operators. "
            f"XR-only: {xr_details}. Spatial-only: {spatial_details}."
        )

    inferred_modes = _infer_supported_modes(xr_only=xr_only, spatial_only=spatial_only)
    if not modes:
        return inferred_modes
    if "spatial" in modes and xr_only:
        details = _format_mode_operator_details(xr_only)
        raise PackageCliError(
            "Manifest runtime.supported_modes includes spatial, but package uses XR-only operators: "
            f"{details}. Remove spatial from supported modes or remove the XR-only operators."
        )
    if "xr" in modes and spatial_only:
        details = _format_mode_operator_details(spatial_only)
        raise PackageCliError(
            "Manifest runtime.supported_modes includes xr, but package uses Spatial-only operators: "
            f"{details}. Remove xr from supported modes or remove the Spatial-only operators."
        )
    return list(supported_modes)


def _infer_supported_modes(
    *,
    xr_only: Sequence[tuple[str, str]],
    spatial_only: Sequence[tuple[str, str]],
) -> list[str]:
    if xr_only:
        return ["xr"]
    if spatial_only:
        return ["spatial"]
    return ["xr", "spatial"]


def _infer_pipeline_modes(pipeline_specs: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    xr_only = []
    spatial_only = []
    for pipeline_id, spec in pipeline_specs:
        for op_type in _pipeline_operator_type_names(spec):
            if op_type in _XR_ONLY_OPERATORS:
                xr_only.append((pipeline_id, op_type))
            if op_type in _SPATIAL_ONLY_OPERATORS:
                spatial_only.append((pipeline_id, op_type))
    if xr_only and spatial_only:
        xr_details = _format_mode_operator_details(xr_only)
        spatial_details = _format_mode_operator_details(spatial_only)
        raise PackageCliError(
            "Package pipelines mix XR-only and Spatial-only operators. "
            f"XR-only: {xr_details}. Spatial-only: {spatial_details}."
        )
    return _infer_supported_modes(xr_only=xr_only, spatial_only=spatial_only)


def _pipeline_operator_type_names(spec: Mapping[str, Any]) -> Iterable[str]:
    for op in spec.get("operators", []):
        if not isinstance(op, Mapping):
            continue
        op_type = _normalize_operator_type_name(str(op.get("type") or op.get("operator_type") or ""))
        if op_type:
            yield op_type


def _normalize_operator_type_name(value: str) -> str:
    normalized = value.strip().upper()
    if normalized.startswith("XR_SECURE_MR_OPERATOR_TYPE_"):
        normalized = normalized[len("XR_SECURE_MR_OPERATOR_TYPE_") :]
    if normalized.endswith("_PICO"):
        normalized = normalized[: -len("_PICO")]
    return normalized


def _format_mode_operator_details(items: Sequence[tuple[str, str]]) -> str:
    return ", ".join(f"{pipeline}:{op_type}" for pipeline, op_type in items)


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


def _resolve_package_file(root: Path, package_path: str, *, label: str) -> Path:
    path = Path(package_path)
    if path.is_absolute():
        raise PackageCliError(f"Package {label} path must be package-relative: {package_path}")
    normalized = _normalize_package_path(package_path)
    resolved_root = root.resolve()
    resolved_path = (root / normalized).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise PackageCliError(f"Package {label} path escapes package root: {package_path}") from exc
    return resolved_path


def _normalize_package_path(path: str) -> str:
    raw_path = Path(path)
    if raw_path.is_absolute() or path.replace("\\", "/").startswith("/"):
        raise PackageCliError(f"Invalid package-relative path: {path}")
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
