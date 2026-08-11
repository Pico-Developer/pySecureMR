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

"""Safe zip extraction helpers."""

from __future__ import annotations

import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


class ZipSafetyError(ValueError):
    """Raised when a zip entry is unsafe to extract."""


def safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract a zip archive while rejecting traversal and absolute paths.

    Backslash-separated relative paths are accepted so archives produced on
    Windows can still be consumed. Absolute Windows paths and drive-relative
    paths are rejected.
    """
    root = destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for member in archive.infolist():
        relative_parts = _safe_zip_parts(member.filename)
        target = root.joinpath(*relative_parts).resolve()
        if target != root and root not in target.parents:
            raise ZipSafetyError(f"Zip entry escapes extraction root: {member.filename}")
        if _is_symlink(member):
            raise ZipSafetyError(f"Zip entry is a symlink: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, open(target, "wb") as output:
            shutil.copyfileobj(source, output)


def _safe_zip_parts(name: str) -> tuple[str, ...]:
    normalized_name = name.replace("\\", "/")
    windows_path = PureWindowsPath(name)
    posix_path = PurePosixPath(normalized_name)
    if (
        not normalized_name
        or normalized_name.startswith("/")
        or normalized_name.startswith("\\")
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ZipSafetyError(f"Unsafe zip entry path: {name}")
    parts = posix_path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ZipSafetyError(f"Unsafe zip entry path: {name}")
    return parts


def _is_symlink(member: zipfile.ZipInfo) -> bool:
    mode = member.external_attr >> 16
    return stat.S_ISLNK(mode)
