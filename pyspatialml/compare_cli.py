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

"""Tensor output comparison command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from securemr.py2smr.verifier import compare_outputs


class CompareCliError(RuntimeError):
    """Raised when tensor comparison cannot be completed."""


def compare_paths(
    expected: Path,
    actual: Path,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    as_json: bool = False,
) -> int:
    """Compare expected and actual tensor files/directories."""
    if expected.is_file() and actual.is_file():
        _validate_npy_file(expected)
        _validate_npy_file(actual)
        expected_tensors = {expected.name: np.load(expected)}
        actual_tensors = {expected.name: np.load(actual)}
    else:
        expected_tensors = _load_tensors(expected)
        actual_tensors = _load_tensors(actual)
    payload = _compare_tensor_maps(expected_tensors, actual_tensors, rtol=rtol, atol=atol)
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        _print_human_summary(expected, actual, payload)
    return 0 if payload["passed"] else 4


def print_compare_error(exc: Exception) -> None:
    """Print a concise compare command error."""
    print(f"Error [PSM_COMPARE]: {exc}", file=sys.stderr)


def _load_tensors(path: Path) -> dict[str, np.ndarray]:
    if path.is_file():
        _validate_npy_file(path)
        return {path.name: np.load(path)}
    if path.is_dir():
        tensors = {}
        for item in sorted(path.glob("*.npy")):
            tensors[item.name] = np.load(item)
        if not tensors:
            raise CompareCliError(f"Directory contains no .npy files: {path}")
        return tensors
    raise CompareCliError(f"Path not found: {path}")


def _validate_npy_file(path: Path) -> None:
    if path.suffix != ".npy":
        raise CompareCliError(f"Only .npy files are supported: {path}")


def _compare_tensor_maps(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    missing_actual = sorted(set(expected) - set(actual))
    extra_actual = sorted(set(actual) - set(expected))
    common_names = sorted(set(expected) & set(actual))
    comparisons = []
    all_passed = not missing_actual

    for name in common_names:
        exp = expected[name]
        act = actual[name]
        comparison = {
            "name": name,
            "shape_expected": list(exp.shape),
            "shape_actual": list(act.shape),
            "dtype_expected": str(exp.dtype),
            "dtype_actual": str(act.dtype),
            "passed": False,
            "max_abs_diff": None,
            "max_rel_diff": None,
            "mean_abs_diff": None,
            "error": None,
        }
        if exp.shape != act.shape:
            comparison["error"] = f"shape mismatch: expected {exp.shape}, actual {act.shape}"
            comparisons.append(comparison)
            all_passed = False
            continue

        try:
            result = compare_outputs({name: exp}, {name: act}, rtol=rtol, atol=atol)
        except Exception as exc:  # noqa: BLE001
            comparison["error"] = str(exc)
            comparisons.append(comparison)
            all_passed = False
            continue

        exp64 = exp.astype(np.float64)
        act64 = act.astype(np.float64)
        abs_diff = np.abs(exp64 - act64)
        comparison["passed"] = bool(result.success)
        comparison["max_abs_diff"] = float(result.max_abs_diff.get(name, 0.0))
        comparison["max_rel_diff"] = float(result.max_rel_diff.get(name, 0.0))
        comparison["mean_abs_diff"] = float(np.mean(abs_diff)) if abs_diff.size else 0.0
        comparison["error"] = result.error_message
        comparisons.append(comparison)
        if not result.success:
            all_passed = False

    return {
        "passed": bool(all_passed),
        "rtol": float(rtol),
        "atol": float(atol),
        "comparisons": comparisons,
        "missing_actual": missing_actual,
        "extra_actual": extra_actual,
    }


def _print_human_summary(expected: Path, actual: Path, payload: Mapping[str, Any]) -> None:
    print(f"Compare: {expected} vs {actual}")
    print(f"rtol: {payload['rtol']} atol: {payload['atol']}")
    print(f"passed: {'yes' if payload['passed'] else 'no'}")
    print(f"Compared tensors: {len(payload['comparisons'])}")
    for item in payload["comparisons"]:
        print(item["name"])
        print(f"  shape: {tuple(item['shape_expected'])} vs {tuple(item['shape_actual'])}")
        print(f"  dtype: {item['dtype_expected']} vs {item['dtype_actual']}")
        if item["max_abs_diff"] is not None:
            print(f"  max_abs_diff: {item['max_abs_diff']:.6g}")
            print(f"  max_rel_diff: {item['max_rel_diff']:.6g}")
            print(f"  mean_abs_diff: {item['mean_abs_diff']:.6g}")
        print(f"  passed: {'yes' if item['passed'] else 'no'}")
        if item.get("error"):
            print(f"  error: {item['error']}")
    if payload["missing_actual"]:
        print("Missing actual:")
        for name in payload["missing_actual"]:
            print(f"  {name}")
    if payload["extra_actual"]:
        print("Extra actual:")
        for name in payload["extra_actual"]:
            print(f"  {name}")
