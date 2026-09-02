import json

import numpy as np
import pytest

from pyspatialml import compare_cli


def test_compare_paths_identical_files_pass(capsys, tmp_path):
    expected = tmp_path / "expected.npy"
    actual = tmp_path / "actual.npy"
    np.save(expected, np.array([1.0, 2.0], dtype=np.float32))
    np.save(actual, np.array([1.0, 2.0], dtype=np.float32))

    assert compare_cli.compare_paths(expected, actual) == 0

    captured = capsys.readouterr()
    assert "passed: yes" in captured.out
    assert "max_abs_diff: 0" in captured.out


def test_compare_paths_mismatch_returns_compare_exit(capsys, tmp_path):
    expected = tmp_path / "expected.npy"
    actual = tmp_path / "actual.npy"
    np.save(expected, np.array([1.0, 2.0], dtype=np.float32))
    np.save(actual, np.array([1.0, 3.0], dtype=np.float32))

    assert compare_cli.compare_paths(expected, actual, rtol=1e-6, atol=1e-6) == 4

    captured = capsys.readouterr()
    assert "passed: no" in captured.out
    assert "max_abs_diff: 1" in captured.out


def test_compare_paths_json_output(capsys, tmp_path):
    expected = tmp_path / "expected.npy"
    actual = tmp_path / "actual.npy"
    np.save(expected, np.array([1.0, 2.0], dtype=np.float32))
    np.save(actual, np.array([1.0, 2.001], dtype=np.float32))

    assert compare_cli.compare_paths(expected, actual, rtol=1e-2, atol=1e-2, as_json=True) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["comparisons"][0]["name"] == "expected.npy"
    assert payload["comparisons"][0]["max_abs_diff"] > 0


def test_compare_directories_report_missing_and_extra(capsys, tmp_path):
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    np.save(expected / "a.npy", np.array([1], dtype=np.float32))
    np.save(expected / "missing.npy", np.array([1], dtype=np.float32))
    np.save(actual / "a.npy", np.array([1], dtype=np.float32))
    np.save(actual / "extra.npy", np.array([1], dtype=np.float32))

    assert compare_cli.compare_paths(expected, actual) == 4

    captured = capsys.readouterr()
    assert "Missing actual:" in captured.out
    assert "missing.npy" in captured.out
    assert "Extra actual:" in captured.out
    assert "extra.npy" in captured.out


def test_compare_rejects_non_npy_file(tmp_path):
    expected = tmp_path / "expected.bin"
    actual = tmp_path / "actual.bin"
    expected.write_bytes(b"1")
    actual.write_bytes(b"1")

    with pytest.raises(compare_cli.CompareCliError, match="Only .npy files"):
        compare_cli.compare_paths(expected, actual)
