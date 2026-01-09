import subprocess
import sys
from pathlib import Path

import pytest
from ppadb.client import Client as AdbClient


ROOT = Path(__file__).resolve().parents[1]
MNIST_ROOT = ROOT / "examples" / "mnistwild"


def has_connected_device() -> bool:
    client = AdbClient(host="127.0.0.1", port=5037)
    return bool(client.devices())


def run_cli(args: list[str], timeout: int = 180) -> None:
    subprocess.run(args, check=True, timeout=timeout)


def test_model_inspect_mnist() -> None:
    if not has_connected_device():
        pytest.skip("No Android device connected")
    run_cli(
        [
            sys.executable,
            "-m",
            "securemr.inspect.model_cli",
            "--model",
            str(MNIST_ROOT / "mnist.serialized.bin"),
            "--json",
            str(MNIST_ROOT / "mnist.serialized.json"),
        ]
    )


def test_pipeline_inspect_mnist() -> None:
    if not has_connected_device():
        pytest.skip("No Android device connected")
    run_cli(
        [
            sys.executable,
            "-m",
            "securemr.inspect.pipeline_cli",
            "--pipeline",
            str(MNIST_ROOT / "mnist_pipeline.json"),
        ],
        timeout=240,
    )


def test_pipeline_inspect_mnist_with_input() -> None:
    if not has_connected_device():
        pytest.skip("No Android device connected")
    run_cli(
        [
            sys.executable,
            "-m",
            "securemr.inspect.pipeline_cli",
            "--pipeline",
            str(MNIST_ROOT / "mnist_pipeline.json"),
            "--input",
            str(MNIST_ROOT / "number_5.png"),
            "--input-tensor",
            "left_rgb",
        ],
        timeout=240,
    )


def test_pipeline_inspect_mnist_with_outputs() -> None:
    if not has_connected_device():
        pytest.skip("No Android device connected")
    run_cli(
        [
            sys.executable,
            "-m",
            "securemr.inspect.pipeline_cli",
            "--pipeline",
            str(MNIST_ROOT / "mnist_pipeline.json"),
            "--input",
            str(MNIST_ROOT / "number_5.png"),
            "--input-tensor",
            "left_rgb",
            "--output",
            str(MNIST_ROOT / "number_5_target_output" / "predicted_class.bin"),
            "--output",
            str(MNIST_ROOT / "number_5_target_output" / "predicted_score.bin"),
        ],
        timeout=240,
    )
