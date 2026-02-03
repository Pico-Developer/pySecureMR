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

"""
QnnModelV2 class uses model_inspect APK to run inference on Android devices.

Unlike QnnModel which uses qnn-net-run binary directly, QnnModelV2 uses the
model_inspect APK for inference, which is useful for testing models in the
SecureMR pipeline.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..core.utils import DEBUG_QNN, TORCH_INSTALLED
from ..inspect.inspect_utils import (
    AdbError,
    DeviceContext,
    capture_adb,
    capture_adb_binary,
    ensure_screen_on,
    install_apk,
    run_adb,
    select_device,
    start_logcat,
    turn_screen_off,
)

if TORCH_INSTALLED:
    import torch

__all__ = ["QnnModelV2"]

PACKAGE_NAME = "com.bytedance.pico.secure_mr_demo.model_inspect"
COMPONENT = f"{PACKAGE_NAME}/android.app.NativeActivity"
OUTPUT_PREFIX = "model_inspect_output_"


def get_output_info_from_json(context_binary_json: str) -> tuple:
    """
    Get the output node IDs and shapes from the context binary JSON file.

    Args:
        context_binary_json: Path to the context binary JSON file.

    Returns:
        output_ids: List of output node names.
        output_shapes: List of output shapes.
    """
    with open(context_binary_json, "r") as f:
        data = json.load(f)
    try:
        graph_outputs = data["info"]["graphs"][0]["info"]["graphOutputs"]
        output_ids = [output["info"]["name"] for output in graph_outputs]
        output_shapes = [output["info"]["dimensions"] for output in graph_outputs]
    except KeyError as e:
        raise RuntimeError(f"Invalid JSON structure: missing key {e}")
    return output_ids, output_shapes


class QnnModelV2:
    """
    A class to run QNN model inference on Android devices using model_inspect APK.

    Unlike QnnModel which uses qnn-net-run binary, this class uses the model_inspect
    APK which is part of the SecureMR pipeline.

    Methods:
        __call__(x, is_nhwc=False):
            Runs inference on the input data.
    """

    def __init__(
        self,
        context_binary: str,
        context_binary_json: str,
        *,
        output_node_ids: Optional[str] = None,
        duration: int = 20,
        device_id: Optional[str] = None,
        apk_path: Optional[str] = None,
    ):
        """
        Constructor for QnnModelV2.

        Args:
            context_binary: Path to the context binary file (.bin).
            context_binary_json: Path to the context binary JSON file.
            output_node_ids: List of output node IDs (comma-separated string).
            duration: Time to wait for model_inspect to complete (seconds).
            device_id: ADB device ID (optional, auto-select if not provided).
            apk_path: Path to model_inspect APK (optional, use default if not provided).
        """
        self.context_binary = os.path.abspath(context_binary)
        self.context_binary_json = os.path.abspath(context_binary_json)
        self.duration = duration

        if not os.path.exists(self.context_binary):
            raise FileNotFoundError(f"Context binary not found: {self.context_binary}")
        if not os.path.exists(self.context_binary_json):
            raise FileNotFoundError(f"Context binary JSON not found: {self.context_binary_json}")

        # Get output info from JSON
        _output_ids, _output_shapes = get_output_info_from_json(self.context_binary_json)
        if output_node_ids is None:
            self.output_node_ids = _output_ids
            self.output_shapes = _output_shapes
        else:
            self.output_node_ids = output_node_ids.split(",") if "," in output_node_ids else [output_node_ids]
            self.output_shapes = []
            for _id in self.output_node_ids:
                if _id in _output_ids:
                    self.output_shapes.append(_output_shapes[_output_ids.index(_id)])
                else:
                    self.output_shapes.append([])

        # Setup device
        self.device = select_device(device_id)
        self.apk_path = self._resolve_apk_path(apk_path)
        self._install_apk()

        # Device paths
        self.device_tmp_dir = f"/sdcard/Android/data/{PACKAGE_NAME}/files"
        self.device_output_dirs = [
            f"{self.device_tmp_dir}/model_inspect",
            "/data/local/tmp/securemr_model_inspect",
        ]

        # Push model files to device
        self._push_model_files()

        # Temp directory for local operations
        self.temp_dir = tempfile.mkdtemp()
        self._input_shapes = None

    def _resolve_apk_path(self, user_path: Optional[str]) -> Path:
        """Resolve the APK path."""
        if user_path:
            return Path(user_path)
        return Path(__file__).resolve().parent.parent / "inspect" / "apks" / "model_inspect-debug.apk"

    def _install_apk(self) -> None:
        """Install the model_inspect APK on the device."""
        if not self.apk_path.exists():
            raise FileNotFoundError(f"APK not found: {self.apk_path}")
        print(f"Installing model_inspect APK from {self.apk_path}...")
        install_apk(self.device, str(self.apk_path), PACKAGE_NAME)

    def _push_model_files(self) -> None:
        """Push model files to the device."""
        # Clean and create device directories
        run_adb(["shell", "rm", "-rf", self.device_tmp_dir], self.device, check=False)
        run_adb(["shell", "mkdir", "-p", self.device_tmp_dir], self.device)
        for output_dir in self.device_output_dirs:
            run_adb(["shell", "rm", "-f", f"{output_dir}/*"], self.device, check=False)

        # Push model files
        print("Pushing model files to device...")
        run_adb(["push", self.context_binary, f"{self.device_tmp_dir}/model.serialized.bin"], self.device)
        run_adb(["push", self.context_binary_json, f"{self.device_tmp_dir}/model.serialized.json"], self.device)

    def set_input_shapes(self, input_shapes):
        """Set input shapes for the model."""
        while np.asarray(input_shapes).ndim > 2:
            input_shapes = input_shapes[0]
        self._input_shapes = input_shapes

    def __del__(self):
        """Clean up resources when the object is deleted."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir) and (not DEBUG_QNN):
            shutil.rmtree(self.temp_dir)

    def _pull_outputs(self, local_dir: Path) -> bool:
        """Pull output files from the device."""
        local_dir.mkdir(parents=True, exist_ok=True)
        outputs_pulled = False
        for directory in self.device_output_dirs:
            listing = capture_adb(["shell", "ls", directory], self.device, check=False)
            for filename in listing.splitlines():
                filename = filename.strip().replace("\r", "")
                if not filename or not filename.startswith(OUTPUT_PREFIX):
                    continue
                if not filename.endswith(".bin"):
                    continue
                device_path = f"{directory}/{filename}"
                data = capture_adb_binary(["shell", "cat", device_path], self.device, check=False)
                if not data:
                    continue
                with open(local_dir / filename, "wb") as file_handle:
                    file_handle.write(data)
                outputs_pulled = True
        return outputs_pulled

    def _run_model_inspect(self, input_path: Optional[str] = None) -> Path:
        """
        Run model_inspect on the device and return the output directory.

        Args:
            input_path: Path to the input tensor file (NHWC float32).

        Returns:
            Path to the local output directory.
        """
        device_input_path = f"{self.device_tmp_dir}/input.bin"

        # Clean output directories
        for output_dir in self.device_output_dirs:
            run_adb(["shell", "rm", "-f", f"{output_dir}/*"], self.device, check=False)

        # Push input if provided
        if input_path:
            run_adb(["push", input_path, device_input_path], self.device)
            run_adb(["shell", "setprop", "debug.securemr.model_inspect.input", device_input_path], self.device)
        else:
            run_adb(["shell", "setprop", "debug.securemr.model_inspect.input", "''"], self.device)

        run_adb(
            ["shell", "setprop", "debug.securemr.model_inspect.model_dir", self.device_tmp_dir],
            self.device,
        )

        ensure_screen_on(self.device)

        # Stop existing process
        existing_pid = capture_adb(["shell", "pidof", PACKAGE_NAME], self.device, check=False).strip()
        if existing_pid:
            run_adb(["shell", "am", "force-stop", PACKAGE_NAME], self.device, check=False)
            time.sleep(3)

        # Launch app
        run_adb(["shell", "am", "force-stop", "com.bytedance.pico.openmr"], self.device, check=False)
        time.sleep(2)
        print(f"Launching model_inspect app (waiting {self.duration}s)...")
        run_adb(["shell", "am", "start", "-n", COMPONENT], self.device)
        time.sleep(self.duration)
        run_adb(["shell", "am", "force-stop", PACKAGE_NAME], self.device, check=False)

        # Pull outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_output_dir = Path(self.temp_dir) / f"model_inspect_outputs_{timestamp}"
        outputs_pulled = self._pull_outputs(local_output_dir)

        if outputs_pulled:
            print(f"Outputs saved under {local_output_dir}")
        else:
            print("Warning: No output files pulled from device.")

        turn_screen_off(self.device)
        return local_output_dir

    def __call__(self, x, is_nhwc=False):
        """Run inference on the model.

        Args:
            x: Input tensor for the model (numpy array or torch tensor).
               Expected shape: (N, C, H, W) if is_nhwc=False, or (N, H, W, C) if is_nhwc=True.
            is_nhwc: Whether the input tensor is in NHWC format.

        Returns:
            Model outputs as numpy array or torch tensor (matching input type).
        """
        if TORCH_INSTALLED and isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
            return_torch = True
        else:
            x_np = np.asarray(x)
            return_torch = False

        # Ensure 4D input
        if x_np.ndim == 3:
            x_np = x_np[np.newaxis, ...]

        assert x_np.ndim == 4, f"Expected 4D input, got {x_np.ndim}D"

        # Convert to NHWC if needed (model_inspect expects NHWC)
        if not is_nhwc:
            x_np = x_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # Process each sample in the batch
        batch_size = x_np.shape[0]
        all_outputs = {output_id: [] for output_id in self.output_node_ids}

        for i in range(batch_size):
            # Save input to temp file
            input_file = os.path.join(self.temp_dir, f"input_{i}.bin")
            x_np[i:i+1].astype(np.float32).tofile(input_file)

            # Run model_inspect
            output_dir = self._run_model_inspect(input_file)

            # Read outputs
            for output_id in self.output_node_ids:
                output_file = output_dir / f"{OUTPUT_PREFIX}{output_id}.bin"
                if not output_file.exists():
                    # Try with underscore prefix
                    output_file = output_dir / f"{OUTPUT_PREFIX}_{output_id}.bin"
                if not output_file.exists():
                    raise FileNotFoundError(
                        f"Output file not found for {output_id}. "
                        f"Available files: {list(output_dir.glob('*.bin'))}"
                    )
                output_data = np.fromfile(output_file, dtype=np.float32)
                all_outputs[output_id].append(output_data)

        # Stack outputs
        results = []
        for output_id, output_shape in zip(self.output_node_ids, self.output_shapes):
            stacked = np.stack(all_outputs[output_id], axis=0)
            # Reshape if shape is known
            if output_shape:
                try:
                    # Output shape from JSON doesn't include batch dimension
                    target_shape = [batch_size] + list(output_shape)
                    stacked = stacked.reshape(target_shape)
                except ValueError:
                    pass  # Keep flat if reshape fails
            if return_torch:
                results.append(torch.from_numpy(stacked))
            else:
                results.append(stacked)

        if len(self.output_node_ids) == 1:
            return results[0]
        return results
