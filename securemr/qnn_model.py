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
QnnModel class is used to run inference on different targets (host or android).

It is very seful for checking the correctness of the model on android platform.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np

from .utils import DEBUG_QNN, TORCH_INSTALLED

if TORCH_INSTALLED:
    import torch

from ppadb.client import Client as AdbClient


def get_output_node_ids(context_binary: str, QNN_SDK_ROOT: str) -> List[str]:
    """
    Get the output node IDs from the context binary file.

    Args:
        context_binary: context binary file path
        QNN_SDK_ROOT: qnn sdk path

    Returns:
        output_ids: output index list of context binary.
    """
    bin_file = Path(QNN_SDK_ROOT) / "bin/x86_64-linux-clang/qnn-context-binary-utility"

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as tmp_json:
        cmd = [
            str(bin_file),
            "--context_binary",
            context_binary,
            "--json_file",
            tmp_json.name,
        ]
        subprocess.run(cmd, check=True)
        tmp_json.seek(0)
        data = json.load(tmp_json)
        try:
            graph_outputs = data["info"]["graphs"][0]["info"]["graphOutputs"]
            output_ids = [output["info"]["name"] for output in graph_outputs]
        except KeyError as e:
            raise RuntimeError(f"Invalid JSON structure: missing key {e}")
    return output_ids


def set_host_or_android():
    """Set the target platform for QNN models.

    Returns:
        The selected platform.
    """
    client = AdbClient(host="127.0.0.1", port=5037)
    if client.devices():
        return "android"
    else:
        return "host"


class QnnModel:
    """
    A class to represent a QNN model for running inference on different targets (host or android).

    Methods:
        __call__(x, is_nhwc=False):
            Runs inference on the input data.
        qnn_net_run(input_list, output_dir):
            Runs the QNN model on the host platform.
        sampleapp_build():
            Builds the sample application for the Android target.
        sampleapp_run(input_list, output_dir):
            Runs the QNN model on the Android platform.
    """

    def __init__(
        self,
        context_binary: str,
        target: str = "host",
        output_node_ids: str = None,
        name="sampleapp_test",
    ):
        """
        Construct for QnnModel.

        Args:
            context_binary : str
                The path to the context binary file.
            target : str
                The target platform for running the model ('host' or 'android').
            output_node_ids : str
                List of output node IDs for the model, split by comma
        """
        self.QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT")
        assert self.QNN_SDK_ROOT, "Please set QNN_SDK_ROOT env."

        if target is None:
            target = set_host_or_android(target)

        target_list = ["host", "android"]
        assert target in target_list
        target_names = ["x86_64-linux-clang", "aarch64-android"]
        self.target = target_names[target_list.index(target)]

        self.temp_dir = tempfile.mkdtemp()
        cache_context_binary = os.path.join(self.temp_dir, os.path.basename(context_binary))
        shutil.copy(context_binary, cache_context_binary)
        self.context_binary = cache_context_binary
        if output_node_ids is None:
            self.output_node_ids = get_output_node_ids(context_binary, self.QNN_SDK_ROOT)
        else:
            self.output_node_ids = output_node_ids.split(",")
        if self.target == "aarch64-android":
            # Default is "127.0.0.1" and 5037
            # Allow configurable ADB host and port for Docker environments
            adb_host = os.getenv("ADB_HOST", "host.docker.internal" if os.path.exists("/.dockerenv") else "127.0.0.1")
            adb_port = int(os.getenv("ADB_PORT", "5037"))
            try:
                client = AdbClient(host=adb_host, port=adb_port)
                if devices := client.devices():
                    self.adb = devices[0]
                else:
                    raise RuntimeError("No android devices found.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to ADB at {adb_host}:{adb_port}. Error: {str(e)}\n"
                    "Please ensure ADB is running on the host and properly forwarded to Docker."
                )
            self.remote_dir = f"/data/local/tmp/{name}"
            self.adb.shell(f"rm -rf {self.remote_dir}")
            self.adb.shell(f"mkdir -p {self.remote_dir}")
            (
                self.binfile,
                self.cpu_libraries,
                self.dsp_libraries,
            ) = self.sampleapp_build()
        else:
            pass

    def __del__(self):
        """Clean up resources when the object is deleted."""
        if os.path.exists(self.temp_dir) and (not DEBUG_QNN):
            shutil.rmtree(self.temp_dir)

    def __call__(self, x, is_nhwc=False):
        """Run inference on the model.

        Args:
            x: Input tensor for the model.
            is_nhwc: Whether the input tensor is in NHWC format.

        Returns:
            Model outputs.
        """
        assert x.ndim == 4
        if not is_nhwc:
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC

        is_numpy = isinstance(x, np.ndarray)

        with tempfile.TemporaryDirectory() as temp_calib_dir:
            temp_calib_dir = self.remote_dir if self.target == "aarch64-android" else temp_calib_dir
            list_txt = os.path.join(temp_calib_dir, "input_list.txt")
            list_fid = open(list_txt, "w")
            cnt = 0
            for x_i in x:
                raw_filename = os.path.join(temp_calib_dir, f"{cnt:06d}.raw")
                if is_numpy:
                    x_i[None, :, :, :].astype(np.float32).tofile(raw_filename)
                else:
                    x_i.unsqueeze(0).numpy().astype(np.float32).tofile(raw_filename)
                list_fid.write(raw_filename + "\n")
                cnt += 1
            list_fid.close()

            output_dir = os.path.join(temp_calib_dir, "output_dir")
            os.makedirs(output_dir, exist_ok=True)

            if self.target == "aarch64-android":
                self.sampleapp_run(list_txt, output_dir)
            elif self.target == "x86_64-linux-clang":
                self.qnn_net_run(list_txt, output_dir)
            else:
                raise NotImplementedError

            res = []
            for output_node_id in self.output_node_ids:
                output_file = f"{output_node_id}.raw"
                preds = []
                for cnt in range(x.shape[0]):
                    output_raw_file = os.path.join(output_dir, f"Result_{cnt}/{output_file}")
                    if not os.path.exists(output_raw_file):
                        output_file = f"_{output_file}"
                        output_raw_file = os.path.join(output_dir, f"Result_{cnt}/{output_file}")
                    assert os.path.exists(output_raw_file), f"{output_raw_file} not exists."
                    preds.append(np.fromfile(output_raw_file, dtype=np.float32))
                if is_numpy:
                    res.append(np.asarray(preds, dtype=np.float32))
                else:
                    res.append(torch.tensor(np.asarray(preds, dtype=np.float32)).squeeze(1))
            if len(self.output_node_ids) == 1:
                return res[0]
            else:
                return res

    def qnn_net_run(self, input_list, output_dir):
        """Run the QNN network with the given inputs.

        Args:
            input_list: Path to the input list file.
            output_dir: Directory to save the output.

        Returns:
            Success status of the run.
        """
        QNN_SDK_ROOT = self.QNN_SDK_ROOT
        cmd = f"""\
        {QNN_SDK_ROOT}/bin/{self.target}/qnn-net-run \
            --backend {QNN_SDK_ROOT}/lib/{self.target}/libQnnHtp.so \
            --retrieve_context {self.context_binary} \
            --input_list {input_list} \
            --output_dir {output_dir}
        """
        os.system(cmd)

    def sampleapp_build(self):
        """Build the sample application for the model.

        Returns:
            Success status of the build.
        """
        binfile = f"{self.QNN_SDK_ROOT}/bin/aarch64-android/qnn-net-run"
        cpu_libraries, dsp_libraries = [], []
        lib_dir1 = f"{self.QNN_SDK_ROOT}/lib/aarch64-android"
        lib_dir2 = f"{self.QNN_SDK_ROOT}/lib/hexagon-v69/unsigned"
        # cpu_libraries.append(f"{root}/libs/arm64-v8a/libc++_shared.so")
        # cpu_libraries.extend([os.path.join(lib_dir1, x) for x in os.listdir(lib_dir1) if x.endswith('.so')])
        # dsp_libraries.extend([os.path.join(lib_dir2, x) for x in os.listdir(lib_dir2) if x.endswith('.so')])

        cpu_lib_names = [
            "libQnnChrometraceProfilingReader.so",
            "libQnnCpu.so",
            "libQnnDsp.so",
            "libQnnDspNetRunExtensions.so",
            "libQnnDspV66Stub.so",
            "libQnnGpu.so",
            "libQnnGpuNetRunExtensions.so",
            "libQnnHta.so",
            "libQnnHtaNetRunExtensions.so",
            "libQnnHtp.so",
            "libQnnHtpNetRunExtensions.so",
            "libQnnHtpPrepare.so",
            "libQnnHtpProfilingReader.so",
            "libQnnHtpV68Stub.so",
            "libQnnHtpV69Stub.so",
            "libQnnHtpV73Stub.so",
            "libQnnHtpV75Stub.so",
            "libQnnSaver.so",
            "libQnnSystem.so",
        ]
        dsp_lib_names = ["libQnnHtpV69Skel.so"]
        cpu_libraries.extend([os.path.join(lib_dir1, x) for x in cpu_lib_names])
        dsp_libraries.extend([os.path.join(lib_dir2, x) for x in dsp_lib_names])
        return binfile, cpu_libraries, dsp_libraries

    def sampleapp_run(self, input_list, output_dir):
        """Run the sample application with the given inputs.

        Args:
            input_list: Path to the input list file.
            output_dir: Directory to save the output.

        Returns:
            Model outputs.
        """

        def _push(src, dst=""):
            if not os.path.exists(src):
                raise FileNotFoundError(f"Source file not found: {src}")
            remote_path = f"{self.remote_dir}/{dst}{os.path.basename(src)}"
            self.adb.push(src, remote_path)
            # Ensure proper permissions on pushed file
            self.adb.shell(f"chmod 644 {remote_path}")
            # Verify file was pushed successfully
            if not self.adb.shell(f"ls {remote_path} 2>/dev/null").strip():
                raise RuntimeError(f"Failed to push file to device: {remote_path}")

        res = self.adb.shell(f"ls {self.remote_dir}/qnn-net-run; echo $?")
        if "No such file or directory" in res:
            if self.binfile:
                _push(self.binfile)
            self.adb.shell(f"mkdir -p {self.remote_dir}/cpu")
            self.adb.shell(f"mkdir -p {self.remote_dir}/dsp")
            for libfile in self.cpu_libraries:
                _push(libfile, "cpu/")
            for libfile in self.dsp_libraries:
                _push(libfile, "dsp/")
            _push(self.context_binary)

        # Ensure input_list exists and is accessible
        if not os.path.exists(input_list):
            raise FileNotFoundError(f"Input list file not found: {input_list}")

        new_input_list = os.path.splitext(input_list)[0] + "_android.txt"
        temp_input_list = None
        try:
            temp_input_list = os.path.join(os.path.dirname(input_list), "temp_input_list.txt")
            with open(temp_input_list, "w") as fid:
                with open(input_list, "r") as src:
                    for line in src:
                        raw_file = line.strip()
                        if not os.path.exists(raw_file):
                            raise FileNotFoundError(f"Raw file not found: {raw_file}")
                        _push(raw_file)
                        fid.write(f"{self.remote_dir}/{os.path.basename(raw_file)}\n")

            # Push the temporary input list to device and verify
            remote_input_list = f"{self.remote_dir}/input_list.txt"
            _push(temp_input_list)
            if not self.adb.shell(f"ls {remote_input_list} 2>/dev/null").strip():
                raise RuntimeError(f"Failed to push input list to device at {remote_input_list}")
            new_input_list = remote_input_list
        finally:
            if temp_input_list and os.path.exists(temp_input_list):
                try:
                    os.unlink(temp_input_list)  # Clean up temporary file
                except OSError:
                    pass  # Ignore cleanup errors

        new_output_dir = f"{self.remote_dir}/output_dir"
        self.adb.shell(f"rm -rf {new_output_dir}; mkdir -p {new_output_dir}")

        # Verify library paths
        cpu_lib_path = f"{self.remote_dir}/cpu"
        dsp_lib_path = f"{self.remote_dir}/dsp"
        lib_check = self.adb.shell(f"ls {cpu_lib_path}/libQnnHtp.so 2>/dev/null || echo 'missing'")
        if "missing" in lib_check:
            raise RuntimeError(f"Required library libQnnHtp.so not found in {cpu_lib_path}")

        cmd = f"""\
        export LD_LIBRARY_PATH={cpu_lib_path}:$LD_LIBRARY_PATH; \
        export ADSP_LIBRARY_PATH=\"{dsp_lib_path}\"; \
        export CDSP_ID=0;
        cd {self.remote_dir};\
        chmod +x {self.remote_dir}/qnn-net-run; {self.remote_dir}/qnn-net-run \
            --backend libQnnHtp.so \
            --retrieve_context {os.path.basename(self.context_binary)} \
            --input_list {new_input_list} \
            --output_dir {new_output_dir} 2>&1
        """

        # Execute command and capture output
        cmd_output = self.adb.shell(cmd)

        # Check for common QNN errors
        error_patterns = ["Error", "error", "Could not readInputListsV2", "failed", "Failed"]
        for pattern in error_patterns:
            if pattern in cmd_output:
                # Debug information
                debug_info = self.adb.shell(
                    f"ls -l {self.remote_dir}; cat {new_input_list} 2>/dev/null || echo 'Cannot read input list'"
                )
                raise RuntimeError(f"QNN execution failed with error:\n{cmd_output}\n\nDebug info:\n{debug_info}")

        temp_dir = os.path.dirname(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)

        # Verify output directory with retries
        max_verify_retries = 3
        for attempt in range(max_verify_retries):
            if self.adb.shell(f"ls {new_output_dir} 2>/dev/null").strip():
                break
            if attempt == max_verify_retries - 1:
                raise RuntimeError(
                    f"Output directory {new_output_dir} not created on device.\nCommand output:\n{cmd_output}"
                )
            time.sleep(1)

        # Pull output directory with retries
        max_retries = 3
        for attempt in range(max_retries):
            pull_result = self.adb.shell(f"ls {new_output_dir}/Result_*")
            if not pull_result or "No such file" in pull_result:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                raise RuntimeError(f"No output files found in {new_output_dir} after {max_retries} attempts")

            pull_cmd = f"adb pull {new_output_dir} {temp_dir}/"
            if os.system(pull_cmd) != 0:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                raise RuntimeError(f"Failed to pull output files from device after {max_retries} attempts")
            break
