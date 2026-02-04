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

It is very useful for checking the correctness of the model on android platform.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import glob
import sys
from pathlib import Path
from typing import List

import numpy as np

from ..core.utils import DEBUG_QNN, TORCH_INSTALLED

if TORCH_INSTALLED:
    import torch

from ppadb.client import Client as AdbClient


def get_output_node_ids(context_binary: str, QNN_SDK_ROOT: str, context_binary_json: str = None) -> List[str]:  # noqa
    """
    Get the output node IDs from the context binary file.

    Args:
        context_binary: context binary file path
        QNN_SDK_ROOT: qnn sdk path
        context_binary_json: if provided, skip qnn-context-binary-utility

    Returns:
        output_ids: output index list of context binary.
        output_shapes: output shape list of context binary.
    """
    if context_binary_json:
        with open(context_binary_json, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    else:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as tmp_json:
            bin_file = Path(QNN_SDK_ROOT) / "bin/x86_64-linux-clang/qnn-context-binary-utility"
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
        output_shapes = [output["info"]["dimensions"] for output in graph_outputs]
    except KeyError as e:
        raise RuntimeError(f"Invalid JSON structure: missing key {e}")
    return output_ids, output_shapes


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

TARGETS = {
    "host": "x86_64-linux-clang",
    "android": "aarch64-android",
}


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
        *,
        target: str = "host",   # "auto", "android", "host"
        output_node_ids: str = None,
        name="sampleapp_test",
        context_binary_json: str = None,
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
        if not sys.platform.startswith("linux"):
            raise RuntimeError("QNN runtime is only supported on Linux.")
        self.QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT")
        assert self.QNN_SDK_ROOT, "Please set QNN_SDK_ROOT env."

        self.temp_dir = tempfile.mkdtemp()
        cache_context_binary = os.path.join(self.temp_dir, os.path.basename(context_binary))
        shutil.copy(context_binary, cache_context_binary)
        self.context_binary = cache_context_binary
        _output_node_ids, _output_shapes = get_output_node_ids(context_binary, self.QNN_SDK_ROOT, context_binary_json)
        if output_node_ids is None:
            self.output_node_ids = _output_node_ids
            self.output_shapes = _output_shapes
        else:
            self.output_node_ids = output_node_ids.split(",") if "," in output_node_ids else output_node_ids
            self.output_shapes = []
            for _id in self.output_node_ids:
                assert _id in _output_node_ids
                self.output_shapes.append(
                    _output_shapes[_output_node_ids.index(_id)]
                    )

        self.set_target(target)
        self._input_shapes = None

    def set_target(self, target):
        if target == "auto":
            target = set_host_or_android(target)
        assert target in TARGETS
        self.target = TARGETS[target]
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
            name = Path(self.context_binary).stem
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
    
    def set_input_shapes(self, input_shapes):
        while np.asarray(input_shapes).ndim > 2:
            input_shapes = input_shapes[0]
        self._input_shapes = input_shapes

    def __del__(self):
        """Clean up resources when the object is deleted."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir) and (not DEBUG_QNN):
            shutil.rmtree(self.temp_dir)

    def __call__(self, x, is_nhwc=False):
        """Run inference on the model.

        Args:
            x: Input tensor for the model.
            is_nhwc: Whether the input tensor is in NHWC format.

        Returns:
            Model outputs.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        assert x.ndim == 4
        if not is_nhwc:
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC

        is_numpy = isinstance(x, np.ndarray)

        with tempfile.TemporaryDirectory() as temp_calib_dir:
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
        QNN_SDK_ROOT = self.QNN_SDK_ROOT  # noqa
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

    def _generate_fake_data(self, num_samples: int = 100, output_path: str = None):
        """Generate fake quantization data similar to generate_fake_quanti_data.py."""
        if output_path is None:
            output_path = os.path.join(self.temp_dir, "fake_data_list.txt")
        
        outdir = os.path.join(os.path.dirname(output_path), "fake_raw")
        os.makedirs(outdir, exist_ok=True)
        
        # For QNN context binary, we need to get input shapes from the binary
        # For simplicity, we'll assume common input shapes for now
        # In a real implementation, you would parse the context binary to get actual input shapes
        if self._input_shapes is not None:
            input_shapes = self._input_shapes 
        else:
            input_shapes = [[1, 3, 224, 224]]  # Default shape
        
        list_fid = open(output_path, "w")
        
        for i in range(num_samples):
            filename_line = ""
            for input_idx, shape in enumerate(input_shapes):
                raw_filename = os.path.join(outdir, f"fake_{input_idx}_{i:06d}.raw")
                
                # Generate random data
                data = np.random.rand(*shape).astype(np.float32)
                data.tofile(raw_filename)
                
                if filename_line:
                    filename_line += f" {raw_filename}"
                else:
                    filename_line = raw_filename
            
            list_fid.write(f"{filename_line}\n")
        
        list_fid.close()
        return output_path

    def benchmark(self, input_list_path: str = None, runs: int = 100, runtimes: List[str] = None, 
                  measurements: List[str] = None, output_dir: str = None):
        """Run benchmark on android device using QNN benchmark tool.
        
        Args:
            input_list_path: Path to the input list file for benchmark
            runs: Number of test runs (default: 100)
            runtimes: List of runtimes to test (e.g., ["HTP_v69"], default: ["HTP_v69"])
            measurements: List of measurements to collect (e.g., ["timing"], default: ["timing"])
            output_dir: Output directory for benchmark results
            
        Returns:
            Path to benchmark results directory
        """
        assert self.target == "aarch64-android", "benchmark only valid when target='android'"
        if runtimes is None:
            runtimes = ["HTP_v69"]
        if measurements is None:
            measurements = ["timing"]
        
        # Import the benchmark config generation function
        try:
            from .generate_benchmark_json import generate_benchmark_config
        except ImportError:
            raise ImportError("generate_benchmark_json.py not found in current package")
        
        # Create temporary directory for benchmark files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate benchmark JSON configuration
            benchmark_json_path = os.path.join(temp_dir, "benchmark_config.json")

            if not output_dir:
                output_dir = os.path.join(temp_dir, "benchmark_output")
            
            # Get model name from context binary
            model_name = Path(self.context_binary).stem
            
            # If input_list_path is not provided, generate fake data
            if input_list_path is None:
                input_list_path = self._generate_fake_data(num_samples=runs, output_path=os.path.join(temp_dir, "fake_data_list.txt"))
            
            # Generate benchmark config using imported function
            generate_benchmark_config(
                model_path=self.context_binary,
                input_list_path=input_list_path,
                output_json=benchmark_json_path,
                task_name=model_name,
                model_name=model_name,
                runs=runs,
                runtimes=runtimes,
                measurements=measurements,
                version="qnn",
                cache=True,
                output=output_dir
            )
            
            # Run the benchmark using qnn_bench.py
            qnn_bench_path = Path(self.QNN_SDK_ROOT) / "benchmarks"/ "QNN" / "qnn_bench.py"
            
            # Set ANDROID_NDK_ROOT environment variable
            env = os.environ.copy()
            env["ANDROID_NDK_ROOT"] = "/home/bingwen/opt/android-ndk-r26c"
            
            benchmark_cmd = [
                "python3", str(qnn_bench_path), "-c", benchmark_json_path
            ]

            # Bugfix for ADSP_LIBRARY_PATH, benchmark.py required "artifacts/dsp/lib/xxx.so"
            # Upload first if need and link to it.
            res = self.adb.shell(f"ls {self.remote_dir}/dsp; echo $?")
            if "No such file or directory" in res:
                self.adb.shell(f"mkdir -p {self.remote_dir}/dsp")
                for dsp_library in self.dsp_libraries:
                    lib_name = os.path.basename(dsp_library)
                    self.adb.push(dsp_library, f"{self.remote_dir}/dsp/{lib_name}")
            dsp_path = "/data/local/tmp/qnn_benchmark/artifacts/dsp/"
            res = self.adb.shell(f"ls {dsp_path}; echo $?")
            if "No such file or directory" in res:
                self.adb.shell(f"mkdir -p {dsp_path}")
                self.adb.shell(f"ln -s {self.remote_dir}/dsp {dsp_path}/lib")
            
            # Execute benchmark
            os.system(" ".join(benchmark_cmd))
            # result = subprocess.run(benchmark_cmd, env=env, capture_output=True, text=True)
            # if result.returncode != 0:
            #     raise RuntimeError(f"Benchmark failed: {result.stderr}")
            
            # print result
            result_csv = glob.glob(os.path.join(output_dir, "latest_results/benchmark_stats_*.csv"))
            if not result_csv:
                print("No result csv found.")
                return
            os.system(f"cat {result_csv[0]} | grep NetRun | grep Inference")


    def sampleapp_run(self, input_list, output_dir):
        """Run the sample application with the given inputs.

        Args:
            input_list: Path to the input list file.
            output_dir: Directory to save the output.

        Returns:
            Model outputs.
        """

        def _push(src, dst="", verbose=False):
            if not os.path.exists(src):
                raise FileNotFoundError(f"Source file not found: {src}")
            remote_path = f"{self.remote_dir}/{dst}{os.path.basename(src)}"
            if verbose: print(f">> adb push {src} {remote_path}")
            self.adb.push(src, remote_path)
            # Ensure proper permissions on pushed file
            if verbose: print(f">> adb shell chmod 644 {remote_path}")
            self.adb.shell(f"chmod 644 {remote_path}")
            # Verify file was pushed successfully
            if verbose: print(f">> adb shell ls {remote_path} 2>/dev/null")
            if not self.adb.shell(f"ls {remote_path} 2>/dev/null").strip():
                raise RuntimeError(f"Failed to push file to device: {remote_path}")
            if verbose: print("")
        
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
            _push(temp_input_list)
            remote_input_list = f"{self.remote_dir}/temp_input_list.txt"
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
        error_patterns = ["Error", "error", "Could not readInputListsV2",
                          "failed", "Failed", "failure"]
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
