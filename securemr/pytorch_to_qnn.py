# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import tempfile
import shutil
from typing import Dict, List
from .qnn_model import QnnModel
from .utils import run
from .utils import DEBUG_QNN

__all__ = ["pytorch_to_qnn"]


def pytorch_to_qnn(
    torch_model: torch.nn.Module,
    input_shape: str,
    qnn_pytorch_convert_kwargs: str | List = "",
    qnn_model_lib_generator_kwargs: str | List = "",
    qnn_context_binary_generator_kwargs: str | List = "",
    output: str = None,
    via_onnx: bool = False,
    ) -> QnnModel:
    """
    Convert pytorch model to qnn model.
    """
    QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT", None)
    if not QNN_SDK_ROOT:
        raise RuntimeError("QNN_SDK_ROOT not found. Please source qnn environment or install qnn first.")

    # dump torch_model to a temp dir
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    torch_model.eval()
    if via_onnx:
        input_shape_list = [int(i) for i in input_shape.split(",")]
        example_inputs = (torch.randn(*input_shape_list),)

        model_path = os.path.join(temp_dir, "model.onnx")
        if hasattr(torch_model, "to_onnx"):
            torch_model.to_onnx(model_path, export_params=True, input_sample=example_inputs, input_names=("input", ))
        else:
            onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True, input_names=("input", ))
            onnx_program.save(model_path)
        convert_bin = "qnn-onnx-converter"
    else:
        model_path = os.path.join(temp_dir, "model.pt")
        torch.save(torch_model, model_path)
        convert_bin = "qnn-pytorch-converter"

    so_target = "x86_64-linux-clang"
    htp_backend = f"{QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so"

    # qnn-pytorch-converter
    if isinstance(qnn_pytorch_convert_kwargs, list):
        kwargs = " ".join(qnn_pytorch_convert_kwargs)
    else:
        kwargs = qnn_pytorch_convert_kwargs

    cmd = f"{convert_bin} --input_network {model_path} --float_bitwidth 16 --input_dim 'input' {input_shape} {kwargs}"
    run(cmd)

    # qnn-model-lib-generator
    if isinstance(qnn_model_lib_generator_kwargs, list):
        kwargs = " ".join(qnn_model_lib_generator_kwargs)
    else:
        kwargs = qnn_model_lib_generator_kwargs
    cmd = f"qnn-model-lib-generator -c model.cpp -b model.bin -o model_targets -t {so_target} {kwargs}"
    run(cmd)
    
    # qnn-context-binary-generator
    if isinstance(qnn_context_binary_generator_kwargs, list):
        kwargs = " ".join(qnn_context_binary_generator_kwargs)
    else:
        kwargs = qnn_context_binary_generator_kwargs
    cmd = f"qnn-context-binary-generator --backend {htp_backend} --model model_targets/{so_target}/libmodel.so --binary_file model.serialized {kwargs}"
    run(cmd)
    context_binary_file = os.path.join(temp_dir, "output/model.serialized.bin")
    assert os.path.exists(context_binary_file)
    cmd = f"qnn-context-binary-utility --context_binary {context_binary_file} --json_file {context_binary_file}.json"
    run(cmd)

    os.chdir(original_dir)

    try:
        assert os.path.exists(context_binary_file)
        qnn_model = QnnModel(context_binary_file)
    except Exception as e:
        if DEBUG_QNN:
            print(f"\033[0;33m Oooooops! Debug qnn convert in {temp_dir}\033[0m")
        else:
            print(f"\033[0;33m Oooooops! Set `DEBUG_QNN=1` to debug.\033[0m")
        raise e

    if output:
        os.makedirs(output, exist_ok=True)
        shutil.copy(context_binary_file, f"{output}/{os.path.basename(context_binary_file)}")
        shutil.copy(context_binary_file + ".json", f"{output}/{os.path.basename(context_binary_file)}.json")
    
    if not DEBUG_QNN:
        shutil.rmtree(temp_dir)
    return qnn_model
