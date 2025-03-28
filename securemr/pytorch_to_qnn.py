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

    ) -> QnnModel:
    """
    Convert pytorch model to qnn model.
    """
    QNN_SDK_ROOT = os.getenv("QNN_SDK_ROOT", None)
    if not QNN_SDK_ROOT:
        raise RuntimeError("QNN_SDK_ROOT not found. Please source qnn environment or install qnn first.")

    # dump torch_model to a temp dir
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model.pt")

    torch_model.eval()
    torch.save(torch_model, model_path)

    original_dir = os.getcwd()
    os.chdir(temp_dir)

    so_target = "x86_64-linux-clang"
    htp_backend = f"{QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so"

    # qnn-pytorch-converter
    if isinstance(qnn_pytorch_convert_kwargs, list):
        kwargs = " ".join(qnn_pytorch_convert_kwargs)
    else:
        kwargs = qnn_pytorch_convert_kwargs
    cmd = f"qnn-pytorch-converter --input_network {model_path} --float_bitwidth 16 --input_dim 'input' {input_shape} {kwargs}"
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
    os.chdir(original_dir)

    assert os.path.exists(context_binary_file)
    
    try:
        qnn_model = QnnModel(context_binary_file, "host")
    except Exception as e:
        if DEBUG_QNN:
            print(f"\033[0;33m Oooooops! Debug qnn convert in {temp_dir}\033[0m")
        else:
            print(f"\033[0;33m Oooooops! Set `DEBUG_QNN=1` to debug.\033[0m")
        raise e
    
    if not DEBUG_QNN:
        shutil.rmtree(temp_dir)
    return qnn_model
