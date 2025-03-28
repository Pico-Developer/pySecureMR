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

try:
    from ._bindings._securemr import EDataType, BaseType, Tensor, TensorMat, TensorFactory
    from ._bindings._securemr import EOperatorType, OperatorFactory
    from ._bindings._securemr import TensorPoint2Float, TensorPoint2Double, TensorPoint2Int
    from ._bindings._securemr import TensorPoint3Float, TensorPoint3Double, TensorPoint3Int
    LD_LIBRARY_PATH_IS_SET = True
    
except ImportError as e:
    import os
    bindings_path = os.path.join(os.path.dirname(__file__), "_bindings")
    print('\033[91m' + f"ImportError! Are you forget to set LD_LIBRARY_PATH to {bindings_path}?" + '\033[0m')
    LD_LIBRARY_PATH_IS_SET = False

from .qnn_model import QnnModel
from .pytorch_to_qnn import pytorch_to_qnn
from .utils import TORCH_INSTALLED

__version__ = "0.0.1"
