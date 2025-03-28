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

import securemr
from securemr import TORCH_INSTALLED

import numpy as np
import pytest

if TORCH_INSTALLED:
    import torch
    import torchvision.models as models

    torch.set_printoptions(precision=4, sci_mode=False)

np.set_printoptions(precision=4, suppress=True)


@pytest.mark.skipif(not TORCH_INSTALLED, reason="torch is required")
def test_pytorch_to_qnn_and_compare():
    torch_model = models.resnet18(pretrained=True)
    torch_model.eval()

    qnn_model = securemr.pytorch_to_qnn(torch_model, "1,3,224,224")
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)

    out1 = torch_model(input_data)
    out2 = qnn_model(input_data)
    np.testing.assert_allclose(out1.detach().numpy(), out2.numpy(), atol=1e-2)
