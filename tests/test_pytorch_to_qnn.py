import pytest
import numpy as np
import securemr
from securemr import TORCH_INSTALLED

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
