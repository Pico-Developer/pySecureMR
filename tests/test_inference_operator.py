import numpy as np
import pytest
import securemr as smr
from pathlib import Path

from securemr.operators import inference_operator as inference_module


class _CaptureResult:
    def __init__(self) -> None:
        self.raw = None

    def load_from_raw_byte_arrays(self, payload: bytes) -> None:
        self.raw = payload


@pytest.fixture
def mnist_qnn_model_path() -> str:
    root = Path(__file__).resolve().parent.parent
    model_path = root / "examples" / "mnistwild" / "mnist.serialized.bin"
    assert model_path.exists(), "MNIST serialized model is required for the test"
    return str(model_path)


def test_model_inference_operator_qnn_backend(monkeypatch, mnist_qnn_model_path):
    operator = inference_module.ModelInferenceOperator(mnist_qnn_model_path, device="host")

    input_np = np.full((28, 28, 1), 0.5, dtype=np.float32)
    operand = smr.TensorMat.from_numpy(input_np)
    operator.data_as_operand(operand, 0)

    out1 = _CaptureResult()
    out2 = _CaptureResult()
    operator.connect_result_to_data_array(0, out1)
    operator.connect_result_to_data_array(1, out2)
    
    operator.forward()
    assert operator._backend == "qnn"
    assert operator.output_shapes == [[1], [1]]

    # TODO: check output data
