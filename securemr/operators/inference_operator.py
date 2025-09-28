from typing import Any, Optional, Tuple, List
import os
import numpy as np
import securemr as smr

from .custom_operator import CustomOperatorBase
from ..qnn_model import QnnModel
from ..onnx_to_qnn import onnx_to_qnn as convert_onnx_to_qnn

try:
    import onnxruntime as ort
except Exception as e:
    ort = None


class PyOperatorBase:
    """Pysecure-like operator base class."""

    def __init__(self, **kwargs):
        self._operands: dict[int, Any] = {}
        self._results: dict[int, Any] = {}

    # Match pysecure API
    def data_as_operand(self, tensor: Any, idx: int) -> None:
        self._operands[idx] = tensor

    def connect_result_to_data_array(self, idx: int, tensor: Any) -> None:
        self._results[idx] = tensor

    def forward(self, stream_id: int = 0) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class ModelInferenceOperator(PyOperatorBase, CustomOperatorBase):
    """Operator that runs inference via onnxruntime or securemr QnnModel.

    Behavior:
      - If model_path endswith .bin: use securemr.QnnModel.
      - If model_path endswith .onnx and onnx_to_qnn=False: use onnxruntime.
      - If model_path endswith .onnx and onnx_to_qnn=True: convert via securemr.onnx_to_qnn then use QnnModel.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        convert_output_dir: Optional[str] = None,
        onnx_to_qnn: bool = False,
        qnn_is_nhwc: bool = True,
        operand_names: List[str] = ["input"],
        result_names: List[str] = ["predictions"],
    ):
        PyOperatorBase.__init__(self)
        CustomOperatorBase.__init__(self, operand_names=operand_names, result_names=result_names)

        assert os.path.exists(model_path), f"Model not found: {model_path}"

        self._backend: Optional[str] = None  # "ort" or "qnn"
        self._session = None
        self._input_name: Optional[str] = None
        self._model = None
        self._qnn_is_nhwc = bool(qnn_is_nhwc)

        dev = str(device).lower()
        is_cpu = dev == "cpu"

        if model_path.endswith(".bin"):
            target = "android" if str(device) == "android" else "host"
            self._model = QnnModel(model_path, target)
            self._backend = "qnn"
        elif model_path.endswith(".onnx"):
            if onnx_to_qnn:
                if target is None:
                    target = "host" if is_cpu else "host"
                self._model = convert_onnx_to_qnn(model_path, output=convert_output_dir)
                self._model.set_target(target)
                self._backend = "qnn"
            else:
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                so.intra_op_num_threads = 1
                providers: list[Any] = []
                if is_cpu:
                    providers.append("CPUExecutionProvider")
                else:
                    # Parse device id for CUDA provider
                    dev_id = 0
                    if dev.startswith("cuda") and ":" in dev:
                        try:
                            dev_id = int(dev.split(":", 1)[1])
                        except Exception:
                            dev_id = 0
                    else:
                        try:
                            dev_id = int(dev)
                        except Exception:
                            dev_id = 0
                    providers.append(
                        (
                            "CUDAExecutionProvider",
                            {
                                "device_id": dev_id,
                                "arena_extend_strategy": "kNextPowerOfTwo",
                                "cudnn_conv_use_max_workspace": "1",
                                "do_copy_in_default_stream": "1",
                            },
                        )
                    )
                self._session = ort.InferenceSession(model_path, session_options=so, providers=providers)
                _inp = self._session.get_inputs()[0]
                self._input_name = _inp.name
                self._expected_shape = _inp.shape  # e.g., [1, 3, 480, 640]
                self._backend = "ort"
        else:
            raise NotImplementedError("Unsupported model format. Expect .onnx or .bin")

    def forward(self, stream_id: int = 0) -> None:
        self.compute(stream_id, self._operands, self._results)

    def set_expected_shapes(
        self,
        output_shape: Tuple[int, ...],
        tensor_layout_shape: Tuple[int, ...],
    ) -> None:
        self._output_shape = tuple(int(s) for s in output_shape)
        self._tensor_layout_shape = tuple(int(s) for s in tensor_layout_shape)

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        return self._output_shape

    @property
    def tensor_layout_shape(self) -> Optional[Tuple[int, ...]]:
        return self._tensor_layout_shape

    def compute(self, task_id: int, operands, results) -> None:
        if not operands:
            raise ValueError("ModelInferenceOperator requires at least one operand")

        input_tensor = operands[0]
        prepared = self._prepare_input(input_tensor)
        y_np = self.forward_numpy(prepared)

        if self._output_shape is None:
            self._output_shape = tuple(int(s) for s in y_np.shape)
        if self._tensor_layout_shape is None:
            layout_tensor = smr.TensorMat.from_numpy(y_np)
            self._tensor_layout_shape = tuple(int(s) for s in layout_tensor.numpy().shape)

        flat = np.ascontiguousarray(y_np, dtype=np.float32).reshape(-1)
        target_shape = self._tensor_layout_shape
        assert target_shape is not None, "tensor layout shape unavailable"
        prepared_output = flat.reshape(target_shape)

        if results and results[0] is not None and hasattr(results[0], "load_from_raw_byte_arrays"):
            results[0].load_from_raw_byte_arrays(prepared_output.tobytes())
        else:
            results[0] = smr.TensorMat.from_numpy(prepared_output)

    def _prepare_input(self, tensor: Any) -> np.ndarray:
        arr = tensor.numpy()
        if arr.ndim == 4 and arr.shape[1] in (1, 3):
            prepared = arr.astype(np.float32, copy=False)
        else:
            if arr.ndim != 3:
                raise ValueError(f"Unsupported tensor rank {arr.ndim} for model inference input")
            # Stored tensors are width-major; swap to height-major before layout conversion.
            arr_swapped = arr.transpose(1, 0, 2)
            prepared = arr_swapped.astype(np.float32, copy=False)

        if prepared.max() > 1.0:
            prepared = prepared / 255.0

        if prepared.ndim == 3:
            prepared = prepared.transpose(2, 0, 1)[None, ...]

        return np.ascontiguousarray(prepared, dtype=np.float32)

    def forward_numpy(self, x_np: np.ndarray) -> np.ndarray:
        if self._backend == "ort":
            # Ensure dtype float32 and NCHW order.
            if x_np.dtype != np.float32:
                x_np = x_np.astype(np.float32)
            try:
                exp = getattr(self, "_expected_shape", None)
                if (
                    exp
                    and len(exp) == 4
                    and isinstance(exp[2], int)
                    and isinstance(exp[3], int)
                    and exp[2] is not None
                    and exp[3] is not None
                    and x_np.shape[2] == exp[3]
                    and x_np.shape[3] == exp[2]
                ):
                    # Swap H and W to match model expectation
                    x_np = x_np.transpose(0, 1, 3, 2)
            except Exception:
                pass
            outputs = self._session.run(None, {self._input_name: x_np})
            y = outputs[0]
            if y.ndim == 3 and y.shape[0] == 1:
                y = y[0]
        elif self._backend == "qnn":
            # QNN models commonly expect NHWC layout; convert if needed
            x_qnn = x_np
            if x_qnn.dtype != np.float32:
                x_qnn = x_qnn.astype(np.float32)
            if self._qnn_is_nhwc and x_qnn.ndim == 4 and x_qnn.shape[1] in (1, 3):
                # Convert NCHW -> NHWC
                x_qnn = x_qnn.transpose(0, 2, 3, 1).copy()
            try:
                outputs = self._model(x_qnn, is_nhwc=self._qnn_is_nhwc)
            except Exception as e:
                raise RuntimeError(f"QNN inference failed: {e}")
            if isinstance(outputs, list) and len(outputs) == 4:
                y = outputs[-1]
            else:
                y = outputs
            y = y.reshape(1, -1, 20)
        else:
            raise RuntimeError("RunOnnxOperator not initialized")

        def _to_numpy_arr(obj):
            if isinstance(obj, np.ndarray):
                return obj
            try:
                return obj.detach().cpu().numpy()
            except Exception:
                pass
            try:
                return np.array(obj)
            except Exception:
                return None

        y_np = _to_numpy_arr(y)
        if y_np is None:
            raise RuntimeError("Unable to convert model output to numpy array")

        # Ensure feature dimension is last; if output comes as (F, N) transpose it
        if y_np.ndim == 2 and y_np.shape[0] < y_np.shape[1] and y_np.shape[0] >= 19:
            y_np = y_np.T

        return np.ascontiguousarray(y_np)
