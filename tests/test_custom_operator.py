import time
import json

import numpy as np

import securemr as smr
from securemr.operators import CustomOperatorBase, create_operator_configs
from securemr.serialization import Pipeline as SerializablePipeline, DeserializedPipeline


class _AddOperator(CustomOperatorBase):
    def __init__(self) -> None:
        super().__init__(
            operand_names=["lhs", "rhs"],
            result_names=["out"],
        )
        self.verified_operands = []
        self.verified_results = []
        self.task_ids = []

    def verify_operand(self, index, tensor) -> bool:
        self.verified_operands.append(index)
        return tensor is not None

    def verify_result(self, index, tensor) -> bool:
        self.verified_results.append(index)
        return tensor is not None

    def compute(self, task_id: int, operands, results) -> None:
        self.task_ids.append(task_id)
        lhs = operands[0].numpy().astype(np.float32)
        rhs = operands[1].numpy().astype(np.float32)
        out = np.ascontiguousarray(lhs + rhs)
        results[0].load_from_raw_byte_arrays(out.tobytes())


def _make_scalar(value: float) -> smr.TensorMat:
    array = np.array([[value]], dtype=np.float32)
    return smr.TensorMat.from_numpy(array)


def _make_output() -> smr.TensorMat:
    return smr.TensorMat.from_numpy(np.zeros((1, 1), dtype=np.float32))


def test_custom_operator_roundtrip():
    implementation = _AddOperator()
    configs = create_operator_configs(implementation)
    assert len(configs) == 1
    assert configs[0].startswith("token:")

    try:
        lhs = _make_scalar(1.5)
        rhs = _make_scalar(2.5)
        out = _make_output()
        flag = lhs.get_type_flag()

        pipeline = smr.Pipeline()
        ph_lhs = pipeline.allocate_placeholder([1, 1], flag)
        ph_rhs = pipeline.allocate_placeholder([1, 1], flag)
        ph_out = pipeline.allocate_placeholder([1, 1], flag)

        op_id = pipeline.allocate_operator(smr.EOperatorType.PYTHON_CUSTOM, configs)
        native_op = pipeline.query_operator(op_id)

        assert native_op.get_operand_cnt() == 2
        assert native_op.get_results_cnt() == 1
        assert native_op.get_operand_idx_from_name("lhs") == 0
        assert native_op.get_operand_idx_from_name("rhs") == 1
        assert native_op.get_result_idx_from_name("out") == 0

        assert native_op.verify_compatibility_data_array_as_operand(0, lhs)
        assert native_op.verify_compatibility_data_array_as_operand(1, rhs)
        assert native_op.verify_compatibility_data_array_as_result(0, out)

        tensor_lhs = pipeline.query_local_tensor(ph_lhs)
        tensor_rhs = pipeline.query_local_tensor(ph_rhs)
        tensor_out = pipeline.query_local_tensor(ph_out)

        native_op.data_as_operand(tensor_lhs, 0)
        native_op.data_as_operand(tensor_rhs, 1)
        native_op.connect_result_to_data_array(0, tensor_out)

        placeholder_map = {
            int(ph_lhs): lhs,
            int(ph_rhs): rhs,
            int(ph_out): out,
        }

        task = smr.Task(pipeline, placeholder_map)
        task.verify_all_place_holder_contained()
        task.setup_place_holder_mapping()

        pool = smr.ThreadPool2()
        pool.enqueue(task)

        for _ in range(100):
            if not pipeline.cannot_modified():
                break
            time.sleep(0.01)

        result = out.numpy()
        assert np.allclose(result.squeeze(), 4.0)
        assert implementation.task_ids == [task.id]
        assert implementation.verified_operands == [0, 1]
        assert implementation.verified_results == [0]
    finally:
        implementation.release()


def test_custom_operator_serialization_roundtrip():
    implementation = _AddOperator()
    configs = create_operator_configs(implementation)

    try:
        flag = _make_scalar(0.0).get_type_flag()

        pipeline = SerializablePipeline()

        lhs_id = pipeline.allocate_placeholder([1, 1], flag, name="lhs")
        rhs_id = pipeline.allocate_placeholder([1, 1], flag, name="rhs")
        out_id = pipeline.allocate_placeholder([1, 1], flag, name="out")

        op_id = pipeline.allocate_operator(smr.EOperatorType.PYTHON_CUSTOM, configs)
        proxy = pipeline.query_operator(op_id)
        proxy.data_as_operand(pipeline.query_local_tensor(lhs_id), 0)
        proxy.data_as_operand(pipeline.query_local_tensor(rhs_id), 1)
        proxy.connect_result_to_data_array(0, pipeline.query_local_tensor(out_id))

        pipeline.set_inputs(["lhs", "rhs"])
        pipeline.set_outputs(["out"])

        spec_snapshot = json.loads(json.dumps(pipeline.spec))

        deserialized = DeserializedPipeline(spec_snapshot)
        try:
            lhs = np.array([[3.0]], dtype=np.float32)
            rhs = np.array([[4.0]], dtype=np.float32)

            result_tensor = deserialized({"lhs": lhs, "rhs": rhs})
            output = result_tensor.numpy()

            assert np.allclose(output.squeeze(), 7.0)
            assert implementation.task_ids, "compute should have been invoked"
        finally:
            deserialized.close()
    finally:
        implementation.release()
