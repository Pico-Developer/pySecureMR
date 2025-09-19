import time
import json

import numpy as np

import securemr as smr


def _float_tensor_from_numpy(arr: np.ndarray) -> smr.Tensor:
    # dtype flag: BaseType.MAT | channels, plus EDataType.FLOAT32
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    flag = int(smr.EDataType.FLOAT32) | smr.BaseType.MAT | (smr.BaseType.CHANNEL_MASK & channels)
    t = smr.TensorFactory.create(list(arr.shape[:2]), flag)
    t.load_from_raw_byte_arrays(np.ascontiguousarray(arr.astype(np.float32)).tobytes())
    return t


def _tensor_to_numpy(t: smr.Tensor, shape_hw) -> np.ndarray:
    raw = t.to_bytes()
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(*shape_hw)


def test_pipeline_arithmetic_add_scalar():
    # Create a simple pipeline: out = in + 2.0
    input_shape = [4, 4]
    flag = int(smr.EDataType.FLOAT32) | smr.BaseType.MAT | (smr.BaseType.CHANNEL_MASK & 1)

    # Global tensors backing the placeholders
    input_np = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2, 4.2],
            [1.3, 2.3, 3.3, 4.3],
        ],
        dtype=np.float32,
    )
    expected_np = input_np + 2.0

    global_input = _float_tensor_from_numpy(input_np)
    global_output = smr.TensorFactory.create(input_shape, flag)

    p = smr.Pipeline()
    ph_in = p.allocate_placeholder(input_shape, flag)
    ph_out = p.allocate_placeholder(input_shape, flag)
    op_id = p.allocate_operator(smr.EOperatorType.ARITHMETIC_COMPOSE, ["{0} + 2.0"])

    op = p.query_operator(op_id)
    t_in = p.query_local_tensor(ph_in)
    t_out = p.query_local_tensor(ph_out)

    # Connect operator IO
    op.data_as_operand(t_in, 0)
    op.connect_result_to_data_array(0, t_out)

    # Build placeholder mapping and run
    ph_map = {
        int(ph_in): global_input,
        int(ph_out): global_output,
    }

    task = smr.Task(p, ph_map, 0, None)
    task.verify_all_place_holder_contained()
    task.setup_place_holder_mapping()

    pool = smr.ThreadPool2()
    pool.enqueue(task)

    # wait until pipeline is free
    for _ in range(100):
        if not p.cannot_modified():
            break
        time.sleep(0.01)

    out_np = _tensor_to_numpy(global_output, input_shape)
    assert np.allclose(out_np, expected_np, rtol=1e-4, atol=1e-4)
