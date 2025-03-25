import securemr as smr
import cv2
import numpy as np
import pathlib
import os

'''
Dump TensorMat to file (cpp snippet)

    std::string opName("ConvertColor");
  	for (unsigned int inputIdx = 0; inputIdx < inputValues.size(); inputIdx++) {
    	auto& inputName = m_inputs[inputIdx];
    	auto& inputTensor = inputValues[inputIdx];
    	char* inputTensorRawData = nullptr;
    	auto handle = inputTensor->getRawByteArray(&inputTensorRawData);

    	std::string filename = "/sdcard/" + opName + "_input.raw";
    	std::ofstream file(filename, std::ios::binary);
    	if (file) {
    	  file.write(static_cast<const char*>(inputTensorRawData, inputTensor->getRawByteArrayLength());
    	  Utils::Log::Write(Utils::Log::Level::Info, formatMessage("Saved input tensor '", inputName, "' to ", filename));
          file.close();
    	}
    	inputTensor->yieldRawByteAccess(handle);	
    }

'''


def preprocess(test_image):
    img = cv2.imread(test_image)
    x = smr.TensorMat.from_numpy(img)

    op1 = smr.OperatorFactory.create(smr.EOperatorType.GET_AFFINE)
    op2 = smr.OperatorFactory.create(smr.EOperatorType.APPLY_AFFINE)
    image_width = 3248
    image_height = 2464
    crop_x1 = 1444
    crop_y1 = 1332
    crop_x2 = 2045
    crop_y2 = 1933
    crop_width = 224
    crop_height = 224

    src_points = smr.TensorPoint2Float.from_numpy(np.array([
        [crop_x1, crop_y1],
        [crop_x2, crop_y1],
        [crop_x2, crop_y2],
        ], dtype=np.float32))
    dst_points = smr.TensorPoint2Float.from_numpy(np.array([
        [0, 0],
        [crop_width, 0],
        [crop_width, crop_height],
        ], dtype=np.float32))
    affine_mat = smr.TensorMat.from_numpy(np.zeros((2, 3), dtype=np.float32))

    op1.data_as_operand(src_points, 0)
    op1.data_as_operand(dst_points, 1)
    op1.connect_result_to_data_array(0, affine_mat)
    op1.compute(0)
    
    # crop image
    assert img.shape[:2] == (image_height, image_width)
    y1 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.UINT8) 
    op2.data_as_operand(affine_mat, 0)
    op2.data_as_operand(x, 1)
    op2.connect_result_to_data_array(0, y1)
    op2.compute(0)

    # to gray
    ConvertColorOp = smr.OperatorFactory.create(smr.EOperatorType.CONVERT_COLOR, [str(cv2.COLOR_BGR2GRAY)])
    y2 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.UINT8) 
    ConvertColorOp.data_as_operand(y1, 0)
    ConvertColorOp.connect_result_to_data_array(0, y2)
    ConvertColorOp.compute(0)
    
    # uint8 to float32
    y3 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.FLOAT32) 
    op3 = smr.OperatorFactory.create(smr.EOperatorType.ASSIGNMENT)
    op3.data_as_operand(y2, 0)
    op3.connect_result_to_data_array(0, y3)
    op3.compute(0)

    # 255.0 -> 1.0
    op4 = smr.OperatorFactory.create(smr.EOperatorType.ARITHMETIC_COMPOSE, ["{0} / 255.0"])
    y4 = smr.TensorMat((crop_width, crop_height), 1, smr.EDataType.FLOAT32) 
    op4.data_as_operand(y3, 0)
    op4.connect_result_to_data_array(0, y4)
    op4.compute(0)

    return y4


def main():
    root = pathlib.Path(__file__).parent.resolve()
    test_image = root / "number_5.png"
    x = preprocess(str(test_image)).numpy()
    
    context_binary_file = root / "mnist.serialized.bin"
    model = smr.QnnModel(context_binary_file, "host", name="mnistwild_test")
    # # You can also run QnnModel on android device, but root is required
    # model = smr.QnnModel(context_binary_file, "android", name="mnistwild_test")
    
    x = x[None, :, :, None] # HxW to NHWC
    score, idx = model(x, is_nhwc=True)
    print("number: ", int(idx.squeeze()))
    print("score: ", score.squeeze())


def debug():
    input_file = "/sdcard/input_input_1.raw"
    output_file1 = "/sdcard/output__538.raw"
    output_file2 = "/sdcard/output__539.raw"

    os.system(f"adb pull {input_file}")
    os.system(f"adb pull {output_file1}")
    os.system(f"adb pull {output_file2}")

    root = pathlib.Path(__file__).parent.resolve()
    context_binary_file = root / "mnist.serialized.bin"
    model = smr.QnnModel(context_binary_file, "android", name="mnistwild_test")
    
    with open(os.path.basename(input_file), "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    x = data.reshape(1, 224, 224, 1)

    with open(os.path.basename(output_file1), "rb") as f:
        idx_device = np.fromfile(f, dtype=np.float32)
    with open(os.path.basename(output_file2), "rb") as f:
        score_device = np.fromfile(f, dtype=np.int32)

    score, idx = model(x, is_nhwc=True)
    print("number: ", int(idx.squeeze()))
    print("score: ", score.squeeze())

    print("number(device): ", idx_device)
    print("score(device): ", score_device)
    os.system(f"rm {os.path.basename(input_file)} {os.path.basename(output_file1)} {os.path.basename(output_file2)}")


def debug2():
    input_file = "/sdcard/Arithmetic_input.raw"
    output_file = "/sdcard/Arithmetic_output.raw"

    os.system(f"adb pull {input_file}")
    os.system(f"adb pull {output_file}")

    with open(os.path.basename(input_file), "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    x1 = data.reshape(1, 224, 224, 1)

    with open(os.path.basename(output_file), "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    x2 = data.reshape(1, 224, 224, 1)

    os.system(f"rm {os.path.basename(input_file)} {os.path.basename(output_file)}")

    __import__('ipdb').set_trace()
    pass


def debug3():
    input_file = "/sdcard/Assignment_input.raw"
    output_file = "/sdcard/Assignment_output.raw"

    os.system(f"adb pull {input_file}")
    os.system(f"adb pull {output_file}")

    with open(os.path.basename(input_file), "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)
    x1 = data.reshape(1, 224, 224, 1)

    with open(os.path.basename(output_file), "rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    x2 = data.reshape(1, 224, 224, 1)

    os.system(f"rm {os.path.basename(input_file)} {os.path.basename(output_file)}")
    __import__('ipdb').set_trace()
    pass


if __name__ == "__main__":
    main()
    # debug()
    # debug2()
    # debug3()
