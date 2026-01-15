
## arithmetic used as transpose operator

There is no `transpose` operator inside securemr operator, but it can be implemented by `arithmetic`:
"T({0} / 2)" in `arithmetic` means divide first tensor and then transpose it.

---

## argMax + arithmetic (index+1)

When you need to convert argMax result (0-based) to 1-based index for slicing, use this pattern:

```cpp
.argMax(scores, bestIndex)           // bestIndex = 0-based
.arithmetic("({0} + 1)", {bestIndex}, bestIndexPlusOne)  // Convert to 1-based
.assignment(bestIndex, (*slice2)[0][0])
.assignment(bestIndexPlusOne, (*slice2)[0][1])
```

Use case: Get the index of maximum score, then use it as end index in tensor slicing.

Reference: `pose_detection.cpp:309-314`, `face_tracking.cpp:263-268`, `whackamole_detection.cpp:437-438`

---

## assignment + slice (tensor indexing)

To extract a specific row/element from tensor using computed indices:

```cpp
// Create slice indices tensor
auto srcSlice = std::make_shared<PipelineTensor>(pipeline, TensorAttribute_SliceArray{.size = 2});
srcSlice->setData(reinterpret_cast<int8_t*>(new int32_t[4]{0, -1, 0, -1}), 4 * sizeof(int32_t));

// Use computed indices to fill slice
.assignment((*indices)[{{i, i+1}, {0, 1}}], (*srcSlice)[0][0])
.assignment((*indicesPlusOne)[{{i, i+1}, {0, 1}}], (*srcSlice)[0][1])
.assignment((*src)[srcSlice], (*dst)[{{i, i+1}, {0, -1}}]);
```

Use case: Extract data from tensor using dynamically computed indices (e.g., NMS results).

Reference: `pose_detection.cpp:313-316`, `yolo_object_detection.cpp:139-141`

---

## getAffine (3-point to transform matrix)

Compute affine transformation matrix from 3 corresponding points (source -> destination):

```cpp
const auto roiPoints = std::make_shared<PipelineTensor>(
    pipeline, TensorAttribute_Point2Array{.size = 3, .dataType = ...});
.assignment(bestHipKeypoint, (*roiPoints)[0])
.assignment(bestHeadKeypoint, (*roiPoints)[1])
.assignment(bestLeftKeypoint, (*roiPoints)[2])
.getAffine(roiPoints, std::array{128.0f, 128.0f, 128.0f, 0.0f, 255.0f, 128.0f}, roiAffine);
```

Parameters: source points, target points [cx, cy, cx, tx, ty], output matrix.

Use case: ROI alignment in detection + landmark pipeline, computing transform for cropping.

Reference: `pose_detection.cpp:330`, `whackamole_detection.cpp:456`

---

## compareTo + all (multi-condition AND)

Combine multiple boolean conditions with logical AND:

```cpp
.compareTo(*score > std::vector<float>{0.55}, scoreDetected)
.compareTo(*uv > uvThreshold, uvDetected)
.all(uvDetected, uvDetectedAll)
.assignment(uvDetectedAll, (*temp)[{{0, 1}, {0, 1}}])
.assignment(scoreDetected, (*temp)[{{1, 2}, {0, 1}}])
.all(temp, isFaceDetectedPlaceholder);
```

Use case: Require multiple conditions to be true simultaneously (e.g., score threshold AND position valid).

Reference: `face_tracking.cpp:270-276`

---

## transform + arithmetic (3D pose estimation)

Transform 3D points with rotation, translation, and scale to get pose matrix:

```cpp
.transform(rvecTensor, pointXYZ, svecTensor, currentPosition)  // Get 4x4 transform matrix
.arithmetic("({0} * {1})", {leftEyeTransform, currentPosition}, currentPosition);  // Apply camera transform
```

Use case: Convert 2D landmarks to 3D world coordinates, apply camera-space to world transform.

Reference: `face_tracking.cpp:336`, `yolo_object_detection.cpp:230`

---

## elementwise MULTIPLY (coordinate sign flip)

Flip coordinate signs using elementwise multiplication with a sign vector:

```cpp
// For Y/Z axis flip: [1, -1, -1]
static float XYZ_MULTIPLIER[]{1.0, -1.0, -1.0};
auto multiplier = std::make_shared<PipelineTensor>(..., reinterpret_cast<int8_t*>(XYZ_MULTIPLIER), ...);
.elementwise(Pipeline::ElementwiseOp::MULTIPLY, {point, multiplier}, point);
```

Use case: Convert between different coordinate systems (e.g., camera space to world space).

Reference: `pose_detection.cpp:323`, `face_tracking.cpp:334`, `yolo_object_detection.cpp:228`

---

## Pipeline Chaining (detection -> affine update -> landmark)

Chain multiple pipelines where one pipeline's output conditions the next:

```cpp
// Detection pipeline outputs ROI affine and detection confidence
const auto detection = m_secureMrDetectionPipeline->submit(
    {{smallF32ImagePlaceholder, resizedLeftFp32Global},
     {isPoseDetectedPlaceholder, isPoseDetectedGlobal},
     {roiAffinePh1, roiAffineGlobal}}, pre, nullptr);

// Affine update pipeline (only runs if detection succeeds)
const auto affine = m_secureMrAffineUpdatePipeline->submit(
    {{roiAffinePh2, roiAffineGlobal}, {roiAffinePh3, roiAffineUpdatedGlobal}}, 
    detection, isPoseDetectedGlobal);  // Use detection as pre-condition

// Landmark pipeline (only runs if affine update completes)
return m_secureMrLandmarkPipeline->submit(
    {{largeU8ImagePlaceholder, vstOutputLeftUint8Global},
     {bodyLandmarkPlaceholder, bodyLandmarkGlobal},
     {roiAffinePh4, roiAffineUpdatedGlobal}},
    affine, nullptr);
```

Use case: Multi-stage processing where later stages depend on earlier results.

Reference: `pose_detection.cpp:376-386`

---

## sortMatByRow + assignment (top-k selection)

Sort matrix rows and extract top-k elements:

```cpp
.sortMatByRow(scores, sortedScores, sortedIndicesPerRow)
.assignment((*sortedScores)[{{0, 8400}, {0, 1}}], bestScores)
.assignment((*sortedIndicesPerRow)[{{0, 8400}, {0, 1}}], bestIndices);
```

Use case: Select highest scoring predictions for NMS input.

Reference: `yolo_object_detection.cpp:439-441`

---

## nms + CopyTensorBySlice (object detection post-processing)

Combine NMS with index-based class label lookup:

```cpp
.nms(bestScores, boxes, nmsScoresPlaceholder, nmsBoxesPlaceholder, nmsIndices, 0.5);

// Then use nmsIndices to lookup class names
CopyTensorBySlice(m_secureMrModelInferencePipeline, bestIndices, classesSelectPlaceholder, nmsIndices, NUMBER_OF_OBJECTS);
```

Use case: NMS gives box indices, use those indices to lookup corresponding class labels.

Reference: `yolo_object_detection.cpp:462-464`

---

## runJavascript (custom logic)

Execute JavaScript code for complex data transformations not supported by built-in operators:

```cpp
if (std::vector<char> javascriptCode; LoadModelData(JSCODE_PATH, javascriptCode)) {
    pipeline->runJavascript(javascriptCode.data(), javascriptCode.size(),
        {{"inputArray", inputTensor}},  // Inputs
        {{"outputArray", outputTensor}}  // Outputs
    );
}
```

Use case: Complex game logic, custom collision detection, state machines.

Reference: `whackamole_detection.cpp:479-493` (joint transforms for whackamole game)

---

## execRenderCommand + DrawText (UI text rendering)

Render dynamic text labels on 3D objects:

```cpp
auto textArrayTensor = std::make_shared<PipelineTensor>(pipeline,
    TensorAttribute{.dimensions = {80, 13}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});

// Fill with class names from COCO dataset
for (int i = 0; i < classes.size(); i++) {
    auto textTensor = MakeScalarStringTensor(pipeline, className);
    pipeline->assignment(textTensor, (*textArrayTensor)[{{i, i+1}, {0, -1}}]);
}

auto drawTextCmd = std::make_shared<RenderCommand_DrawText>();
drawTextCmd->text = textArrayTensor;
drawTextCmd->startPosition = startTensor;
drawTextCmd->fontSize = fontSizeTensor;
drawTextCmd->colors = colorTensor;
pipeline->execRenderCommand(drawTextCmd);
```

Use case: Display detection labels, scores, or debug info in AR/VR.

Reference: `yolo_object_detection.cpp:235-267`, `mnistwild.cpp:342-349`

---

## Multiple gltf placeholder (batch rendering)

Render multiple instances of the same 3D model at different positions:

```cpp
// Create multiple global tensors from same GLTF data
gltfAsset1 = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
gltfAsset2 = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());

// Create placeholders
gltfPlaceholderTensor1 = PipelineTensor::PipelinePlaceholderLike(pipeline, gltfAsset1);
gltfPlaceholderTensor2 = PipelineTensor::PipelinePlaceholderLike(pipeline, gltfAsset2);

// Render each at different position
pipeline->execRenderCommand(std::make_shared<RenderCommand_Render>(gltfPlaceholderTensor1, pose1));
pipeline->execRenderCommand(std::make_shared<RenderCommand_Render>(gltfPlaceholderTensor2, pose2));
```

Use case: Render multiple detections, particles, or repeated objects.

Reference: `yolo_object_detection.cpp:596-604`

---

## cameraAccess + applyAffine + arithmetic (image preprocessing pipeline)

Complete image preprocessing pipeline for model input:

```cpp
.cameraAccess(nullptr, vstOutputLeftUint8Placeholder, nullptr, nullptr)  // Get camera image
.applyAffine(affineMatReshape, vstOutputLeftUint8Placeholder, resizedUint8)  // Resize via affine
.assignment(resizedUint8, resizedFp32Placeholder)
.arithmetic("({0} / 255.0)", {resizedFp32Placeholder}, resizedFp32Placeholder);  // Normalize
```

Use case: Prepare camera frame for neural network inference (resize + normalize).

Reference: `pose_detection.cpp:181-185`, `whackamole_detection.cpp:295-299`

---

## arithmetic interpolation (smooth transitions)

Create smooth transitions between values:

```cpp
.arithmetic("({0} * 0.95 + {1} * 0.05)", {previousPosition, currentPosition}, interpolatedResult)
.assignment(interpolatedResult, previousPosition);
```

Use case: Reduce jitter in tracking, smooth camera or object movements.

Reference: `face_tracking.cpp:369-372`

---

## debugRenderText (on-screen debugging)

Render debug information directly on screen:

```cpp
.debugRenderText(gltfPlaceholder, scoreTensor, 256, 256, 
    std::tuple<float, float>{0.5F, 0.5F},  // Position
    64.0F,  // Font size
    std::array<std::array<uint8_t, 4>, 2>{{{255, 255, 255, 255}, {0, 0, 0, 255}}},  // Colors
    static_cast<uint16_t>(1));  // Material ID
```

Use case: Display scores, counts, or debug info in VR demo.

Reference: `whackamole_detection.cpp:567-569`

---

## slice assignment with computed indices (tensor scatter)

Write to specific positions in a tensor using computed indices:

```cpp
.assignment((*pointXYZ)[{{0, 1},{2, 3}}], depth)
.elementwise(Pipeline::ElementwiseOp::MIN, {depth, minDepth}, minDepth)
.assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{0, 1}}])
.assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{1, 2}}])
.assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{2, 3}}])
.elementwise(Pipeline::ElementwiseOp::MULTIPLY, {pointXYZ, pointXYZAdj}, pointXYZ);
```

Use case: Selective update of tensor elements based on computed values.

Reference: `yolo_object_detection.cpp:220-226`

---

## newTextureToGLTF + UpdateMaterial (dynamic textures)

Update GLTF material textures at runtime:

```cpp
.newTextureToGLTF(gltfPlaceholder, texturePlaceholder, newTextureId);
auto updateMaterialCmd = std::make_shared<RenderCommand_UpdateMaterial>();
updateMaterialCmd->gltfTensor = gltfPlaceholder;
updateMaterialCmd->materialIds = std::vector<uint16_t>{0};
updateMaterialCmd->attribute = RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_BASE_COLOR;
updateMaterialCmd->materialValues = newTextureId;
pipeline->execRenderCommand(updateMaterialCmd);
```

Use case: Display model inference results as texture on 3D object (e.g., cropped image on TV screen).

Reference: `mnistwild.cpp:350-357`, `whackamole_detection.cpp:206-213`
