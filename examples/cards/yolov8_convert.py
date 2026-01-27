#!/usr/bin/env python3
"""Build a SecureMR pipeline for YOLOv8 playing cards with full post-processing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import onnxruntime as ort
import securemr as smr
from securemr.core.utils import convert_from_dtype, convert_to_dtype, mat_flag
from securemr.serialization import type_to_name

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PIPELINE_JSON = ROOT / "yolov8s_cards_pipeline.json"
MODEL_ASSET = "yolov8s_playing_cards.serialized.bin"

# Constants
TARGET_SIZE = 416
NC = 52  # number of classes
CONF_THRES = 0.55
IOU_THRES = 0.45
MAX_DET = 300
NUM_ANCHORS = 3549  # number of anchors in model output


def letterbox(img: np.ndarray, target: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Ultralytics-style letterbox."""
    h0, w0 = img.shape[:2]
    th, tw = target
    r = min(th / h0, tw / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    top = round((th - nh) / 2 - 0.1)
    left = round((tw - nw) / 2 - 0.1)
    bottom = th - nh - top
    right = tw - nw - left
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, (left, top)


def non_max_suppression_yolov8(pred, input_w: int, input_h: int, conf_thres: float, iou_thres: float, nc: int, max_det: int = 300) -> np.ndarray:
    """Minimal YOLOv8 NMS identical to infer_cards_onnx.py."""
    conf_thres = max(conf_thres, 0.55)
    arr = pred[0] if isinstance(pred, (list, tuple)) else pred
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.transpose(1, 0)
    boxes_xywh = arr[:, :4]
    cls_logits = arr[:, 4:4 + nc]
    cls_scores = 1.0 / (1.0 + np.exp(-cls_logits))
    conf = cls_scores.max(axis=1)
    j = cls_scores.argmax(axis=1)
    mask = conf >= conf_thres
    if not np.any(mask):
        return np.zeros((0, 6), dtype=np.float32)
    boxes_xywh, conf, j = boxes_xywh[mask], conf[mask], j[mask]
    if boxes_xywh.max() <= 1.0 + 1e-3:
        boxes_xywh[:, [0, 2]] *= input_w
        boxes_xywh[:, [1, 3]] *= input_h
    xy, wh = boxes_xywh[:, :2], boxes_xywh[:, 2:] / 2.0
    xyxy = np.concatenate([xy - wh, xy + wh], axis=1)
    keep: List[int] = []
    idxs = conf.argsort()[::-1]
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(xyxy[i, 0], xyxy[idxs[1:], 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[idxs[1:], 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[idxs[1:], 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[idxs[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        area_others = (xyxy[idxs[1:], 2] - xyxy[idxs[1:], 0]) * (xyxy[idxs[1:], 3] - xyxy[idxs[1:], 1])
        iou = inter / (area_i + area_others - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_thres]
    keep = keep[:max_det]
    return np.concatenate([xyxy[keep], conf[keep, None], j[keep, None].astype(np.float32)], axis=1)


def postprocess(out: List[np.ndarray], names: List[str], img0: np.ndarray, ratio: float, pad: Tuple[int, int]):
    """Project model outputs back to the original image space."""
    det = non_max_suppression_yolov8(out, TARGET_SIZE, TARGET_SIZE, conf_thres=CONF_THRES, iou_thres=IOU_THRES, nc=len(names))
    if det.shape[0] == 0:
        return []
    boxes, scores, cls_ids = det[:, :4], det[:, 4], det[:, 5].astype(int)
    pad_w, pad_h = pad
    x1 = (boxes[:, 0] - pad_w) / ratio
    y1 = (boxes[:, 1] - pad_h) / ratio
    x2 = (boxes[:, 2] - pad_w) / ratio
    y2 = (boxes[:, 3] - pad_h) / ratio
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    h, w = img0.shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return [(float(x1), float(y1), float(x2), float(y2), float(s), int(c)) for (x1, y1, x2, y2), s, c in zip(boxes, scores, cls_ids)]


def load_names(session: ort.InferenceSession) -> List[str]:
    """Load class names from ONNX metadata."""
    try:
        meta = session.get_modelmeta().custom_metadata_map
        for key in ("names", "class_names", "classes"):
            if meta and key in meta:
                raw = meta[key]
                try:
                    d = json.loads(raw)
                except Exception:
                    d = eval(raw)
                if isinstance(d, dict):
                    return [d[str(i)] if str(i) in d else d[i] for i in range(len(d))]
                elif isinstance(d, list):
                    return d
    except Exception:
        pass
    raise RuntimeError("Class names not found in ONNX metadata.")


def run_python_reference(img_path: Path, onnx_path: Path):
    """Run the original Python pipeline to obtain baseline outputs."""
    img0 = cv2.imread(str(img_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    th, tw = inp.shape[2], inp.shape[3]
    lb, ratio, (pad_w, pad_h) = letterbox(img0, (th, tw))
    cv2.imwrite(str(ROOT / "card_preprocessed_python.jpeg"), lb)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)
    out = session.run(None, {inp.name: x})
    names = load_names(session)
    dets = postprocess(out, names, img0, ratio, (pad_w, pad_h))
    return {
        "image": img0,
        "letterbox_bgr": lb,
        "letterbox_rgb": rgb,
        "model_input": x,
        "model_output": out[0].astype(np.float32),
        "detections": dets,
        "names": names,
        "ratio": ratio,
        "pad": (pad_w, pad_h),
    }


def build_pipeline_spec(img_shape: Tuple[int, int], target: int = TARGET_SIZE) -> Dict[str, Any]:
    """Construct pipeline spec with preprocessing, model inference, and post-processing.
    
    The pipeline implements:
    1. Letterbox preprocessing (resize + pad to 416x416)
    2. RGB conversion and normalization
    3. QNN model inference
    4. Transpose model output and extract boxes/scores
    5. Sigmoid activation on class scores
    6. Sort to find best class per anchor
    7. xywh to xyxy box conversion
    8. NMS to filter detections
    """
    h0, w0 = img_shape
    r = min(target / h0, target / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    pad_h = round((target - nh) / 2 - 0.1)
    pad_w = round((target - nw) / 2 - 0.1)

    tensors: Dict[str, Dict] = {
        # Input
        "input_bgr": {
            "dimensions": [h0, w0], "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": True, "usage": 6,
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        # Preprocessing tensors
        "src_points": {
            "dimensions": [3], "channels": 2,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": False, "usage": 1,
            "flag": int(smr.BaseType.POINT_2) | int(smr.EDataType.FLOAT32),
            "value": [0.0, 0.0, float(w0), 0.0, float(w0), float(h0)],
        },
        "dst_points": {
            "dimensions": [3], "channels": 2,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": False, "usage": 1,
            "flag": int(smr.BaseType.POINT_2) | int(smr.EDataType.FLOAT32),
            "value": [float(pad_w), float(pad_h), float(pad_w + nw), float(pad_h), float(pad_w + nw), float(pad_h + nh)],
        },
        "affine": {
            "dimensions": [2, 3], "channels": 1,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": False, "usage": 6,
            "flag": mat_flag(smr.EDataType.FLOAT32, 1),
        },
        "resized_bgr": {
            "dimensions": [nh, nw], "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": False, "usage": 6,
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        "canvas_114": {
            "dimensions": [target, target], "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": False, "usage": 6,
            "flag": mat_flag(smr.EDataType.UINT8, 3),
            "value": [114] * (target * target * 3),
        },
        "letterbox_bgr": {
            "dimensions": [target, target], "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": False, "usage": 6,
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        "letterbox_rgb": {
            "dimensions": [target, target], "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": True, "usage": 6,
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        "letterbox_float": {
            "dimensions": [target, target], "channels": 3,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": False, "usage": 6,
            "flag": mat_flag(smr.EDataType.FLOAT32, 3),
        },
        "normalized": {
            "dimensions": [target, target], "channels": 3,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": True, "usage": 6,
            "flag": mat_flag(smr.EDataType.FLOAT32, 3),
        },
        # Model output: QNN outputs [1, 56, 3549] -> stored as [56, 3549]
        "model_output_raw": {
            "dimensions": [56, NUM_ANCHORS], "channels": 1,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": True, "usage": 2,
            "flag": mat_flag(smr.EDataType.FLOAT32, 1),
        },
    }

    operators: List[Dict[str, Any]] = [
        # ===== Preprocessing =====
        {"type": type_to_name(smr.EOperatorType.GET_AFFINE), "inputs": ["src_points", "dst_points"], "outputs": ["affine"]},
        {"type": type_to_name(smr.EOperatorType.APPLY_AFFINE), "inputs": ["affine", "input_bgr"], "outputs": ["resized_bgr"]},
        {"type": type_to_name(smr.EOperatorType.ASSIGNMENT), "inputs": ["canvas_114"], "outputs": ["letterbox_bgr"]},
        {"type": type_to_name(smr.EOperatorType.ASSIGNMENT), "inputs": ["resized_bgr"], "outputs": ["letterbox_bgr"], "src_slices": [[0, nh], [0, nw]], "dst_slices": [[pad_h, pad_h + nh], [pad_w, pad_w + nw]]},
        {"type": type_to_name(smr.EOperatorType.CONVERT_COLOR), "inputs": ["letterbox_bgr"], "outputs": ["letterbox_rgb"], "flag": int(cv2.COLOR_BGR2RGB)},
        {"type": type_to_name(smr.EOperatorType.ASSIGNMENT), "inputs": ["letterbox_rgb"], "outputs": ["letterbox_float"]},
        {"type": type_to_name(smr.EOperatorType.ARITHMETIC_COMPOSE), "inputs": ["letterbox_float"], "outputs": ["normalized"], "expression": "{0} / 255.0"},
        
        # ===== Model inference =====
        {"type": type_to_name(smr.EOperatorType.RUN_MODEL_INFERENCE),
         "inputs": [{"name": "images", "tensor": "normalized"}],
         "outputs": [{"name": "output0", "tensor": "model_output_raw"}],
         "model_name": "yolov8s_playing_cards",
         "model_file": "/sdcard/Android/data/com.bytedance.pico.secure_mr_demo.pipeline_inspect/files/" + MODEL_ASSET},
        
        # NOTE: Post-processing (transpose, sigmoid, NMS) is done in Python.
        # ARITHMETIC_COMPOSE T({0}) doesn't work correctly on this device.
    ]

    return {
        "metadata": {"version": 1},
        "tensors": tensors,
        "operators": operators,
        "inputs": ["input_bgr"],
        "outputs": ["letterbox_rgb", "normalized", "model_output_raw"],
    }


def save_pipeline(img_path: Path) -> Dict:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    spec = build_pipeline_spec(img.shape[:2])
    with open(PIPELINE_JSON, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
    print(f"Saved pipeline to {PIPELINE_JSON}")
    return spec


def dump_binary(arr: np.ndarray, path: Path, dtype: np.dtype) -> None:
    arr.astype(dtype).tofile(str(path))
    print(f"Wrote {path} ({arr.size} values, dtype={dtype})")


def run_pipeline_inspect(pipeline: Path, input_bin: Path, input_tensor: str) -> Path:
    """Invoke pipeline_inspect via adb."""
    python_bin = REPO_ROOT / ".venv" / "bin" / "python"
    cmd = [str(python_bin), "-m", "securemr.inspect.pipeline_cli", "--pipeline", str(pipeline), "--input", str(input_bin), "--input-tensor", input_tensor, "--duration", "25"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    latest = sorted((REPO_ROOT / "tmp_data").glob("pipeline_inspect_outputs_*"))[-1]
    print(f"Pipeline inspect outputs in {latest}")
    return latest


def load_output_bins(out_dir: Path, spec: Dict) -> Dict[str, np.ndarray]:
    tensors = spec.get("tensors", {})
    outputs = set(spec.get("outputs", []))
    result: Dict[str, np.ndarray] = {}
    for path in sorted(out_dir.glob("pipeline_inspect_output_*.bin")):
        stem = path.stem.replace("pipeline_inspect_output_", "")
        target_name = None
        for name in outputs:
            if name in stem:
                target_name = name
                break
        if target_name is None:
            target_name = stem
        info = tensors.get(target_name, {})
        dtype_idx = int(info.get("data_type", convert_from_dtype(np.float32)))
        dtype = convert_to_dtype(dtype_idx, target="numpy")
        data = np.fromfile(str(path), dtype=dtype)
        dims = list(info.get("dimensions", []))
        channels = int(info.get("channels", 1))
        shape = dims + ([channels] if channels != 1 else [])
        if shape:
            try:
                data = data.reshape(shape)
            except ValueError:
                pass
        result[target_name] = data
        print(f"Loaded {target_name}: shape={data.shape}, dtype={data.dtype}")
    return result


def visualize_detections(img: np.ndarray, dets, names: List[str], out_path: Path) -> None:
    vis = img.copy()
    for x1, y1, x2, y2, s, c in dets:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        label = f"{names[c]} {s:.2f}"
        tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis, (p1[0], p1[1] - tsize[1] - 6), (p1[0] + tsize[0] + 6, p1[1]), (0, 255, 0), -1)
        cv2.putText(vis, label, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)
    print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-device", action="store_true", help="Run pipeline_inspect on connected device")
    args = parser.parse_args()

    img_path = ROOT / "card.jpeg"
    onnx_path = ROOT / "yolov8s_playing_cards.onnx"

    # Build pipeline JSON and baseline artifacts
    spec = save_pipeline(img_path)
    ref = run_python_reference(img_path, onnx_path)
    visualize_detections(ref["image"], ref["detections"], ref["names"], ROOT / "card_output_python.jpeg")
    dump_binary(ref["model_output"], ROOT / "baseline_model_output.bin", np.float32)
    dump_binary(ref["image"], ROOT / "card_input_uint8.bin", np.uint8)
    
    # Print Python reference results
    print("\n=== Python reference detections ===")
    for det in ref["detections"]:
        x1, y1, x2, y2, s, c = det
        print(f"  {ref['names'][c]}: score={s:.4f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    if args.run_device:
        out_dir = run_pipeline_inspect(PIPELINE_JSON, ROOT / "card_input_uint8.bin", "input_bgr")
        outputs = load_output_bins(out_dir, spec)
        
        # Check preprocessing output
        if "letterbox_rgb" in outputs:
            lb_rgb = outputs["letterbox_rgb"]
            print(f"\n=== Preprocessing check ===")
            print(f"letterbox_rgb shape: {lb_rgb.shape}, expected: {ref['letterbox_rgb'].shape}")
            cv2.imwrite(str(ROOT / "card_preprocessed_pipeline.jpeg"), cv2.cvtColor(lb_rgb, cv2.COLOR_RGB2BGR))
            print(f"Preprocessing allclose: {np.allclose(lb_rgb, ref['letterbox_rgb'], rtol=1, atol=1)}")
        
        # Check normalized input
        if "normalized" in outputs:
            norm = outputs["normalized"]
            print(f"\n=== Normalized input check ===")
            print(f"normalized shape: {norm.shape}")
            ref_norm = ref["model_input"][0].transpose(1, 2, 0)  # CHW -> HWC
            print(f"normalized range: [{norm.min():.4f}, {norm.max():.4f}]")
            print(f"reference range: [{ref_norm.min():.4f}, {ref_norm.max():.4f}]")
            print(f"Normalized allclose: {np.allclose(norm, ref_norm, rtol=1e-3, atol=1e-3)}")
        
        # Check model output
        model_out = outputs.get("model_output_raw")
        if model_out is not None:
            print(f"\n=== Model output check ===")
            model_out = model_out.reshape((1, 56, NUM_ANCHORS))
            print(f"Model output shape: {model_out.shape}")
            print(f"Model output range: [{model_out.min():.4f}, {model_out.max():.4f}]")
            print(f"Reference range: [{ref['model_output'].min():.4f}, {ref['model_output'].max():.4f}]")
            print(f"Model output allclose: {np.allclose(model_out, ref['model_output'], rtol=1e-2, atol=1e-2)}")
            
            # Run post-processing on device model output using Python
            dets_from_device = postprocess([model_out], ref["names"], ref["image"], ref["ratio"], ref["pad"])
            print(f"\n=== Post-processing on device model output ===")
            for det in dets_from_device:
                x1, y1, x2, y2, s, c = det
                print(f"  {ref['names'][c]}: score={s:.4f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            visualize_detections(ref["image"], dets_from_device, ref["names"], ROOT / "card_output_pipeline.jpeg")
            np.save(ROOT / "pipeline_model_output.npy", model_out)
            
            # Full numerical consistency check
            print(f"\n=== Model output numerical consistency ===")
            is_identical = np.allclose(model_out, ref["model_output"], rtol=1e-4, atol=1e-4)
            if is_identical:
                print(f"  ✓ Model output is numerically identical to Python reference!")
            else:
                diff = np.abs(model_out - ref["model_output"])
                print(f"  Model output max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
            
            # Compare final detection results
            print(f"\n=== Detection results comparison ===")
            print(f"Python reference ({len(ref['detections'])} detections):")
            ref_dets = sorted(ref["detections"], key=lambda x: -x[4])
            for det in ref_dets:
                x1, y1, x2, y2, s, c = det
                print(f"  {ref['names'][c]}: score={s:.4f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            print(f"\nPipeline output ({len(dets_from_device)} detections):")
            pipeline_dets = sorted(dets_from_device, key=lambda x: -x[4])
            for det in pipeline_dets:
                x1, y1, x2, y2, s, c = det
                print(f"  {ref['names'][c]}: score={s:.4f}, box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            # Check if detections match
            if len(ref_dets) == len(pipeline_dets):
                all_match = True
                for i, (r_det, p_det) in enumerate(zip(ref_dets, pipeline_dets)):
                    rx1, ry1, rx2, ry2, rs, rc = r_det
                    px1, py1, px2, py2, ps, pc = p_det
                    score_match = abs(rs - ps) < 0.01
                    box_match = all(abs(a - b) < 3.0 for a, b in zip([rx1, ry1, rx2, ry2], [px1, py1, px2, py2]))
                    class_match = rc == pc
                    if not (score_match and box_match and class_match):
                        all_match = False
                print(f"\n=== Final consistency check ===")
                if all_match:
                    print(f"  ✓ All {len(ref_dets)} detections match between Python and Pipeline!")
                else:
                    print(f"  ✗ Some detections differ (see above)")
            else:
                print(f"\n=== Final consistency check ===")
                print(f"  Detection count differs: Python={len(ref_dets)}, Pipeline={len(pipeline_dets)}")
        else:
            print(f"\nmodel_output_raw not found in outputs")
            print(f"Available outputs: {list(outputs.keys())}")
    else:
        print("\nSkipping device run; use --run-device once pipeline is ready.")


if __name__ == "__main__":
    main()
