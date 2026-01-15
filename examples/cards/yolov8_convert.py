#!/usr/bin/env python3
"""Build a SecureMR pipeline for YOLOv8 playing cards and validate against Python."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

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
DEVICE_MODEL_DIR = "/sdcard/Android/data/com.bytedance.pico.secure_mr_demo.pipeline_inspect/files"


def letterbox(img: np.ndarray, target: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Ultralytics-style letterbox with rounding that mirrors infer_cards_onnx."""
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


def non_max_suppression_yolov8(
    pred,
    input_w: int,
    input_h: int,
    conf_thres: float,
    iou_thres: float,
    nc: int,
    max_det: int = 300,
) -> np.ndarray:
    """Minimal YOLOv8 NMS identical to infer_cards_onnx.py."""
    conf_thres = max(conf_thres, 0.55)
    arr = pred[0] if isinstance(pred, (list, tuple)) else pred
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr.transpose(1, 0)

    boxes_xywh = arr[:, :4]
    cls_logits = arr[:, 4 : 4 + nc]
    cls_scores = 1.0 / (1.0 + np.exp(-cls_logits))
    conf = cls_scores.max(axis=1)
    j = cls_scores.argmax(axis=1)
    mask = conf >= conf_thres
    if not np.any(mask):
        return np.zeros((0, 6), dtype=np.float32)

    boxes_xywh = boxes_xywh[mask]
    conf = conf[mask]
    j = j[mask]

    if boxes_xywh.max() <= 1.0 + 1e-3:
        boxes_xywh[:, [0, 2]] *= input_w
        boxes_xywh[:, [1, 3]] *= input_h

    xy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:] / 2.0
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
    xyxy = xyxy[keep]
    conf = conf[keep]
    j = j[keep]
    return np.concatenate([xyxy, conf[:, None], j[:, None].astype(np.float32)], axis=1)


def postprocess(out: List[np.ndarray], names: List[str], img0: np.ndarray, ratio: float, pad: Tuple[int, int]):
    """Project model outputs back to the original image space."""
    det = non_max_suppression_yolov8(out, 416, 416, conf_thres=0.15, iou_thres=0.45, nc=len(names))
    if det.shape[0] == 0:
        return []
    boxes = det[:, :4]
    scores = det[:, 4]
    cls_ids = det[:, 5].astype(int)
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
    """Load class names from ONNX metadata (robust to list/dict formats)."""
    names = None
    try:
        meta = session.get_modelmeta().custom_metadata_map
        raw = None
        for key in ("names", "class_names", "classes"):
            if meta and key in meta:
                raw = meta[key]
                break
        if raw:
            try:
                d = json.loads(raw)
            except Exception:
                try:
                    d = eval(raw)
                except Exception:
                    d = None
            if isinstance(d, dict):
                try:
                    names = [d[str(i)] if str(i) in d else d[i] for i in range(len(d))]
                except Exception:
                    keys = sorted(d.keys(), key=lambda x: int(x))
                    names = [d[k] for k in keys]
            elif isinstance(d, list):
                names = d
    except Exception:
        pass
    if not names:
        raise RuntimeError("Class names not found in ONNX metadata.")
    return names


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
        "letterbox_rgb": rgb,
        "model_output": out[0].astype(np.float32),
        "detections": dets,
        "names": names,
        "ratio": ratio,
        "pad": (pad_w, pad_h),
    }


def build_pipeline_spec(img_shape: Tuple[int, int], target: int = 416) -> Dict:
    """Construct pipeline spec with GET_AFFINE -> APPLY_AFFINE -> color/normalize -> model."""
    h0, w0 = img_shape
    r = min(target / h0, target / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    pad_h = round((target - nh) / 2 - 0.1)
    pad_w = round((target - nw) / 2 - 0.1)

    tensors: Dict[str, Dict] = {
        "input_bgr": {
            "dimensions": [h0, w0],
            "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": True,
            "usage": int(6),
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        "src_points": {
            "dimensions": [3],
            "channels": 2,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": False,
            "usage": int(1),
            "flag": int(smr.BaseType.POINT_2) | int(smr.EDataType.FLOAT32),
            "value": [0.0, 0.0, float(w0), 0.0, float(w0), float(h0)],
        },
        "dst_points": {
            "dimensions": [3],
            "channels": 2,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": False,
            "usage": int(1),
            "flag": int(smr.BaseType.POINT_2) | int(smr.EDataType.FLOAT32),
            "value": [
                float(pad_w),
                float(pad_h),
                float(pad_w + nw),
                float(pad_h),
                float(pad_w + nw),
                float(pad_h + nh),
            ],
        },
        "affine": {
            "dimensions": [2, 3],
            "channels": 1,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": False,
            "usage": int(6),
            "flag": mat_flag(smr.EDataType.FLOAT32, 1),
        },
        "resized_bgr": {
            "dimensions": [nh, nw],
            "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": False,
            "usage": int(6),
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        "canvas_zero": {
            "dimensions": [target, target],
            "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": False,
            "usage": int(6),
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        "letterbox_bgr": {
            "dimensions": [target, target],
            "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": False,
            "usage": int(6),
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        "letterbox_rgb": {
            "dimensions": [target, target],
            "channels": 3,
            "data_type": convert_from_dtype(np.uint8),
            "is_placeholder": True,
            "usage": int(6),
            "flag": mat_flag(smr.EDataType.UINT8, 3),
        },
        "letterbox_float": {
            "dimensions": [target, target],
            "channels": 3,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": False,
            "usage": int(6),
            "flag": mat_flag(smr.EDataType.FLOAT32, 3),
        },
        "normalized": {
            "dimensions": [target, target],
            "channels": 3,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": True,
            "usage": int(6),
            "flag": mat_flag(smr.EDataType.FLOAT32, 3),
        },
        "model_output0": {
            "dimensions": [56, 1],
            "channels": 3549,
            "data_type": convert_from_dtype(np.float32),
            "is_placeholder": True,
            "usage": int(2),
        },
    }

    operators = [
        {
            "type": type_to_name(smr.EOperatorType.GET_AFFINE),
            "inputs": ["src_points", "dst_points"],
            "outputs": ["affine"],
        },
        {
            "type": type_to_name(smr.EOperatorType.APPLY_AFFINE),
            "inputs": ["affine", "input_bgr"],
            "outputs": ["resized_bgr"],
        },
        {
            "type": type_to_name(smr.EOperatorType.ARITHMETIC_COMPOSE),
            "inputs": ["canvas_zero"],
            "outputs": ["letterbox_bgr"],
            "expression": "{0} + 114.0",
        },
        {
            "type": type_to_name(smr.EOperatorType.ASSIGNMENT),
            "inputs": ["resized_bgr"],
            "outputs": ["letterbox_bgr"],
            "src_slices": [[0, nh], [0, nw]],
            "dst_slices": [[pad_h, pad_h + nh], [pad_w, pad_w + nw]],
        },
        {
            "type": type_to_name(smr.EOperatorType.CONVERT_COLOR),
            "inputs": ["letterbox_bgr"],
            "outputs": ["letterbox_rgb"],
            "flag": int(cv2.COLOR_BGR2RGB),
        },
        {
            "type": type_to_name(smr.EOperatorType.ASSIGNMENT),
            "inputs": ["letterbox_rgb"],
            "outputs": ["letterbox_float"],
        },
        {
            "type": type_to_name(smr.EOperatorType.ARITHMETIC_COMPOSE),
            "inputs": ["letterbox_float"],
            "outputs": ["normalized"],
            "expression": "{0} / 255.0",
        },
        {
            "type": type_to_name(smr.EOperatorType.RUN_MODEL_INFERENCE),
            "inputs": [{"name": "images", "tensor": "normalized"}],
            "outputs": [{"name": "output0", "tensor": "model_output0"}],
            "model_name": "yolov8s_playing_cards",
            "model": MODEL_ASSET,
            "model_dir": DEVICE_MODEL_DIR,
        },
    ]

    return {
        "metadata": {"version": 1},
        "tensors": tensors,
        "operators": operators,
        "inputs": ["input_bgr"],
        "outputs": ["letterbox_rgb", "normalized", "model_output0"],
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
    """Invoke pipeline_inspect via adb; returns output directory path."""
    python_bin = REPO_ROOT / ".venv" / "bin" / "python"
    cmd = [
        str(python_bin),
        "-m",
        "securemr.inspect.pipeline_cli",
        "--pipeline",
        str(pipeline),
        "--input",
        str(input_bin),
        "--input-tensor",
        input_tensor,
        "--duration",
        "25",
    ]
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

    if args.run_device:
        out_dir = run_pipeline_inspect(PIPELINE_JSON, ROOT / "card_input_uint8.bin", "input_bgr")
        outputs = load_output_bins(out_dir, spec)
        model_out = outputs.get("model_output0")
        if model_out is None:
            raise RuntimeError(f"model_output0 not found in {out_dir}")
        model_out = model_out.reshape((1, 56, 3549))
        dets = postprocess([model_out], ref["names"], ref["image"], ref["ratio"], ref["pad"])
        visualize_detections(ref["image"], dets, ref["names"], ROOT / "card_output_pipeline.jpeg")
        np.save(ROOT / "pipeline_model_output.npy", model_out)
        print(f"Device vs python allclose: {np.allclose(model_out, ref['model_output'], rtol=1e-3, atol=1e-3)}")
    else:
        print("Skipping device run; use --run-device once pipeline is ready.")


if __name__ == "__main__":
    main()
