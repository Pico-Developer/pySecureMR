import cv2
import numpy as np
import onnxruntime as ort
import argparse

EXPECTED_SPADES = ["10S", "JS", "QS", "KS", "AS"]


def letterbox(img, target_size=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    th, tw = target_size
    r = min(th / h0, tw / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    # Ultralytics rounding style to mirror scale_boxes
    top = round((th - nh) / 2 - 0.1)
    left = round((tw - nw) / 2 - 0.1)
    bottom = th - nh - top
    right = tw - nw - left
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def nms(boxes, scores, iou_th=0.5):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        iou = inter / (area_i + area_others - inter + 1e-6)
        idxs = idxs[1:][iou <= iou_th]
    return keep


def xywh_to_xyxy(xywh):
    xy = xywh[:, :2]
    wh = xywh[:, 2:] / 2.0
    tl = xy - wh
    br = xy + wh
    return np.concatenate([tl, br], axis=1)


def non_max_suppression_yolov8(pred, input_w, input_h, conf_thres=0.25, iou_thres=0.45, nc=52, max_det=300):
    conf_thres = max(conf_thres, 0.55)
    arr = pred[0].transpose(1, 0)

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

    xyxy = xywh_to_xyxy(boxes_xywh)
    

    offsets = (j.astype(np.float32) * 7680.0)[:, None]
    keep = nms(xyxy + offsets, conf, iou_th=iou_thres)[:max_det]
    xyxy = xyxy[keep]
    conf = conf[keep]
    j = j[keep]
    return np.concatenate([xyxy, conf[:, None], j[:, None].astype(np.float32)], axis=1)


def load_names(session):
    names = None
    try:
        meta = session.get_modelmeta().custom_metadata_map
        raw = None
        for k in ("names", "class_names", "classes"):
            if meta and k in meta:
                raw = meta[k]
                break
        if raw:
            try:
                import json

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
    return names


def preprocess_image(img0, th, tw):
    img_lb, ratio, (pad_w, pad_h) = letterbox(img0, (th, tw))
    cv2.imwrite("card_preprocessed.jpeg", img_lb)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    x = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = np.expand_dims(x, 0)
    return x, ratio, pad_w, pad_h, img_lb


def postprocess_outputs(out, names, tw, th, ratio, pad_w, pad_h, conf_thres, iou_thres, img0):
    detections = []
    if not isinstance(out, (list, tuple)) or len(out) == 0:
        return detections
    y = out[0]
    det = non_max_suppression_yolov8(
        y,
        input_w=tw,
        input_h=th,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        nc=len(names),
        max_det=300,
    )
    if det.shape[0] == 0:
        return detections
    boxes = det[:, :4]
    scores = det[:, 4]
    cls_ids = det[:, 5].astype(int)
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
    for i in range(boxes.shape[0]):
        x1i, y1i, x2i, y2i = boxes[i]
        si = float(scores[i])
        ci = int(cls_ids[i])
        detections.append((x1i, y1i, x2i, y2i, si, ci))
    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()
    src_path = "card.jpeg"
    onnx_path = "yolov8s_playing_cards.onnx"
    img0 = cv2.imread(src_path)
    if img0 is None:
        raise FileNotFoundError(src_path)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]) 
    inp = session.get_inputs()[0]
    shape = inp.shape
    th = shape[2] if isinstance(shape[2], int) else 640
    tw = shape[3] if isinstance(shape[3], int) else 640
    print(f"input shape: {shape}")
    x, ratio, pad_w, pad_h, img_lb = preprocess_image(img0, th, tw)
    out = session.run(None, {inp.name: x})
    
    names = load_names(session)
    if not names or len(names) != 52:
        raise Exception("Canmot find the channels names")
    detections = postprocess_outputs(out, names, tw, th, ratio, pad_w, pad_h, args.conf, args.iou, img0)

    vis = img0.copy()
    for x1, y1, x2, y2, s, c in detections:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        label = (names[c] if names and c < len(names) else f"class{c}") + f" {s:.2f}"
        tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(vis, (p1[0], p1[1] - tsize[1] - 6), (p1[0] + tsize[0] + 6, p1[1]), (0, 255, 0), -1)
        cv2.putText(vis, label, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    for x1, y1, x2, y2, s, c in detections:
        name = names[c] if names and c < len(names) else f"class{c}"
        print({"name": name, "class": c, "confidence": round(s, 4), "box": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)}})

    cv2.imwrite("card_output.jpeg", vis)

    vis_exp = img0.copy()
    found = []
    for x1, y1, x2, y2, s, c in detections:
        nm = names[c] if names and c < len(names) else f"class{c}"
        if nm in EXPECTED_SPADES:
            found.append(nm)
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(vis_exp, p1, p2, (255, 0, 0), 2)
            label = nm + f" {s:.2f}"
            tsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_exp, (p1[0], p1[1] - tsize[1] - 6), (p1[0] + tsize[0] + 6, p1[1]), (255, 0, 0), -1)
            cv2.putText(vis_exp, label, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    missing = [nm for nm in EXPECTED_SPADES if nm not in set(found)]
    print({"expected": EXPECTED_SPADES, "found": sorted(set(found)), "missing": missing})
    cv2.imwrite("card_output_expected.jpeg", vis_exp)


if __name__ == "__main__":
    main()
