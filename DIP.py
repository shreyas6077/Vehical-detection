import os
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CACHE = {}


def _get_models():
    """Loads and caches the YOLO models once so repeated API calls are fast."""
    if "aerial" not in _MODEL_CACHE or "street" not in _MODEL_CACHE:
        print("Loading Aerial (OBB) and Street-View models...")
        obb_path = os.path.join(BASE_DIR, "yolov8m-obb.pt")
        std_path = os.path.join(BASE_DIR, "yolov8m.pt")
        _MODEL_CACHE["aerial"] = YOLO(obb_path if os.path.exists(obb_path) else "yolov8m-obb.pt")
        _MODEL_CACHE["street"] = YOLO(std_path if os.path.exists(std_path) else "yolov8m.pt")
    return _MODEL_CACHE["aerial"], _MODEL_CACHE["street"]


def _can_show_gui(show_window):
    """Return True only when GUI output is both requested and available."""
    if not show_window:
        return False
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        return False
    try:
        cv2.imshow("__gui_check__", np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.destroyAllWindows()
        return True
    except Exception:
        return False


def enhance_contrast(image):
    """Enhances local contrast using CLAHE to reveal dark cars on dark asphalt/shadows."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def compute_iou_and_containment(boxA, boxB):
    """Returns standard IoU and Intersection-over-Minimum-Area (to prevent nested double counts)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0, 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(areaA + areaB - inter_area)
    io_min = inter_area / float(min(areaA, areaB))
    return iou, io_min


def get_tiles(image, overlap=0.2):
    """Generates full image + 2x2 overlapping tiles to catch small top-angle vehicles."""
    h, w = image.shape[:2]
    tiles = [(image, 0, 0, "Full Image")]  # Always include the full image

    # Only slice if the image is reasonably large
    if h >= 600 and w >= 600:
        mid_x, mid_y = w // 2, h // 2
        pad_x, pad_y = int(w * overlap / 2), int(h * overlap / 2)

        coords = [
            (0, 0, mid_x + pad_x, mid_y + pad_y, "Tile 1 (Top-Left)"),
            (mid_x - pad_x, 0, w, mid_y + pad_y, "Tile 2 (Top-Right)"),
            (0, mid_y - pad_y, mid_x + pad_x, h, "Tile 3 (Bottom-Left)"),
            (mid_x - pad_x, mid_y - pad_y, w, h, "Tile 4 (Bottom-Right)")
        ]
        for x1, y1, x2, y2, label in coords:
            tiles.append((image[y1:y2, x1:x2], x1, y1, label))

    return tiles


def detect_and_count_vehicles_accurate(
    image_path,
    output_path="accurate_output.jpg",
    conf_threshold=0.28,
    iou_threshold=0.40,
    scene_mode="both",
    use_clahe=True,
    use_tiling=True,
    show_window=None
):
    if show_window is None:
        show_window = os.name == "nt" or bool(os.environ.get("DISPLAY"))

    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise ValueError(f"Error: Could not load image from '{image_path}'")

    h, w = orig_image.shape[:2]
    aerial_model, street_model = _get_models()

    proc_image = enhance_contrast(orig_image) if use_clahe else orig_image.copy()

    aerial_classes = {
        9: "Large Vehicle",
        10: "Small Vehicle"
    }

    street_classes = {
        2: "Small Vehicle",
        3: "Motorcycle",
        5: "Large Vehicle",
        7: "Large Vehicle"
    }

    raw_detections = []
    tiles = get_tiles(proc_image) if use_tiling else [(proc_image, 0, 0, "Full Image")]

    mode = (scene_mode or "both").lower()
    run_aerial = mode in ("aerial", "obb", "both", "auto", "hybrid")
    run_street = mode in ("street", "standard", "both", "auto", "hybrid")
    # Fallback: if an unrecognized mode string is sent, run both models
    if not run_aerial and not run_street:
        run_aerial = True
        run_street = True

    for tile_img, offset_x, offset_y, tile_name in tiles:
        if run_aerial:
            obb_res = aerial_model(tile_img, imgsz=1024, conf=conf_threshold, verbose=False)[0]
            if obb_res.obb is not None and len(obb_res.obb) > 0:
                for i in range(len(obb_res.obb)):
                    cls_id = int(obb_res.obb.cls[i])
                    if cls_id in aerial_classes:
                        conf = float(obb_res.obb.conf[i])

                        x1, y1, x2, y2 = obb_res.obb.xyxy[i].cpu().numpy()
                        x1, x2 = x1 + offset_x, x2 + offset_x
                        y1, y2 = y1 + offset_y, y2 + offset_y

                        poly = obb_res.obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                        poly[:, 0] += offset_x
                        poly[:, 1] += offset_y

                        raw_detections.append({
                            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "poly": poly,
                            "conf": conf,
                            "label": aerial_classes[cls_id],
                            "source": "YOLOv8m-OBB (Aerial)",
                            "model": "yolov8m-obb",
                            "tile": tile_name
                        })

        if run_street:
            std_res = street_model(
                tile_img, imgsz=1280, conf=conf_threshold, augment=True, verbose=False
            )[0]
            if std_res.boxes is not None and len(std_res.boxes) > 0:
                for box in std_res.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in street_classes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, x2 = x1 + offset_x, x2 + offset_x
                        y1, y2 = y1 + offset_y, y2 + offset_y

                        poly = np.array([
                            [int(x1), int(y1)],
                            [int(x2), int(y1)],
                            [int(x2), int(y2)],
                            [int(x1), int(y2)]
                        ], dtype=np.int32)

                        raw_detections.append({
                            "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "poly": poly,
                            "conf": conf,
                            "label": street_classes[cls_id],
                            "source": "YOLOv8m (Street)",
                            "model": "yolov8m",
                            "tile": tile_name
                        })

    final_detections = []
    if raw_detections:
        boxes_xywh = [
            [d["xyxy"][0], d["xyxy"][1], d["xyxy"][2] - d["xyxy"][0], d["xyxy"][3] - d["xyxy"][1]]
            for d in raw_detections
        ]
        confs = [d["conf"] for d in raw_detections]

        nms_indices = cv2.dnn.NMSBoxes(boxes_xywh, confs, conf_threshold, iou_threshold)
        if len(nms_indices) > 0:
            nms_indices = np.array(nms_indices).flatten()

            candidates = [raw_detections[i] for i in nms_indices]
            candidates.sort(key=lambda x: x["conf"], reverse=True)

            for cand in candidates:
                is_duplicate = False
                for kept in final_detections:
                    iou, io_min = compute_iou_and_containment(cand["xyxy"], kept["xyxy"])
                    if iou > iou_threshold or io_min > 0.70:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    final_detections.append(cand)

    counts = {"Small Vehicle": 0, "Large Vehicle": 0, "Motorcycle": 0}
    colors = {
        "Small Vehicle": (0, 255, 0),    # Green
        "Large Vehicle": (0, 165, 255),  # Orange
        "Motorcycle": (255, 255, 0)      # Cyan
    }

    output_img = orig_image.copy()
    serializable_detections = []

    for idx, det in enumerate(final_detections, start=1):
        label = det["label"]
        conf = float(det["conf"])
        poly = det["poly"]
        counts[label] += 1
        color = colors.get(label, (0, 255, 0))

        cv2.polylines(output_img, [poly], isClosed=True, color=color, thickness=2)

        center_x = int(np.mean(poly[:, 0]))
        center_y = int(np.mean(poly[:, 1]))
        cv2.circle(output_img, (center_x, center_y), 3, (0, 0, 255), -1)

        top_x = int(poly[:, 0].min())
        top_y = int(poly[:, 1].min())
        cv2.putText(output_img, f"#{idx} {label} {conf:.2f}",(top_x, max(top_y - 6, 15)),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        int_box = [int(round(v)) for v in det["xyxy"]]
        box_w = max(0, int_box[2] - int_box[0])
        box_h = max(0, int_box[3] - int_box[1])

        # Complete detection log entry supporting all frontend property names
        serializable_detections.append({
            "id": idx,
            "label": label,
            "type": label,
            "class": label,
            "conf": round(conf, 4),
            "confidence": round(conf, 4),
            "confidence_pct": f"{conf * 100:.1f}%",
            "xyxy": int_box,
            "bbox": int_box,
            "box": int_box,
            "x1": int_box[0],
            "y1": int_box[1],
            "x2": int_box[2],
            "y2": int_box[3],
            "width": box_w,
            "height": box_h,
            "center": [center_x, center_y],
            "center_x": center_x,
            "center_y": center_y,
            "poly": poly.tolist(),
            "source": det.get("source", "YOLOv8m"),
            "model": det.get("model", "yolov8m"),
            "tile": det.get("tile", "Full Image")
        })

    total_vehicles = len(final_detections)
    #cv2.rectangle(output_img, (10, 10), (300, 145), (0, 0, 0), -1)
    #cv2.putText( output_img, f"Total Vehicles: {total_vehicles}", (20, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    y_pos = 72
    for v_type, count in counts.items():
        #cv2.putText(output_img, f"{v_type}s: {count}", (20, y_pos),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 24

    print(f"Total Vehicles Detected: {total_vehicles}")
    print("Breakdown:", counts)

    cv2.imwrite(output_path, output_img)

    if _can_show_gui(show_window):
        cv2.imshow("Top-Angle & Street Vehicle Counter", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "total_vehicles": total_vehicles,
        "counts": counts,
        "raw_candidates": len(raw_detections),
        "tiles_processed": len(tiles),
        "image_size": f"{w}x{h}",
        "image_dimensions": {"width": int(w), "height": int(h)},
        "output_path": output_path,
        "detections": serializable_detections,
    }


if __name__ == "__main__":
    detect_and_count_vehicles_accurate(r"c:\Users\Acer\Documents\ML\traffic.jpg")