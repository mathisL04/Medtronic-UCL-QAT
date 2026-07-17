from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Configuration
MODEL_PATH = "/workspace/runs_sanoscience/yolo26n_sanoscience_tool_sample/weights/best.pt"

VAL_IMAGES_DIR = Path("/workspace/datasets/sanoscience_yolo_tool_sample/images/val")
VAL_LABELS_DIR = Path("/workspace/datasets/sanoscience_yolo_tool_sample/labels/val")

OUTPUT_CSV = Path("/workspace/runs_sanoscience/yolo26n_sanoscience_tool_sample/evaluation_val_custom.csv")

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
IMG_SIZE = 640

DEVICE = 0  # use GPU


def yolo_txt_to_xyxy(label_path: Path, img_w: int, img_h: int):
    """
    Read YOLO ground-truth label file and convert normalized xywh to pixel xyxy.
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id, x_c, y_c, w, h = map(float, parts)

            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h

            x1 = x_c - w / 2
            y1 = y_c - h / 2
            x2 = x_c + w / 2
            y2 = y_c + h / 2

            boxes.append([x1, y1, x2, y2])

    return boxes


def box_iou(box_a, box_b):
    """
    Compute IoU between two boxes in xyxy format.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def match_predictions_to_ground_truth(pred_boxes, gt_boxes, iou_threshold):
    """
    Greedy matching between predictions and ground-truth boxes.
    """
    matched_gt = set()
    matched_pred = set()

    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue

            iou = box_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_pred.add(pred_idx)
            matched_gt.add(best_gt_idx)

    true_positives = len(matched_gt)
    false_positives = len(pred_boxes) - len(matched_pred)
    false_negatives = len(gt_boxes) - len(matched_gt)

    return true_positives, false_positives, false_negatives


def main():
    print("=" * 80)
    print("Custom Sanoscience YOLO evaluation")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"Validation images: {VAL_IMAGES_DIR}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")

    model = YOLO(MODEL_PATH)

    image_paths = sorted(VAL_IMAGES_DIR.glob("*.jpg"))

    rows = []

    total_images = 0
    image_level_correct = 0

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    missed_images = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: could not read image {image_path}")
            continue

        img_h, img_w = image.shape[:2]

        label_path = VAL_LABELS_DIR / f"{image_path.stem}.txt"
        gt_boxes = yolo_txt_to_xyxy(label_path, img_w, img_h)

        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )[0]

        pred_boxes = []

        if results.boxes is not None:
            for box in results.boxes.xyxy.cpu().numpy():
                pred_boxes.append(box.tolist())

        tp, fp, fn = match_predictions_to_ground_truth(
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            iou_threshold=IOU_THRESHOLD,
        )

        # Simple image-level score:
        # 1 if at least one labelled tool was correctly detected, else 0
        image_correct = 1 if tp > 0 else 0

        if image_correct == 0:
            missed_images.append(image_path.name)

        total_images += 1
        image_level_correct += image_correct

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        rows.append({
            "image": image_path.name,
            "gt_boxes": len(gt_boxes),
            "pred_boxes": len(pred_boxes),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "image_correct": image_correct,
        })

    image_accuracy = image_level_correct / total_images if total_images else 0
    object_recall = total_tp / total_gt if total_gt else 0
    object_precision = total_tp / total_pred if total_pred else 0

    if object_precision + object_recall > 0:
        f1 = 2 * object_precision * object_recall / (object_precision + object_recall)
    else:
        f1 = 0

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 80)
    print("Evaluation results")
    print("=" * 80)

    print(f"Images evaluated: {total_images}")
    print(f"Images with at least one correct detection: {image_level_correct}")
    print(f"Image-level accuracy: {image_accuracy:.3f} ({image_accuracy * 100:.1f}%)")

    print("\nObject-level metrics")
    print(f"Ground-truth boxes: {total_gt}")
    print(f"Predicted boxes: {total_pred}")
    print(f"True positives: {total_tp}")
    print(f"False positives: {total_fp}")
    print(f"False negatives: {total_fn}")
    print(f"Precision: {object_precision:.3f} ({object_precision * 100:.1f}%)")
    print(f"Recall: {object_recall:.3f} ({object_recall * 100:.1f}%)")
    print(f"F1 score: {f1:.3f}")

    print("\nMissed images:")
    if missed_images:
        for name in missed_images:
            print(f"  {name}")
    else:
        print("  None")

    print(f"\nDetailed CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
