import json
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- CONFIGURATION ---
PREDICTIONS_FILE = "lidar_valid_bbox_results.json"
INSTANCES_FILE = "datasets/snowpoles/annotations/instances_lidar_valid.json"
IMAGES_FOLDER = "datasets/snowpoles/lidar_valid"
SCORE_THRESHOLD = 0.1  # Optional: filter out low-confidence predictions
IOU_THRESHOLD = 0.5

# --- LOAD DATA ---
with open(INSTANCES_FILE, "r") as f:
    instances = json.load(f)
with open(PREDICTIONS_FILE, "r") as f:
    predictions = json.load(f)

# Build mapping from image_id to list of ground truth annotations (bboxes & category_ids)
gt_by_image = {}
for ann in instances["annotations"]:
    img_id = ann["image_id"]
    if img_id not in gt_by_image:
        gt_by_image[img_id] = []
    gt_by_image[img_id].append(ann)

# Build mapping from image_id to predictions (filtered by score threshold)
pred_by_image = {}
for pred in predictions:
    if pred["score"] < SCORE_THRESHOLD:
        continue
    img_id = pred["image_id"]
    if img_id not in pred_by_image:
        pred_by_image[img_id] = []
    pred_by_image[img_id].append(pred)


# --- UTILITY FUNCTIONS ---
# Convert COCO bbox [x, y, w, h] into [x1, y1, x2, y2]
def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return torch.tensor([x, y, x + w, y + h], dtype=torch.float32)


# Compute Intersection-over-Union (IoU) between two boxes in [x1, y1, x2, y2] format.
def compute_iou(box1, box2):
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else torch.tensor(0.0)


# --- MAIN EVALUATION LOOP ---
# Total counts for the entire dataset
total_tp = 0
total_fp = 0
total_fn = 0

# For each image in the annotations, compute true positives, false positives, and false negatives.
all_image_ids = [img["id"] for img in instances["images"]]
for image_id in all_image_ids:
    gt_annots = gt_by_image.get(image_id, [])
    pred_annots = pred_by_image.get(image_id, [])

    # Track matched ground truths to prevent double counting
    matched_gt_indices = set()
    tp = 0
    fp = 0

    # Loop over predictions for this image
    for pred in pred_annots:
        pred_box = xywh_to_xyxy(pred["bbox"])
        pred_label = pred["category_id"]
        best_iou = 0.0
        best_gt_idx = -1

        # Try to match this prediction with a ground truth box of the same class.
        for idx, gt in enumerate(gt_annots):
            if idx in matched_gt_indices:
                continue
            # Require that the class matches before testing IoU.
            if pred_label != gt["category_id"]:
                continue

            gt_box = xywh_to_xyxy(gt["bbox"])
            iou = compute_iou(pred_box, gt_box).item()
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        # If the best match is above the IOU threshold, count as a true positive.
        if best_iou >= IOU_THRESHOLD and best_gt_idx != -1:
            tp += 1
            matched_gt_indices.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_annots) - len(matched_gt_indices)

    total_tp += tp
    total_fp += fp
    total_fn += fn

# --- COMPUTE CLASSIC PRECISION & RECALL ---
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

print("Overall Dataset Metrics:")
print(f"Total True Positives:  {total_tp}")
print(f"Total False Positives: {total_fp}")
print(f"Total False Negatives: {total_fn}")
print(f"Classic Precision: {precision:.4f}")
print(f"Classic Recall:    {recall:.4f}")

# --- OPTIONAL: Visualize one image (e.g., IMAGE_ID = 6) ---
# If you want to see predictions on one image, load its file name using the images list.
id_to_filename = {img["id"]: img["file_name"] for img in instances["images"]}
if 6 in id_to_filename:
    image_path = os.path.join(IMAGES_FOLDER, id_to_filename[6])
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw predictions on the image
    for pred in pred_by_image.get(6, []):
        x, y, w, h = pred["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f'{pred["score"]:.2f}', color='white', backgroundcolor='red', fontsize=8)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
