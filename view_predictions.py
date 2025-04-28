from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def draw_boxes(image, boxes, color, label_prefix=""):
    for box in boxes:
        x1, y1, x2, y2, cls = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, f"{label_prefix}{int(cls)}", (int(x1), int(y1)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def yolo_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [x1, y1, x2, y2]

def load_labels(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            cls, cx, cy, w, h = map(float, line.strip().split())
            xyxy = yolo_to_xyxy([cx, cy, w, h], img_w, img_h)
            boxes.append(xyxy + [int(cls)])
    return boxes

def visualize_preds_and_labels(image_path, label_path, model_path="yolov8n.pt"):
    model = YOLO(model_path)
    results = model(image_path, conf=0.25)[0]  

    pred_boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls = int(box.cls[0].item())
        pred_boxes.append([x1, y1, x2, y2, cls])

    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    label_boxes = load_labels(label_path, img_w, img_h)

    draw_boxes(image, pred_boxes, (0, 255, 0), label_prefix="Pred:")
    draw_boxes(image, label_boxes, (255, 0, 0), label_prefix="Label:")

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Predicted vs Ground Truth Poles")
    plt.show()

image_path = "/datasets/tdt4265/ad/open/Poles/rgb/images/valid"
label_path = "/datasets/tdt4265/ad/open/Poles/rgb/labels/valid"

images = [str(p) for p in Path(image_path).glob("*.PNG")]
labels = [str(p) for p in Path(label_path).glob("*.txt")]

images.sort()
labels.sort()

for i in range(len(images)):
    visualize_preds_and_labels(images[i], labels[i], model_path="test_model.pt")
    input()