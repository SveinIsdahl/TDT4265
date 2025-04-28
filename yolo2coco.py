import os
import json
from PIL import Image
from pathlib import Path

def yolo_to_coco(yolo_dir, output_json_path, class_names):
    image_dir = Path(yolo_dir) / "images"
    label_dir = Path(yolo_dir) / "labels"
    image_paths = list(image_dir.glob("*.PNG")) + list(image_dir.glob("*.png"))

    print("image count:", len(image_paths))

    images = []
    annotations = []
    ann_id = 1

    for img_id, img_path in enumerate(image_paths, start=1):
        # Image info
        with Image.open(img_path) as im:
            width, height = im.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height
        })

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_center, y_center, w, h = map(float, parts)

                x = (x_center - w / 2) * width
                y = (y_center - h / 2) * height
                w *= width
                h *= height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, "w") as f:
        json.dump(coco_dict, f, indent=4)
    print(f"Saved COCO annotations to {output_json_path}")


if __name__ == "__main__":
    yolo_dir = "./lidar_dataset_links/train" 
    output_json_path = "lidar_train_coco.json" 
    class_names = ["pole"] 

    yolo_to_coco(yolo_dir, output_json_path, class_names)