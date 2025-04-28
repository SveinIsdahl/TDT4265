import cv2
import os

# --- CONFIG ---
#image_folder = "/datasets/tdt4265/ad/open/Poles/rgb/images/train"       # Folder with .jpg or .png files
image_folder = "/datasets/tdt4265/ad/open/Poles/lidar/combined_color/train"       # Folder with .jpg or .png files

#label_folder = "/datasets/tdt4265/ad/open/Poles/rgb/labels/train"       # Folder with YOLO .txt files
label_folder = "/datasets/tdt4265/ad/open/Poles/lidar/labels/train"       # Folder with YOLO .txt files

image_exts = [".jpg", ".jpeg", ".png"]
# --------------

def draw_yolo_labels(img, label_path, class_names):
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, x, y, bw, bh = map(float, parts)
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            class_name = class_names.get(int(cls_id), str(int(cls_id)))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def main():
    class_names = {0:"pole"}

    image_files = [f for f in os.listdir(image_folder)
                   if os.path.splitext(f)[1].lower() in image_exts]

    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        label_path = os.path.join(label_folder, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            continue

        img = draw_yolo_labels(img, label_path, class_names)

        cv2.imshow("YOLO Viewer", img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
