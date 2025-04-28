from ultralytics import RTDETR
import torch
model = RTDETR("rtdetr-l.pt")

# Train the model
model.train(
            project="rtdetr_files",
                name="rtdetr_training",
                    data="rgb_data.yaml", 
                        epochs=150,
                            batch=8, 
                                imgsz=1024,
                                degrees=10,
                                translate=0.2,
                                mixup=0.1
                                )

results = model.val()

torch.save(model.state_dict(), "rtdetr.pth")

lidar.train(
            project="rtdetr_files_lidar",
                name="rtdetr_training_lidar",
                    data="lidar_data.yaml", 
                        epochs=150,
                            batch=10, 
                                imgsz=1024
                                )

results = lidar.val()

from pathlib import Path

dataset_path = "/datasets/tdt4265/ad/open/Poles"
rgb_valid_path = dataset_path + "/rgb/images/valid/"
lidar_valid_path = dataset_path + "/lidar/combined_color/valid/"

rgb_valid_images = [str(p) for p in Path(rgb_valid_path).glob("*.PNG")]
lidar_valid_images = [str(p) for p in Path(lidar_valid_path).glob("*.png")]

for i in range(1):
    results = model(rgb_valid_images[i])
    results[0].show()  # Display predictions
    
lidar.predict(
    source="lidar_dataset_links/test/images",
    project="testt_lidar",
    name="testt_lidar",
    save_txt=True,
    save_conf=True # <--- This adds the probability of each predicted box
    )
    
all.train(
            project="rtdetr_files_all",
                name="rtdetr_training_all",
                    data="all_data.yaml", 
                        epochs=150,
                            batch=10, 
                                imgsz=1024
                                )

results = all.val()
