from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    project="yolo_files_27",
    name="yolo_training_27",
    data="rgb_data.yaml", 
    epochs=100,
    batch=16, 
    imgsz=1024,
)
results = model.val()

model.predict(
    source="rgb_dataset_links/test/images",
    project="testt_rgb111",
    name="testt_rgb111",
    save_txt=True,
    save_conf=True # <--- This adds the probability of each predicted box
    )