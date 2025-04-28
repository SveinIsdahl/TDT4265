import torchvision
# from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, SSDLite320_MobileNet_V3_Large
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16
# from torchvision.transforms.v2 import functional as F, Transform, Compose, ToImage, ToDtype # Removed import
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import glob
import numpy as np
import torchmetrics
import time
import torch


DATASET_PATH = 'lidar_dataset_links'
NUM_CLASSES = 1 + 1
BATCH_SIZE = 48
NUM_EPOCHS = 5000
LEARNING_RATE = 0.0005
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class PoleDataset(Dataset):
    def __init__(self, root_dir, train=True, transforms=None):
        self.root_dir = root_dir
        self.train = train

        self.img_dir = os.path.join(root_dir, 'train' if train else 'valid', 'images')
        self.label_dir = os.path.join(root_dir, 'train' if train else 'valid', 'labels')

        # Use glob to find all image files and assume corresponding label files exist
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
        print(f"Found {len(self.img_files)} images in {'train' if train else 'valid'} set.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.png', '.txt'))


        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:

                        class_id, cx_n, cy_n, w_n, h_n = map(float, line.strip().split())


                        cx = cx_n * img_w
                        cy = cy_n * img_h
                        w = w_n * img_w
                        h = h_n * img_h

                        xmin = cx - w / 2
                        ymin = cy - h / 2
                        xmax = cx + w / 2
                        ymax = cy + h / 2


                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(img_w, xmax)
                        ymax = min(img_h, ymax)


                        if xmax > xmin and ymax > ymin:
                           boxes.append([xmin, ymin, xmax, ymax])

                           labels.append(int(class_id) + 1)
                    except ValueError:
                         print(f"Warning: Skipping invalid line in {label_path}: '{line.strip()}'")



        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if boxes.shape[0] == 0:
             boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.shape[0] > 0 else torch.tensor(0.0)
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)


  

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    weights = None

    model = torchvision.models.detection.ssd300_vgg16(weights=weights, num_classes=num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    start_time = time.time()
    total_loss = 0

    for i, (images, targets) in enumerate(data_loader):

        images = [torch.tensor(np.array(img)).permute(2, 0, 1).float().to(device) / 255.0 for img in images]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        total_loss += loss_value


        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    avg_epoch_loss = total_loss / len(data_loader)
    total_epoch_time = time.time() - start_time
    print(f"--- Epoch {epoch+1} Finished ---")
    print(f"Average Training Loss: {avg_epoch_loss:.4f}")
    print(f"Total Epoch Time: {total_epoch_time:.2f}s")
    print("------------------------------")

import torch
from typing import Tuple, List

from torch import nn

class Backbone(nn.Module):
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, output_channels[0], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[0], 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[1], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[1], 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, output_channels[2], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[2], 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[3], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[3], 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(output_channels[4], 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, output_channels[5], kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
            ),
        ])
        
    def forward(self, x):

        out_features = []
        
        for idx, extractor in enumerate(self.extractor):
            x = extractor(x)
            out_features.append((x))
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
        return tuple(out_features)



@torch.no_grad() 
def evaluate(model, data_loader, device, metrics_calculator):
    model.eval() 
    metrics_calculator.reset()
    start_time = time.time()


    for images, targets in data_loader:

        images = [torch.tensor(np.array(img)).permute(2, 0, 1).float().to(device) / 255.0 for img in images]
        targets_metric = [{k: v.to(device) for k, v in t.items()} for t in targets]


        outputs = model(images)


        metrics_calculator.update(outputs, targets_metric)


    metrics = metrics_calculator.compute()
    eval_time = time.time() - start_time
    print(f"Validation Finished | Time: {eval_time:.2f}s")
    return metrics


if not os.path.isdir(DATASET_PATH):
    print(f"Error: Dataset directory not found at {DATASET_PATH}")
    exit()

print(f"Using device: {DEVICE}")

dataset_train = PoleDataset(DATASET_PATH, train=True, transforms=None)
dataset_valid = PoleDataset(DATASET_PATH, train=False, transforms=None)


data_loader_train = DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    collate_fn=collate_fn
)
data_loader_valid = DataLoader(
    dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    collate_fn=collate_fn
)

# Model
model = get_model(num_classes=NUM_CLASSES)
model.to(DEVICE)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)


metrics = torchmetrics.detection.MeanAveragePrecision(box_format='xyxy', iou_type='bbox') # 'bbox' needed for mAP
metrics.to(DEVICE)

# Training Loop
print("Starting Training...")
for epoch in range(NUM_EPOCHS):
    train_one_epoch(model, optimizer, data_loader_train, DEVICE, epoch)

    eval_metrics = evaluate(model, data_loader_valid, DEVICE, metrics)

    map_50_95 = eval_metrics.get('map', torch.tensor(-1.0)).item() # Corresponds to mAP@0.50:0.95
    map_50 = eval_metrics.get('map_50', torch.tensor(-1.0)).item() # Corresponds to mAP@0.50

    print(f"--- Epoch {epoch+1} Validation Metrics ---")
    print(f"mAP@0.50:0.95 (COCO style): {map_50_95:.4f}")
    print(f"mAP@0.50 (Pascal VOC style): {map_50:.4f}")
    print(f"Mean Average Recall (MAR) large: {eval_metrics.get('mar_large', torch.tensor(-1.0)).item():.4f}")
    print("--------------------------------------")


print("Training finished!")
MODEL_SAVE_PATH = 'ssd_pole_detector.pth'


torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"Model saved to {MODEL_SAVE_PATH}")
