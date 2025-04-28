import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class FlatImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Define dataset and loader
transform = transforms.ToTensor()
dataset = FlatImageFolder('./datasets/snowpoles/lidar_train', transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# Compute mean and std
mean = 0.
std = 0.
n_images = 0

for images in loader:
    batch_size = images.size(0)
    images = images.view(batch_size, 3, -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_images += batch_size

mean /= n_images
std /= n_images

print(f"Mean: {mean}")
print(f"Std: {std}")
