import os
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

class CustomDataset(VisionDataset):
    def __init__(self, csv_file, root_dir, label_map, transforms=None):

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_map = label_map
        self.transforms = transforms

        self.image_names = self.annotations['image_name'].unique()
        self.image_annotations = {img: self.annotations[self.annotations['image_name'] == img] for img in self.image_names}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        annotations = self.image_annotations[img_name]
        boxes = []
        labels = []

        for _, row in annotations.iterrows():
            x_min = row['bbox_x']
            y_min = row['bbox_y']
            x_max = x_min + row['bbox_width']
            y_max = y_min + row['bbox_height']
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.label_map[row['label_name']])
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            image = self.transforms(image)

        return image, target

# Create label mapping
annotations = pd.read_csv('CSV_FILE_PATH')
label_names = annotations['label_name'].unique()


label_map = {name: idx for idx, name in enumerate(label_names, start=1)}

# Add a background class with index 0
label_map['background'] = 0

num_classes = len(label_map)

# Example usage
transform = T.Compose([T.ToTensor()])
dataset = CustomDataset(csv_file='CSV_FILE_PATH', root_dir='ROOT_DIR', label_map=label_map, transforms=transform)
