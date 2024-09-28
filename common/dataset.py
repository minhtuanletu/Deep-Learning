import os
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose
from torch.utils.data import Dataset

class Custom_Dataset(Dataset):
    def __init__(self, root_dataset_folder: str, transform: Compose):
        super(Custom_Dataset, self).__init__()
        self.root_dataset_folder = root_dataset_folder
        self.transform = transform
        self.images_path = []
        self.labels = []
        self.name_labels = []
        self.load_dataset()

    def load_dataset(self):
        classes_name = os.listdir(self.root_dataset_folder)
        for class_name in classes_name:
            self.name_labels.append(class_name)
            images_folder = os.path.join(self.root_dataset_folder, class_name)
            for dir in os.listdir(images_folder):
                image_path = os.path.join(images_folder, dir)
                self.images_path.append(image_path)
                self.labels.append(classes_name.index(class_name))

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert('RGB')
        x = self.transform(image)
        label = self.labels[idx]
        y = [0] * len(self.name_labels)
        y[label] = 1
        y = torch.FloatTensor(y)
        return x, y
