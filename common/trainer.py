import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from common.dataset import *
from model.InceptionNet.InceptionNetv1 import *
from model.InceptionNet.InceptionNetv2 import *
from model.Resnet.Resnet18 import *
from model.Resnet.Resnet34 import *
from model.Resnet.Resnet50 import *
from model.ViT.vit import *
from model.VGG.VGG16 import VGG16
from model.VGG.VGG19 import VGG19

class Trainer:
    def __init__(self, train_path: str, val_path: str, test_path: str, transform: Compose, 
                 batch_size: int, optimizer: str, loss_func: str, epochs: int, model: str,
                 in_channels: int, input_size: tuple):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load Dataset
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.transform = transform
        self.train_dataset = Custom_Dataset(self.train_path, self.transform)
        self.val_dataset = Custom_Dataset(self.val_path, self.transform)
        self.test_dataset = Custom_Dataset(self.val_path, self.transform)
        # Create Model
        self.in_channels = in_channels
        self.num_classes = len(os.listdir(self.train_path))
        self.input_size = input_size
        if model == 'vgg16':
            self.model = VGG16(input_channels=self.in_channels, output_classes=self.num_classes, image_size=self.input_size).to(self.device)
        elif model == 'vgg19':
            self.model = VGG19(input_channels=self.in_channels, output_classes=self.num_classes, image_size=self.input_size).to(self.device)
        elif model == 'resnet18':
            self.model = Resnet18(input_channels=self.in_channels, output_classes=self.num_classes, image_size=self.input_size).to(self.device)
        elif model == 'resnet34':
            self.model = Resnet34(input_channels=self.in_channels, output_classes=self.num_classes, image_size=self.input_size).to(self.device)
        elif model == 'resnet50':
            self.model = Resnet50(input_channels=self.in_channels, output_classes=self.num_classes, image_size=self.input_size).to(self.device)
        # elif model == 'vit':
        #     self.model = ViT(input_channels=self.in_channels, output_classes=self.num_classes, image_size=self.input_size).to(self.device)
        else:
            self.model = VGG16(input_channels=self.in_channels, output_classes=self.num_classes, image_size=self.input_size).to(self.device)
        # Create Dataloader
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        # Training parameters
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        elif optimizer == 'adamw':
            self.optimizer == torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-5)
        if loss_func == 'bce':
            self.loss_func = torch.nn.BCELoss()
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        
    def train_func(self):
        self.model.train()
        loss_value = 0
        for _, (x, y) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            loss.backward()
            self.optimizer.step()
            loss_value += loss.item()
        loss_value = loss_value / len(self.train_dataloader)
        return loss_value
    
    def val_func(self):
        self.model.eval()
        loss_value = 0
        for _, (x, y) in enumerate(self.val_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            loss_value += loss.item()
        loss_value = loss_value / len(self.val_dataloader)
        return loss_value
    
    def test_func(self):
        self.model.eval()
        loss_value = 0
        for _, (x, y) in enumerate(self.test_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            loss_value += loss.item()
        loss_value = loss_value / len(self.test_dataloader)
        return loss_value
        
    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_func()
            val_loss = self.val_func()
            print(f"Epoch: {epoch:2d} - Train loss: {train_loss:.2f} - Val loss: {val_loss:.2f}")
            
    def inference(self, image: Image):
        self.model.eval()
        with torch.no_grad():
            x = self.transform(image)
            x = torch.unsqueeze(x, 0)
            y_pred = self.model(x)
            idx = torch.argmax(y_pred, dim=-1)
            return idx
        
    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path, map_location=self.device, weights_only=True))