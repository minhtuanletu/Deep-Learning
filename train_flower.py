import os
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from common.metrics import Compute_Metrics
from common.dataset import Custom_Dataset
from common.early_stopping import EarlyStopping

class Trainer:
    def __init__(self, train_path: str, val_path: str, test_path: str, transform: Compose, 
                 batch_size: int, optimizer: str, loss_func: str, epochs: int, model: str,
                 in_channels: int, input_size: tuple):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = Compute_Metrics()
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
        if model == 'custom':
            self.model = Custom_Model(input_channels=self.in_channels, output_classes=self.num_classes, image_size=self.input_size).to(self.device)
        # Create Dataloader
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True)
        # Training parameters
        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        elif optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-3)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-3)
        if loss_func == 'bce':
            self.loss_func = torch.nn.BCELoss()
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.early_stop = EarlyStopping(patience=5, delta=1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        
    def train_func(self):
        self.model.train()
        loss_value = 0
        accuracy_value, precision_value, recall_value, f1_value = 0, 0, 0, 0
        for _, (x, y) in enumerate(self.train_dataloader):
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.loss_func(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value += loss.item()
            accuracy, precision, recall, f1 = self.metrics.compute(y, y_pred)
            accuracy_value += accuracy
            precision_value += precision
            recall_value += recall
            f1_value += f1
        loss_value = loss_value / len(self.train_dataloader)
        accuracy_value = accuracy_value / len(self.train_dataloader)
        precision_value = precision_value / len(self.train_dataloader)
        recall_value = recall_value / len(self.train_dataloader)
        f1_value = f1_value / len(self.train_dataloader)
        return loss_value, accuracy_value, precision_value, recall_value, f1_value
    
    def val_func(self):
        self.model.eval()
        with torch.no_grad():
            loss_value = 0
            accuracy_value, precision_value, recall_value, f1_value = 0, 0, 0, 0
            for _, (x, y) in enumerate(self.val_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)
                loss_value += loss.item()
                accuracy, precision, recall, f1 = self.metrics.compute(y, y_pred)
                accuracy_value += accuracy
                precision_value += precision
                recall_value += recall
                f1_value += f1
            loss_value = loss_value / len(self.val_dataloader)
            accuracy_value = accuracy_value / len(self.val_dataloader)
            precision_value = precision_value / len(self.val_dataloader)
            recall_value = recall_value / len(self.val_dataloader)
            f1_value = f1_value / len(self.val_dataloader)
            return loss_value, accuracy_value, precision_value, recall_value, f1_value
    
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
            train_loss, train_accuracy_value, train_precision_value, train_recall_value, train_f1_value = self.train_func()
            val_loss, val_accuracy_value, val_precision_value, val_recall_value, val_f1_value = self.val_func()
            print(f"Epoch: {epoch:2d} - Train loss: {train_loss:.2f} - Val loss: {val_loss:.2f} - Train accuracy {train_accuracy_value:.2f} - Val accuracy {val_accuracy_value:.2f} - Train precision {train_precision_value:.2f} - Val precision {val_precision_value:.2f} - Train recall: {train_recall_value:.2f} - Val recall: {val_recall_value:.2f} - Train f1: {train_f1_value:.2f} - Val f1: {val_f1_value:.2f}")
            self.lr_scheduler.step(val_loss)
            if self.early_stop(val_loss):
                print("Early Stopping!!!")
                break
                
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

class CNN_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm='batch_norm', drop_out=0.5):
        super(CNN_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm = norm
        self.drop_out = drop_out
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        )        
        if norm == 'batch_norm':
            self.cnn.append(torch.nn.BatchNorm2d(self.out_channels))
        elif norm == 'layer_norm':
            self.cnn.append(torch.nn.LayerNorm(self.out_channels))
            
        if self.drop_out != 0:
            self.cnn.append(torch.nn.Dropout(self.drop_out))
            
    def forward(self, x):
        return self.cnn(x)

class Residual_Block(torch.nn.Module):
    def __init__(self, input_channels, output_channels, drop_out, activate_function='relu'):
        super(Residual_Block, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.drop_out = drop_out
        if activate_function == 'relu':
            self.activate_function = torch.nn.ReLU()
        elif activate_function == 'leaky_relu':
            self.activate_function = torch.nn.LeakyReLU()
        elif activate_function == 'tanh':
            self.activate_function = torch.nn.Tanh()
        self.residual = torch.nn.Sequential(
            CNN_Block(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1, norm='none', drop_out=0.5),
            self.activate_function,
            CNN_Block(in_channels=self.output_channels, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1, norm='none', drop_out=0.5)
        )
        if self.input_channels != self.output_channels:
            self.shortcut_connection = CNN_Block(in_channels=self.input_channels, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, norm='none', drop_out=0.5)
        else:
            self.shortcut_connection = None

    def forward(self, x):
        if self.shortcut_connection is not None:
            x_shortcut = self.shortcut_connection(x)
        else:
            x_shortcut = x
        output = self.residual(x)
        return self.activate_function(output + x_shortcut)
    
class MLP_Block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, activate_function='relu', drop_out=0.5):
        super(MLP_Block, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.activate_function = activate_function
        self.drop_out = drop_out
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features=self.input_channel, out_features=self.output_channel))
        if self.activate_function == 'relu':
            self.mlp.append(torch.nn.ReLU())
        elif self.activate_function == 'leaky_relu':
            self.mlp.append(torch.nn.LeakyReLU())
        elif self.activate_function == 'tanh':
            self.mlp.append(torch.nn.Tanh())
        elif self.activate_function == 'softmax':
            self.mlp.append(torch.nn.Softmax(dim=-1))
        if self.drop_out != 0:
            self.mlp.append(torch.nn.Dropout(self.drop_out))
    
    def forward(self, x):
        return self.mlp(x)      
    
class Custom_Model(torch.nn.Module):
    def __init__(self, input_channels, output_classes, image_size):
        super(Custom_Model, self).__init__()
        self.input_channels = input_channels
        self.output_classes = output_classes
        self.image_size = image_size
        self.block_1 = torch.nn.Sequential(
            CNN_Block(in_channels=self.input_channels, out_channels=16, kernel_size=7, stride=2, padding=3, norm='batch_norm', drop_out=0.5),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block_2 = torch.nn.Sequential(
            Residual_Block(input_channels=16, output_channels=16, drop_out=0.5, activate_function='relu'),
        )
        self.block_3 = torch.nn.Sequential(
            Residual_Block(input_channels=16, output_channels=32, drop_out=0.5, activate_function='relu'),
        )
        self.block_4 = torch.nn.Sequential(
            Residual_Block(input_channels=32, output_channels=64, drop_out=0.5, activate_function='relu'),
        )
        self.block_5 = torch.nn.Sequential(
            Residual_Block(input_channels=64, output_channels=128, drop_out=0.5, activate_function='relu'),
        )
        self.feature_extraction = torch.nn.Sequential(
            self.block_1, self.block_2, self.block_3, self.block_4, self.block_5,
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Flatten(start_dim=1)
        )
        self.output_size = self.compute_output_dim()
        self.mlp = torch.nn.Sequential(
            MLP_Block(input_channel=self.output_size, output_channel=1000, activate_function='relu', drop_out=0.5),
            MLP_Block(input_channel=1000, output_channel=128, activate_function='relu', drop_out=0.5),
            MLP_Block(input_channel=128, output_channel=self.output_classes, activate_function='none', drop_out=0)
        )
        
    def compute_output_dim(self):
        x = torch.randn((1, self.input_channels, self.image_size[0], self.image_size[1]))
        output = self.feature_extraction(x)
        output_size = output.shape
        return output_size[-1]
    
    def forward(self, x):
        feature = self.feature_extraction(x)
        output = self.mlp(feature)
        return output
    
train_path = 'dataset/Flower_Recognition_augment/train'
val_path = 'dataset/Flower_Recognition_augment/val'
test_path = 'dataset/Flower_Recognition_augment/test'
batch_size = 128
optimizer = 'adam'
loss_func = 'cross_entropy'
epochs = 20
model = 'custom'
in_channels = 3
input_size = (160, 160)
save_path = 'weight_models/flower.pt'
transform = v2.Compose([
    v2.Resize(input_size, antialias=True),
    v2.PILToTensor(),
    v2.ToDtype(torch.float)
])
trainer = Trainer(train_path=train_path, val_path=val_path, test_path=test_path, transform=transform, 
                 batch_size=batch_size, optimizer=optimizer, loss_func=loss_func, epochs=epochs, model=model,
                 in_channels=in_channels, input_size=input_size)

trainer.train()
trainer.save_model(save_path)
torch.cuda.empty_cache()