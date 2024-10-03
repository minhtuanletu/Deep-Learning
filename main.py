import torch
from torchvision.transforms import v2
from common.trainer import Trainer

train_path = 'dataset/Flower_Recognition/train'
val_path = 'dataset/Flower_Recognition/val'
test_path = 'dataset/Flower_Recognition/test'
batch_size = 64
optimizer = 'adam'
loss_func = 'cross_entropy'
epochs = 50
model = 'vgg16'
in_channels = 3
input_size = (224, 224)
save_path = ''
transform = v2.Compose([
    v2.Resize(input_size, antialias=True),
    v2.PILToTensor(),
    v2.ToDtype(torch.float),
])
trainer = Trainer(train_path=train_path, val_path=val_path, test_path=test_path, transform=transform, 
                 batch_size=batch_size, optimizer=optimizer, loss_func=loss_func, epochs=epochs, model=model,
                 in_channels=in_channels, input_size=input_size)
trainer.train()
trainer.save_model('weight_models/flower.pt')