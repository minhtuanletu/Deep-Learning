from common.trainer import Trainer

train_path = ''
val_path = ''
test_path = ''
transform = ''
batch_size = 128
optimizer = 'adam'
loss_func = 'cross_entropy'
epochs = 1000
model = 'vgg16'
in_channels = 3
input_size = (224, 224)

trainer = Trainer(train_path=train_path, val_path=val_path, test_path=test_path, transform=transform, 
                 batch_size=batch_size, optimizer=optimizer, loss_func=loss_func, epochs=epochs, model=model,
                 in_channels=in_channels, input_size=input_size)