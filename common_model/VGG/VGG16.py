import torch

class CNN_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm='batch_norm', drop_out=0):
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

class MLP_Block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, activate_function='relu', drop_out=0):
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
        if self.drop_out != 0:
            self.mlp.append(torch.nn.Dropout(self.drop_out))
    
    def forward(self, x):
        return self.mlp(x)        
        
class VGG16(torch.nn.Module):
    def __init__(self, input_channels: int, output_classes: int, image_size: tuple):
        super(VGG16, self).__init__()
        self.input_channels = input_channels
        self.output_classes = output_classes
        self.image_size = image_size
        self.block_1 = torch.nn.Sequential(
            CNN_Block(in_channels=self.input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            CNN_Block(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.block_2 = torch.nn.Sequential(
            CNN_Block(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            CNN_Block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.block_3 = torch.nn.Sequential(
            CNN_Block(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            CNN_Block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            CNN_Block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.block_4 = torch.nn.Sequential(
            CNN_Block(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            CNN_Block(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            CNN_Block(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.block_5 = torch.nn.Sequential(
            CNN_Block(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            CNN_Block(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            CNN_Block(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, norm='batch_norm', drop_out=0.2),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.feature_extraction = torch.nn.Sequential(
            self.block_1, self.block_2, self.block_3, self.block_4, self.block_5,
            torch.nn.Flatten(start_dim=1)
        )
        self.output_size = self.compute_output_size()
        self.MLP = torch.nn.Sequential(
            MLP_Block(input_channel=self.output_size, output_channel=4096, activate_function='relu', drop_out=0),
            MLP_Block(input_channel=4096, output_channel=1000, activate_function='relu', drop_out=0),
            MLP_Block(input_channel=1000, output_channel=1000, activate_function='relu', drop_out=0),
            MLP_Block(input_channel=1000, output_channel=self.output_classes, activate_function='batch_norm', drop_out=0.2),
        )
    
    def compute_output_size(self):
        x = torch.randn((1, self.input_channels, self.image_size[0], self.image_size[1]))
        y = self.feature_extraction(x)
        output_size = y.shape[-1]
        return output_size
    
    def forward(self, x):
        feature = self.feature_extraction(x)
        output = self.MLP(feature)
        return output