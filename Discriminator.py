import torch
import torch.nn as nn

ksize = 3
class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, ksize, stride, bias = False, padding = 1),
            nn.BatchNorm2d(output_channels), 
            nn.LeakyReLU(0.2),
        )
        
    def forward (self, x):
        return self.conv(x)


class Discriminator (nn.Module):
    def __init__(self, input_channels = 1, features = [64, 128, 256, 512]): 
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels*2, features[0], kernel_size = ksize, stride = 2, padding = 1),
            nn.LeakyReLU(0.2),
        )
        
        layers = []
        input_channels = features[0]
        for feature in features[1:]: 
            layers.append(
                CNNBlock(input_channels, feature, stride = 1 if feature == features[-1] else 2),
            )
            input_channels = feature
        
        layers.append(
            nn.Conv2d(
                input_channels, 1, kernel_size = ksize, stride = 1, padding = 1
                ), 
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x = torch.cat([x,y], dim = 1)
        x = self.initial(x)
        return self.model(x)
    
    