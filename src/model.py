import torch
import torch.nn as nn
# class Convolution()

architecture = [
    # 0 -> Name, 1 -> Kernel, 2 -> out_channels, 3 -> stride, 4 -> padding
    ("Conv", (7, 7), 64, 2, 3),
    ("MaxPool", (2, 2), None, 2, 0),
    
    ("Conv", (3, 3), 192, 1, 1),
    ("MaxPool", (2, 2), None, 2, 0),
    
    ("Conv", (1, 1), 128, 1, 0),
    ("Conv", (3, 3), 256, 1, 1),
    ("Conv", (1, 1), 256, 1, 0),
    ("Conv", (3, 3), 512, 1, 1),
    ("MaxPool", (2, 2), None, 2, 0),
    
    ("Conv", (1, 1), 256, 1, 0),
    ("Conv", (3, 3), 512, 1, 1),
    ("Conv", (1, 1), 256, 1, 0),
    ("Conv", (3, 3), 512, 1, 1),
    ("Conv", (1, 1), 256, 1, 0),
    ("Conv", (3, 3), 512, 1, 1),
    ("Conv", (1, 1), 256, 1, 0),
    ("Conv", (3, 3), 512, 1, 1),
    ("Conv", (1, 1), 512, 1, 0),
    ("Conv", (3, 3), 1024, 1, 1),
    ("MaxPool", (2, 2), None, 2, 0),
    
    ("Conv", (1, 1), 512, 1, 0),
    ("Conv", (3, 3), 1024, 1, 1),
    ("Conv", (1, 1), 512, 1, 0),
    ("Conv", (3, 3), 1024, 1, 1),
    ("Conv", (3, 3), 1024, 1, 1),
    ("Conv", (3, 3), 1024, 2, 1),
    
    ("Conv", (3, 3), 1024, 1, 1),
    ("Conv", (3, 3), 1024, 1, 1), 
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs)
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        output = self.leakyrelu(self.batchNorm(self.conv(x)))
        return output


class YoloV1(torch.nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        super(YoloV1, self).__init__()
        self.input_shape = (448, 448, 3)
        self.architechture = architecture
        self.in_channels = input_channels
        self.net = self._create_conv_layers(self.architechture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.net(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for layer in architecture:
            if layer[0] == 'Conv':
                conv = CNNBlock(in_channels, out_channels=layer[2], kernel_size=layer[1], stride=layer[3], padding=layer[4])
                layers.append(conv)
                in_channels = layer[2]
            if layer[0] == 'MaxPool':
                maxPool = nn.MaxPool2d(kernel_size=layer[1], stride=layer[3])
                layers.append(maxPool)

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C= split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + 5 * B))
        )


# def test(S=7,B=2,C=20):
#     model = YoloV1(input_channels=3, split_size=7, num_boxes=2, num_classes=C)
#     x = torch.randn((2, 3, 448, 448))

#     output = model(x)
#     print(output.shape)

# test()
