"""
Feature extractor for few show segmentation (modified vgg16)
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        self.model = nn.Sequential(
            self._make_layer(2, in_channels, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(2, 64, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 128, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 256, 512),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(3, 512, 512, dilation=2, relu=False)
        )

        self._init_weights()

    def forward(self, x):
        return self.model(x)

    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, relu=True):
        """
        Make a [(conv->relu)...] layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, dilation=dilation, padding=dilation))
            if i != n_convs - 1 or relu:
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self):
        if self.pretrained_path:
            state_dict = torch.load(self.pretrained_path, map_location='cpu')
            self.load_state_dict(state_dict)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

# sanity check
if __name__ == '__main__':
    encoder = Encoder().cuda()
    x = torch.rand(1, 3, 128, 128).cuda()
    y = encoder(x)
    print(y.shape)