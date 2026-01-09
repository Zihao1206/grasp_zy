import torch.nn as nn
import torch.utils.data
# from graspData import GraspData
import torch.nn.functional as F


class VGG(torch.nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        self.input_channel = input_channels

        self.features = self.make_features()
        self.up_pooling = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, dilation=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, dilation=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, dilation=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=2, dilation=4, output_padding=1),
            nn.ReLU(inplace=True),
        )

        self.pos_output = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.sin_output = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.cos_output = nn.Conv2d(8, 1, kernel_size=3, padding=1)
        self.width_output = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.features(x)
        x = self.up_pooling(x)

        pos = self.pos_output(x)
        sin = self.sin_output(x)
        cos = self.cos_output(x)
        width = self.width_output(x)
        return pos, cos, sin, width

    def make_features(self):
        layers = []
        vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        in_channels = 4
        for v in vgg16:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def compute_loss(self, xc, yc):

        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(xc)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }













