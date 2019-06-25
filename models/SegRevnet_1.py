import torch
from torch import nn
from reversible_blocks.revop import ReversibleBlock


class ConvBnReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1,
                 has_bn=False, has_relu=True):
        super(ConvBnReLu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding, dilation=dilation)

        if has_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):

        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


def make_block(channels, num_layers_in_block):
    layers = []

    for i in range(num_layers_in_block - 1):
        layers.append(ConvBnReLu(channels, channels, has_bn=True, has_relu=True))
    layers.append(ConvBnReLu(channels, channels, has_bn=True, has_relu=False))

    return nn.Sequential(*layers)


class SegRevnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, channels=30, num_layers_in_block=2, num_blocks=5):
        super(SegRevnet, self).__init__()
        self.name = 'Revnet'
        self.initial_laeyr= ConvBnReLu(in_channels=in_channels, out_channels=channels,
                                        kernel_size=1, has_bn=True, has_relu=True, padding=0)
        # F = ConvBnReLu(channels // 2, channels // 2)
        # G = ConvBnReLu(channels // 2, channels // 2)
        # self.Y = ReversibleBlock(F, G)

        main = []

        for i in range(num_blocks):
            F = make_block(channels // 2, num_layers_in_block)
            G = make_block(channels // 2, num_layers_in_block)
            Y = ReversibleBlock(F, G)
            main.append(Y)
            main.append(nn.ReLU())

        self.main = nn.Sequential(*main)
        self.final_layer = ConvBnReLu(in_channels=channels, out_channels=num_classes,
                                       kernel_size=1, has_bn=False, has_relu=False, padding=0)

    def forward(self, x):
        x = self.initial_laeyr(x)
        x = self.main(x)
        x = self.final_layer(x)
        # x = self.Y(x)
        # x = self.final_layer(x)
        return x


if __name__ == "__main__":
    net = SegRevnet(in_channels=3, channels=10,
                    num_layers_in_block=2, num_blocks=1,
                    num_classes=5)

    inp = torch.rand((1, 3, 512, 512))
    conv = nn.Conv2d(in_channels=3, out_channels=5,
                     kernel_size=3, padding=1, dilation=1)
    out = net(inp)
    print(out.shape)
