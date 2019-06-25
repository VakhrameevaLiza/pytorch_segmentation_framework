import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()

        self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),

                        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
        )

    def forward(self, inp):
        return self.block(inp)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.ReLU()
        )

    def forward(self, inp):
        return self.block(inp)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()

        self.up = nn.Sequential(
                        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                        nn.ReLU()
        )

        self.block = nn.Sequential(
                        nn.Conv2d(2 * out_ch, out_ch, kernel_size=3, padding=1),
                        nn.ReLU(),

                        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                        nn.ReLU(),
        )

    def forward(self, tail, inp):
        x = self.up(inp)
        #x = nn.functional.interpolate(inp, scale_factor=2)
        cat = torch.cat((tail, x), dim=1)
        return self.block(cat)


class Unet(nn.Module):
    name='Unet'
    def __init__(self, num_classes, in_channels=3):
        super(Unet, self).__init__()

        self.down_block_1 = DownBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down_block_2 = DownBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down_block_3 = DownBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down_block_4 = DownBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block_1 = ConvBlock(512, 1024)
        self.conv_block_2 = ConvBlock(1024, 1024)

        self.up_block_1 = UpBlock(1024, 512)
        self.up_block_2 = UpBlock(512, 256)
        self.up_block_3 = UpBlock(256, 128)
        self.up_block_4 = UpBlock(128, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)


    def forward(self, inp):
        x1 = self.down_block_1(inp)
        x2 = self.down_block_2(self.pool1(x1))
        x3 = self.down_block_3(self.pool2(x2))
        x4 = self.down_block_4(self.pool3(x3))

        x = self.conv_block_1(self.pool4(x4))
        x = self.conv_block_2(x)

        x = self.up_block_1(x4, x)
        x = self.up_block_2(x3, x)
        x = self.up_block_3(x2, x)
        x = self.up_block_4(x1, x)

        x = self.out_conv(x)
        return x


if __name__ == "__main__":
    unet = Unet(11)
    t = torch.ones((1, 3, 128, 256))
    out = unet(t)
    print(out.shape)
