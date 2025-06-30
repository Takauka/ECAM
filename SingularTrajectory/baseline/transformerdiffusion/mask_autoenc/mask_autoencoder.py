"""Mask convolutional autoencoder model for map mask encoding."""

from torch import nn


class PatchEncoder(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(11*11, output_size),
        )

    def forward(self, x):
        return self.net(x)


class PatchDecoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 upsample_arch: str):
        super().__init__()

        if upsample_arch == 'conv_transpose':
            self.net = nn.Sequential(
                nn.Linear(input_size, 11*11),
                nn.Unflatten(1, (1, 11, 11)),
                nn.ConvTranspose2d(1, 32, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0,
                                   output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=0),
                nn.Sigmoid(),
            )

        elif upsample_arch == 'pixel_shuffle':
            self.net = nn.Sequential(
                nn.Linear(input_size, 25*25),
                nn.Unflatten(1, (1, 25, 25)),
                nn.Conv2d(1, 128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.PixelShuffle(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.PixelShuffle(2),
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(
                f'Upsample method {upsample_arch} not supported'
                ' (choose from "conv_transpose" or "pixel_shuffle")')

    def forward(self, x):
        return self.net(x)


class MaskConvAutoencoder(nn.Module):
    def __init__(self,
                 bottleneck_size: int,
                 upsample_arch: str):
        super().__init__()

        self.encoder = PatchEncoder(bottleneck_size)
        self.decoder = PatchDecoder(bottleneck_size, upsample_arch)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
