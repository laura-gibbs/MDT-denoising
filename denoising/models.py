import torch.nn as nn
# import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),# stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),# stride=2),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )

        self.fully_connected = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1),# stride=2),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        out = self.fully_connected(dec)

        return out

