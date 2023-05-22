import torch.nn as nn
from autoencoder.models import BaseAE


class Autoencoder(BaseAE):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(16, 8, kernel_size=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(f'before: {x.shape}')
        x = self.encoder(x)
        print(f'after: {x.shape}')
        x = self.decoder(x)
        print(f'finally: {x.shape}')
        return x

    def loss_function(self, outputs, inputs):
        criterion = nn.MSELoss()
        loss = criterion(outputs, inputs)
        return loss