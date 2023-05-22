import torch.nn as nn
from autoencoder.models import BaseAE

class aeMNIST(BaseAE):
    def __init__(self,
                 in_channels: int,
                 **kwargs):
        super(aeMNIST, self).__init__()
        self.z_size = 32

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=4, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
        #     nn.ReLU(inplace=True),
        # )
        #
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=16, out_channels=in_channels, kernel_size=4, stride=2),
        #     nn.Tanh()
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=in_channels, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input):
        # print(f'before: {x.shape}')
        z = self.encoder(input)
        # print(f'after: {x.shape}')
        output = self.decoder(z)
        # print(f'finally: {x.shape}')

        return [output, input]

    def loss_function(self, *args, **kwargs):
        outputs = args[0]
        inputs = args[1]
        criterion = nn.MSELoss()
        loss = criterion(outputs, inputs)
        return loss

