import torch
import torch.nn as nn
from models import BaseAE

class aeResNet(BaseAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 **kwargs):
        super(aeResNet, self).__init__()

        self.encoder = nn.Sequential(
            ResBlock(in_channels=in_channels, out_channels=16, kernel_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=16, out_channels=32, kernel_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=32, out_channels=64, kernel_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=64, out_channels=128, kernel_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=128, out_channels=256, kernel_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=256, out_channels=512, kernel_size=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.encoder_fc = nn.Linear(512, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 512)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=in_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )


        # self.encoder = nn.Sequential(
        #     ResBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=4),
        #     nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=4, stride=2),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(inplace=True),
        #
        #     ResBlock(in_channels=16, out_channels=16, kernel_size=4),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(inplace=True),
        #
        #     ResBlock(in_channels=32, out_channels=32, kernel_size=4),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(inplace=True),
        #
        #     ResBlock(in_channels=64, out_channels=64, kernel_size=4),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
        #     nn.BatchNorm2d(128),
        # )
        #
        # self.encoder_fc = nn.Linear(2048, latent_dim)
        #
        # self.decoder_fc = nn.Linear(latent_dim, 2048)
        #
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=16, out_channels=in_channels, kernel_size=4, stride=2),
        #     nn.Sigmoid(),
        # )

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        output = self.encoder(input)
        output = torch.flatten(output, start_dim=1)
        output = self.encoder_fc(output)
        output = torch.flatten(output, start_dim=1)

        return output

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        output = self.decoder_fc(z)
        # output = output.view(-1, 256, 3, 3)
        # output = output.view(-1, 128, 4, 4)
        output = output.view(-1, 512, 1, 1)
        output = self.decoder(output)
        return output

    def forward(self, input):
        # print(f'before: {input.shape}')
        output = self.encode(input)
        # print(f'after: {output.shape}')
        output = self.decode(output)
        # print(f'finally: {output.shape}')
        return [output, input]

    def loss_function(self, *args, **kwargs):
        outputs = args[0]
        inputs = args[1]
        criterion = nn.MSELoss()
        loss = criterion(outputs, inputs)
        return loss

    # def loss_function(self, *args, **kwargs):
    #     outputs = args[0]
    #     inputs = args[1]
    #
    #     channel_mean = torch.mean(inputs, dim=(0, 2, 3))
    #     channel_weights = 1.0 / channel_mean
    #     channel_weights /= torch.sum(channel_weights)
    #
    #     squared_error = torch.square(inputs - outputs)
    #     weighted_error = squared_error * channel_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    #     return torch.mean(weighted_error)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size-2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        fx = self.conv1(x)
        # print(f'fx1: {fx.shape}')
        fx = self.bn1(fx)
        fx = self.relu(fx)
        fx = self.conv2(fx)
        # print(f'fx2: {fx.shape}')
        fx[:, 0:self.in_channels, :, :] += x
        # print(f'fx3: {fx.shape}')
        out = fx
        # print(f'out1: {out.shape}')
        out = self.relu(out)
        out = self.bn2(out)
        # print(f'out2: {out.shape}')
        return out


