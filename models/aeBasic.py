import torch
import torch.nn as nn
from autoencoder.models import BaseAE

class aeBasic(BaseAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 **kwargs):
        super(aeBasic, self).__init__()

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=4, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
        #     nn.LeakyReLU(inplace=True),
        # )
        #
        # self.encoder_fc = nn.Linear(2048, latent_dim)
        #
        # self.decoder_fc = nn.Linear(latent_dim, 2048)
        #
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=16, out_channels=in_channels, kernel_size=4, stride=2),
        #     # nn.LeakyReLU(inplace=True),
        #     # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        #     # nn.Tanh(),
        #     nn.Sigmoid(),
        #     # ClampLayer()
        # )

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
        #     # nn.LeakyReLU(inplace=True),
        # )
        #
        # self.encoder_fc = nn.Linear(3200, latent_dim)
        #
        # self.decoder_fc = nn.Linear(latent_dim, 3200)
        #
        # self.decoder = nn.Sequential(
        #     # nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
        #     # nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.ConvTranspose2d(in_channels=16, out_channels=in_channels, kernel_size=3, stride=2, output_padding=1),
        #     # nn.LeakyReLU(inplace=True),
        #     # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
        #     # nn.Tanh(),
        #     nn.Sigmoid(),
        #     # ClampLayer()
        # )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )

        self.encoder_fc = nn.Linear(2048, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 2048)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=in_channels, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

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
        output = output.view(-1, 128, 4, 4)
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
    #     mask = (inputs > 0) | (outputs > 0)
    #     masked_inputs = inputs[mask]
    #     masked_outputs = outputs[mask]
    #
    #     criterion = nn.MSELoss()
    #     loss = criterion(masked_outputs, masked_inputs)
    #
    #     return loss


class CutOff(nn.Module):
    def __init__(self, bottom, top):
        super(CutOff, self).__init__()
        self.bottom = bottom
        self.top = top

    def forward(self, x):
        out = x + (self.bottom+self.top)/2
        out[x < self.bottom] = self.bottom
        out[x > self.top] = self.top
        return out

    def backward(self, grad_output):
        out = grad_output
        out[grad_output < self.bottom] = 0
        out[grad_output > self.top] = 0
        out[grad_output != 0] = 1
        return out

class ClampLayer(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0, max=1)


