import torch
import torch.nn as nn

from .losses import RMSELoss, RMSLELoss

class regBasicDINOv2(nn.Module):
    def __init__(self,
                 in_size,
                 **kwargs):
        super(regBasicDINOv2, self).__init__()

        # self.network = nn.Sequential(nn.Linear(in_size, 32),
        #                              nn.BatchNorm1d(32),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Linear(32, 16),
        #                              nn.BatchNorm1d(16),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Linear(16, 1),
        #                              nn.Sigmoid())

        # self.network = nn.Sequential(nn.Linear(in_size, 256),
        #                              nn.BatchNorm1d(256),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Linear(256, 128),
        #                              nn.BatchNorm1d(128),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Linear(128, 64),
        #                              nn.BatchNorm1d(64),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Linear(64, 1),
        #                              nn.LeakyReLU(inplace=True),
        #                              # nn.Sigmoid()
        #                              )

        self.network = nn.Sequential(nn.Linear(in_size, 128),
                                     nn.BatchNorm1d(128),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(128, 64),
                                     nn.BatchNorm1d(64),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(64, 1),
                                     # nn.LeakyReLU(inplace=True),
                                     )

    def forward(self, input):
        output = self.network(input)
        # print('---')
        # print(output)
        output = torch.exp(output * 1)  # Scale the output to the range [0, 20000]
        # print(output)
        return output

    def loss_function(self, predictions, labels):
        # criterion = RMSLELoss()
        criterion = RMSELoss()
        # criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        loss = criterion(predictions, labels)
        return loss
