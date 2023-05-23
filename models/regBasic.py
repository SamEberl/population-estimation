import torch
import torch.nn as nn


class regBasic(nn.Module):
    def __init__(self,
                 in_size,
                 **kwargs):
        super(regBasic, self).__init__()

        self.network = nn.Sequential(nn.Linear(in_size, 32),
                                     nn.BatchNorm1d(32),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(32, 16),
                                     nn.BatchNorm1d(16),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Linear(16, 1),
                                     nn.Sigmoid())

    def forward(self, input):
        output = self.network(input)
        output = (output * 45000)  # Scale the output to the range [0, 20000]
        return output

    def loss_function(self, predictions, labels):
        criterion = RMSLELoss()
        # criterion = nn.MSELoss()
        loss = criterion(predictions, labels)
        return loss


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
