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

        # self.fc1 = nn.Linear(in_size, 32)
        # self.bn1 = nn.BatchNorm1d(32),
        # self.fc2 = nn.Linear(32, 16)
        # self.bn2 = nn.BatchNorm1d(16),
        # self.fc3 = nn.Linear(16, 1)
        # self.relu = nn.LeakyReLU

    def forward(self, input):
        output = self.network(input)
        output = (output * 45000) - 100  # Scale the output to the range [0, 20000]
        return output

    # def forward(self, input):
    #     output = self.bn1(self.relu(self.fc1(input)))
    #     output = self.bn2(self.relu(self.fc2(output)))
    #     output = torch.sigmoid(self.fc3(output))  # Apply sigmoid to map the output to the range [0, 1]
    #     output = output * 20000  # Scale the output to the range [0, 20000]
    #     return output

    def loss_function(self, predictions, labels):
        criterion = nn.MSELoss()
        loss = criterion(predictions, labels)
        return loss
