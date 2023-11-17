import torch
import torch.nn as nn
from timm import create_model
from .losses import *


class fixMatch(nn.Module):
    def __init__(self,
                 pretrained_weights='resnet18',
                 in_channels=3,
                 nbr_outputs=1,
                 supervised_criterion='MSE',
                 drop_rate=0,
                 **kwargs):
        super(fixMatch, self).__init__()
        self.model = create_model(pretrained_weights, pretrained=True, drop_rate=drop_rate, num_classes=0, in_chans=in_channels)
        self.fc_preds = nn.Linear(self.model.num_features, nbr_outputs)
        self.uncertainty = nn.Linear(self.model.num_features, nbr_outputs)
        # factor to scale unsupervised_loss to be similar to supervised_loss
        self.unsupervised_factor = 1  # 1_000_000 / self.model.num_features

        supervised_losses = {'L1': nn.L1Loss(),
                             'MSE': nn.MSELoss(),
                             'RMSE': RMSELoss(),
                             'RMSLE': RMSLELoss(),
                             'Aleatoric': AleatoricLoss(),
                             'LinUncertainty': LinUncertaintyLoss(),
                             'SquaredUncertainty': SquaredUncertaintyLoss()}

        self.supervised_criterion = supervised_losses[supervised_criterion]

    def forward(self, x):
        features = self.model(x)
        prediction = self.fc_preds(features).flatten()
        prediction = torch.pow(2, prediction)
        uncertainty = self.uncertainty(features).flatten()
        uncertainty = torch.pow(2, uncertainty)
        return prediction, features, uncertainty

    def loss_supervised(self, predictions, labels):
        loss = self.supervised_criterion(predictions, labels)
        return loss

    def loss_supervised_w_uncertainty(self, predictions, labels, log_var):
        loss = self.supervised_criterion(predictions, labels, log_var)
        return loss

    def loss_unsupervised(self, student_features, teacher_features, mask):
        squared_difference = torch.sum((student_features - teacher_features) ** 2, dim=1)
        masked_squared_difference = squared_difference * mask
        # To ensure that we compute the mean correctly, we should divide by the number of '1's in the mask.
        mse = torch.sum(masked_squared_difference) / torch.sum(mask)
        loss = mse * self.unsupervised_factor
        return loss
