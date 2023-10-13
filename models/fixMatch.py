import torch
import torch.nn as nn
from timm import create_model
from .losses import RMSELoss, RMSLELoss


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
        self.fc_log_var = nn.Linear(self.model.num_features, nbr_outputs)
        self.unsupervised_factor = 1_000_000 / self.model.num_features

        supervised_losses = {'L1': nn.L1Loss(),
                             'MSE': nn.MSELoss(),
                             'RMSE': RMSELoss(),
                             'RMSLE': RMSLELoss()}
        self.supervised_criterion = supervised_losses[supervised_criterion]

    def forward(self, x):
        features = self.model(x)
        prediction = self.fc_preds(features).flatten()
        prediction = torch.pow(2, prediction)

        log_var = self.fc_log_var(features).flatten()
        return prediction, log_var, features

    def loss_supervised(self, predictions, log_var, labels):
        loss = self.supervised_criterion(predictions, labels)
        loss = 0.5 * torch.exp(-log_var) * loss + (1/2 * log_var)
        return loss

    def loss_unsupervised(self, student_features, teacher_features):
        loss = 0 * self.unsupervised_factor
        return loss
