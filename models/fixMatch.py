import torch
import torch.nn as nn
from timm import create_model
from .losses import RMSELoss, RMSLELoss


class fixMatch(nn.Module):
    def __init__(self,
                 model_name='resnet18',
                 in_channels=3,
                 nbr_outputs=1,
                 supervised_criterion='MSE',
                 drop_rate=0,
                 **kwargs):
        super(fixMatch, self).__init__()
        self.model = create_model(model_name, pretrained=True, drop_rate=drop_rate, num_classes=0, in_chans=in_channels)
        self.fc = nn.Linear(self.model.num_features, nbr_outputs)

        supervised_losses = {'L1': nn.L1Loss(),
                             'MSE': nn.MSELoss(),
                             'RMSE': RMSELoss(),
                             'RMSLE': RMSLELoss()}
        self.supervised_criterion = supervised_losses[supervised_criterion]

    def forward(self, x):
        features = self.model(x)
        print(f'features: {features}')
        print(f'shape: {features.shape}')
        prediction = self.fc(features).flatten()
        prediction = torch.pow(2, prediction)
        return prediction, features

    def loss_supervised(self, predictions, labels):
        loss = self.supervised_criterion(predictions, labels)
        return loss

    def loss_unsupervised(self, student_features, teacher_features):
        loss = 0
        return loss
