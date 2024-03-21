import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from .losses import *


class fixMatch(nn.Module):
    def __init__(self,
                 pretrained_weights='resnet18',
                 pretrained=True,
                 in_channels=3,
                 nbr_outputs=1,
                 supervised_criterion='MSE',
                 unsupervised_criterion='contrastive',
                 unsupervised_factor=1e5,
                 drop_rate=0,
                 drop_path_rate=0,
                 projection_size=128,
                 **kwargs):
        super(fixMatch, self).__init__()
        self.model = create_model(pretrained_weights, pretrained=pretrained,
                                  drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                                  num_classes=0, in_chans=in_channels)
        #self.set_dropout(drop_rate)
        self.projection = nn.Sequential(
            nn.Linear(self.model.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, projection_size),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.model.num_features, self.model.num_features // 8),
            nn.ReLU(),
            nn.Linear(self.model.num_features // 8, nbr_outputs)
        )
        self.fc_uncertainty = nn.Sequential(
            nn.Linear(self.model.num_features, self.model.num_features // 8),
            nn.ReLU(),
            nn.Linear(self.model.num_features // 8, nbr_outputs)
        )
        # factor to scale unsupervised_loss to be similar to supervised_loss
        self.unsupervised_factor = unsupervised_factor  # 1_000_000 / self.model.num_features

        supervised_losses = {'L1': F.l1_loss,
                             'MSE': F.mse_loss,
                             'RMSE': RMSELoss(),
                             'RMSLE': RMSLELoss(),
                             'Aleatoric': AleatoricLoss(),
                             'AleatoricModified': AleatoricLossModified(),
                             'AleatoricLinDecay': AleatoricLinDecayLoss(),
                             'L1UncertaintyLoss': L1UncertaintyLoss(),
                             'SquaredUncertainty': SquaredUncertaintyLoss()}

        self.supervised_criterion = supervised_losses[supervised_criterion]
        self.uncertainty_criterion = UncertaintyLoss()

        unsupervised_losses = {'contrastive': ContrastiveLoss(),
                               'triplet': TripletLoss(),
                               'tripletModified': TripletLossModified(),
                               'cosine': CosineSimilarity(),
                               'MSE': F.mse_loss}

        self.unsupervised_criterion = unsupervised_losses[unsupervised_criterion]

    def set_dropout(self, drop_rate):
        # Loop through modules
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                #module.p = drop_rate
                print(f'drop_rate: {module.p}')

    def forward(self, x):
        features = self.model(x)
        projection = self.projection(features)
        prediction = self.fc(features).flatten()
        #features_u = torch.cat((features, prediction.unsqueeze(1)), dim=1)
        uncertainty = self.fc_uncertainty(features).flatten()
        return prediction, projection, uncertainty

    def loss_supervised(self, predictions, labels):
        loss = self.supervised_criterion(predictions, labels)
        return loss

    def loss_uncertainty(self, predictions, labels, uncertainties):
        loss = self.uncertainty_criterion(predictions, labels, uncertainties)
        return loss

    def loss_supervised_w_uncertainty(self, predictions, labels, log_var):
        loss = self.supervised_criterion(predictions, labels, log_var)
        return loss

    def loss_supervised_w_uncertainty_decay(self, predictions, labels, log_var, cur_step, total_step):
        loss = self.supervised_criterion(predictions, labels, log_var, cur_step, total_step)
        return loss

    def loss_unsupervised(self, student_features, teacher_features, dearanged_teacher_features, mask, margin=1):
        loss = self.unsupervised_criterion(student_features, teacher_features, dearanged_teacher_features, mask, margin)
        loss = loss * self.unsupervised_factor
        return loss


