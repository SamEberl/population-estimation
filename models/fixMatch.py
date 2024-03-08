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
                 **kwargs):
        super(fixMatch, self).__init__()
        self.model = create_model(pretrained_weights, pretrained=pretrained, drop_rate=drop_rate, num_classes=0, in_chans=in_channels)
        self.add_dropout_to_convnext(drop_rate)
        self.fc = nn.Sequential(
            nn.Linear(self.model.num_features, self.model.num_features // 2),
            nn.ReLU(),
            nn.Linear(self.model.num_features // 2, self.model.num_features // 4),
            nn.ReLU(),
            nn.Linear(self.model.num_features // 4, self.model.num_features // 8),
            nn.ReLU(),
            nn.Linear(self.model.num_features // 8, nbr_outputs)
        )
        self.fc_uncertainty = nn.Sequential(
            nn.Linear(self.model.num_features, self.model.num_features // 2),
            nn.ReLU(),
            nn.Linear(self.model.num_features // 2, self.model.num_features // 4),
            nn.ReLU(),
            nn.Linear(self.model.num_features // 4, self.model.num_features // 8),
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
                               'cosine': CosineSimilarity()}

        self.unsupervised_criterion = unsupervised_losses[unsupervised_criterion]

    def add_dropout_to_convnext(self, drop_rate):
        # Loop through modules
        for name, module in self.model.named_modules():
            # Check if module is a bottleneck layer (might involve specific class names)
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.9
            print(f'module: {module}')
            """
            if isinstance(module, Bottleneck):  # Replace with appropriate class name
                # Insert dropout layer after the activation (e.g., ReLU) within the block
                module.relu = torch.nn.Sequential(module.relu, torch.nn.Dropout(p=drop_rate))  # Adjust dropout rate (p)
            """
        exit()


    def forward(self, x):
        features = self.model(x)
        prediction = self.fc(features).flatten()
        # prediction = torch.pow(2, prediction)
        #prediction = torch.sigmoid(prediction) * 40_000
        uncertainty = self.fc_uncertainty(features).flatten()
        #uncertainty = torch.sigmoid(uncertainty) * 5_000
        # uncertainty = torch.pow(2, uncertainty)
        return prediction, features, uncertainty

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


