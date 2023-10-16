import torch
import torch.nn as nn

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(pred, actual))


class AleatoricLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, log_var):
        loss = (pred - actual)**2
        loss = 0.5 * torch.exp(-log_var) * loss + (0.5 * log_var)
        loss = torch.sum(loss) / pred.numel()
        return loss


class LinUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, log_var):
        loss = (pred - actual)**2
        uncertainty_loss = torch.abs(log_var - loss)
        loss = torch.sum(loss + uncertainty_loss) / pred.numel()
        return loss


class SquaredUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, log_var):
        loss = (pred - actual)**2
        uncertainty_loss = (((log_var - loss)**2)/loss)
        loss = torch.sum(loss + 0.1*uncertainty_loss) / pred.numel()
        return loss
