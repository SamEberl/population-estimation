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


class AleatoricLossModified(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, log_var):
        loss = (pred - actual)**2
        loss = 0.5 * torch.exp(-log_var*0.001) * loss + (0.5 * log_var)
        loss = torch.sum(loss) / pred.numel()
        return loss

class AleatoricLinDecayLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, log_var, cur_step, total_step):
        loss = (pred - actual)**2
        if cur_step < 0.2*total_step:
            loss = 0.5 * torch.exp(-log_var) * loss + (0.5 * log_var)
        else:
            loss = 0.5 * torch.exp(-log_var) * loss + (0.5 * log_var) * (1-(cur_step/total_step))
        loss = torch.sum(loss) / pred.numel()
        return loss


def maskedBias(pred, actual):
    mask = actual != -1
    diff = (pred - actual)
    masked_diff = diff * mask
    # To ensure that we compute the mean correctly, we should divide by the number of '1's in the mask.
    pred_numel = torch.sum(mask)
    if pred_numel > 0:
        bias = torch.sum(masked_diff) / pred_numel
    else:
        bias = torch.tensor([-1], dtype=torch.float32)
        if torch.cuda.is_available():
            bias = bias.cuda()
    return bias


def maskedL1Loss(pred, actual):
    mask = actual != -1
    l1_loss = torch.abs(pred - actual)
    masked_l1_loss = l1_loss * mask
    # To ensure that we compute the mean correctly, we should divide by the number of '1's in the mask.
    pred_numel = torch.sum(mask)
    if pred_numel > 0:
        loss = torch.sum(masked_l1_loss) / pred_numel
    else:
        loss = torch.tensor([-1], dtype=torch.float32)
        if torch.cuda.is_available():
            loss = loss.cuda()
    return loss


def maskedMSELoss(pred, actual):
    mask = actual != -1
    mse_loss = torch.pow(pred - actual, 2)
    masked_mse_loss = mse_loss * mask
    # To ensure that we compute the mean correctly, we should divide by the number of '1's in the mask.
    pred_numel = torch.sum(mask)
    if pred_numel > 0:
        loss = (torch.sum(masked_mse_loss) / pred_numel)
    else:
        loss = torch.tensor([-1], dtype=torch.float32)
        if torch.cuda.is_available():
            loss = loss.cuda()
    return loss


def maskedRMSELoss(pred, actual):
    mask = actual != -1
    mse_loss = torch.pow(pred - actual, 2)
    masked_mse_loss = mse_loss * mask
    # To ensure that we compute the mean correctly, we should divide by the number of '1's in the mask.
    pred_numel = torch.sum(mask)
    if pred_numel > 0:
        loss = torch.sqrt(torch.sum(masked_mse_loss) / pred_numel)
    else:
        loss = torch.tensor([-1], dtype=torch.float32)
        if torch.cuda.is_available():
            loss = loss.cuda()
    return loss


class LinUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, log_var):
        mask = actual != -1
        squared_diff = (pred - actual) ** 2
        #uncertainty_loss = torch.abs(log_var - squared_diff)
        uncertainty_loss = torch.sqrt(torch.sum(torch.pow(log_var - squared_diff, 2)))
        masked_squared_diff = squared_diff * mask
        masked_uncertainty_loss = uncertainty_loss * mask
        # To ensure that we compute the mean correctly, we should divide by the number of '1's in the mask.
        pred_numel = torch.sum(mask)
        if pred_numel > 0:
            loss = (torch.sum(masked_squared_diff) + uncertainty_loss) / pred_numel
        else:
            loss = torch.tensor([-1], dtype=torch.float32)
            if torch.cuda.is_available():
                loss = loss.cuda()
        # if pred_numel == 39:
            # print(f'\n---------------')
            # print(f'log_var     : {log_var}')
            # print(f'squared_diff: {squared_diff}')
            # print(f'log_var-diff: {log_var-squared_diff}')
            # print(f'uncertainty : {uncertainty_loss}')
            # print(f'pred_numel: {pred_numel}')
            # print(f'pred: {torch.sum(pred)} | actual: {torch.sum(actual)} | log_var: {torch.sum(log_var)}')
            # print(f'mse   : {torch.sum(masked_squared_diff)}')
            # print(f'uncert: {torch.sum(uncertainty_loss)}')
            # print(f'loss: {loss} \n')
        return loss


class SquaredUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, actual, log_var):
        loss = (pred - actual)**2
        uncertainty_loss = (((log_var - loss)**2)/loss)
        loss = torch.sum(loss + 0.1*uncertainty_loss) / pred.numel()
        return loss


# def contrastiveLoss(self, student_features, teacher_features, mask):
#     squared_difference = torch.sum((student_features - teacher_features) ** 2, dim=1)
#     masked_squared_difference = squared_difference * mask
#     # To ensure that we compute the mean correctly, we should divide by the number of '1's in the mask.
#     mse = torch.sum(masked_squared_difference) / torch.sum(mask)
#     loss = mse * self.unsupervised_factor
#     return loss


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_features, teacher_features, mask, Y, margin=1.0):
        """
        Compute the Contrastive Loss for batches of vectors P1 and P2 with a feature mask,
        using only the unmasked features for the distance calculation.

        :param P1: Batch of vectors (PyTorch tensor)
        :param P2: Batch of vectors (PyTorch tensor)
        :param Y: Batch of binary labels (1 for positive pairs, 0 for negative pairs)
        :param mask: Tensor of the same shape as P1 and P2, with 1s for features to use and 0s for features to ignore
        :param margin: Margin for the loss (default is 1.0)
        :return: Masked contrastive loss for the batch
        """
        # Calculate Euclidean distances
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(student_features - teacher_features, 2), dim=1))

        # Apply the mask to the squared differences
        masked_distance = euclidean_distance * mask

        # Compute loss for each pair and then average over the batch
        loss_contrastive = (torch.sum(Y * torch.pow(masked_distance, 2) +
                                      (1 - Y) * torch.pow(torch.clamp(margin - masked_distance, min=0.0), 2)) / torch.sum(mask))

        # Scale to be on the same magnitude as the supervised loss
        loss = loss_contrastive #* self.unsupervised_factor

        return loss


class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchor, positive, negative, mask, margin):
        # Compute the Euclidean distance between anchor and positive
        positive_distance = (anchor - positive).pow(2).sum(1)
        # Compute the Euclidean distance between anchor and negative
        negative_distance = (anchor - negative).pow(2).sum(1)
        # Compute the loss
        losses = torch.relu(positive_distance - negative_distance + margin)
        losses_masked = losses * mask
        if torch.sum(mask) != 0:
            triplet_loss = torch.sum(losses_masked) / torch.sum(mask)
        else:
            triplet_loss = torch.tensor([0], dtype=torch.float32)
            if torch.cuda.is_available():
                triplet_loss = triplet_loss.cuda()
        return triplet_loss


class TripletLossModified(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchor, positive, negative, mask, margin):
        # Compute the Euclidean distance between anchor and positive
        positive_distance = torch.sqrt((anchor - positive).pow(2).sum(1))
        # Compute the Euclidean distance between anchor and negative
        negative_distance = torch.sqrt((anchor - negative).pow(2).sum(1))
        # Compute the loss
        losses = torch.relu(positive_distance - negative_distance + margin)
        losses_masked = losses * mask
        if torch.sum(mask) != 0:
            triplet_loss = torch.sum(losses_masked) / torch.sum(mask)
        else:
            triplet_loss = torch.tensor([0], dtype=torch.float32)
            if torch.cuda.is_available():
                triplet_loss = triplet_loss.cuda()

        # mean_distances = torch.mean((anchor - positive).pow(2).sum() + (anchor - negative).pow(2).sum())
        mean_distances = torch.mean(positive_distance + negative_distance)
        new_term = torch.abs((0.5 * mean_distances) - 1)

        loss = triplet_loss + new_term
        return loss