import torch
import torch.nn as nn
import torch.nn.functional as F


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

    @staticmethod
    def forward(pred, actual, log_var):
        loss = (pred - actual)**2
        loss = 0.5 * torch.exp(-log_var) * loss + (0.5 * log_var)
        loss = torch.sum(loss) / pred.numel()
        return loss


class AleatoricLossModified(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual, log_var):
        loss = (pred - actual)**2
        loss = 0.5 * torch.exp(-log_var*0.001) * loss + (0.5 * log_var)
        loss = torch.sum(loss) / pred.numel()
        return loss

class AleatoricLinDecayLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual, log_var, cur_step, total_step):
        loss = (pred - actual)**2
        if cur_step < 0.2*total_step:
            loss = 0.5 * torch.exp(-log_var) * loss + (0.5 * log_var)
        else:
            loss = 0.5 * torch.exp(-log_var) * loss + (0.5 * log_var) * (1-(cur_step/total_step))
        loss = torch.sum(loss) / pred.numel()
        return loss


class L1UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual, var):
        # TODO separate mse and uncertainty for better logging?
        mse_loss = (pred - actual)**2
        uncertainty_loss = (torch.sqrt(mse_loss) - var)**2
        loss = mse_loss + uncertainty_loss
        loss = torch.sum(loss) / pred.numel()
        return loss


class CalcBias(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual):
        diff = (pred - actual)
        bias = torch.sum(diff) / diff.numel()
        return bias


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual):
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


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual):
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


class MaskedRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual):
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


class MaskedLinUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual, log_var):
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


class UncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(predictions, labels, uncertainties):
        diff = torch.abs(predictions - labels)
        mse = F.mse_loss(uncertainties, diff)
        return mse


class SquaredUncertaintyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, actual, log_var):
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

    @staticmethod
    def forward(anchor, positive, negative, mask, margin=1.0):
        """
        Compute the Contrastive Loss for batches of vectors P1 and P2 with a feature mask,
        using only the unmasked features for the distance calculation.

        :param mask: Tensor of the same shape as P1 and P2, with 1s for features to use and 0s for features to ignore
        :param margin: Margin for the loss (default is 1.0)
        :return: Masked contrastive loss for the batch
        """
        # Calculate Euclidean distances
        euclidean_distance_pos = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=1))
        euclidean_distance_neg = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=1))

        # Compute loss for each pair and then average over the batch
        loss_contrastive_pos = torch.pow(euclidean_distance_pos, 2)
        loss_contrastive_neg = torch.pow(torch.clamp(margin - euclidean_distance_neg, min=0.0), 2)

        # Scale to be on the same magnitude as the supervised loss
        #TODO Handle mask being all zeros
        loss = torch.sum((loss_contrastive_pos + loss_contrastive_neg) * mask) / torch.sum(mask)

        return loss


class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(anchor, positive, negative, mask, margin):
        # L2 Normalize the embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Compute the Euclidean distance between anchor and positive/negative
        positive_distance = (anchor - positive).pow(2).sum(1)
        negative_distance = (anchor - negative).pow(2).sum(1)

        # Compute the loss
        losses = torch.relu(positive_distance - negative_distance + margin)
        if mask == None:
            return torch.sum(losses)
        else:
            losses_masked = losses * mask
            # Calculate the average loss only for non-zero mask elements
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

    @staticmethod
    def forward(anchor, positive, negative, mask, margin):
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


class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, anchor, positive, negative, mask, margin):
        # Cosine similarity
        sim_pos = F.cosine_similarity(anchor, positive, dim=1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=1)

        # Compute the loss
        losses = torch.relu(sim_neg - sim_pos + margin)

        if mask is not None:
            losses_masked = losses * mask
            # Calculate the average loss only for non-zero mask elements
            if torch.sum(mask) != 0:
                cos_similarity_loss = torch.sum(losses_masked) / torch.sum(mask)
            else:
                cos_similarity_loss = torch.tensor([0], dtype=torch.float32)
                if torch.cuda.is_available():
                    cos_similarity_loss = cos_similarity_loss.cuda()
        else:
            # If mask is not provided, calculate the average loss
            cos_similarity_loss = torch.mean(losses)

        return cos_similarity_loss

