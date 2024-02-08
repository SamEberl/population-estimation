import torch
import numpy as np
import os
from datetime import datetime

def calc_r2(numbers, split):
    # Filtering out the zeros and calculating the sum of the remaining numbers
    non_zero_numbers = [num for num in numbers if num != 0]
    total_sum = sum(non_zero_numbers)

    # Counting the number of non-zero elements
    non_zero_count = len(non_zero_numbers)

    # Avoiding division by zero
    if non_zero_count == 0:
        return 0

    # Calculating the mean
    mean = total_sum / non_zero_count

    r2 = 0
    if split == 'train':
        r2 = 1-(mean/7373152.5)
    elif split == 'test' or split == 'val' or split == 'valid':
        r2 = 1-(mean/9868522.0)

    return r2


def calc_bias(numbers):
    """
    Computes the mean of a list of numbers, excluding zeros from both the sum and the count.
    """
    # Filtering out the zeros and calculating the sum of the remaining numbers
    non_zero_numbers = [num for num in numbers if num != 0]
    total_sum = sum(non_zero_numbers)

    # Counting the number of non-zero elements
    non_zero_count = len(non_zero_numbers)

    # Avoiding division by zero
    if non_zero_count == 0:
        return 0

    # Calculating the mean
    mean = total_sum / non_zero_count
    return mean


class MetricsLogger:
    def __init__(self, writer):
        self.metrics = {}
        self.writer = writer
        self.uncertainties = {}
        self.last_epoch = False

    def add_metric(self, metric_name, split, value):
        """Accumulate train metrics."""
        metric_name = f'{metric_name}/{split}'
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.metrics[metric_name].append(value)

    def write(self, step_nbr):
        """Write all accumulated metrics to TensorBoard."""
        for metric_name, values in self.metrics.items():
            if 'Observe-Bias' in metric_name:
                bias = calc_bias(values)
                self.writer.add_scalar(metric_name, bias, global_step=step_nbr)
            elif 'Observe-R2' in metric_name:
                if 'train' in metric_name:
                    r2 = calc_r2(values, 'train')
                    self.writer.add_scalar(metric_name, r2, global_step=step_nbr)
                elif 'valid' in metric_name:
                    r2 = calc_r2(values, 'valid')
                    self.writer.add_scalar(metric_name, r2, global_step=step_nbr)
                else:
                    print('wrong split when computing r2')
            else:
                mean_value = sum(values) / len(values)
                self.writer.add_scalar(metric_name, mean_value, global_step=step_nbr)

    def clear(self):
        # Clear metrics after logging
        self.metrics = {}
        self.uncertainties = {}
        # self.metrics.clear()
        # self.uncertainties.clear()

    def add_uncertainty(self, uncertainty_name, uncertainty):
        uncertainty = uncertainty.to('cpu').detach().numpy()
        if uncertainty_name not in self.uncertainties:
            self.uncertainties[uncertainty_name] = np.array([])
        self.uncertainties[uncertainty_name] = np.concatenate((self.uncertainties[uncertainty_name], uncertainty), axis=0)

    def save_uncertainties(self, config):
        if bool(self.uncertainties):
            if config['model_params']['retrain']:
                path = f"/home/sameberl/computed_numpy/{config['model_params']['retrain_from']}"
            else:
                path = f"/home/sameberl/computed_numpy/{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
            for uncertainty_name, uncertainty in self.uncertainties.items():
                np.save(os.path.join(path, f'{uncertainty_name}.npy'), uncertainty)
