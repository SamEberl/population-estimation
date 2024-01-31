import torch

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
    elif split == 'test' or split == 'val':
        r2 = 1-(mean/9868522.0)
    else:
        print('wrong split when computing r2')
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
            print(f'metric_name: {metric_name}')
            if 'Observe-Bias' in metric_name:
                if 'train' in metric_name:
                    calc_bias(values)
                elif 'val' in metric_name:
                    calc_bias(values)
            elif 'Observe-R2' in metric_name:
                if 'train' in metric_name:
                    calc_r2(values, 'train')
                elif 'val' in metric_name:
                    calc_r2(values, 'val')
            else:
                mean_value = sum(values) / len(values)
                self.writer.add_scalar(metric_name, mean_value, step_nbr)

    def clear(self):
        # Clear metrics after logging
        self.metrics.clear()

