import numpy as np
import torch
from functools import partial
from scipy.interpolate import PchipInterpolator

class UncertaintyJudge:
    def __init__(self, use_judge):
        self.preds = []
        self.uncertainties = []
        self.threshold_func = None
        self.use_judge = use_judge
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.median_x_values_torch = None
        self.y_values_torch = None

    @staticmethod
    def power_law(x, a, b):
        return a * x ** (b)

    @staticmethod
    def linear_interp1d(x, y):
        def interpolator(x_new):
            """
            Interpolates the y values for new x values.
            """
            # Ensure x_new is a tensor for operations
            # x_new = torch.as_tensor(x_new, dtype=x.dtype, device=x.device)

            # Find indices of the closest points to the right of the new x values
            idx = torch.searchsorted(x, x_new)
            idx = torch.clamp(idx, 1, len(x) - 1)

            x_left, x_right = x[idx - 1], x[idx]
            y_left, y_right = y[idx - 1], y[idx]

            # Linear interpolation formula
            y_new = y_left + (x_new - x_left) * (y_right - y_left) / (x_right - x_left)
            return y_new

        return interpolator

    def add_pred_var_pair(self, preds, uncertainties):
            self.preds.extend(preds.cpu().detach().numpy().flatten())
            self.uncertainties.extend(uncertainties.cpu().detach().numpy().flatten())

    def calc_threshold_func(self, window=50, percentile=50, smoothing=10):
        self.median_x_values_torch = None
        self.y_values_torch = None

        x_values = np.exp(np.array(self.uncertainties))  # np.exp to get variance instead of s
        y_values = np.array(self.preds)

        y_values, indices = np.unique(y_values, return_index=True)
        x_values = x_values[indices]

        median_x_values = []
        for i in range(len(y_values)):
            window_below = i - window // 2
            window_above = i + window // 2
            if window_below < 0:
                window_above = i + 1
                window_below = 0
            elif window_above > len(y_values):
                window_below = i - 1
                window_above = len(y_values)
            data_in_window = x_values[window_below:window_above]
            median_x = np.percentile(data_in_window, percentile)
            median_x_values.append(median_x)

        # Keep only the first occurrence of each unique value in median_x_values
        median_x_values = np.array(median_x_values)
        median_x_values, unique_indices = np.unique(median_x_values, return_index=True)
        y_values = y_values[unique_indices]

        max_y = 0
        indices_to_drop = []
        for i in range(len(y_values)):
            if y_values[i] <= max_y:
                indices_to_drop.append(i)
            else:
                max_y = y_values[i]
        median_x_values = np.delete(median_x_values, indices_to_drop)[::smoothing]
        y_values = np.delete(y_values, indices_to_drop)[::smoothing]

        self.median_x_values_torch = torch.from_numpy(median_x_values).to(self.device)
        self.y_values_torch = torch.from_numpy(y_values).to(self.device)

        self.threshold_func = self.linear_interp1d(self.median_x_values_torch, self.y_values_torch)
        self.clear()

    def evaluate_threshold_func(self, pred, uncertainty):
        threshold = self.threshold_func(pred)
        pred_var = torch.exp(uncertainty)
        mask = pred_var < threshold
        return mask

    def clear(self):
        self.preds = []
        self.uncertainties = []