import statsmodels.api as sm
import numpy as np
from functools import partial
from scipy.interpolate import PchipInterpolator

class UncertaintyJudge:
    def __init__(self, use_judge):
        self.preds = []
        self.uncertainties = []
        self.threshold_func = None
        self.use_judge = use_judge

    @staticmethod
    def power_law(x, a, b):
        return a * x ** (b)

    def add_pred_var_pair(self, preds, uncertainties):
            self.preds.extend(preds.cpu().detach().numpy().flatten())
            self.uncertainties.extend(uncertainties.cpu().detach().numpy().flatten())

    def calc_threshold_func_archive(self, q=0.6):
        """
        Fit a quantile regression model with a quantile value of 0.6.
        Meaning 60% of the datapoints are larger than what the function predicts.
        """

        preds_log = np.log10(np.array(self.preds))
        uncertainties_log = sm.add_constant(np.log10(np.array(self.uncertainties)))

        model = sm.QuantReg(preds_log, uncertainties_log)
        #result = model.fit(q=q, maxiter=2000)
        result = model.fit(q=q)
        intercept, slope = result.params

        a = 10 ** intercept
        b = slope

        self.threshold_func = partial(self.power_law, a=a, b=b)
        self.clear()

    def calc_threshold_func(self, window=50, percentile=50, smoothing=10):
        x_values = np.exp(np.array(self.uncertainties))  # np.exp to get variance instead of s
        y_values = np.array(self.preds)

        y_values, indices = np.unique(y_values, return_index=True)
        x_values = x_values[indices]

        median_x_values = []
        for i in range(len(y_values)):
            window_below = i - window // 2
            window_above = i + window // 2
            if window_below < 0:
                # window_above -= window_below
                window_above = i + 1
                window_below = 0
            elif window_above > len(y_values):
                # window_below -= (window_above - len(y_values))
                window_below = i - 1
                window_above = len(y_values)
            data_in_window = x_values[window_below:window_above]
            # median_x = np.median(data_in_window)
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

        median_x_values = np.append(0, median_x_values)
        y_values = np.append(0, y_values)

        median_x_values = np.append(median_x_values, median_x_values[-1] * 2)
        y_values = np.append(y_values, max_y * 2)

        # interp_function = interp1d(median_x_values, median_y_values, kind='linear', fill_value='extrapolate')
        # spl = CubicSpline(median_x_values, y_values, bc_type='natural')
        # spl = Akima1DInterpolator(median_x_values, y_values)
        self.threshold_func = PchipInterpolator(median_x_values, y_values, extrapolate=True)
        self.clear()

    def evaluate_threshold_func(self, pred, uncertainty):
        pred = np.array(pred.cpu().detach().numpy().flatten())
        threshold = self.threshold_func(pred)

        pred_var = np.exp(uncertainty.cpu().detach().numpy().flatten())
        mask = pred_var < threshold
        return mask

    def clear(self):
        self.preds = []
        self.uncertainties = []