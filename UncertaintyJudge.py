import statsmodels.api as sm
import numpy as np
from functools import partial

class UncertaintyJudge:
    def __init__(self):
        self.preds = []
        self.uncertainties = []
        self.threshold_func = None

    @staticmethod
    def power_law(x, a, b):
        return a * x ** (b)

    def add_pred_var_pair(self, pred, uncertainty):
        self.preds.append(pred.cpu().detach().numpy())
        self.uncertainties.append(uncertainty.cpu().detach().numpy())

    def calc_threshold_func(self, q=0.6):
        """
        Fit a quantile regression model with a quantile value of 0.6.
        Meaning 60% of the datapoints are larger than what the function predicts.
        """

        preds_log = np.log10(np.array(self.preds))
        uncertainties_log = sm.add_constant(np.log10(np.array(self.uncertainties)))

        model = sm.QuantReg(preds_log, uncertainties_log)
        result = model.fit(q=q)
        intercept, slope = result.params

        a = 10 ** intercept
        b = slope

        self.threshold_func = partial(self.power_law, a=a, b=b)
        self.clear()

    def evaluate_threshold_func(self, pred, uncertainty):
        threshold = self.threshold_func(pred)
        mask = uncertainty < threshold
        return mask

    def clear(self):
        self.preds = []
        self.uncertainties = []