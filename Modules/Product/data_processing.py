import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def clipper(data, data_def=None):
    """clip data based on min_high and max_low in data_def

    Parameters
    ----------
    data : pd.DataFrame
        input data
    data_def : pd.DataFrame or None
        dataset including clip data - default : None
        index : feature names in data
        columns -> min_high & max_low : float or None (for unspecified bound)

    Returns
    ----------
    pd.DataFrame
    """

    data = data.copy()
    cols = data_def.loc[np.logical_or(~pd.isnull(data_def.min_high), ~pd.isnull(data_def.max_low))].index
    for c in cols:
        mn = data_def.max_low[c]
        mx = data_def.min_high[c]
        data[c] = np.clip(data[c].values, mn, mx)
    return data


def age_transformer():
    """ generates sklearn transformer to divide data by 10

    Returns
    -------
    sklearn.preprocessing instance
    """

    def _div_(x):
        return x / 100

    def _prod_(x):
        return x * 100

    return FunctionTransformer(_div_, _prod_, validate=True)


class PipeTransform:
    def __init__(self, preprocessor, fit_dataset):
        """
        Parameters
        ----------
        preprocessor : dict
            map from preprocessor object to list of columns
            { preprocessor_1 : [col set1],...}
            preprocessor should have fit and transform methods
        fit_dataset : pd.DataFrame
        """
        self.processor = []
        self.columns = []
        for p in preprocessor.keys():
            self.processor.append(copy.deepcopy(p))
            self.columns.append(preprocessor[p])
            self.processor[-1].fit(fit_dataset[self.columns[-1]])

    def transform(self, data):
        """
        Parameters
        ----------
        data : pd.DataFrame

        Returns
        ----------
        pd.DataFrame
        """
        new_data = data.copy()
        for p, c in zip(self.processor, self.columns):
            new_data[c] = p.transform(new_data[c])
        return new_data

    def inv_transform(self, data):
        """
        Parameters
        ----------
        data : pd.DataFrame

        Returns
        ----------
        pd.DataFrame
        """
        new_data = data.copy()
        for p, c in zip(self.processor, self.columns):
            new_data[c] = p.inverse_transform(new_data[c])
        return new_data
