import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from torch.utils.data import Dataset


def fix_missing(df, col, name, na_dict):
    """
    Fill missing data in a column of df with the median, and add a {name}_na column
    which specifies if the data was missing.

    :param df: data frame to fix
    :param col: column of data to fix
    :param name: the name of the new filled column in df
    :param na_dict: a dictionary of values to create na's of and the value to insert.
    :return: fixed data frame

    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, np.NaN, 3], 'col2' : [5, 2, 2]})
    >>> df
       col1 col2
    0     1    5
    1   nan    2
    2     3    2
    >>> fix_missing(df, df['col1'], 'col1', {})
    """
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict


class TabularDataset(Dataset):
    def __init__(self, data, cat_cols=None, output_col=None):
        """
        :param data: pandas data frame
        :param cat_cols: list of strings
            The names of columns containing categorical data.
        :param output_col: string
            The name of the output variable column.
        """
        super(TabularDataset, self).__init__()

        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = data[self.cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]
