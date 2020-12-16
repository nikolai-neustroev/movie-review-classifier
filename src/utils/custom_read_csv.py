from typing import Tuple

import numpy as np
import pandas as pd


def custom_read_csv(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Splits two-column Pandas DataFrame into two numpy arrays.

    Parameters
    ----------
    file_path : str
        csv file to read as Pandas DataFrame.

    Returns
    -------
    sentences : np.ndarray
        Column 0 as numpy array.
    labels : np.ndarray
        Column 1 as numpy array. The elements converted to int.

    """
    df = pd.read_csv(file_path)
    sentences = df.iloc[:, 0].to_numpy()
    labels = df.iloc[:, 1].astype('category').cat.codes.to_numpy()
    return sentences, labels
