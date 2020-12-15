from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.read_params import read_params


def custom_train_test_split(file_path: str, test_size: float, split_seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splitting data into train and test sets.

    Parameters
    ----------
    file_path : str
        Dataset path.
    test_size : float
        Proportion of the dataset to include in the train split.
    split_seed : int
        Seed for reproducible splitting.

    Returns
    -------
    train : pd.DataFrame
        Train set.
    test : pd.DataFrame
        Test set.

    """
    df = pd.read_csv(file_path)
    train, test = train_test_split(df, test_size=test_size, random_state=split_seed)
    return train, test


if __name__ == '__main__':
    params = read_params()

    train_df, test_df = custom_train_test_split(
        params['file_path'],
        params['test_size'],
        params['split_seed']
    )

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
