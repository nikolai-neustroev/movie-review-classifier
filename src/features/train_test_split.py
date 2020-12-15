import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.read_params import read_params


def custom_train_test_split(file_path, test_size, split_seed):
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
