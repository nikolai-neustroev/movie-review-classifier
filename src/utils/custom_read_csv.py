import pandas as pd


def custom_read_csv(file_path):
    df = pd.read_csv(file_path)
    sentences = df.iloc[:, 0].to_numpy()
    labels = df.iloc[:, 1].astype('category').cat.codes.to_numpy()
    return sentences, labels