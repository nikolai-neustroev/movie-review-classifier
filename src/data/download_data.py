from typing import NoReturn

import kaggle

from src.utils.read_params import read_params


def get_data(dataset: str, path: str) -> NoReturn:
    """Authenticate and download Kaggle dataset.

    Parameters
    ----------
    dataset : str
        Kaggle dataset name in "username/dataset_name" format.
    path : str
        Directory path.

    """
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=path, unzip=True)


if __name__ == '__main__':
    params = read_params()
    get_data(params['kaggle_dataset'], 'data/raw')
