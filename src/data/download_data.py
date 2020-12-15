import kaggle

from src.utils.read_params import read_params


def get_data(dataset, path):
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=path, unzip=True)


if __name__ == '__main__':
    params = read_params()
    get_data(params['kaggle_dataset'], 'data/raw')
