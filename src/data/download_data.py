import yaml

import kaggle


def get_data(dataset, path):
	kaggle.api.authenticate()
	kaggle.api.dataset_download_files(dataset, path=path, unzip=True)


if __name__== '__main__':
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    get_data(params['kaggle_dataset'], 'data/raw')
