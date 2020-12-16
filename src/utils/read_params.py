import yaml


def read_params() -> dict:
    """Reads the params.yaml file, where information
    on data and model's hyper-parameters are stored.

    Returns
    -------
    params : dict
        params.yaml as Python dictionary

    """
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        return params


if __name__ == '__main__':
    pass
