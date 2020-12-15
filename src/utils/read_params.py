import yaml


def read_params():
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        return params


if __name__ == '__main__':
    pass
