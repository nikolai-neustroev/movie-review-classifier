import os

import tensorflow as tf
import pandas as pd
import yaml


def read_params():
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
        return params


def export_model(model, timestamp, base_path):
    path = os.path.join(base_path, str(timestamp))
    tf.saved_model.save(model, path)


def save_history(history, timestamp, dir):
    hist_df = pd.DataFrame(history.history)
    # directory = 'data/fit_history/lstm_history'
    if not os.path.exists(dir):
        os.makedirs(dir)
    hist_df.to_json(f'{dir}/{timestamp}.json')


if __name__ == '__main__':
    pass
