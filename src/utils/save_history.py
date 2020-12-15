import os

import pandas as pd


def save_history(history, timestamp, dir):
    hist_df = pd.DataFrame(history.history)
    # directory = 'data/fit_history/lstm_history'
    if not os.path.exists(dir):
        os.makedirs(dir)
    hist_df.to_json(f'{dir}/{timestamp}.json')


if __name__ == '__main__':
    pass
