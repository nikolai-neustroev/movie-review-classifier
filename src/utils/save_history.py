import os
from typing import NoReturn

import pandas as pd
import tensorflow as tf


def save_history(history: tf.keras.callbacks.History, timestamp: int, dir: str) -> NoReturn:
    """Saves model fit history as JSON file.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Model fit history as History object.
    timestamp : int
        Timestamp used as model identifier.
    dir : str
        Directory where JSON file will be saved.

    """
    hist_df = pd.DataFrame(history.history)
    # directory = 'data/fit_history/lstm_history'
    if not os.path.exists(dir):
        os.makedirs(dir)
    hist_df.to_json(f'{dir}/{timestamp}.json')


if __name__ == '__main__':
    pass
