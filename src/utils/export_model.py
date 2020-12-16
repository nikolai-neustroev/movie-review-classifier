import os
from typing import NoReturn

import tensorflow as tf


def export_model(
        model: tf.python.keras.engine.sequential.Sequential,
        timestamp: int,
        base_path: str
    ) -> NoReturn:
    """

    Parameters
    ----------
    model : tf.python.keras.engine.sequential.Sequential
        TensorFlow Sequential model to save as protobuf.
    timestamp : int
        Timestamp used as model identifier.
    base_path : str
        Parent directory of the model directory.

    """
    path = os.path.join(base_path, str(timestamp))
    tf.saved_model.save(model, path)


if __name__ == '__main__':
    pass
