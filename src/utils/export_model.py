import os

import tensorflow as tf


def export_model(model, timestamp, base_path):
    path = os.path.join(base_path, str(timestamp))
    tf.saved_model.save(model, path)


if __name__ == '__main__':
    pass
