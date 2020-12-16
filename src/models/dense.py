import re
import string
from time import time

import numpy as np
import tensorflow as tf

from src.utils.save_history import save_history
from src.utils.export_model import export_model
from src.utils.read_params import read_params
from src.utils.custom_read_csv import custom_read_csv


def custom_standardization(input_data: tf.Tensor) -> tf.Tensor:
    """Standardization preprocessing pipeline for input data.

    Parameters
    ----------
    input_data : tf.Tensor
        Raw input data.

    Returns
    -------
    tf.Tensor
        Preprocessed input data.

    """
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


def get_vectorize_layer(max_features=10000, sequence_length=250) \
        -> tf.keras.layers.experimental.preprocessing.TextVectorization:
    """Transforms a batch of strings into either a list of token indices or a dense representation.

    Parameters
    ----------
    max_features : int
        The maximum size of the vocabulary for this layer.
    sequence_length : int
        Dimension padded or truncated to exactly sequence_length values.

    Returns
    -------
    vectorize_layer : tf.keras.layers.experimental.preprocessing.TextVectorization
        Text vectorization layer.

    """
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    return vectorize_layer


def build_model(vocab_size: int, embedding_dim: int, max_length: int, units: int) -> tf.keras.Sequential:
    """Builds core model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the dense embedding.
    max_length : int
        Length of input sequences.
    units : int
        Number of units in Dense layers.

    Returns
    -------
    model : tf.python.keras.engine.sequential.Sequential
        Sequential model with Dense layers.

    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D()
    ])
    return model


def build_export_model(
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        sentences: np.ndarray,
        lstm_units: int
    ) -> tf.keras.Sequential:
    """Builds model with input and output layers.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the dense embedding.
    max_length : int
        Length of input sequences.
    sentences : np.ndarray
        Data to adapt the input layer.
    lstm_units : int
        Number of units in Dense layers.

    Returns
    -------

    """
    vectorize_layer = get_vectorize_layer(vocab_size, max_length)
    vectorize_layer.adapt(sentences)
    model = build_model(vocab_size, embedding_dim, max_length, lstm_units)
    export_model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer,
        model,
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    export_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return export_model


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    params = read_params()
    timestamp = int(time())

    training_sentences, training_labels = custom_read_csv('data/processed/train.csv')
    model = build_export_model(
        params['vocab_size'],
        params['embedding_dim'],
        params['max_length'],
        training_sentences,
        params['dense_units']
    )

    testing_sentences, testing_labels = custom_read_csv('data/processed/test.csv')

    history = model.fit(
        training_sentences,
        training_labels,
        epochs=params['num_epochs'],
        validation_data=(testing_sentences, testing_labels)
        # TODO: Add checkpoint and early stopping callbacks
    )

    save_history(history, timestamp, 'data/fit_history/dense_history')
    export_model(model, timestamp, base_path="models/dense/")
