from time import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils.save_history import save_history
from src.utils.export_model import export_model
from src.utils.read_params import read_params
from src.utils.custom_read_csv import custom_read_csv


def get_tokenizer(training_sentences: np.ndarray, vocab_size: int, oov_tok: str) -> Tokenizer:
    """Returns Tokenizer fitted on given text.

    Parameters
    ----------
    training_sentences : np.ndarray
        Numpy array of texts on which the tokenizer will be fitted.
    vocab_size : int
        Size of the tokenizer vocabulary.
    oov_tok : str
        Token for the out-of-vocabulary words.

    Returns
    -------
    tokenizer : Tokenizer
        Tokenizer fitted on given text.

    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    return tokenizer


def get_padded_seq(
        tokenizer: Tokenizer,
        sentences: np.ndarray,
        max_length: int,
        trunc_type: str
    ) -> np.ndarray:
    """Tokenizes texts and pads sequences to the same length.

    Parameters
    ----------
    tokenizer : Tokenizer
        Tokenizer used to make sequences.
    sentences : np.ndarray
        Numpy array of texts.
    max_length : int
        Maximum length of all sequences.
    trunc_type : str
        Pad either before or after each sequence.

    Returns
    -------
    padded : np.ndarray
        Padded sequence.
    """
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
    return padded


def build_model(
        vocab_size: int,
        embedding_dim: int,
        max_length: int,
        lstm_units: int
    ) -> tf.python.keras.engine.sequential.Sequential:
    """Defining the model architecture.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimension of the dense embedding.
    max_length : int
        Length of input sequences.
    lstm_units : int
        Number of units in LSTM layers.

    Returns
    -------
    model : tf.python.keras.engine.sequential.Sequential
        Sequential model with LSTM layers.

    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    params = read_params()
    timestamp = int(time())

    training_sentences, training_labels = custom_read_csv('data/processed/train.csv')
    tokenizer = get_tokenizer(training_sentences, params['vocab_size'], params['oov_tok'])
    padded = get_padded_seq(
        tokenizer,
        training_sentences,
        max_length=params['max_length'],
        trunc_type=params['trunc_type']
    )

    testing_sentences, testing_labels = custom_read_csv('data/processed/test.csv')
    testing_padded = get_padded_seq(
        tokenizer,
        testing_sentences,
        max_length=params['max_length'],
        trunc_type=params['trunc_type']
    )

    model = build_model(  # TODO: fine-tune the model
        params['vocab_size'],
        params['embedding_dim'],
        params['max_length'],
        params['lstm_units']
    )
    history = model.fit(
        padded,
        training_labels,
        epochs=params['num_epochs'],
        validation_data=(testing_padded, testing_labels)
        # TODO: Add checkpoint and early stopping callbacks
    )

    save_history(history, timestamp, 'data/fit_history/lstm_history')
    export_model(model, timestamp, base_path="models/lstm/")
