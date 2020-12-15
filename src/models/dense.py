import re
import string
from time import time

import tensorflow as tf
import pandas as pd

from src.utils.save_history import save_history
from src.utils.export_model import export_model
from src.utils.read_params import read_params


def custom_read_csv(file_path):
    df = pd.read_csv(file_path)
    sentences = df.iloc[:, 0].to_numpy()
    labels = df.iloc[:, 1].astype('category').cat.codes.to_numpy()
    return sentences, labels


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


def get_vectorize_layer(max_features=10000, sequence_length=250):
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    return vectorize_layer


def build_model(vocab_size, embedding_dim, max_length, units):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D()
    ])
    return model


def build_export_model(vocab_size, embedding_dim, max_length, sentences, lstm_units):
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
