from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
dataset = 'gestures.csv'
model_save_path = 'gestures_classifier_test.hdf5'
tflite_save_path = 'gestures_classifier_test.tflite'
NUM_CLASSES = 3


def prepare_data():
    x_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32',
                           usecols=list(range(1, (21 * 2) + 1)), skiprows=1)
    for x_row in x_dataset:
        first_x = x_row[0]
        first_y = x_row[1]
        x_row[0::2] -= first_x
        x_row[1::2] -= first_y
    y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32',
                           usecols=0, skiprows=1)
    y_dataset -= 1
    return x_dataset, y_dataset


def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))


def update_model():
    x_dataset, y_dataset = prepare_data()
    x_train, x_test, y_train, y_test = \
        train_test_split(x_dataset, y_dataset,
                         train_size=0.75, random_state=RANDOM_SEED)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=False
    )
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    begin = datetime.now()
    model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback, es_callback]
    )
    end = datetime.now()

    print(f'Time: {end - begin}')

    val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)
    Y_pred = model.predict(x_test)
    print_confusion_matrix(y_test, np.argmax(Y_pred, axis=1))
    model.save(model_save_path, include_optimizer=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    open(tflite_save_path, 'wb').write(tflite_quantized_model)


if __name__ == '__main__':
    update_model()
