from alibi_detect.cd import ClassifierDrift
import torch
import torch.nn as nn
import torchvision.models as models
from alibi_detect.saving import save_detector, load_detector
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input

def classifier_drift(X_ref, X_h0, X_c, X_c_names,  save=False, load=False):
    tf.random.set_seed(0)

    model = tf.keras.Sequential(
      [
          Input(shape=(32, 32, 3)),
          Conv2D(8, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(16, 4, strides=2, padding='same', activation=tf.nn.relu),
          Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu),
          Flatten(),
          Dense(2, activation='softmax')
      ]
    )

    cd = ClassifierDrift(X_ref, model, p_val=.05, train_size=.75, epochs=1)

    if load:
        filepath = "pretrained_detector_path"
        cd = load_detector(filepath)

    if save:
        filepath = "detector_save_path"
        save_detector(cd, filepath)

    labels = ['No!', 'Yes!']

    t = timer()
    preds = cd.predict(X_h0)
    dt = timer() - t
    print('No corruption')
    print(f'Drift? {labels[preds["data"]["is_drift"]]}')
    print(f'p-value: {preds["data"]["p_val"]:.3f}')
    print(f'Time (s) {dt:.3f}')

    if isinstance(X_c, list):
        for x, c in zip(X_c, X_c_names):
            t = timer()
            preds = cd.predict(x)
            dt = timer() - t
            print('')
            print(f'Corruption type : {c}')
            print(f'Drift? {labels[preds["data"]["is_drift"]]}')
            print(f'p-value : {preds["data"]["p_val"]:.3f}')
            print(f'Time (s) : {dt:.3f}')
