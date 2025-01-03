from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from alibi_detect.cd import MMDDrift
from alibi_detect.models.tensorflow import scale_by_instance
from alibi_detect.utils.fetching import fetch_tf_model
from alibi_detect.saving import save_detector, load_detector
from alibi_detect.datasets import fetch_cifar10c, corruption_types_cifar10c
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer, Reshape
from alibi_detect.cd.tensorflow import preprocess_drift

from timeit import default_timer as timer

# import torch
# import torch.nn as nn
from alibi_detect.cd.pytorch import preprocess_drift

def mmd_drift(image_by_type, X_c, X_c_names, save=False, load=False):
    tf.random.set_seed(0)

    # define encoder
    encoding_dim = 32
    encoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(32, 32, 3)),
            Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
            Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
            Flatten(),
            Dense(encoding_dim, )
        ]
    )

    # BBSDs : Black-Box Shift Detection. Based on Detecting and correcting for label shift w/ black pox predictors
    # X_ref_bbsds = scale_by_instance(image_by_type[0])
    # X_h0_bbsds = scale_by_instance(image_by_type[1])
    # X_c_bbsds = scale_by_instance(X_c)

    # define preprocessing function
    preprocess_fn = partial(preprocess_drift, model=encoder_net, batch_size=512)

    # initialise drift detector
    cd = MMDDrift(image_by_type[0], backend='tensorflow', p_val=.05,
                  preprocess_fn=preprocess_fn, n_permutations=100)

    if load:
        filepath = "pretrained_detector_path"
        cd = load_detector(filepath)

    if save:
        filepath = "detector_save_path"
        save_detector(cd, filepath)

    labels = ['No!', 'Yes!']

    t = timer()
    preds = cd.predict(image_by_type[1])
    dt = timer() - t
    print('No corruption')
    print('Drift? {}'.format(labels[preds['data']['is_drift']]))
    print(f'p-value: {preds["data"]["p_val"]:.3f}')
    print(f'Time (s) {dt:.3f}')

    if isinstance(X_c, list):
        for x, c in zip(X_c, X_c_names):
            t = timer()
            preds = cd.predict(x)
            dt = timer() - t
            print('')
            print(f'Corruption type: {c}')
            print('Drift? {}'.format(labels[preds['data']['is_drift']]))
            print(f'p-value: {preds["data"]["p_val"]:.3f}')
            print(f'Time (s) {dt:.3f}')
