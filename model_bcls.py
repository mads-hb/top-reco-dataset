#!/usr/bin/env python3

import os
import numpy as np
import awkward as ak
from argparse import ArgumentParser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # "3"
import tensorflow as tf
from tensorflow import keras
from general import feat, table_to_numpy, accuracy_max, TitledModel, plot_loss_history, HDF5File


def load_data(path, *args):
    with HDF5File(path, "r") as f:
        data = f["data"]
        offset = f["offset"]
        scale = f["scale"]
    flav = data["jet_flav"] * scale["jet_flav"] + offset["jet_flav"]
    data["jet_flav"] = ak.values_astype(flav + 0.5 * np.sign(flav), int)
    data = data[(ak.sum(data["jet_flav"] == 5, axis=1) == 1) & (ak.sum(data["jet_flav"] == -5, axis=1) == 1)]
    data["weight"] = data["weight"] * scale["weight"] + offset["weight"]

    flav = np.asarray(ak.fill_none(ak.pad_none(data["jet_flav"], ak.max(ak.num(data["jet_flav"]))), 0))
    label = ((flav == 5)[:, :, None] & (flav == -5)[:, None, :]).astype(float).reshape((-1, flav.shape[1] ** 2))
    weight = abs(np.asarray(data["weight"]))
    data = [table_to_numpy(data, arg) for arg in args]
    return data, label, [weight]*len(label) if isinstance(label, list) else weight


def make_model(input_shapes, input_titles, num_layers_per_input, num_layers, num_nodes, activation, dropout,
               num_outputs, output_titles, lr, decay):
    if not isinstance(activation, (list, tuple)):
        activation = [activation]
    if not isinstance(num_nodes, (list, tuple)):
        num_nodes = [num_nodes]

    inputs = []
    for input_shape in input_shapes:
        inputs.append(keras.Input(shape=input_shape[1:]))
    xs = []
    for i, num_layer_input in enumerate(num_layers_per_input):
        x = inputs[i]
        for j in range(num_layer_input):
            act = activation[j % len(activation)]
            nn = num_nodes[j % len(num_nodes)]
            x = keras.layers.Dense(nn, activation=act)(x)
            if dropout:
                x = keras.layers.Dropout(dropout)(x)
        if len(inputs[i].shape) == 3:
            x = keras.layers.Flatten()(x)
        xs.append(x)
    x = keras.layers.Concatenate()(xs)

    for i in range(num_layers):
        act = activation[i % len(activation)]
        nn = num_nodes[i % len(num_nodes)]
        x = keras.layers.Dense(nn, activation=act)(x)
        if dropout:
            x = keras.layers.Dropout(dropout)(x)
    if num_outputs == 1:
        cls_activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        cls_activation = "softmax"
        loss = "categorical_crossentropy"
    cls = keras.layers.Dense(num_outputs, activation=cls_activation, name="cls")(x)
    model = TitledModel(inputs=inputs, outputs=[cls], input_titles=input_titles, output_titles=output_titles)
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=lr, decay=decay),
        metrics=[],
        weighted_metrics=[accuracy_max]
    )

    return model


def load_normalization(path):
    with HDF5File(path, "r") as f:
        return ak.to_list(f["offset"])[0], ak.to_list(f["scale"])[0]


if __name__ == "__main__":
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = ArgumentParser()
    parser.add_argument("traindata")
    parser.add_argument("validatedata")
    args = parser.parse_args()

    features1 = feat("jet") + ["jet_btag"]
    features2 = feat("alep") + feat("lep") + ["met_pt", "met_phi", "met_x", "met_y"]
    train = load_data(args.traindata, features1, features2)
    validate = load_data(args.validatedata, features1, features2)
    model = make_model([t.shape for t in train[0]], [features1, features2], [2, 0], 2, 200, "relu", 0.25, train[1].shape[1], [], 0.0003, 0)
    print(f"Got {train[0][0].shape[0]} events, {train[0][0].shape[1]} jets and {train[1].shape[1]} outputs")
    earlystop = keras.callbacks.EarlyStopping(patience=10, monitor="val_loss", mode="min")
    checkpoint = keras.callbacks.ModelCheckpoint("model_bcls.hdf5", save_best_only=True, verbose=1)

    epochs = 500
    batch_size = 2**13
    history = model.fit(
        train[0], train[1], sample_weight=train[2], batch_size=batch_size,
        epochs=epochs, validation_data=validate,
        callbacks=[earlystop, checkpoint])
    plot_loss_history(history, "loss_bcls.svg")
