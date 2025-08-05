#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from argparse import ArgumentParser

from general import table_to_numpy, TitledModel, plot_loss_history, custom_objects, HDF5File, MaxPassLayer, feat, load_normalization


def load_data(path, inputs, targets):
    with HDF5File(path, "r") as f:
        data = f["data"]
        offset = f["offset"]
        scale = f["scale"]
    data["weight"] = data["weight"] * scale["weight"] + offset["weight"]

    weight = abs(np.asarray(data["weight"]))
    data_x = [table_to_numpy(data[input]) for input in inputs]
    if isinstance(targets[0], list):
        data_y = []
        for target in targets:
            data_y.append(table_to_numpy(data, target))
    else:
        data_y = table_to_numpy(data, targets)
    return data_x, data_y, weight


def make_model(input_shapes, input_titles, prev_models, num_layers_per_input, num_layers, num_nodes, activation, dropout,
               output_titles, lr, decay):
    if not isinstance(activation, (list, tuple)):
        activation = [activation]
    if not isinstance(num_nodes, (list, tuple)):
        num_nodes = [num_nodes]

    inputs = []
    for input_shape in input_shapes:
        inputs.append(keras.Input(shape=input_shape[1:]))
    xs = []
    input_offset = 0
    for prev_model in prev_models:
        xs.append(prev_model(inputs[input_offset:input_offset + len(prev_model.input_titles)]))
        input_offset += len(prev_model.input_titles)
    xs += inputs[input_offset:]
    xs_next = []
    for i, num_layer_input in enumerate(num_layers_per_input):
        x = xs[i]
        for j in range(num_layer_input):
            act = activation[j % len(activation)]
            nn = num_nodes[j % len(num_nodes)]
            x = keras.layers.Dense(nn, activation=act)(x)
            if dropout:
                x = keras.layers.Dropout(dropout)(x)
        if len(xs[i].shape) == 3:
            x = keras.layers.Flatten()(x)
        xs_next.append(x)
    del xs
    x = keras.layers.Concatenate()(xs_next)

    for i in range(num_layers):
        act = activation[i % len(activation)]
        nn = num_nodes[i % len(num_nodes)]
        x = keras.layers.Dense(nn, activation=act, name=f"reg_tt{i}")(x)
        if dropout:
            x = keras.layers.Dropout(dropout)(x)
    reg_activation = "linear"
    loss = "mean_squared_error"
    if isinstance(output_titles[0], list):
        reg = []
        for output_title in output_titles:
            pname = output_title[0].split("_")[0]
            reg.append(keras.layers.Dense(len(output_title), activation=reg_activation, name=pname)(x))
    else:
        reg = keras.layers.Dense(len(output_titles), activation=reg_activation, name="reg_tt")(x)
    model = TitledModel(input_titles, output_titles, inputs=inputs, outputs=reg)
    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=lr, decay=decay),
        metrics=[],
        weighted_metrics=[])

    return model


if __name__ == "__main__":
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = ArgumentParser()
    parser.add_argument("traindata")
    parser.add_argument("validatedata")
    parser.add_argument("-b", "--modelb", default="model_bcls.hdf5")
    args = parser.parse_args()

    model_b = keras.models.load_model(args.modelb, custom_objects=custom_objects)

    b_out = MaxPassLayer(2)([model_b.output, model_b.inputs[0]])
    model_bbar = TitledModel(
        model_b.input_titles,
        [p + t.split("_")[1] for t in model_b.input_titles[0] for p in ("bot_", "abot_")],
        inputs=model_b.inputs, outputs=[b_out])
    model_bbar.trainable = False

    inputs = model_b.input_titles + [["met_pt", "met_phi", "met_x", "met_y"] + feat("lep") + feat("alep")]
    targets = [
        ["top_x", "top_y", "top_z", "top_mass"],
        ["atop_x", "atop_y", "atop_z", "atop_mass"],
    ]
    train = load_data(args.traindata, inputs, targets)
    validate = load_data(args.validatedata, inputs, targets)
    model_tt = make_model([t.shape for t in train[0]], inputs, [model_bbar], [0, 2], 3, 800, "relu", 0.25, targets, 0.001, 0)
    norm = load_normalization(args.traindata)
    keras.models.save_model(model_tt, "model_tt.hdf5")
    earlystop = keras.callbacks.EarlyStopping(patience=6)
    checkpoint = keras.callbacks.ModelCheckpoint(
        "model_tt.hdf5", save_best_only=True, verbose=1)

    epochs = 500
    batch_size = 2**13
    history = model_tt.fit(
        train[0], train[1], sample_weight=train[2], batch_size=batch_size,
        epochs=epochs, validation_data=validate,
        callbacks=[earlystop, checkpoint])
    plot_loss_history(history, "loss_tt.svg")
