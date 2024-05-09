#!/usr/bin/env python3

import os
import numpy as np
import awkward as ak
import vector
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # "3"
import tensorflow as tf
import keras
import mplhep as hep
from argparse import ArgumentParser

from general import custom_objects, table_to_numpy, HDF5File, feat, load_normalization


def load_data(path, input_titles, validate_titles, norm):
    with HDF5File(path, "r") as f:
        data = f["data"]
        offset = f["offset"]
        scale = f["scale"]
    train_offset = ak.with_name(norm[0], "Dataframe")
    train_scale = ak.with_name(norm[1], "Dataframe")
    flav = data["jet_flav"] * scale["jet_flav"] + offset["jet_flav"]
    data["jet_flav"] = ak.values_astype(flav + 0.5 * np.sign(flav), int)
    data = data[(ak.sum(data["jet_flav"] == 5, axis=1) == 1) & (ak.sum(data["jet_flav"] == -5, axis=1) == 1)]
    data["weight"] = data["weight"] * scale["weight"] + offset["weight"]

    flav = np.asarray(ak.fill_none(ak.pad_none(data["jet_flav"], ak.max(ak.num(data["jet_flav"]))), 0))
    label = ((flav == 5)[:, :, None] & (flav == -5)[:, None, :]).astype(float).reshape((-1, flav.shape[1] ** 2))
    weight = abs(np.asarray(data["weight"]))
    data_x = []
    for input in input_titles:
        data_input = ak.with_name(data[input], "Dataframe")
        data_input = (data_input * scale[input] + offset[input] - train_offset[input]) / train_scale[input]
        data_x.append(table_to_numpy(data_input, input))
    validate = ak.with_name(data[validate_titles], "Dataframe") * scale[validate_titles] + offset[validate_titles]
    return data_x, label, weight, validate


def plot_correctness(observable, bins, range, pred_true, mlb_true, found_mlb_jets, xlabel, ylabel, histname):
    hist, edges = np.histogram(observable, bins=bins, range=range)
    pred_hist, edges = np.histogram(observable, bins=bins, range=range, weights=pred_true.astype(float))
    pred_hist /= hist / 100
    mlb_hist, edges = np.histogram(observable[found_mlb_jets], bins=bins, range=range, weights=mlb_true.astype(float))
    mlb_hist /= hist / 100

    plt.step(edges, np.r_[pred_hist[0], pred_hist], label=f"Neural Network, mean: {np.mean(pred_true) * 100:.3}%")
    plt.step(edges, np.r_[mlb_hist[0], mlb_hist], label=f"$m_{{\\mathrm{{lb}}}}$ method, mean: {np.mean(mlb_true) * 100:.3}%")
    plt.margins(x=0)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    hep.cms.label(llabel="Private work (CMS simulation)", exp="", rlabel="", loc=0, fontsize=13)
    plt.gcf().set_size_inches(plt.gcf().get_size_inches() / 1.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "bcls_" + histname + ".svg"))
    plt.close()


def make_vector(data, particle):
    return vector.obj(**{part: data[f"{particle}_{part}"] for part in ["pt", "eta", "phi", "mass"]})


for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

parser = ArgumentParser()
parser.add_argument("traindata")
parser.add_argument("validatedata")
parser.add_argument("--model_bcls", default="model_bcls.hdf5")
parser.add_argument("-o", "--output")
args = parser.parse_args()
if args.output is None:
    args.output = os.path.basename(os.path.dirname(args.validatedata))

norm = load_normalization(args.traindata)
os.makedirs(args.output, exist_ok=True)

model = keras.models.load_model(args.model_bcls, custom_objects=custom_objects)
data_x, label, weight, validate = load_data(
    args.validatedata, model.input_titles,
    ["met_pt", "bot_pt", "abot_pt", "jet_pt"] + feat("lep", "pt") + feat("alep", "pt"),
    norm)
lep = make_vector(validate, "lep")
alep = make_vector(validate, "alep")
mll = (lep + alep).mass
deltarll = lep.deltaR(alep)

prediction = model.predict(data_x, batch_size=2**14)
pred_true = np.all((np.max(prediction, axis=1, keepdims=True) == prediction) == label, axis=1)

is_mlb_bot = abs(validate["jet_pt"] - validate["bot_pt"]) < 1e-8
is_mlb_abot = abs(validate["jet_pt"] - validate["abot_pt"]) < 1e-8
found_mlb_jets = (ak.sum(is_mlb_bot, axis=1) == 1) & (ak.sum(is_mlb_abot, axis=1) == 1)
num_jets = ak.max(ak.num(validate["jet_pt"]))
is_mlb_bot = np.asarray(ak.fill_none(ak.pad_none(is_mlb_bot[found_mlb_jets], num_jets), False))
is_mlb_abot = np.asarray(ak.fill_none(ak.pad_none(is_mlb_abot[found_mlb_jets], num_jets), False))
mlb_true = np.all((is_mlb_bot[:, :, None] & is_mlb_abot[:, None, :]).reshape(-1, num_jets ** 2) == label[found_mlb_jets], axis=1)

plot_correctness(np.asarray(validate["met_pt"]), 40, (0, 200), pred_true, mlb_true, found_mlb_jets, r"MET $p_{\mathrm{T}}$ (GeV)", "Correct / 5 GeV (percent)", "met_pt")
plot_correctness(np.asarray(mll), 40, (0, 200), pred_true, mlb_true, found_mlb_jets, r"$m_{\mathrm{ll}}$ (GeV)", "Correct / 5 GeV (percent)", "mll")
plot_correctness(np.asarray(deltarll), 50, (0, 5), pred_true, mlb_true, found_mlb_jets, r"$\Delta R_{\mathrm{ll}}$", "Correct / 0.1 units (percent)", "deltarll")
