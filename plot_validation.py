#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import awkward as ak
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["figure.figsize"] = [6.4 / 1.3, 4.8 / 1.3]
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from argparse import ArgumentParser
import mplhep as hep
import vector

from general import custom_objects, feat, table_to_numpy, load_normalization, SwitchLayer, HDF5File

particle_labels = {
    "bot": r"Bottom quark",
    "abot": r"Bottom antiquark",
    "lep": r"Lepton",
    "alep": r"Antilepton",
    "top": r"Top quark",
    "atop": r"Top antiquark",
    "ttbar": r"$\mathrm{t}\bar{\mathrm{t}}$",
    "wminus": r"$\mathrm{W}^{-}$ boson",
    "wplus": r"$\mathrm{W}^{+}$ boson",
    "met": "MET",
    "jet": "Jet"
}

obs_labels = {
    "pt": r"$p_{\mathrm{T}}$",
    "eta": r"$\eta$",
    "phi": r"$\varphi$",
    "x": r"$p_{x}$",
    "y": r"$p_{y}$",
    "z": r"$p_{z}$",
    "t": r"Energy",
    "mtt": r"$m_{\mathrm{t}\bar{\mathrm{t}}}$",
    "chel": r"$c_{\mathrm{hel}}$",
    "pttt": r"$p_{\mathrm{T},\mathrm{t}\bar{\mathrm{t}}}$",
}

obs_units = {
    "x": "GeV",
    "y": "GeV",
    "z": "GeV",
    "t": "GeV",
    "pt": "GeV",
    "mass": "GeV",
    "mtt": "GeV",
    "pttt": "GeV",
}


def get_label(observable):
    label = ""
    if "_" in observable:
        part, observable = observable.split("_")
        label += particle_labels.get(part, part) + " "
    label += obs_labels.get(observable, observable)
    if observable in obs_units:
        label += f" ({obs_units[observable]})"
    return label


def load_data(path, inputs, targets, norm):
    with HDF5File(path, "r") as f:
        data = f["data"]
        offset = f["offset"]
        scale = f["scale"]
    train_offset = ak.with_name(norm[0], "Dataframe")
    train_scale = ak.with_name(norm[1], "Dataframe")

    if data["sonnentop_t"].ndim > 1:
        data = data[ak.num(data["sonnentop_t"]) == 1]
        for field in ak.fields(data):
            if field.startswith("sonnen"):
                data[field] = ak.flatten(data[field])
    data["weight"] = data["weight"] * scale["weight"] + offset["weight"]

    weight = np.asarray(data["weight"])
    data_x = []
    for input in inputs:
        data_input = ak.with_name(data[input], "Dataframe")
        data_input = (data_input * scale[input] + offset[input] - train_offset[input]) / train_scale[input]
        data_x.append(table_to_numpy(data_input, input))
    data_y = table_to_numpy(data, targets)
    truth = data_y * table_to_numpy(scale, targets) + table_to_numpy(offset, targets)
    truth = pd.DataFrame(truth, columns=targets)
    return data_x, truth, weight


def plot_observable(observable, norm, pred, truth, weight, bins=50, range=None, sonnen=None):
    weight = np.asarray(weight)
    if range is None:
        offset = norm[0][observable]
        scale = norm[1][observable]
        range = (offset - 1.5 * scale, offset + 1.5 * scale)
        if observable == "top_mass":
            range = (169, 177)
    is_good = pred < 1e30
    mse = np.mean((pred[is_good] - truth[is_good]) ** 2)
    plt.hist(
        np.asarray(pred),
        weights=weight,
        range=range,
        bins=50,
        label=f"Network (MSE: {mse:.3})",
        facecolor=to_rgb("C0") + (0.5,),
        edgecolor="C0")
    plt.hist(
        np.asarray(truth),
        weights=weight,
        range=range,
        bins=50,
        label="Truth",
        facecolor=to_rgb("C1") + (0.5,),
        edgecolor="C1")
    if sonnen is not None:
        sonnen_mse = np.mean((sonnen - truth) ** 2)
        plt.hist(
            np.asarray(sonnen),
            weights=weight,
            range=range,
            bins=50,
            label=f"Analytic (MSE: {sonnen_mse:.3})",
            facecolor=to_rgb("C2") + (0.5,),
            edgecolor="C2")
    plt.xlabel(get_label(observable))
    plt.ylabel("Events / bin")
    plt.margins(x=0)
    plt.legend()
    hep.cms.label(llabel="Private work (CMS simulation)", exp="", rlabel="", loc=0, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"predtruth_{observable}.svg"))
    plt.close()


def make_lorentzvec(prefix, truth, targets):
    vec = {}
    for p in ["x", "y", "z", "t"]:
        name = f"{prefix}_{p}"
        if p == "t":
            p = "E"
        vec[p] = np.asarray(truth[name])
    return vector.obj(**vec)


def compute_chel(top, atop, lep, alep):
    tt = top + atop
    ttboost = -tt.to_beta3()
    top_tt = top.boost(ttboost)
    atop_tt = atop.boost(ttboost)
    lep_tt = lep.boost(ttboost)
    alep_tt = alep.boost(ttboost)
    lep_hel = lep_tt.boost(-atop_tt.to_beta3())
    alep_hel = alep_tt.boost(-top_tt.to_beta3())
    chel = lep_hel.to_xyz().dot(alep_hel.to_xyz()) / lep_hel.mag / alep_hel.mag
    return chel


def featc(particle):
    return [f"{particle}_{x}" for x in ["x", "y", "z", "t"]]


def binned_mean(x, values, edges, idx=None, counts=None, return_unc=True):
    if idx is None:
        idx = np.digitize(x, edges) - 1
    if counts is None:
        counts = np.bincount(idx)
    bincounts = np.bincount(idx, weights=values, minlength=len(edges) - 1)
    mean = np.divide(bincounts, counts, where=counts != 0, out=np.zeros_like(counts, dtype=float))
    if return_unc:
        diff = values - mean[idx]
        diff_up = diff[diff > 0]
        diff_down = diff[diff < 0]
        one_sigma = 0.6827
        if len(diff_up) > 0:
            up = np.quantile(diff_up, one_sigma)
        else:
            up = 0
        if len(diff_down) > 0:
            down = np.quantile(abs(diff_down), one_sigma)
        else:
            down = 0
        return mean, mean + up, mean - down
    else:
        return mean


def plot_resbias(observable, norm, pred, truth, bins=50, range=None, sonnen=None, relative=None):
    xlabel = get_label(observable)
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    if range is None:
        offset = norm[0][observable]
        scale = norm[1][observable]
        if scale < 1e-10:
            return
        range = (offset - 1.5 * scale, offset + 1.5 * scale)
    if relative is None:
        relative = not (range[0] <= 0 and 0 <= range[1])
    edges = np.linspace(range[0], range[1], bins)
    is_inrange = (edges[0] <= truth) & (truth < edges[-1])
    truth = truth[is_inrange]
    pred = pred[is_inrange]
    idx = np.digitize(truth, edges) - 1
    counts = np.bincount(idx, minlength=bins - 1)
    if relative:
        err = (pred - truth) / truth
    else:
        err = pred - truth

    resp, resp_up, resp_down = binned_mean(truth, pred, edges, idx, counts)
    bias, bias_up, bias_down = binned_mean(truth, err, edges, idx, counts)
    resolution = np.sqrt(binned_mean(truth, err ** 2, edges, idx, counts, return_unc=False) - bias ** 2)

    if sonnen is not None:
        sonnen = np.asarray(sonnen)[is_inrange]
        if relative:
            err_son = (sonnen - truth) / truth
        else:
            err_son = sonnen - truth
        resp_son, resp_up_son, resp_down_son = binned_mean(truth, sonnen, edges, idx, counts)
        bias_son, bias_up_son, bias_down_son = binned_mean(truth, err_son, edges, idx, counts)
        resolution_son = np.sqrt(binned_mean(truth, err_son ** 2, edges, idx, counts, return_unc=False) - bias_son ** 2)

    line = plt.step(edges, np.r_[resp[0], resp], label="Network")[0]
    plt.fill_between(edges, np.r_[resp_down[0], resp_down], np.r_[resp_up[0], resp_up], step="pre", color=line.get_color(), alpha=0.5)
    if sonnen is not None:
        line = plt.step(edges, np.r_[resp_son[0], resp_son], label="Analytic")[0]
        plt.fill_between(edges, np.r_[resp_down_son[0], resp_down_son], np.r_[resp_up_son[0], resp_up_son], step="pre", color=line.get_color(), alpha=0.5)
        plt.legend()
    plt.xlabel(f"True {xlabel}")
    if observable in obs_units:
        plt.ylabel(f"Response ({obs_units[observable]})")
    else:
        plt.ylabel("Response")
    plt.grid()
    plt.margins(x=0)
    hep.cms.label(llabel="Private work (CMS simulation)", exp="", rlabel="", loc=0, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"response_{observable}.svg"))
    plt.close()

    overall = np.mean(np.mean(abs(err)))
    line = plt.step(edges, np.r_[bias[0], bias], label=f"Network (mean abs {overall:.3})")[0]
    plt.fill_between(edges, np.r_[bias_down[0], bias_down], np.r_[bias_up[0], bias_up], step="pre", color=line.get_color(), alpha=0.5)
    if sonnen is not None:
        overall = np.mean(np.mean(abs(err_son)))
        line = plt.step(edges, np.r_[bias_son[0], bias_son], label=f"Analytic (mean abs {overall:.3})")[0]
        plt.fill_between(edges, np.r_[bias_down_son[0], bias_down_son], np.r_[bias_up_son[0], bias_up_son], step="pre", color=line.get_color(), alpha=0.5)
        plt.legend()
    if plt.ylim()[0] < -2:
        plt.ylim(-2, plt.ylim()[1])
    if plt.ylim()[1] > 2:
        plt.ylim(plt.ylim()[0], 2)
    plt.xlabel(f"True {xlabel}")
    if relative:
        plt.ylabel("Relative bias")
    else:
        plt.ylabel("Bias")
    plt.grid()
    plt.margins(x=0)
    hep.cms.label(llabel="Private work (CMS simulation)", exp="", rlabel="", loc=0, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"bias_{observable}.svg"))
    plt.close()

    overall = np.sqrt(np.mean(err ** 2))
    plt.step(edges, np.r_[resolution[0], resolution], label=f"Network (mean {overall:.3})")
    if sonnen is not None:
        overall = np.sqrt(np.mean(err_son ** 2))
        plt.step(edges, np.r_[resolution_son[0], resolution_son], label=f"Analytic (mean {overall:.3})")
        plt.legend()
    plt.ylim(0, plt.ylim()[1])
    if plt.ylim()[1] > 2:
        plt.ylim(0, 2)
    plt.xlabel(f"True {xlabel}")
    if relative:
        plt.ylabel("Resolution")
    else:
        plt.ylabel("Standard deviation")
    plt.grid()
    plt.margins(x=0)
    hep.cms.label(llabel="Private work (CMS simulation)", exp="", rlabel="", loc=0, fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"resolution_{observable}.svg"))
    plt.close()


def switch_charge_label(label):
    if label.startswith("alep_"):
        return "lep_" + label[len("alep_"):]
    elif label.startswith("lep_"):
        return "alep_" + label[len("lep_"):]
    elif label.startswith("abot_"):
        return "bot_" + label[len("abot_"):]
    elif label.startswith("bot_"):
        return "abot_" + label[len("bot_"):]
    elif label.startswith("wminus_"):
        return "wplus_" + label[len("wminus_"):]
    elif label.startswith("wplus_"):
        return "wminus_" + label[len("wplus_"):]
    elif label.startswith("atop_"):
        return "top_" + label[len("atop_"):]
    elif label.startswith("top_"):
        return "atop_" + label[len("top_"):]
    else:
        return label


def find_switch(model):
    for layer in model.layers:
        if hasattr(layer, "layers"):
            try:
                return find_switch(layer)
            except ValueError:
                pass
        elif isinstance(layer, SwitchLayer):
            return layer
    else:
        raise ValueError("No switch in model")


def create_particle(prediction, model, index, norm, switch=False, mass=None):
    part = {}
    for i, title in enumerate(model.output_titles[index]):
        p = title.split("_")[-1]
        if switch:
            idx = model.output_titles[index].index(switch_charge_label(title))
        else:
            idx = model.output_titles[index].index(title)
        if p == "t":
            p = "E"
        part[p] = prediction[index][:, idx] * norm[1][title] + norm[0][title]
    if mass is not None:
        part["mass"] = mass
    part = vector.obj(**part)
    return part


def plot_particle(pred, particle_name, norm, truth, weight, sonnenpart=None):
    for observable in feat(particle_name):
        predval = getattr(pred, observable[len(f"{particle_name}_"):])
        if sonnenpart is not None:
            sonnen = getattr(sonnenpart, observable[len(f"{particle_name}_"):])
        else:
            sonnen = None
        plot_observable(observable, norm, predval, truth[observable], weight, sonnen=sonnen)
        plot_resbias(observable, norm, predval, truth[observable], sonnen=sonnen)


def get_partidx(model, particle):
    titles = model.output_titles
    return [title[0].split("_")[0] for title in titles].index(particle)


for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

parser = ArgumentParser()
parser.add_argument("traindata")
parser.add_argument("validatedata")
parser.add_argument("--model_tt", default="model_tt.hdf5")
parser.add_argument("-o", "--output")
args = parser.parse_args()
if args.output is None:
    args.output = os.path.basename(os.path.dirname(args.validatedata))

norm = load_normalization(args.traindata)
validnorm = load_normalization(args.validatedata)
model = keras.models.load_model(args.model_tt, custom_objects=custom_objects)

targets = feat("top") + feat("atop") +\
          feat("wplus") + feat("wminus") +\
          feat("lep") + feat("alep") +\
          feat("genlep") + feat("genalep") +\
          feat("genbot") + feat("genabot") +\
          featc("sonnentop") + featc("sonnenatop") + feat("ttbar") + ["mtt", "chel"]

data_x, truth, weight = load_data(args.validatedata, model.input_titles, targets, norm)

lep = make_lorentzvec("lep", truth, targets)
alep = make_lorentzvec("alep", truth, targets)
sonnentop = make_lorentzvec("sonnentop", truth, targets)
sonnenatop = make_lorentzvec("sonnenatop", truth, targets)

os.makedirs(args.output, exist_ok=True)

prediction = model.predict(data_x, batch_size=2**14)
top_pred = create_particle(prediction, model, get_partidx(model, "top"), norm)
atop_pred = create_particle(prediction, model, get_partidx(model, "atop"), norm)
ttbar_pred = top_pred + atop_pred
atop_pred = ttbar_pred - top_pred
plot_particle(top_pred, "top", validnorm, truth, weight, sonnentop)
plot_particle(atop_pred, "atop", validnorm, truth, weight, sonnenatop)
plot_particle(ttbar_pred, "ttbar", validnorm, truth, weight, sonnentop + sonnenatop)


mtt_pred = ttbar_pred.mass
mtt_sonnen = (sonnentop + sonnenatop).mass
plot_observable("mtt", validnorm, mtt_pred, truth["mtt"], weight, sonnen=mtt_sonnen, range=(300, 1200))
plot_resbias("mtt", validnorm, mtt_pred, truth["mtt"], sonnen=mtt_sonnen, range=(300, 1200))

top = make_lorentzvec("top", truth, targets)
atop = make_lorentzvec("atop", truth, targets)
genlep = make_lorentzvec("genlep", truth, targets)
genalep = make_lorentzvec("genalep", truth, targets)
chel = compute_chel(top, atop, genlep, genalep)
chel_sonnen = compute_chel(sonnentop, sonnenatop, lep, alep)
pred = compute_chel(top_pred, atop_pred, lep, alep)
plot_observable("chel", validnorm, pred, chel, weight, range=(-1, 1), sonnen=chel_sonnen)
plot_resbias("chel", validnorm, pred, chel, sonnen=chel_sonnen, range=(-1, 1), relative=False)

dalpha = top.to_xyz().dot(atop.to_xyz()) / top.mag / atop.mag
dalpha_pred = top_pred.to_xyz().dot(atop_pred.to_xyz()) / top_pred.mag / atop_pred.mag
dalpha_sonnen = sonnentop.to_xyz().dot(sonnenatop.to_xyz()) / sonnentop.mag / sonnenatop.mag
plot_observable("dalpha", validnorm, dalpha_pred, dalpha, weight, range=(-1, 1), sonnen=dalpha_sonnen)
plot_resbias("dalpha", validnorm, dalpha_pred, dalpha, sonnen=dalpha_sonnen, range=(-1, 1), relative=False)
