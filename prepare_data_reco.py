#!/usr/bin/env python3

import os
import sys
import glob
from pathlib import Path
import numpy as np
import awkward as ak
from pepper import HDF5File
from argparse import ArgumentParser
from coffea.nanoevents.methods import vector
from tqdm import tqdm
import warnings
from collections import defaultdict


warnings.filterwarnings("error")
ak.behavior.update(vector.behavior)

RANDOM_SEED = 42

@ak.mixin_class(ak.behavior)
class Dataframe:
    @ak.mixin_class_method(np.ufunc)
    def ufunc(ufunc, method, args, kwargs):  # noqa: N805
        fields = set(ak.fields(args[0]))
        for i in range(1, len(args)):
            if np.isscalar(args[i]):
                args[i] = ak.full_like(args[0], args[i])
            fields &= set(ak.fields(args[i]))
        out = {}
        func = getattr(ufunc, method)
        for field in fields:
            out[field] = func(*[arg[field] for arg in args], **kwargs)
        return ak.Array(out, with_name="Dataframe")

    def _runak(self, func, *args, **kwargs):
        out = {}
        for field in ak.fields(self):
            out[field] = [func(self[field], *args, **kwargs)]
        return ak.Array(out, with_name="Dataframe")

    def mean(self, *args, **kwargs):
        return self._runak(ak.mean, *args, **kwargs)

    def std(self, *args, **kwargs):
        return self._runak(ak.std, *args, **kwargs)



def save_output(path, data, inputdirs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with HDF5File(str(path), "w") as f:
        f["data"] = data
        f["inputdirs"] = str(inputdirs.resolve())


def add_particle(table, prefix, vector, system):
    if system == "cartesian":
        table[f"{prefix}_x"] = vector.x
        table[f"{prefix}_y"] = vector.y
        table[f"{prefix}_z"] = vector.z
        table[f"{prefix}_t"] = vector.t
    elif system == "ptetaphim":
        table[f"{prefix}_pt"] = vector.pt
        table[f"{prefix}_eta"] = vector.eta
        table[f"{prefix}_phi"] = vector.phi
        table[f"{prefix}_mass"] = vector.mass
    elif system == "ptetaphi":
        table[f"{prefix}_pt"] = vector.pt
        table[f"{prefix}_eta"] = vector.eta
        table[f"{prefix}_phi"] = vector.phi
    elif system == "cartesian_transverse":
        table[f"{prefix}_x"] = vector.x
        table[f"{prefix}_y"] = vector.y
    elif system == "ptphi":
        table[f"{prefix}_pt"] = vector.pt
        table[f"{prefix}_phi"] = vector.phi
    if "partonFlavour" in ak.fields(vector):
        table[f"{prefix}_flav"] = vector["partonFlavour"]
    if "btagDeepFlavB" in ak.fields(vector):
        table[f"{prefix}_btag"] = vector["btagDeepFlavB"]


def compute_helframe(top, atop, lep, alep):
    tt = top + atop
    ttboost = -tt.boostvec
    top_tt = top.boost(ttboost)
    atop_tt = atop.boost(ttboost)
    lep_tt = lep.boost(ttboost)
    alep_tt = alep.boost(ttboost)
    lep_hel = lep_tt.boost(-atop_tt.boostvec)
    alep_hel = alep_tt.boost(-top_tt.boostvec)
    return top_tt, atop_tt, lep_hel, alep_hel


def compute_chel(top, atop, lep, alep):
    tt = top + atop
    ttboost = -tt.boostvec
    top_tt = top.boost(ttboost)
    atop_tt = atop.boost(ttboost)
    lep_tt = lep.boost(ttboost)
    alep_tt = alep.boost(ttboost)
    lep_hel = lep_tt.boost(-atop_tt.boostvec)
    alep_hel = alep_tt.boost(-top_tt.boostvec)
    return lep_hel.dot(alep_hel) / lep_hel.rho / alep_hel.rho


def compute_sdmb(top, atop, lep, alep):
    ttbar_boost = -(top + atop).boostvec
    top = top.boost(ttbar_boost)
    atop = atop.boost(ttbar_boost)
    lep = lep.boost(ttbar_boost)
    alep = alep.boost(ttbar_boost)

    top_boost = -top.boostvec
    atop_boost = -atop.boostvec
    lep_hel = lep.boost(atop_boost)
    alep_hel = alep.boost(top_boost)
    lep_unit = lep_hel.unit
    alep_unit = alep_hel.unit

    kaxis = top.unit
    costheta = kaxis.z
    sgncostheta = np.sign(costheta)
    sintheta = np.sqrt(1 - costheta**2)
    naxis = (sgncostheta / sintheta) * ak.zip(
        {"x": kaxis.y, "y": -kaxis.x, "z": np.zeros(len(kaxis))},
        with_name="ThreeVector", behavior=top.behavior)
    raxis = (sgncostheta / sintheta) * ak.zip(
        {"x": -kaxis.x * costheta, "y": -kaxis.y * costheta,
         "z": 1 - kaxis.z * costheta},
        with_name="ThreeVector", behavior=top.behavior)
    b = ak.to_regular(ak.concatenate([
        alep_unit.dot(kaxis)[:, None],
        lep_unit.dot(-kaxis)[:, None],
        alep_unit.dot(raxis)[:, None],
        lep_unit.dot(-raxis)[:, None],
        alep_unit.dot(naxis)[:, None],
        lep_unit.dot(-naxis)[:, None],
    ], axis=1))
    return b


def compute_sdmc(sdmb):
    cii = sdmb[:, ::2] * sdmb[:, 1::2]
    basis = ak.Array(np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]))
    c = ak.sum(cii[:, None, :] * basis[None, :, :], axis=2)
    return c


def make_table(ev, weight, include_DM=False):
    sel = ((ak.num(ev["gent"]) == 2) & (ak.num(ev["genw"]) == 2)
           & (ak.num(ev["genb"]) == 2) & (ak.num(ev["genlepton"]) == 2))
    ev = ev[sel]
    weight = weight[sel]
    top = ak.with_name(ev["gent"][:, 0],
                       "PtEtaPhiMLorentzVector")
    atop = ak.with_name(ev["gent"][:, 1],
                        "PtEtaPhiMLorentzVector")
    ttbar = top + atop
    lep = ak.with_name(ev["genlepton"][:, 0],
                       "PtEtaPhiMLorentzVector")
    alep = ak.with_name(ev["genlepton"][:, 1],
                        "PtEtaPhiMLorentzVector")
    bot = ak.with_name(ev["genb"][:, 0],
                       "PtEtaPhiMLorentzVector")
    abot = ak.with_name(ev["genb"][:, 1],
                        "PtEtaPhiMLorentzVector")
    wplus = ak.with_name(ev["genw"][:, 0],
                         "PtEtaPhiMLorentzVector")
    wminus = ak.with_name(ev["genw"][:, -1],
                          "PtEtaPhiMLorentzVector")
    # save neutrinos as well
    neutrino = ak.with_name(ev["genv"][:, 0],
                       "PtEtaPhiMLorentzVector")
    aneutrino = ak.with_name(ev["genv"][:, 1],
                        "PtEtaPhiMLorentzVector")
    recotop = ak.with_name(ev["recot"][:, 0:1],
                           "PtEtaPhiMLorentzVector")
    recotop["mass"] = 172.5
    recoatop = ak.with_name(ev["recot"][:, 1:2],
                            "PtEtaPhiMLorentzVector")
    recoatop["mass"] = 172.5
    recolep = ak.with_name(ev["recolepton"][:, 0],
                           "PtEtaPhiMLorentzVector")
    recoalep = ak.with_name(ev["recolepton"][:, 1],
                            "PtEtaPhiMLorentzVector")
    recobot = ak.with_name(ev["recob"][:, 0],
                           "PtEtaPhiMLorentzVector")
    recoabot = ak.with_name(ev["recob"][:, 1],
                            "PtEtaPhiMLorentzVector")
    recomet = ak.with_name(ev["MET"], "PtEtaPhiMLorentzVector")
    recopuppimet = ak.with_name(ev["puppimet"], "PtEtaPhiMLorentzVector")
    recojet = ak.with_name(ev["Jet"], "PtEtaPhiMLorentzVector")
    
    table = ak.Array({
        "mtt": (top + atop).mass,
        "weight": weight
    })
    add_particle(table, "top", top, "ptetaphim")
    add_particle(table, "atop", atop, "ptetaphim")
    add_particle(table, "ttbar", ttbar, "ptetaphim")
    add_particle(table, "top", top, "cartesian")
    add_particle(table, "atop", atop, "cartesian")
    add_particle(table, "ttbar", ttbar, "cartesian")
    add_particle(table, "wplus", wplus, "ptetaphim")
    add_particle(table, "wminus", wminus, "ptetaphim")
    add_particle(table, "wplus", wplus, "cartesian")
    add_particle(table, "wminus", wminus, "cartesian")
    add_particle(table, "lep", recolep, "cartesian")
    add_particle(table, "alep", recoalep, "cartesian")
    add_particle(table, "lep", recolep, "ptetaphim")
    add_particle(table, "alep", recoalep, "ptetaphim")
    add_particle(table, "genlep", lep, "cartesian")
    add_particle(table, "genalep", alep, "cartesian")
    add_particle(table, "genlep", lep, "ptetaphim")
    add_particle(table, "genalep", alep, "ptetaphim")
    add_particle(table, "bot", recobot, "cartesian")
    add_particle(table, "abot", recoabot, "cartesian")
    add_particle(table, "bot", recobot, "ptetaphim")
    add_particle(table, "abot", recoabot, "ptetaphim")
    add_particle(table, "genbot", bot, "cartesian")
    add_particle(table, "genabot", abot, "cartesian")
    add_particle(table, "genbot", bot, "ptetaphim")
    add_particle(table, "genabot", abot, "ptetaphim")
    add_particle(table, "genv", neutrino, "cartesian")
    add_particle(table, "genav", aneutrino, "cartesian")
    add_particle(table, "genv", neutrino, "ptetaphim")
    add_particle(table, "genav", aneutrino, "ptetaphim")
    add_particle(table, "met", recomet, "ptphi")
    add_particle(table, "met", recomet, "cartesian_transverse")
    add_particle(table, "puppimet", recopuppimet, "ptphi")
    add_particle(table, "puppimet", recopuppimet, "cartesian_transverse")
    add_particle(table, "sonnentop", recotop, "cartesian")
    add_particle(table, "sonnenatop", recoatop, "cartesian")
    add_particle(table, "jet", recojet, "cartesian")
    add_particle(table, "jet", recojet, "ptetaphim")
    table["mtt"] = (top + atop).mass
    table["chel"] = compute_chel(top, atop, lep, alep)
    table["btag_scores"] = ak.fill_none(ak.pad_none(ev["Jet"]["btagDeepFlavB"][:, :7], 7, axis=1), 0)
    top_tt, atop_tt, lep_hel, alep_hel = compute_helframe(top, atop, lep, alep)
    add_particle(table, "toptt", top_tt, "cartesian")
    add_particle(table, "toptt", top_tt, "ptetaphim")
    add_particle(table, "genlephel", lep_hel, "cartesian")
    add_particle(table, "genlephel", lep_hel, "ptetaphi")
    add_particle(table, "genalephel", alep_hel, "cartesian")
    add_particle(table, "genalephel", alep_hel, "ptetaphi")

    if args.skim is not None:
        table["weight"] = table["weight"] * args.skim
        table = table[::args.skim]
    return table


parser = ArgumentParser()
parser.add_argument("inputdir", type=Path, help="Directory containing input HDF5 files")
parser.add_argument("outputdir", type=Path, help="Directory to save output HDF5 files")
parser.add_argument("-s", "--validationsplit", type=float, default=0.3, help="Fraction of data to use for validation set")
parser.add_argument("-t", "--testsplit", type=float, default=0.2, help="Fraction of data to use for test set (default: 0.2)")
parser.add_argument("-k", "--skim", type=int, help="Skim every Nth event (optional)")
parser.add_argument("--cuts", action="store_true", help="Use only events that pass all cuts")
parser.add_argument("-c", "--counts", help="Number of events to process from each input file (optional)")
parser.add_argument("-S", "--scale", help="Scale factor to apply to event weights")
args = parser.parse_args()

#if args.counts is not None and len(args.inputdir) != len(args.counts):
#    sys.exit("--counts must be present as often as inputdir is given or not present at all")
#if args.scale is not None and len(args.inputdir) != len(args.scale):
#(    sys.exit("--scale must be present as often as inputdir is given or not present at all")

Path(args.outputdir).resolve().mkdir(parents=True, exist_ok=True)

dirname = Path(args.inputdir).resolve()
if not dirname.is_dir():
    sys.exit(f"Input directory {dirname} does not exist or is not a directory")
if args.counts is None:
    events_needed = None
else:
    events_needed = int(args.counts)
if args.scale is None:
    scale = 1
else:
    scale = args.scale
tables = defaultdict(list)
for fname in tqdm(list(dirname.glob("**/*.h5"))):
    if events_needed is not None and events_needed <= 0:
        break
    with HDF5File(str(fname), "r") as a:
        ev = a["events"][:events_needed]
        if args.cuts:
            passes_cuts = np.all([np.asarray(a["cutflags"][field]) for field in a["cutflags"].fields], axis=0)
            passes_cuts = passes_cuts[:events_needed]
            ev = ev[passes_cuts]
        if "systematics" in a:
            weight = a["systematics"][:events_needed]["weight"]
        else:
            weight = a["weight"][:events_needed]
        if args.cuts:
            weight = weight[passes_cuts]
        weight = weight * scale
        tables[fname.parent.stem].append(make_table(ev, weight))
        nevt = len(tables[fname.parent.stem][-1])
        if events_needed is not None:
            events_needed -= nevt
            if events_needed < 0:
                break

for dataset in list(tables.keys()):
    tbls = tables.pop(dataset)
    args.outputdir.resolve().joinpath(dataset).mkdir(exist_ok=True)
    trainpath = args.outputdir.resolve().joinpath(dataset, "traindata.hdf5")
    validatepath = args.outputdir.resolve().joinpath(dataset, "validatedata.hdf5")
    testpath = args.outputdir.resolve().joinpath(dataset, "testdata.hdf5")

    table = ak.with_name(ak.concatenate(tbls), "Dataframe")
    del tbls
    
    rng = np.random.default_rng(RANDOM_SEED)
    shuffledidx = np.arange(len(table))
    rng.shuffle(shuffledidx)
    table = table[shuffledidx]

    valsplitidx = round(len(table) * (1 - args.validationsplit - args.testsplit))
    testsplitidx = round(len(table) * (1 - args.testsplit))
    traindata = table[:valsplitidx]
    validatedata = table[valsplitidx:testsplitidx]
    testdata = table[testsplitidx:]
    print(f"Dataset {dataset}: {len(traindata)} training events, "
          f"{len(validatedata)} validation events, {len(testdata)} test events")

    save_output(trainpath, traindata, args.inputdir)
    save_output(validatepath, validatedata, args.inputdir)
    save_output(testpath, testdata, args.inputdir)
