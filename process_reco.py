#!/usr/bin/env python3
import os
import sys

import awkward as ak
import numba
import numpy as np
from functools import partial, reduce
import uproot

import pepper


class Processor(pepper.ProcessorTTbarLL):
    def __init__(self, config, outdir):
        config["columns_to_save"] = [
            ["genlepton", ["pt", "eta", "phi", "mass", "pdgId"]],
            ["gent", ["pt", "eta", "phi", "mass", "pdgId"]],
            ["genb", ["pt", "eta", "phi", "mass", "pdgId"]],
            ["genv", ["pt", "eta", "phi", "mass", "pdgId"]],
            ["genw", ["pt", "eta", "phi", "mass", "pdgId"]],
            ["recolepton", ["pt", "eta", "phi", "mass", "pdgId"]],
            ["recot", ["pt", "eta", "phi", "mass"]],
            ["recob", ["pt", "eta", "phi", "mass"]],
            ["Jet", ["pt", "eta", "phi", "mass", "partonFlavour", "btagDeepFlavB"], {"leading": (1, 8)}],
            ["MET", ["pt", "phi"]],
            ["chel"],
            ["MT2ll"],
            ["PuppiMET", ["pt", "phi"]],
            # We do not need these for now
            # ["GenMET", ["pt", "phi"]],
            # ["GenJet", ["pt", "eta", "phi", "mass", "partonFlavour"]],
            # ["Lepton", ["pt", "eta", "phi", "mass"]]
        ]
        if config["reco_algorithm"] == "both":
            config["columns_to_save"].append(["recot_sonn", ["pt", "eta", "phi", "mass"]])
        super().__init__(config, outdir)

    def process_selection(self, selector, dsname, is_mc, filler):
        era = self.get_era(selector.data, is_mc)
        if is_mc and "pileup_reweighting" in self.config:
            selector.add_cut("Pileup reweighting", partial(
                self.do_pileup_reweighting, dsname))
        if is_mc and self.config["mc_lumifactors"]:
            if "dataset" in selector.cats:
                selector.add_cut(
                    "Cross section", partial(
                        self.crosssection_scale, selector.cats["dataset"]))
            else:
                selector.add_cut(
                    "Cross section", partial(self.crosssection_scale, dsname))
        selector.add_cut("Lumi", partial(self.good_lumimask, is_mc, dsname))
        selector.set_multiple_columns(self.build_gen_columns)

        pos_triggers, neg_triggers = pepper.misc.get_trigger_paths_for(
            dsname, is_mc, self.config["dataset_trigger_map"],
            self.config["dataset_trigger_order"], era=era)
        selector.add_cut("Trigger", partial(
            self.passing_trigger, pos_triggers, neg_triggers))
        selector.add_cut("MET filters", partial(self.met_filters, is_mc))
        selector.add_cut("No add leps",
                         partial(self.no_additional_leptons, is_mc))
        selector.set_column("Electron", self.pick_electrons)
        selector.set_column("Muon", self.pick_muons)
        selector.set_column("Lepton", partial(
            self.build_lepton_column, is_mc, selector.rng))
        # Wait with hists filling after channel masks are available
        selector.add_cut("At least 2 leps", partial(self.lepton_pair, is_mc),
                         no_callback=True)
        selector.set_multiple_columns(self.channel_masks)
        selector.set_cat("channel", {"is_ee", "is_em", "is_mm"})
        selector.set_column("mll", self.mass_lepton_pair)

        selector.add_cut("Opposite sign", self.opposite_sign_lepton_pair)
        selector.add_cut("Chn trig match",
                         partial(self.channel_trigger_matching, era))
        if "trigger_sfs" in self.config and is_mc:
            selector.add_cut(
                "Trigger SFs", partial(self.apply_trigger_sfs, dsname))

        variation = self.get_jetmet_nominal_arg()
        reapply_jec = ("reapply_jec" in self.config
                       and self.config["reapply_jec"])
        selector.set_multiple_columns(partial(
            self.compute_jet_factors, is_mc, era, reapply_jec, variation.junc,
            variation.jer, selector.rng))
        selector.set_column("OrigJet", selector.data["Jet"])
        selector.set_column("Jet", partial(self.build_jet_column, is_mc))
        smear_met = "smear_met" in self.config and self.config["smear_met"]
        selector.set_column(
            "MET", partial(self.build_met_column, is_mc, variation.junc,
                           variation.jer if smear_met else None, selector.rng,
                           era, variation=variation.met))
        selector.add_cut("Has jet(s)", self.has_jets)
        selector.add_cut("Has btag(s)", partial(self.btag_cut, is_mc))

        reco_alg = self.config["reco_algorithm"]
        selector.set_column("recolepton", self.pick_lepton_pair,
                            all_cuts=True)
        selector.set_column("recob", self.pick_bs_from_lepton_pair,
                            all_cuts=True)
        if reco_alg.lower() == "both":
            selector.set_column("recot", partial(
                self.ttbar_system, "betchart", selector.rng),
                all_cuts=True, no_callback=True)
            selector.set_column("recot_sonn", partial(
                self.ttbar_system, "sonnenschein", selector.rng),
                all_cuts=True, no_callback=True)
        else:
            selector.set_column("recot", partial(
                self.ttbar_system, reco_alg, selector.rng),
                all_cuts=True, no_callback=True)
        selector.set_column("reconu", self.build_nu_column_ttbar_system,
                            all_cuts=True, lazy=True)
        selector.add_cut("Reco", self.has_ttbar_system)
        selector.set_column("chel", self.calculate_chel)

        selector.applying_cuts = False
        selector.add_cut("Req lep pT", self.lep_pt_requirement)
        selector.add_cut("m_ll", self.good_mass_lepton_pair)
        selector.add_cut("Z window", self.z_window,
                         categories={"channel": ["is_ee", "is_mm"]})
        selector.add_cut("Jet pt req", self.jet_pt_requirement)
        selector.add_cut("Req MET", self.met_requirement,
                         categories={"channel": ["is_ee", "is_mm"]})

    def build_lhe_columns(self, data, prefix="gen"):
        part = data["LHEPart"]
        pdgid = abs(part["pdgId"])
        cols = {}
        cols["lepton"] = part[(pdgid == 11) | (pdgid == 13) | (pdgid == 15)][:, -2:]
        cols["lhelepton"] = cols["lepton"][ak.argsort(cols["lepton"]["pdgId"], axis=1, ascending=True)]
        cols["v"] = part[(pdgid == 12) | (pdgid == 14) | (pdgid == 16)][:, -2:]
        cols["v"] = cols["v"][ak.argsort(cols["v"]["pdgId"], axis=1, ascending=False)]
        cols["b"] = part[pdgid == 5][:, -2:]
        cols["b"] = cols["b"][ak.argsort(cols["b"]["pdgId"], axis=1, ascending=False)]
        cols["w"] = cols["lepton"] + cols["v"]
        cols["t"] = cols["b"] + cols["w"]
        cols["lepton"] = cols["lepton"][ak.argsort(cols["lepton"]["pdgId"], axis=1, ascending=False)]

        cols = {prefix + k: v for k, v in cols.items()}

        return cols

    @staticmethod
    @numba.njit
    def get_mother_pdgids_lastcopy(motheridxs, pdgids, offsets):
        # Find PDG ID of the ancestor particle with a different ID
        motherpdgids = np.zeros_like(pdgids)
        islastcopy = np.full_like(pdgids, True).astype(np.dtype("bool"))
        for i in range(len(offsets) - 1):
            offset = offsets[i]
            num_parts = offsets[i + 1] - offset
            for j in range(num_parts):
                pdgid = pdgids[offset + j]
                motheridx = motheridxs[offset + j]
                motheridx_old = j
                while 0 <= motheridx < motheridx_old:
                    motherpdgid = pdgids[offset + motheridx]
                    if motherpdgid != pdgid:
                        motherpdgids[offset + j] = motherpdgid
                        break
                    islastcopy[offset + motheridx] = False
                    motheridx_old = motheridx
                    motheridx = motheridxs[offset + motheridx]
        return motherpdgids, islastcopy
    
    def build_gen_columns(self, data):
        part = data["GenPart"]
        abspdg = abs(part["pdgId"])
        part["mass"] = ak.where(
            abspdg == 5, 4.18, ak.where(
                abspdg == 11, 0.0005, ak.where(
                    abspdg == 13, 0.102, ak.where(
                        abspdg == 15, 1.777, part.mass))))
        motheridx = part["genPartIdxMother"]
        pdgid = part["pdgId"]
        offsets = np.r_[0, np.cumsum(np.asarray(ak.num(part)))]
        motherpdgid, islastcopy = self.get_mother_pdgids_lastcopy(
            np.asarray(ak.flatten(motheridx)),
            np.asarray(ak.flatten(pdgid)),
            offsets)
        part["motherid"] = ak.unflatten(motherpdgid, ak.num(part))
        part = part[part.hasFlags("isFirstCopy")]
        abspdg = abs(part["pdgId"])
        sgn = np.sign(part["pdgId"])

        cols = {}
        cols["genlepton"] = part[
            ((abspdg == 11) | (abspdg == 13) | (abspdg == 15)) & (part.motherid == sgn * -24)]
        cols["genlepton"] = cols["genlepton"][
            ak.argsort(cols["genlepton"]["pdgId"], ascending=False)]

        cols["genv"] = part[((abspdg == 12) | (abspdg == 14) | (abspdg == 16)) & (part.motherid == sgn * 24)]
        cols["genv"] = cols["genv"][
            ak.argsort(cols["genv"]["pdgId"], ascending=False)]

        cols["genb"] = part[(abspdg == 5) & (part.motherid == sgn * 6)]
        cols["genb"] = cols["genb"][
            ak.argsort(cols["genb"]["pdgId"], ascending=False)]

        cols["genw"] = part[(abspdg == 24) & (part.motherid == sgn * 6)]
        cols["genw"] = cols["genw"][
            ak.argsort(cols["genw"]["pdgId"], ascending=False)]

        cols["gent"] = part[(abspdg == 6)]
        cols["gent"] = cols["gent"][
            ak.argsort(cols["gent"]["pdgId"], ascending=False)]

        cols["genS"] = part[(abspdg == 54)]

        cols["genChi"] = part[(abspdg == 52)]
        cols["genChi"] = cols["genChi"][
            ak.argsort(cols["genChi"]["pdgId"], ascending=False)]

        return cols

    def has_gen_particles(self, data):
        mask = ((ak.num(data["genlepton"]) == 2)
                & (ak.num(data["genv"]) == 2)
                & (ak.num(data["genb"]) == 2)
                & (ak.num(data["genw"]) == 2)
                & (ak.num(data["gent"]) == 2))
        return mask

    def btag_cut2(self, data):
        return ak.sum(data["Jet"]["btagDeepFlavB"] > 0.0494, axis=1) >= 2

    def pick_bs(self, data):
        recolepton = data["recolepton"]
        lep = recolepton[:, 0]
        antilep = recolepton[:, 1]
        # Build a reduced jet collection to avoid loading all branches and
        # make make this function faster overall
        columns = ["pt", "eta", "phi", "mass", "btagged", "partonFlavour"]
        jets = ak.with_name(data["Jet"][columns], "PtEtaPhiMLorentzVector")
        btags = jets[data["Jet"].btagged]
        jetsnob = jets[~data["Jet"].btagged]
        num_btags = ak.num(btags)
        b0, b1 = ak.unzip(ak.where(
            num_btags > 1, ak.combinations(btags, 2),
            ak.where(
                num_btags == 1, ak.cartesian([btags, jetsnob]),
                ak.combinations(jetsnob, 2))))
        bs = ak.concatenate([b0, b1], axis=1)
        bs_rev = ak.concatenate([b1, b0], axis=1)
        mass_alb = reduce(
            lambda a, b: a + b, ak.unzip(ak.cartesian([bs, antilep]))).mass
        mass_lb = reduce(
            lambda a, b: a + b, ak.unzip(ak.cartesian([bs_rev, lep]))).mass
        with uproot.open(self.reco_info_filepath) as f:
            mlb_prob = pepper.scale_factors.ScaleFactors.from_hist(f["mlb"])
        p_m_alb = mlb_prob(mlb=mass_alb)
        p_m_lb = mlb_prob(mlb=mass_lb)
        bestbpair_mlb = ak.unflatten(
            ak.argmax(p_m_alb * p_m_lb, axis=1), np.full(len(bs), 1))
        return ak.concatenate([bs[bestbpair_mlb], bs_rev[bestbpair_mlb]],
                              axis=1)

    def build_nu_column(self, data):
        """Get four momenta for the neutrinos coming from top pair decay"""
        lep = data["recolepton"][:, 0:1]
        antilep = data["recolepton"][:, 1:2]
        b = data["recob"][:, 0:1]
        antib = data["recob"][:, 1:2]
        top = data["recot"][:, 0:1]
        antitop = data["recot"][:, 1:2]
        nu = top - b - antilep
        antinu = antitop - antib - lep
        return ak.concatenate([nu, antinu], axis=1)

    def calculate_chel(self, data):
        """Calculate the angle between the leptons in their helicity frame"""
        top = data["recot"]
        lep = data["recolepton"]
        ttbar_boost = -top.sum().boostvec
        top = top.boost(ttbar_boost)
        lep = lep.boost(ttbar_boost)

        top_boost = -top.boostvec
        lep_ZMFtbar = lep[:, 0].boost(top_boost[:, 1])
        lbar_ZMFtop = lep[:, 1].boost(top_boost[:, 0])

        chel = lep_ZMFtbar.dot(lbar_ZMFtop) / lep_ZMFtbar.rho / lbar_ZMFtop.rho
        return chel

if __name__ == "__main__":
    from pepper import runproc
    runproc.run_processor(Processor, mconly=True)
