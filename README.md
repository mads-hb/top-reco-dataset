# Machine learning approach for top pair reconstruction

This repository provides files used to create and execute the neural network used in the thesis of ref. [[1]](#1) for reconstruction of top quark pairs from proton-proton collisions. The code is using the Pepper framework [[2]](#2).

## Installation
Required libraries are Pepper, Tensorflow (GPU installation recommended), and Scikit-hep's vector library.

## Usage
1. Run the Processor (a Pepper processor) using `process_reco.py`. The required Pepper configuration file can be any Pepper configuration file for pp -> ttbar analyses.
2. Run `prepare_data_reco.py`, which creates normalized values for a range of observables the network is working with and saved them into HDF5 files. 
3. Run `model_bcls.py`, which trains a network, whose purpose is to identify the correct pairing of bottom quarks.
4. Run `model_tt.py`, which trains a network to reconstruct the top quarks. It uses the network from the previous step.
5. The files `plot_validation_bcls.py` and `plot_validation.py` execute the networks on data and create plots to judge their performance.


## References
<a id="1">[1]</a> Rübenach, J. (2023) Search for heavy Higgs bosons in conjunction with neural-network-driven reconstruction and upgrade of the Fast Beam Condition Monitor at the CMS experiment. [CERN-THESIS-2023-066](https://cds.cern.ch/record/2861145)
<a id="2">[2]</a> Pepper contributors. Pepper - ParticlE Physics ProcEssoR. 2023. URL: https://gitlab.cern.ch/pepper/
