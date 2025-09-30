# Machine learning approach for top pair reconstruction

This repository provides files used to create the dataset used for the top quark reconstruction shown in [this](https://gitlab.cern.ch/mabaattr/top-reconstruction) repository.

## Installation
This package requires the Pepper framework and is usually installed using an LCG environment. Is has been tested to work with CMS LCG_107a environment. To install, run:
```bash
source environment.sh
python -m pip install -e pepperlib
```

## Usage
1. Run the Processor (a Pepper [2] processor) using `process_reco.py`. The required Pepper configuration file can be any Pepper configuration file for pp -> ttbar analyses:
```bash
python process_reco.py config/config2022post.hjson -c 500 --dataset TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8 -o output --eventdir events/
```
2. Run `prepare_data_reco.py`, which creates normalized values for a range of observables the network is working with and saved them into HDF5 files:
```bash
python prepare_data_reco.py events/ data/hdf5/ --cuts --validationsplit 0.25 --testsplit 0.25 --counts 1500000
```
3. Finally, run `extract_trainarrays.py` to extract the arrays used for training:
```bash
python extract_trainarrays.py data/hdf5/ data/npz/ --verbose
```

## References

<a id="2">[2]</a> Pepper contributors. Pepper - ParticlE Physics ProcEssoR. 2023. URL: https://gitlab.cern.ch/pepper/
