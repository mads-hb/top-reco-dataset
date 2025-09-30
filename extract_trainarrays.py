# import h5py
# import pickle
# import numpy as np
# from pepper.hdffile import HDF5File
# # import vector
# import awkward as ak

# new_data_dir = '/data/dust/user/stafford/For_Emanuele/reconn/2007_data'

# #data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2007_data/_TTbarDMJets_Dilepton_scalar_LO_Mchi_1_Mphi_250_TuneCP5_13TeV_madgraph_mcatnlo_pythia8.h5'
# #data_dir = '/gpfs/dust/cms/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/traindata.hdf5'
# # data_dir = '/data/dust/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/validatedata.hdf5'
# data_dir = '/data/dust/user/stafford/For_Emanuele/reconn/2907_nn_inputs/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/traindata.hdf5'
# with HDF5File(data_dir, "r") as f:
#     df = f["data"]
#     offset = f['offset']
#     scale = f['scale']
# print_data = True
# print(f'the scale is {scale}')
# print(f'and the offset is {offset}')
# if print_data:
#     print('available fields')
#     print(df.fields)
#     #print('for the gen tops this is:')
#     #print(df['gent'].fields)
#     print('for the jets this is')
#     print(df['jet_pt'].fields)
#     print(f'and the jets look like {df["jet_pt"].type}')
#     print(f'the mtt is {df["mtt"].fields}')
#     print(f'the max mtt is {np.max(df["mtt"])}')
#     print(f'the top mass is {np.mean(df["atop_mass"])}')
#     print(f'the weights are in mean {np.mean(df["weight"])}')

# jet_t = df["jet_t"][:, :7]*scale['jet_t'] + offset['jet_t']
# jet_x = df["jet_x"][:, :7]*scale['jet_x'] + offset['jet_x']
# jet_y = df["jet_y"][:, :7]*scale['jet_y'] + offset['jet_y']
# jet_z = df["jet_z"][:, :7]*scale['jet_z'] + offset['jet_z']
# print(f' the maximum jet component is {np.max(jet_z)}')
# breakpoint()
# # jets
# jets = ak.fill_none(ak.pad_none(ak.concatenate([jet_t[:,:, np.newaxis], jet_x[:,:, np.newaxis], jet_y[:,:, np.newaxis], jet_z[:,:, np.newaxis]], axis=2), 7, axis=1), 0)
# # leptons
# leptons = ak.concatenate([df["lep_t"][:, np.newaxis, np.newaxis]*scale['lep_t'] + offset['lep_t'], df["lep_x"][:, np.newaxis, np.newaxis]*scale['lep_x'] + offset['lep_x'], df["lep_y"][:, np.newaxis,  np.newaxis]*scale['lep_y'] + offset['lep_y'], df["lep_z"][:, np.newaxis, np.newaxis]*scale['lep_z'] + offset['lep_z']], axis=2)
# # antileptons
# antileptons = ak.concatenate([df["alep_t"][:, np.newaxis, np.newaxis]*scale['alep_t'] + offset['alep_t'], df["alep_x"][:, np.newaxis, np.newaxis]*scale['alep_x'] + offset['alep_x'], df["alep_y"][:, np.newaxis, np.newaxis]*scale['alep_y'] + offset['alep_y'], df["alep_z"][:, np.newaxis, np.newaxis]*scale['alep_z'] + offset['alep_z']], axis=2)
# # b
# bottoms = ak.concatenate([df["bot_t"][:, np.newaxis, np.newaxis]*scale['bot_t'] + offset['bot_t'], df["bot_x"][:, np.newaxis, np.newaxis]*scale['bot_x'] + offset['bot_x'], df["bot_y"][:, np.newaxis, np.newaxis]*scale['bot_y'] + offset['bot_y'], df["bot_z"][:, np.newaxis, np.newaxis]*scale['bot_z'] + offset['bot_z']], axis=2)
# # antib
# antibottoms = ak.concatenate([df["abot_t"][:, np.newaxis, np.newaxis]*scale['abot_t'] + offset['abot_t'], df["abot_x"][:, np.newaxis, np.newaxis]*scale['abot_x'] + offset['abot_x'], df["abot_y"][:, np.newaxis, np.newaxis]*scale['abot_y'] + offset['abot_y'], df["abot_z"][:, np.newaxis, np.newaxis]*scale['abot_z'] + offset['abot_z']], axis=2)
# # MET
# #met = ak.concatenate([df["met_pt"][:, np.newaxis, np.newaxis]*scale['met_pt'] + offset['met_pt'], df["met_x"][:, np.newaxis, np.newaxis]*scale['met_x'] + offset['met_x'], df["met_y"][:, np.newaxis, np.newaxis]*scale['met_y'] + offset['met_y'], df["met_phi"][:, np.newaxis, np.newaxis]*scale['met_phi'] + offset['met_phi']], axis=2)
# # Save MET x and y components
# met = ak.concatenate([df["met_pt"][:, np.newaxis, np.newaxis]*scale['met_pt'] + offset['met_pt'], df["met_x"][:, np.newaxis, np.newaxis]*scale['met_x'] + offset['met_x'], df["met_y"][:, np.newaxis, np.newaxis]*scale['met_y'] + offset['met_y']], axis=2)

# # top
# top = ak.concatenate([df["top_t"][:, np.newaxis, np.newaxis]*scale['top_t'] + offset['top_t'], df["top_x"][:, np.newaxis, np.newaxis]*scale['top_x'] + offset['top_x'], df["top_y"][:, np.newaxis, np.newaxis]*scale['top_y'] + offset['top_y'], df["top_z"][:, np.newaxis, np.newaxis]*scale['top_z'] + offset['top_z']], axis=2)
# # atop
# antitop = ak.concatenate([df["atop_t"][:, np.newaxis, np.newaxis]*scale['atop_t'] + offset['atop_t'], df["atop_x"][:, np.newaxis, np.newaxis]*scale['atop_x'] + offset['atop_x'], df["atop_y"][:, np.newaxis, np.newaxis]*scale['atop_y'] + offset['atop_y'], df["atop_z"][:, np.newaxis, np.newaxis]*scale['atop_z'] + offset['atop_z']], axis=2)

# print(f'the maximum top component is {np.max(top)}')

# target = ak.concatenate([top, antitop], axis=1)
# print(target.type)
# target = ak.to_numpy(ak.concatenate([top, antitop], axis=1), allow_missing=True)
# inputarr = ak.to_numpy(ak.concatenate([jets, leptons, antileptons, bottoms, antibottoms], axis=1), allow_missing=True)
# metarr = ak.to_numpy(met, allow_missing=True)
# print(target.shape)

# # np.savez('data/train_TTTo2L2Nu_train_scaled.npz', x=inputarr, y=target)
# # np.savez('/data/dust/user/baattrup/top-quark-reconstruction/data/train_TTTo2L2Nu_train_scaled_met.npz', x=inputarr, y=target, met=metarr)
# # np.savez('data/train_TTTo2L2Nu_train_scaled_slimmed.npz', x=inputarr[::100], y=target[::100])
# # np.savez('data/train_TTTo2L2Nu_val_scaled_slimmed.npz', x=inputarr[::100], y=target[::100])


import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional
import typing
import h5py
import numpy as np
import awkward as ak
from pepper.hdffile import HDF5File

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataExtractor:
    """Extracts and processes training arrays from HDF5 files."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> None:
        """Load data from HDF5 file."""
        try:
            with HDF5File(str(self.data_path), "r") as f:
                self.df = f["data"]
            logger.info(f"Successfully loaded data from {self.data_path}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def print_data_info(self) -> None:
        """Print information about the loaded data."""
        if self.df is None:
            logger.warning("No data loaded")
            return
            
        logger.info("Data fields available:")
        logger.info(f"  Main fields: {self.df.fields}")
        logger.info(f"  Jet fields: {self.df['jet_pt'].fields}")
        logger.info(f"  Jet type: {self.df['jet_pt'].type}")
        logger.info(f"  MTT fields: {self.df['mtt'].fields}")
        logger.info(f"  Max MTT: {np.max(self.df['mtt']):.2f}")
        logger.info(f"  Mean top mass: {np.mean(self.df['atop_mass']):.2f}")
        logger.info(f"  Mean weight: {np.mean(self.df['weight']):.4f}")
    
    def extract_btagging_scores(self, max_jets: int = 7) -> ak.Array:
        """Extract and normalize b-tagging scores."""
        btag_scores = self.df["btag_scores"][:, :max_jets]
        return ak.fill_none(ak.pad_none(btag_scores, max_jets, axis=1), 0)

    def extract_jets(self, max_jets: int = 7) -> ak.Array:
        """Extract and normalize jet 4-vectors."""
        jet_t = self.df["jet_t"][:, :max_jets]
        jet_x = self.df["jet_x"][:, :max_jets]
        jet_y = self.df["jet_y"][:, :max_jets]
        jet_z = self.df["jet_z"][:, :max_jets]
        jet_btag = self.df["jet_btag"][:, :max_jets]
        
        # Stack components and pad to fixed length
        jets = ak.concatenate([
            jet_t[:, :, np.newaxis], 
            jet_x[:, :, np.newaxis], 
            jet_y[:, :, np.newaxis], 
            jet_z[:, :, np.newaxis]
        ], axis=2)
        jets_padded = ak.fill_none(ak.pad_none(jets, max_jets, axis=1), 0)
        btag_padded = ak.fill_none(ak.pad_none(jet_btag, max_jets, axis=1), 0)
        # Check shapes
        # assert jets_padded.shape[:-1] == btag_padded.shape, "Jets and b-tagging scores shape mismatch"
        return jets_padded, btag_padded

    def extract_particle_4vector(self, prefix: str) -> ak.Array:
        """Extract and normalize 4-vector for a particle type."""
        components = []
        for suffix in ['t', 'x', 'y', 'z']:
            field = f"{prefix}_{suffix}"
            data = self.df[field]
            components.append(data[:, np.newaxis, np.newaxis])
        
        return ak.concatenate(components, axis=2)
    
    def extract_met(self, *, met_type: typing.Literal["MET", "PuppiMET"]) -> ak.Array:
        """Extract and normalize MET components."""
        key = "met" if met_type == "MET" else "puppimet"
        components = []
        for field in [f'{key}_pt', f'{key}_x', f'{key}_y']:
            data = self.df[field]
            components.append(data[:, np.newaxis, np.newaxis])
        
        return ak.concatenate(components, axis=2)
    
    def extract_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract and combine all data components."""
        # Extract individual components
        jets, btag = self.extract_jets()
        leptons = self.extract_particle_4vector('lep')
        antileptons = self.extract_particle_4vector('alep')
        bottoms = self.extract_particle_4vector('bot')
        antibottoms = self.extract_particle_4vector('abot')
        met = self.extract_met(met_type="PuppiMET")
        # met = self.extract_met(met_type="MET")

        neutrino = self.extract_particle_4vector('genv')
        aneutrino = self.extract_particle_4vector('genav')
        # Extract targets
        top = self.extract_particle_4vector('top')
        antitop = self.extract_particle_4vector('atop')
        
        # Combine inputs and targets
        # fourvectors = ak.concatenate([jets, leptons, antileptons, bottoms, antibottoms, neutrino, aneutrino], axis=1)
        # fourvectors = ak.concatenate([jets, leptons, antileptons, bottoms, antibottoms], axis=1)
        fourvectors = ak.concatenate([leptons, antileptons, bottoms, antibottoms, neutrino, aneutrino], axis=1)
        target = ak.concatenate([top, antitop], axis=1)
        
        # Convert to numpy arrays
        fourvectors = ak.to_numpy(fourvectors, allow_missing=True)
        target = ak.to_numpy(target, allow_missing=True)
        met = ak.to_numpy(met, allow_missing=True)
        btag = ak.to_numpy(btag, allow_missing=True)

        scalars = {"met": met, "btagging_scores": btag}
        logger.info(f"Fourvectors shape: {fourvectors.shape}")
        logger.info(f"Target shape: {target.shape}")
        
        return scalars, fourvectors, target

    def save_data(self, scalars: dict[str, np.ndarray], fourvectors: np.ndarray, target: np.ndarray,
                  output_path: str) -> None:
        """Save processed data to npz file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)


        np.savez_compressed(output_path, x=fourvectors, y=target, **scalars)
        logger.info(f"Saved data to {output_path}")


def process_hdf5_file(input_path: Path, output_path: Path, verbose: bool = False) -> None:
    """Process a single HDF5 file and save the extracted data."""
    extractor = DataExtractor(input_path)
    extractor.load_data()
    
    if verbose:
        extractor.print_data_info()
    
    scalars, fourvectors, target = extractor.extract_all_data()
    extractor.save_data(scalars, fourvectors, target, output_path)
    logger.info(f"Processing complete for {input_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract training arrays from HDF5 files")
    parser.add_argument("input",
                        type=Path,
                        help="Path to input HDF5 file or directory containing HDF5 files")
    parser.add_argument("output",
                        type=Path,
                        help="Path to output npz file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed data info")
    
    args = parser.parse_args()
    # _SUFFIX = ""
    _SUFFIX = "toy_ellbnu_slimmed"
    # Initialize extractor and process data
    if not args.input.exists():
        sys.exit(f"Input path {args.input} does not exist.")
    if args.input.is_dir():
        for hdf5_file in args.input.glob("**/*.hdf5"):
            relative_path = hdf5_file.relative_to(args.input)
            output_file = args.output / relative_path.with_stem(f"{relative_path.stem}{_SUFFIX}").with_suffix(".npz")
            process_hdf5_file(hdf5_file, output_file, verbose=args.verbose)
    else:
        process_hdf5_file(args.input, args.output, verbose=args.verbose)

if __name__ == "__main__":
    main()
