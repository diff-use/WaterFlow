
import copy
import os
import pathlib
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from SLAE.model.decoder import FILL
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from loguru import logger
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from torch_geometric.data.data import BaseData
from concurrent.futures import ThreadPoolExecutor, as_completed

from SLAE.io.atom_tensor import atomarray_to_tensors
# AtomWorks imports
try:
    from atomworks.io.parser import parse
    from atomworks.io.utils.io_utils import load_any
    import biotite.structure as struc
    ATOMWORKS_AVAILABLE = True
except ImportError:
    ATOMWORKS_AVAILABLE = False
    logger.warning("AtomWorks not available. Please install with: pip install atomworks")

IndexType = Union[slice, Tensor, np.ndarray, Sequence]

from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def _init_worker():
    # Prevent per-process thread explosions
    import torch
    torch.set_num_threads(1)

def _process_one(pdb_input: str, pdb_dir: str, processed_dir: str, fmt: str, remove_sidechains: bool):

    pdb = Path(pdb_input).name
    in_path = Path(pdb_dir) / f"{pdb_input}.{fmt}"
    out_path = Path(processed_dir) / f"{pdb}.pt"
    if out_path.exists():
        return (pdb, True, None)

    if not in_path.exists():
        return (pdb, False, f"missing: {in_path}")

    try:
        result = parse(filename=str(in_path), hydrogen_policy="remove")
        asym_unit = result["asym_unit"]
        model = asym_unit[0]

        coords, residue_type, chains, residue_id = atomarray_to_tensors(model)

        # FIX #4: Move NaN handling to preprocessing (was in get())
        nan_mask = torch.isnan(coords)
        if nan_mask.any():
            coords[nan_mask] = FILL

        graph = Data(
            coords=coords,
            residue_type=residue_type,
            chains=chains,
            residue_id=residue_id,
        )
        graph.id = pdb

        if remove_sidechains:
            graph.coords_true = graph.coords.clone()
            graph.coords[:, 4:, :] = FILL

        # FIX #1: Precompute pos, residue_index, atom_type during preprocessing
        # This avoids doing atom37_to_atoms conversion during training
        from SLAE.io.atom_tensor import atom37_to_atoms
        pos, residue_index, atom_type = atom37_to_atoms(coords)
        graph.pos = pos
        graph.residue_index = residue_index
        graph.atom_type = atom_type.long()

        tmp_path = out_path.with_suffix(".pt.tmp")
        torch.save(graph, tmp_path)
        os.replace(tmp_path, out_path)  # atomic rename on same filesystem

        return (pdb, True, None)
    except Exception as e:
        return (pdb, False, repr(e))

class ProteinDataset(Dataset):
    """Dataset for loading protein structures loads all chains in a PDB file as a single graph.
    """

    def __init__(
        self,
        pdb_codes: List[str],
        pdb_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        graph_labels: Optional[List[torch.Tensor]] = None,
        node_labels: Optional[List[torch.Tensor]] = None,
        overwrite: bool = False,
        format: str = "pdb",
        in_memory: bool = False,
        store_het: bool = False,
        out_names: Optional[List[str]] = None,
        crop: Optional[bool] = False,
        crop_len: Optional[int] = None,
        rand_slice: Optional[bool] = False,
        slice_range: Optional[List[float]] = None,
        remove_sidechains: bool = False,
    ):
        self.pdb_codes = [pdb.split('.')[0] for pdb in pdb_codes]
        self.get_pdb_codes = [pdb.split('/')[-1] for pdb in self.pdb_codes]
        self.pdb_dir = Path(pdb_dir)
        self._processed_dir = processed_dir
        self.overwrite = overwrite
        self.node_labels = node_labels
        self.graph_labels = graph_labels
        self.format = format
        self.in_memory = in_memory
        self.store_het = store_het
        self.out_names = out_names
        self.rand_slice = rand_slice
        self.slice_range = slice_range
        self._processed_files = []



        self.skip_pdb_list = set()
        self.crop = crop
        self.crop_len = crop_len

        self.remove_sidechains = remove_sidechains


        super().__init__()
        self.structures = pdb_codes
        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = [
                torch.load(pathlib.Path(processed_dir) / f , weights_only=False)
                for f in tqdm(self._processed_files)#processed_file_names)
            ]

    def len(self) -> int:
        """Return length of the dataset."""
        return len(self.pdb_codes)



    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        """Returns the processed file names.

        This will either be a list in format [``{pdb_code}.pt``] or
        a list of [{pdb_code}_{chain(s)}.pt].

        :return: List of processed file names.
        :rtype: Union[str, List[str], Tuple]
        """
        if self._processed_files:
            return self._processed_files
        if self.overwrite:
            return ["this_forces_a_processing_cycle"]
        if self.out_names is not None:
            return [f"{name}.pt" for name in self.out_names]
        else:
            return [f"{pdb}.pt" for pdb in self.pdb_codes]
        
    def process(self, max_workers: int = 12, chunksize: int = 8):
        # Read skip list once
        skip_list_path = Path(self._processed_dir) / "skip_list.txt"
        if skip_list_path.exists():
            self.skip_pdb_list = set(skip_list_path.read_text().splitlines())
        else:
            self.skip_pdb_list = set()

        # Build todo list
        todo = []
        for pdb in self.pdb_codes:
            pdb_input = Path(pdb).name
            if pdb_input in self.skip_pdb_list:
                continue
            out_path = Path(self._processed_dir) / f"{pdb_input}.pt"
            if (not self.overwrite) and out_path.exists():
                continue
            todo.append(pdb)

        from tqdm import tqdm
        failed = []

        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as ex:
            futures = []
            for pdb in todo:
                pdb_input = Path(pdb).name
                futures.append(ex.submit(
                    _process_one,
                    pdb,
                    str(self.pdb_dir),
                    str(self._processed_dir),
                    self.format,
                    self.remove_sidechains,
                ))

            for fut in tqdm(as_completed(futures), total=len(futures)):
                pdb, ok, err = fut.result()
                if not ok:
                    failed.append((pdb_input, err))
                    self.skip_pdb_list.add(pdb_input)

        # Write skip list once (no races)
        skip_list_path.write_text("\n".join(sorted(self.skip_pdb_list)))

        # Optionally log failures
        for pdb, err in failed[:20]:
            logger.error(f"Failed {pdb}: {err}")
        logger.info(f"Processed {len(todo) - len(failed)} / {len(todo)}; skipped total {len(self.skip_pdb_list)}")

    # def process(self):
    #     """Process raw data into PyTorch Geometric Data objects 

    #     Processed data are stored in ``self.processed_dir`` as ``.pt`` files.
    #     """
    #     # skip list is used to store PDBs that failed to process
    #     skip_list_path = Path(self._processed_dir) / "skip_list.txt"
    #     # read in a list of PDBs to skip
    #     if skip_list_path.exists():
    #         with open(skip_list_path, "r") as f:
    #             self.skip_pdb_list = set(f.read().splitlines())
        

    #     if not self.overwrite :
               
    #         index_pdb_tuples = [
    #             (i, pdb)
    #             for i, pdb in enumerate(self.pdb_codes)
    #             if (not os.path.exists(
    #                 Path(self._processed_dir) / f"{pdb}.pt"
    #             ) 
    #             ) and pdb not in self.skip_pdb_list
    #         ]
    #         logger.info(
    #             f"Processing {len(index_pdb_tuples)} unprocessed structures"
    #         )
           
    #     else:
    #         index_pdb_tuples = [
    #             (i, pdb) for i, pdb in enumerate(self.pdb_codes) if pdb not in self.skip_pdb_list
    #         ]

        
    #     for index_pdb_tuple in tqdm(index_pdb_tuples):
    #         try:
    #             (
    #                 i,
    #                 pdb,
    #             ) = index_pdb_tuple  # NOTE: here, we unpack the tuple to get each PDB's original index in `self.pdb_codes`
    #             path = self.pdb_dir / f"{pdb}.{self.format}"
    #             if path.exists():
    #                 path = str(path)
    #             else:
    #                 logger.error(f"{pdb} not found in raw directory. Are you sure it's available and has the format {self.format}?")
    #                 # add to skip list
    #                 self.skip_pdb_list.add(pdb)
    #                 continue

    #             # Load structure using atomworks
    #             result = parse(filename=path,
    #                     hydrogen_policy="remove")
                
    #             asym_unit = result["asym_unit"]
    #             if len(asym_unit) > 1:
    #                 logger.warning(f"{pdb} has multiple models, using the first one.")
    #             model = asym_unit[0]

                
    #             coords, residue_type, chains, residue_id = atomarray_to_tensors(model)


    #             # Create PyG Data object
    #             graph = Data(
    #                 coords=coords,
    #                 residue_type=residue_type,
    #                 chains=chains,
    #                 residue_id=residue_id,
    #                 #residues=residue_id  # Keep for compatibility
    #             )
    #         except Exception as e:
    #             logger.error(f"Error processing {pdb}: {e}")  # type: ignore
    #             self.skip_pdb_list.add(pdb)   
    #             continue

    #         fname = f"{Path(pdb).name}.pt"

                
    #         graph.id = fname.split(".")[0]

    #         torch.save(graph, Path(self._processed_dir) / fname)

    #         self._processed_files.append(fname)

    #     # Save the skip list
    #     with open(skip_list_path, "w") as f:
    #         f.write("\n".join(list(self.skip_pdb_list)))

    #     logger.info(f"Skipped {len(self.skip_pdb_list)} structures")
    #     logger.info("Completed processing.")

    ###############################################################
    # MODIFIED
    def get(self, idx: int) -> Data:
        if self.in_memory:
            data = copy.deepcopy(self.data[idx])
            if data is None:
                logger.error(f"Error loading {self.get_pdb_codes[idx]}")
                return None
        else:
            fname = f"{self.get_pdb_codes[idx]}.pt"

        try:
            new_data = torch.load(Path(self._processed_dir) / fname, map_location='cpu', weights_only = False)

            # FIX #4: NaN handling now done in preprocessing, removed from here

            # remove side chain atoms in [N, 37, 3], keep only backbone atoms (N, CA, C, O) at indices [0, 1, 2, 3]
            # change [:, 4:, :] to FILL
            if self.remove_sidechains:
                new_data.coords_true = new_data.coords.clone()
                new_data.coords[:, 4:, :] = FILL


            if self.crop and self.crop_len is not None and new_data.coords.shape[0] > self.crop_len:
                new_data.coords = new_data.coords[:self.crop_len,:, :]
                if hasattr(new_data, 'coords_true'):
                    new_data.coords_true = new_data.coords_true[:self.crop_len,:, :]
                new_data.residue_type = new_data.residue_type[:self.crop_len] 
                new_data.chains = new_data.chains[:self.crop_len] 
                new_data.residue_id = new_data.residue_id[:self.crop_len]
                #new_data.residues = new_data.residues[:self.crop_len]
            if self.rand_slice:
                start_idx = 0
                data_len = new_data.coords.shape[0]
                end_idx = data_len
                
                # every 1 in 10 times, do NOT crop
                if torch.rand(1) > 0.10: # 0.03:                  
                    # sample a slice fraction from 0.6 to 0.9
                    if self.slice_range is not None:
                        min_percent, max_percent = self.slice_range
                    else: 
                        min_percent, max_percent = 0.6, 0.9
                    slice_fraction = (torch.rand(1) * (max_percent - min_percent) + min_percent).item()
                    #slice_fraction = (torch.rand(1) * 0.3 + 0.6).item()
                    #logger.info(f"Slicing {fname} with fraction {slice_fraction}")
                    window_size = int(data_len * slice_fraction)
                    if window_size == data_len:
                        #logger.info(f"Window size is the same as data length, not slicing {fname}")
                        start_idx = 0
                    else:
                        start_idx = torch.randint(0, data_len - window_size + 1, (1,)).item()
                    end_idx = start_idx + window_size
                    new_data.coords = new_data.coords[start_idx:end_idx,:, :]
                    if hasattr(new_data, 'coords_true'):
                        new_data.coords_true = new_data.coords_true[start_idx:end_idx,:, :]
                    new_data.residue_type = new_data.residue_type[start_idx:end_idx]
                    new_data.chains = new_data.chains[start_idx:end_idx]
                    new_data.residue_id = new_data.residue_id[start_idx:end_idx]
                    #new_data.residues = new_data.residues[start_idx:end_idx]
                new_data.slice_idx = torch.tensor([start_idx, end_idx])
                #else:
                    #logger.info(f"Not slicing {fname}")
        except Exception as e:
            logger.error(f"Error loading {fname}: {e}")
            return None


        if new_data is None:
            logger.error(f"Error loading {fname} as it is None")
            return None
        
        # Set this to ensure proper batching behaviour
        new_data.x = torch.zeros(new_data.coords.shape[0])  # type: ignore




        return new_data
    

    def __getitem__(self,
        idx: Union[int, np.integer, IndexType],) -> Union['Dataset', BaseData]:

        data = self.get(self.indices()[idx])
        
        if data is None:
            logger.error(f"Error loading {self.pdb_codes[idx]}")
            
        return data


        
