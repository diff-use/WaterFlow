
from os import PathLike
from pathlib import Path

import numpy as np
import torch

from atomworks.common import exists
from atomworks.constants import AF3_EXCLUDED_LIGANDS, GAP, STANDARD_AA, STANDARD_DNA, STANDARD_RNA

from atomworks.ml.transforms.af3_reference_molecule import GetAF3ReferenceMoleculeFeatures
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
    CopyAnnotation,
)

from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RandomRoute,
    SubsetToKeys,
    Transform,
)
from atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.transforms.center_random_augmentation import CenterRandomAugmentation
from atomworks.ml.transforms.chirals import AddAF3ChiralFeatures
from atomworks.ml.transforms.covalent_modifications import (
    FlagAndReassignCovalentModifications,
)
from atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3

from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
)

from atomworks.ml.transforms.base import Transform
from atomworks.ml.example_id import parse_example_id
from SLAE.io.atom_tensor import atomarray_to_tensors
from SLAE.util.constants import FILL
from torch_geometric.data import Data

class ToGraph(Transform):
    """Convert an AtomArray to atom-level tensors for model input."""

    def __init__(self):
        super().__init__()

    def forward(self, data: dict) -> dict:
        # subset to only the query pn_unit(s)!
        # IMPORTANT!
        data["atom_array"] = data["atom_array"][data["atom_array"].pn_unit_iid == data["query_pn_unit_iids"]]
        atom_array = data["atom_array"]
        coords, residue_type, chains, residue_id = atomarray_to_tensors(atom_array)

        nan_mask = torch.isnan(coords)
        if nan_mask.any():
            coords[nan_mask] = FILL

        # remove coords [N, 37, 3] that have rows of all FILL values in the 37 and 3 dimensions, then adjust residue_type, chains, residue_id accordingly
        valid_res_mask = ~(coords == FILL).all(dim=-1).all(dim=-1)
        residue_type = residue_type[valid_res_mask]
        chains = chains[valid_res_mask]
        residue_id = residue_id[valid_res_mask]

        coords = coords[valid_res_mask]
        # Create PyG Data object
        
        graph = Data(
            coords=coords,
            residue_type=residue_type,
            chains=chains,
            residue_id=residue_id,
            #residues=residue_id  # Keep for compatibility
        )
        graph.x = torch.zeros(graph.coords.shape[0]) 
        example_id = parse_example_id(data['example_id'])
        graph.id = f"{example_id['pdb_id']}_{example_id['assembly_id']}_{example_id['query_pn_unit_iids'][0]}"
        data["graph"] = graph

        return data
    
def build_transform_pipeline(
    *,
    # Training or inference (required)
    is_inference: bool,  # If True, we skip cropping, etc.
    # Crop params
    crop_size: int = 512,
    crop_center_cutoff_distance: float = 15.0,
    crop_contiguous_probability: float = 1, # Default to contiguous cropping
    crop_spatial_probability: float = 0,
    max_atoms_in_crop: int | None = None,
    # Undesired res names
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
) -> Transform:
    """Build the AF3 pipeline with specified parameters.

    This function constructs a pipeline of transforms for processing protein structures
    in a manner similar to AlphaFold 3. The pipeline includes steps for removing hydrogens,
    adding annotations, atomizing residues, cropping

    Args:
        crop_size (int, optional): The size of the crop. Defaults to 384.
        crop_center_cutoff_distance (float, optional): The cutoff distance for spatial cropping.
            Defaults to 15.0.
        crop_contiguous_probability (float, optional): The probability of using contiguous cropping.
            Defaults to 0.5.
        crop_spatial_probability (float, optional): The probability of using spatial cropping.
            Defaults to 0.5.
        conformer_generation_timeout (float, optional): The timeout for conformer generation in seconds.
            Defaults to 10.0.

    Returns:
        Transform: A composed pipeline of transforms.

    Raises:
        AssertionError: If the crop probabilities do not sum to 1.0, if the crop size is not positive,
        or if the crop center cutoff distance is not positive.

    Note:
        The cropping method is chosen randomly based on the provided probabilities.
        The pipeline includes steps for processing the structure, adding annotations,
        and generating features required for AF3-like predictions.

    Reference:
        `AlphaFold 3 Supplementary Information <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf>`_
    """

    if (crop_contiguous_probability > 0 or crop_spatial_probability > 0) and not is_inference:
        assert np.isclose(
            crop_contiguous_probability + crop_spatial_probability, 1.0, atol=1e-6
        ), "Crop probabilities must sum to 1.0"
        assert crop_size > 0, "Crop size must be greater than 0"
        assert crop_center_cutoff_distance > 0, "Crop center cutoff distance must be greater than 0"


    transforms = [
        AddData({"is_inference": is_inference}),
        RemoveHydrogens(),
        FilterToSpecifiedPNUnits(
            extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"
        ),  # Filter to non-clashing PN units
        RemoveTerminalOxygen(),
        RemoveUnresolvedPNUnits(),
        RemovePolymersWithTooFewResolvedResidues(min_residues=10),
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        # NOTE: For inference, we must keep UNL to support ligands that are not in the CCD
        HandleUndesiredResTokens(undesired_res_tokens=undesired_res_names),  # e.g., non-standard residues
        FlagAndReassignCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(allow_overwrite=True),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
    ]

    # Crop

    # ... crop around our query pn_unit(s) early, since we don't need the full structure moving forward
    cropping_transform = Identity()
    if crop_size is not None:
        cropping_transform = RandomRoute(
            transforms=[
                CropContiguousLikeAF3(
                    crop_size=crop_size,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms_in_crop,
                ),
                CropSpatialLikeAF3(
                    crop_size=crop_size,
                    crop_center_cutoff_distance=crop_center_cutoff_distance,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms_in_crop,
                ),
            ],
            probs=[crop_contiguous_probability, crop_spatial_probability],
        )

    transforms.append(
        ConditionalRoute(
            condition_func=lambda data: data.get("is_inference", False),
            transform_map={
                True: Identity(),
                False: cropping_transform,
                # Default to Identity during inference (`is_inference == True`)
            },
        )
    )

    # convert to Graph Data
    transforms.append(ToGraph())


    """
    keys_to_keep = [
        "atom_array",
        "example_id",
        "extra_info",
        "graph",
    ]


    transforms += [
        # Subset to only keys necessary
        SubsetToKeys(keys_to_keep)
    ]
    """
    # ... compose final pipeline
    pipeline = Compose(transforms)

    return pipeline