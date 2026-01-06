import pyrosetta
from pyrosetta import rosetta
import pandas as pd
import os

# Initialize PyRosetta (only needs to be done once)
# mute output to keep the console clean
pyrosetta.init("-mute all")

def get_pairwise_scores(pdb_path_list):
    """
    Calculates pairwise fa_sol, fa_elec, and total hbond scores for a list of PDBs.
    
    Args:
        pdb_path_list (list): List of file paths to PDB files.
        
    Returns:
        dict: A dictionary where keys are PDB filenames and values are lists of 
              pairwise interaction dictionaries.
    """
    
    # 1. Setup the standard full-atom ScoreFunction (usually ref2015)
    scorefxn = pyrosetta.create_score_function("ref2015")
    
    # Define the specific ScoreTypes corresponding to the image description
    score_type_sol = rosetta.core.scoring.fa_sol
    score_type_elec = rosetta.core.scoring.fa_elec
    
    # Define the list of hbond terms to sum up
    # (hbond_sr_bb = short range backbone, lr = long range, sc = sidechain)
    hbond_types = [
        rosetta.core.scoring.hbond_sr_bb,
        rosetta.core.scoring.hbond_lr_bb,
        rosetta.core.scoring.hbond_bb_sc,
        rosetta.core.scoring.hbond_sc
    ]
    
    all_data = {}

    for pdb_path in pdb_path_list:
        if not os.path.exists(pdb_path):
            print(f"Skipping {pdb_path}: File not found.")
            continue
            
        print(f"Processing {pdb_path}...")
        
        try:
            # Load the Pose
            pose = pyrosetta.pose_from_pdb(pdb_path)
            
            # Score the pose first to populate the energy graph
            scorefxn(pose)
            
            n_res = pose.total_residue()
            pairwise_data = []

            # Loop through every pair of residues (Upper Triangle)
            for i in range(1, n_res + 1):
                res_i = pose.residue(i)
                for j in range(i + 1, n_res + 1):
                    res_j = pose.residue(j)
                    
                    # Create an EnergyMap to store the specific interaction energies
                    emap = rosetta.core.scoring.EMapVector()
                    
                    # Evaluate Context-Independent (ci) and Context-Dependent (cd) 2-body energies
                    scorefxn.eval_ci_2b(res_i, res_j, pose, emap)
                    scorefxn.eval_cd_2b(res_i, res_j, pose, emap)
                    
                    # Extract the specific terms requested in the image
                    val_sol = emap[score_type_sol]
                    val_elec = emap[score_type_elec]
                    
                    # Sum the hydrogen bonding terms
                    val_hbond = sum([emap[t] for t in hbond_types])
                    
                    # Only store pairs with non-zero interaction (optional optimization)
                    # If you strictly need ALL pairs, remove the `if` check below.
                    if abs(val_sol) > 0.001 or abs(val_elec) > 0.001 or abs(val_hbond) > 0.001:
                        pairwise_data.append({
                            "residue_1": i,
                            "residue_2": j,
                            "fa_sol": val_sol,
                            "fa_elec": val_elec,
                            "hbond": val_hbond
                        })

            all_data[pdb_path] = pairwise_data

        except Exception as e:
            print(f"Error processing {pdb_path}: {e}")

    return all_data


# list of paths
pdb_dataset = [
    "/sb/wankowicz_lab/data/srivasv/pdb_redo_data/6eey/6eey_final.pdb",
    "/sb/wankowicz_lab/data/srivasv/pdb_redo_data/1a2p/1a2p_final.pdb",
]

results = get_pairwise_scores(pdb_dataset)

for key in results.keys():
    print(results[key][0])

# if results:
#     first_pdb = list(results.keys())[0]
#     df = pd.DataFrame(results[first_pdb])
#     print(f"\nResults for {first_pdb}:")
#     print(df.head())