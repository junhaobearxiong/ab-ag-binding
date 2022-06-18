from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities
# from Bio.PDB.Polypeptide import is_aa

import itertools
import pandas as pd
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed


# find all residues between a pair of ab and ag chains 
# with any heavy atom < 5A away from each other
# return list of tuples: [(tuple(ids), set(contacts))]
def get_contacts_ab_ag(pdb_file, pdb_id, chain_id_ab, chain_id_ag, radius=5, heavy=True):
    parser = PDBParser(PERMISSIVE=1)
    struct = parser.get_structure(pdb_id, pdb_file)
    assert len(struct) == 1
    chain_ab = struct[0][chain_id_ab]
    # the max index of chothia numbering scheme
    if heavy:
        max_idx = 113
    else:
        max_idx = 107

    contacts = []   # all contacts of a single ab chain
    # `chain_id_ag` is taken from `summary`, which might contains multiple chains
    for chain_id_ag_single in chain_id_ag.split(" | "):
        chain_ag = struct[0][chain_id_ag_single]
        contacts_ag = set()    # all contacts of an ab chain with a single ag chain 
        for res_ab in chain_ab:
            # TODO: some aa in ab chains have residue number > max of numbering scheme, ignoring those for now
            # also ignoring het residues
            res_ab_id = res_ab.get_id()
            if res_ab_id[0] == " " and res_ab_id[1] <= max_idx:
                # can also avoid iterating over atoms and define contact based on CA distance
                # TODO: not sure if need to check for hydrogen
                for atom_ab in res_ab:
                    atoms_ag = unfold_entities(chain_ag, 'A')
                    ns = NeighborSearch(atoms_ag)
                    res_contact_ag = ns.search(center=atom_ab.get_coord(), level="R", radius=radius)
                    contacts_ag.update([(res_ab_id, res_ag.get_id()) for res_ag in res_contact_ag if res_ag.get_id()[0] == " "])
        contacts += [((pdb_id, chain_id_ab, chain_id_ag_single), contacts_ag)]
    return contacts


if __name__ == "__main__":
    data_folder_name = "ag-protein_res-3.2"
    input_dir = f"/storage/bear/sabdab/{data_folder_name}/"
    output_dir = f"/storage/bear/ab-ag-binding/outputs/sabdab/{data_folder_name}/"
    summary = pd.read_csv(input_dir + "20220609_0108913_summary.tsv", delimiter='\t')
    chain = "H"

    if chain == "H":
        chain_col = "Hchain"
    elif chain == "L":
        chain_col = "Lchain"
    else:
        raise ValueError("`chain` needs to be `H` or `L`")
    # TODO: for now, only take complex that has only protein (no peptide, carb, or nucleid acid)
    summary_subset = summary[
        ~summary[chain_col].isnull() & 
        ~summary["antigen_chain"].isnull() &
        summary["antigen_type"].isin(["protein", "protein | protein", "protein | protein | protein", "protein | protein | protein | protein"])
    ].reset_index(drop=True)  
    
    outputs = Parallel(n_jobs=-1)(delayed(get_contacts_ab_ag)(
            pdb_file=input_dir + f"chothia/{summary_subset.loc[i, 'pdb']}.pdb",
            pdb_id=summary_subset.loc[i, "pdb"],
            chain_id_ab=summary_subset.loc[i, chain_col],
            chain_id_ag=summary_subset.loc[i, "antigen_chain"]
        )
        # for i in tqdm(range(20))
        for i in tqdm(range(summary_subset.shape[0]))
    )
    print("finish processing all structures")

    # need to unpack nested list
    contacts_dict = dict([contacts_tuple for contacts_list in outputs for contacts_tuple in contacts_list])
    with open (output_dir + f"contacts_{chain}chain.pkl", "wb") as f:
        pickle.dump(contacts_dict, f)