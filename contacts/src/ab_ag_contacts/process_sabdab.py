# helper functions for processing sabdab data
import itertools
import pandas as pd
import pickle
from prody.atomic import subset
from tqdm import tqdm
from joblib import Parallel, delayed
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import Select, PDBIO
from Bio.PDB.Polypeptide import is_aa
from utils import MAX_RESNUM_HEAVY, MAX_RESNUM_LIGHT


''' general helper functions '''
# for a pdb_id, return Dict{chain_id: chain_type}
def get_chain_type_dict(pdb_id, summary):
    subset = summary.loc[summary["pdb"] == pdb_id]
    chain_type_dict = {}
    for chain_id in subset["Hchain"][~subset["Hchain"].isna()].values:
        chain_type_dict[chain_id] = "heavy"
    for chain_id in subset["Lchain"][~subset["Lchain"].isna()].values:
        # if a light chain id == a heavy chain id, this chain is a scfv 
        if chain_id in chain_type_dict.keys():
            chain_type_dict[chain_id] = "scfv"
        else:
            chain_type_dict[chain_id] = "light"
    for chain_id in subset["antigen_chain"][~subset["antigen_chain"].isna()].values:
        for chain_id_single in chain_id.split(" | "):
            chain_type_dict[chain_id_single] = "antigen"
    return chain_type_dict


# subset the summary csv file of a batch of sabdab downloads
# ids: List[(pdb_id, chain_id)] for chain_type
# when not None, subset to rows with the corresponding ids
def subset_summary(file_path, ids=None, chain_type=None):
    summary = pd.read_csv(file_path, delimiter='\t')
    summary = summary[
        ~summary["Hchain"].isnull() & 
        ~summary["antigen_chain"].isnull() &
        summary["antigen_type"].isin(["protein", "protein | protein", "protein | protein | protein", "protein | protein | protein | protein"])
    ].reset_index(drop=True)

    if (ids is not None and chain_type is None) or (ids is None and chain_type is not None):
        raise ValueError("`ids` and `chain_type` both need to be `None` or `not None`")

    if ids is not None and chain_type is not None:
        all_ids = get_all_ids_chain_type(summary, chain_type)
        row_idx = [i for i in range(len(all_ids)) if all_ids[i] in ids]
        summary = summary.loc[row_idx].reset_index(drop=True)

    return summary


# get all (pdb_id, chain_id) pairs in a summary table for a given chain_type
# for antigen, each chain_id is a single chain
def get_all_ids_chain_type(summary, chain_type):
    if chain_type == "heavy":
        chain_col = "Hchain"
    elif chain_type == "light":
        chain_col = "Lchain"
    elif chain_type == "antigen":
        chain_col = "antigen_chain"
    elif chain_type == "scfv":
        chain_col = "scfv"
    else:
        raise ValueError("`chain_type` must be 'heavy', 'light', 'antigen' or 'scfv")

    if chain_col == "scfv":
        mask = summary[chain_col]
    else:
        mask = ~summary[chain_col].isna()
    subset = summary.loc[mask].reset_index(drop=True)

    if chain_type == "antigen":
        ids = []
        for i in range(subset.shape[0]):
            for chain_id in subset.loc[i, chain_col].split(" | "):
                ids.append((subset.loc[i, "pdb"], chain_id))
    elif chain_type == "scfv":
        # TODO: 5cbe is the only scfv that has both chains labeled separately
        # when analyzing light chains, should include it
        ids = list(zip(subset["pdb"], subset["Hchain"]))
    else:
        ids = list(zip(subset["pdb"], subset[chain_col]))
    return ids


# get sequence of a single chain
def get_seq_of_chain(pdb_file, pdb_id, chain_id, seq_id=None, seq_description=None):
    for s in SeqIO.parse(pdb_file, "pdb-atom"):
        if chain_id == s.id.split(":")[1]:
            if seq_id is None:
                s.id = f"{pdb_id}_{chain_id}"
            else:
                s.id = seq_id
            if seq_description is None:
                s.description = ""
            else:
                s.description = seq_description
            return s


''' helper functions for pdb file '''
# subset a sabdab pdb file to the relavant chain, residue, atom
class SabdabSelect(Select):
    def __init__(self, chain_type_dict):
        super().__init__()
        self.chain_type_dict = chain_type_dict

    def accept_chain(self, chain):
        # only accept chains present in chain_type_dict
        if chain.get_id() not in self.chain_type_dict.keys():
            return False
        else:
            return True

    def accept_residue(self, residue):
        # only accept standard amino acid residues
        if not is_aa(residue) or residue.get_id()[0] != " ":
            return False
        
        # only accept residues in fv for ab chains 
        chain_id = residue.get_full_id()[2]
        if self.chain_type_dict[chain_id] == "heavy":
            max_resnum = MAX_RESNUM_HEAVY
        elif self.chain_type_dict[chain_id] == "light":
            max_resnum = MAX_RESNUM_LIGHT
        else:
            return True
        # check residue number (assumed to be renumbered by numbering system)
        if residue.get_id()[1] <= max_resnum:
            return True
        else:
            return False

    def accept_atom(self, atom):
        # only accept heavy atoms
        if atom.get_name() == "H":
            return False
        else:
            return True


# for every pdb_id in `summary`, subset the corresponding pdb file in `input_dir` according to `FvSelect`
# save a new pdb file to `output_dir`
def select_fv_all(input_dir, output_dir, summary):

    def _select_and_save_struct(pdb_id, chain_type_dict):
        parser = PDBParser()
        struct = parser.get_structure(pdb_id, input_dir + f"{pdb_id}.pdb")
        io = PDBIO()
        io.set_structure(struct)
        io.save(output_dir + f"{pdb_id}.pdb", SabdabSelect(chain_type_dict))
    
    all_pdb_ids = summary["pdb"].unique()
    all_chain_type_dicts = summary.groupby("pdb").apply(lambda x: get_chain_type_dict(x["pdb"].iloc[0], summary))
    Parallel(n_jobs=-1)(delayed(_select_and_save_struct)(
            pdb_id=pdb_id,
            chain_type_dict=all_chain_type_dicts[pdb_id]
        )
        for pdb_id in tqdm(all_pdb_ids)
    )


# input_dir: where sabdab pdb files are located
# output_dir: where fasta files will be output to 
# summary: summary table for metadata of sabdab structures
# get a single fasta file for every chain in the summary of `chain_type`` 
def get_fasta_for_chain(input_dir, output_dir, summary, chain_type="heavy"):
    all_ids = get_all_ids_chain_type(summary, chain_type)

    sequences = Parallel(n_jobs=-1)(delayed(get_seq_of_chain)(
        pdb_file=input_dir + f"{pdb_id}.pdb",
        pdb_id=pdb_id,
        chain_id=chain_id
    ) for pdb_id, chain_id in tqdm(all_ids))

    SeqIO.write(sequences, output_dir + f"chain_{chain_type}.fasta", format="fasta-2line")


''' helper functions for redundancy removal '''
# return List[(pdb_id, chain_id)] by reading from fasta file 
# assume fasta description is ">`pdb_id`_`chain_id`"
def get_ids_from_fasta(fasta_file_path):
    ids = []
    for s in SeqIO.parse(fasta_file_path, "fasta-2line"):
        pdb_id, chain_id = s.id.split("_")
        ids.append((pdb_id, chain_id))
    return ids


# get a single fasta files for every ag chain in summary 
# that is paired with an ab chain in ab_ids
# used after the ab chains are filtered fore redundancy
# ab_ids: List[(pdb_id, chain_id)]
# ab_chain_type: heavy or light
def get_fasta_for_ag_of_ab(input_dir, output_dir, ab_ids, summary, ab_chain_type="heavy"):
    all_ab_ids = get_all_ids_chain_type(summary, ab_chain_type)
    row_idx = [i for i in range(len(all_ab_ids)) if all_ab_ids[i] in ab_ids]
    ag_ids = get_all_ids_chain_type(summary.loc[row_idx], "antigen") 
    
    sequences = Parallel(n_jobs=-1)(delayed(get_seq_of_chain)(
        pdb_file=input_dir + f"{ag_id[0]}.pdb",
        pdb_id=ag_id[0],
        chain_id=ag_id[1],
    ) for i, ag_id in enumerate(tqdm(ag_ids)))

    SeqIO.write(sequences, output_dir + f"chain_antigen-filtered_by_{ab_chain_type}.fasta", format="fasta-2line")


# compare all pairs of fv region of ab
# return Dict{int: List[(pdb_id, chain_id)])}
def get_ab_clusters(input_dir, output_dir, summary, chain_type):
    
    # check if two ab chains are redundant
    # redundancy is defined based on Ferdous & Martin 2018 (AbDb)
    # if there is any residue (defined by residue number and insertion code)
    # that is present in both chains, but with different residue identity
    # then the two chains are NOT redundant
    # otherwise they are redundant
    def ab_chains_redundant(input_dir, pdb_id1, chain_id1, pdb_id2, chain_id2):
        parser = PDBParser()
        chain1 = parser.get_structure(pdb_id1, input_dir + f"{pdb_id1}.pdb")[0][chain_id1]
        chain2 = parser.get_structure(pdb_id2, input_dir + f"{pdb_id2}.pdb")[0][chain_id2]
        # only need to check for common residues in both chains
        # so can iterate over the shorter chain, and skip any residue that's not in the other
        if len(chain1) <= len(chain2):
            chain_short = chain1
            chain_long = chain2
        else:
            chain_short = chain2
            chain_long = chain1

        for res1 in chain_short.get_list():
            try:
                res2 = chain_long[res1.get_id()]
            except KeyError:
                continue
            if res1.get_resname() != res2.get_resname():
                return False
        return True

    if chain_type != "heavy" and chain_type != "light":
        raise ValueError("`chain_type` must be 'heavy' or 'light'")

    all_ids = get_all_ids_chain_type(summary, "heavy")
    # need to keep in memory for masking later
    all_ids_pairs = list(itertools.combinations(all_ids, 2))

    redundancy_mask = Parallel(n_jobs=-1)(delayed(ab_chains_redundant)(
            input_dir=input_dir,
            pdb_id1=pdb_id1,
            chain_id1=chain_id1,
            pdb_id2=pdb_id2,
            chain_id2=chain_id2
        )
        for (pdb_id1, chain_id1), (pdb_id2, chain_id2) in tqdm(all_ids_pairs)
    )
    print("finished comparing all pairs of chains")

    # assignments: (pdb_id, chain_id) --> cluster_idx
    assignments = {}
    current_cluster_idx = 0
    for i, (id1, id2) in tqdm(enumerate((all_ids_pairs))):
        # if two chains are redundant
        if redundancy_mask[i]:
            # if one is already assigned a cluster
            # assign the other chain to the same cluster
            if id1 in assignments.keys():
                assignments[id2] = assignments[id1]
            elif id2 in assignments.keys():
                assignments[id1] = assignments[id2]
            # if neither is assigned a cluster
            # assign both to a new cluster
            # increment cluster index
            else:
                assignments[id1] = current_cluster_idx
                assignments[id2] = current_cluster_idx
                current_cluster_idx += 1
    print("finished assigning clusters")

    # clusters: cluster_idx --> List[(pdb_id, chain_id)]
    clusters = {}
    for ids, cluster_idx in assignments.items():
        if cluster_idx in clusters.keys():
            clusters[cluster_idx].append(ids)
        else:
            clusters[cluster_idx] = [ids]
    # create cluster of size 1 for each ab that is not redundant with any other ab
    for ids in all_ids:
        if ids not in assignments.keys():
            clusters[current_cluster_idx] = [ids]
            current_cluster_idx += 1    

    output_file_path = output_dir + f"clusters_{chain_type}.pkl"
    with open(output_file_path, "wb") as f:
        pickle.dump(clusters, f)
    print(f"output saved to {output_file_path}")


if __name__ == "__main__":
    # TODO: make these command line arguments
    dir_prefix = "/storage/bear/projects/ab-ag-binding/data/sabdab/ag-protein_res-3.2/"
    summary_file_path = dir_prefix + "summary.tsv"
    summary = subset_summary(summary_file_path)
    dir_prefix += "chothia_selected/"

    # run method that needs to be run
    # get_ab_clusters(input_dir, output_dir, summary, "heavy")
    # get_fasta_for_chain(input_dir, output_dir, summary, "antigen")

    # get a list of ab ids after mmseqs clustering
    ab_ids = get_ids_from_fasta(dir_prefix + "clusters/chain_heavy_clu_rep.fasta")

    # get a single fasta file for ags paired with the filtered abs
    get_fasta_for_ag_of_ab(dir_prefix + "pdb/", dir_prefix + "fasta/", ab_ids, summary)