import pickle 
from joblib import Parallel, delayed
import argparse
from geometricus import MomentInvariants, SplitType, GeometricusEmbedding
import umap
import numpy as np


# for exception handling
def get_invariants(pdb_file, chain, split_type, split_size):
    try:
        invariants = MomentInvariants.from_pdb_file(pdb_file=pdb_file, chain=chain, split_type=split_type, split_size=split_size)
    except AttributeError:
        print(f"caught AttributionError for file: {pdb_file}, chain: {chain}")
        return
    return invariants

# embed a set of pdb chains into shapemer object
def get_shapmers(pdb_files, chain_ids, split_type, split_size, resolution=2, n_jobs=-1):
    if len(pdb_files) != len(chain_ids):
        raise ValueError("`pdb_file_list` and `chain_id_list` need to have the same length")

    invariants = Parallel(n_jobs=n_jobs)(delayed(get_invariants)(
            pdb_file=pdb_files[i],
            chain=chain_ids[i],
            split_type=split_type,
            split_size=split_size
        )
        for i in range(len(pdb_files))
    )
    embedder = GeometricusEmbedding.from_invariants(invariants=invariants, resolution=resolution)
    return embedder


def get_umap(radius_embedder_file, kmer_embedder_file):
    with open(radius_embedder_file, "rb") as f:
        radius_embedder = pickle.load(f)
    with open(kmer_embedder_file, "rb") as f:
        kmer_embedder = pickle.load(f)
    reducer = umap.UMAP(metric="cosine", n_components=2)
    reduced = reducer.fit_transform(np.hstack((radius_embedder.embedding, kmer_embedder.embedding)))
    return reduced


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("protein", metavar="PROTEIN", choices=["ab", "ag"])
    parser.add_argument("-t", "--split_type", metavar="TYPE", choices = ["radius", "kmer"], default="radius")
    parser.add_argument("-s", "--split_size", metavar="SIZE", type=int, default=10)
    parser.add_argument("-r", "--resolution", metavar="RES", type=float, default=2)
    args = parser.parse_args()

    data_folder_name = "ag-protein_res-3.2"
    input_dir = f"/storage/bear/sabdab/{data_folder_name}/chothia/"
    output_dir = f"/storage/bear/ab-ag-binding/outputs/sabdab/{data_folder_name}/"

    with open(output_dir + "contacts_Hchain.pkl", "rb") as f:
        all_contacts = pickle.load(f)

    # get shapemer for all ab and ag chains in with contacts
    shapemer_ids = [None] * len(all_contacts)
    for i, ids in enumerate(all_contacts.keys()):
        pdb_id = ids[0]
        if args.protein == "ab":
            chain_id = ids[1]
        else:
            chain_id = ids[2]
        pdb_file = input_dir + f"{pdb_id}.pdb"
        shapemer_ids[i] = (pdb_file, chain_id)
    pdb_files, chain_ids = list(zip(*set(shapemer_ids)))

    if args.split_type == "radius":
        split_type =  SplitType.RADIUS
    else:
        split_type = SplitType.KMER

    embedder = get_shapmers(pdb_files=pdb_files, chain_ids=chain_ids, split_type=split_type, split_size=args.split_size, resolution=args.resolution)
    with open(output_dir + f"shapemers-{args.split_type}_{args.split_size}-resolution_{args.resolution}-{args.protein}.pkl", "wb") as f:
        pickle.dump(embedder, f)
    print(f"finish {args.protein} type={args.split_type}, size={args.split_size}, res={args.resolution}")