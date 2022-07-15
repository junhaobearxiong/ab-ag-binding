import pickle 
from joblib import Parallel, delayed
import umap
import numpy as np
from geometricus import MomentInvariants, GeometricusEmbedding


class MomentInvariantsSubset(MomentInvariants):
    # subset a MomentInvariant to a list of residues
    # `subset_indices`: list of residue indices to subset, 0-index
    def subset(self, subset_indices):
        for i in subset_indices:
            if i < 0 or i > self.length - 1:
                raise ValueError(f"index {i} in `subset_indices is outside valid range")
        self.subset_indices = subset_indices
        self.length = len(subset_indices)
        self.coordinates = self.coordinates[subset_indices, :]
        self.residue_splits = [self.residue_splits[i] for i in subset_indices]
        self.original_indices = self.original_indices[subset_indices]
        self.sequence = "".join([self.sequence[i] for i in subset_indices])
        self.split_indices = [self.split_indices[i] for i in subset_indices]
        self.moments = self.moments[subset_indices, :]


def get_invariants(pdb_file, chain, split_type, split_size, subset_indices=None):
    invariants = MomentInvariantsSubset.from_pdb_file(pdb_file=pdb_file, chain=chain, split_type=split_type, split_size=split_size)
    if subset_indices is not None:
        invariants.subset(subset_indices)
    return invariants


# embed a set of pdb chains into shapemer object
# `subset_indices`: List[List[int]]
def get_shapemers(pdb_files, chain_ids, split_type, split_size, resolution=2, n_jobs=-1, subset_indices=None):
    if len(pdb_files) != len(chain_ids):
        raise ValueError("`pdb_files` and `chain_ids` need to have the same length")

    if subset_indices is None:
        invariants = Parallel(n_jobs=n_jobs)(delayed(get_invariants)(
                pdb_file=pdb_files[i],
                chain=chain_ids[i],
                split_type=split_type,
                split_size=split_size
            )
            for i in range(len(pdb_files))
        )
    else:
        if len(pdb_files) != len(subset_indices):
            raise ValueError("`pdb_files` and `subset_indices` need to have the same length")
        invariants = Parallel(n_jobs=n_jobs)(delayed(get_invariants)(
                pdb_file=pdb_files[i],
                chain=chain_ids[i],
                split_type=split_type,
                split_size=split_size,
                subset_indices=subset_indices[i]
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
    assert radius_embedder.protein_keys == kmer_embedder.protein_keys

    reducer = umap.UMAP(metric="cosine", n_components=2)
    reduced = reducer.fit_transform(np.hstack((radius_embedder.embedding, kmer_embedder.embedding)))
    return radius_embedder.protein_keys, reduced


# count the number of Ca in each protein in an embedder
def get_protein_size(embedder_file):
    with open(embedder_file, "rb") as f:
        embedder = pickle.load(f)
    protein_keys = embedder.protein_keys
    protein_sizes = list(map(lambda key: len(embedder.invariants[key].residue_splits), protein_keys))
    return protein_keys, protein_sizes