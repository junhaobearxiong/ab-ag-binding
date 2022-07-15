import pickle
import numpy as np
from tqdm import tqdm

from functional_groups import SMARTS_ATOM_MAP
from utils import AA_ATOMS_ORDER, BACKBONE_ATOMS, VDM_TYPES, AA_NAMES


''' functions to compute vdms from coordinates of contacts'''

def get_label(vdm_type):
    aa, ifg, sidechain_bonding, order = vdm_type
    return f"{aa}-{ifg}-sidechain_bonding_{sidechain_bonding}-order_{order}"


def compute_vdms(contacts_file, vdm_dir):
    with open(contacts_file, "rb") as f:
        contacts = pickle.load(f)

    vdms = {vdm_type: [[], []] for vdm_type in VDM_TYPES}

    contained_ifgs = {aa: set() for aa in AA_NAMES}
    for ifg in SMARTS_ATOM_MAP.keys():
        for aa in SMARTS_ATOM_MAP[ifg].keys():
            contained_ifgs[aa].add(ifg)

    remaining_ifgs = set(SMARTS_ATOM_MAP.keys())

    # helper functions to add a vdm for an (aa, ifg) pair
    def update_vdms(aa1_data, aa2_data, aa1_fxl_atoms, aa2_fxl_atoms, order):
        aa1 = aa1_data[0] # the amino acid
        aa2, aa2_atoms = aa2_data[0], aa2_data[1] # the source of ifg

        for ifg in contained_ifgs[aa2]:

            # check if any atom in the ifg is actually part of the contact between the two aa
            # if not skip this ifg
            if not any([atom in aa2_fxl_atoms for atom in SMARTS_ATOM_MAP[ifg][aa2]]):
                continue

            ifg_i = [aa2_atoms.index(atom) for atom in SMARTS_ATOM_MAP[ifg][aa2]]
            # coordinates of the ifg atoms in the aa atom coordinates array
            ifg_pts = [aa2_data[2][i] for i in ifg_i]

            sidechain_bonding = not all([atom in BACKBONE_ATOMS for atom in aa1_fxl_atoms])

            label = (aa1, ifg, sidechain_bonding, order)
            
            # Separate backbone/ifg and sidechain atoms to perform downstream 
            # rmsd computations on backbone/ifg only
            vdms[label][0].append(np.array(list(aa1_data[2][:3]) + ifg_pts).astype('float32'))
            vdms[label][1].append(aa1_data[2][3:len(AA_ATOMS_ORDER[aa1])].astype('float32'))

            if not ifg in remaining_ifgs:
                continue
            remaining_ifgs.remove(ifg)

    # get vdms for every pair of amino acids in the contact file
    for aa_pair in tqdm(contacts.keys()):
        aa1, aa2 = aa_pair

        for contact in contacts[aa_pair]:
            contact[0][1][:] = [x.strip() for x in contact[0][1]]
            contact[1][1][:] = [x.strip() for x in contact[1][1]]

            # atoms in each aa should match AA_ATOMS_ORDER[aa] exactly up to the
            # second-to-last non-H atom. We ignore the following atoms, which are
            # likely 'OXT', followed by any hydrogens.
            '''
            if len(contact[0][1]) != len(AA_ATOMS_ORDER[aa1]) or \
                    len(contact[1][1]) != len(AA_ATOMS_ORDER[aa2]) or \
                    not all([contact[0][1][i]==AA_ATOMS_ORDER[aa1][i]
                        for i in range(len(AA_ATOMS_ORDER[aa1]))]) or \
                    not all([contact[1][1][i]==AA_ATOMS_ORDER[aa2][i]
                        for i in range(len(AA_ATOMS_ORDER[aa2]))]):
                continue
            '''
            # force atoms in each aa to match AA_ATOMS_ORDER[aa] exactly
            # since we only have heavy atoms
            if not (contact[0][1] == AA_ATOMS_ORDER[aa1]) or not (contact[1][1] == AA_ATOMS_ORDER[aa2]):
                continue

            # these are the atoms that are actually in contact
            aa1_fxl_atoms, aa2_fxl_atoms = set(), set()
            if len(contact[2]) == 0:
                continue

            for pair in contact[2]:
                aa1_fxl_atoms.add(pair[0])
                aa2_fxl_atoms.add(pair[1])
                
            update_vdms(contact[0], contact[1], aa1_fxl_atoms, aa2_fxl_atoms, "ab_ag")
            update_vdms(contact[1], contact[0], aa2_fxl_atoms, aa1_fxl_atoms, "ag_ab")

    # write vdms to file
    for vdm_type in VDM_TYPES:

        vdms[vdm_type][0] = np.array(vdms[vdm_type][0])
        vdms[vdm_type][1] = np.array(vdms[vdm_type][1])

        label = get_label(vdm_type)

        with open(f"{vdm_dir}/{label}.pkl", 'wb') as f:
            pickle.dump(vdms[vdm_type], f)
    
    return vdms



''' functions to compute statistics from vdm coordinates'''

# atoms: np.array([K, 3])
# K: number of pts
def compute_centroid(atoms):
    return np.sum(atoms, axis=0) / atoms.shape[0]


# coords: np.array([N, M, 3])
# N is the number of vdm samples
# M is the number of atoms in each sample, including both the 3 aa backbone and ifg atoms
# return np.array(N)
def compute_centroid_distances(coords):
    distances = np.zeros(coords.shape[0])
    for i in range(distances.size):
        backbone_coords = coords[i][:3]
        ifg_coords = coords[i][3:]
        distances[i] = np.linalg.norm(compute_centroid(backbone_coords) - compute_centroid(ifg_coords))
    return distances


def cosine_sim(x, y):
    return x @ y / (np.linalg.norm(x) * np.linalg.norm(y))


# coords: np.array([N, M, 3])
# N is the number of vdm samples
# M is the number of atoms in each sample, including both the 3 aa backbone and ifg atoms
# return: np.array([N, M-3])
def compute_angles_cosine(coords):
    # number of vdm samples * number of ifg atoms
    if coords.shape[0] > 0:
        angles = np.zeros((coords.shape[0], coords.shape[1] - 3))
        for i in range(angles.shape[0]):
            backbone_centroid = compute_centroid(coords[i][:3])
            ifg_centroid = compute_centroid(coords[i][3:])
            v1 = ifg_centroid - backbone_centroid
            for j in range(angles.shape[1]):
                atom_coord = coords[i][3 + j]
                v2 = atom_coord - ifg_centroid
                angles[i, j] = cosine_sim(v1, v2)
    else:
        angles = np.zeros((0, 0))
    return angles


# v is a 3-length np.array
def cart2sph(v):
    x, y, z = v[0], v[1], v[2]
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, el, az


# coords: np.array([N, M, 3])
# first 3 atoms are backbone atoms of the aa
# the remaining atoms are the ifg atoms
# return np.array([N, 3])
def compute_centered_ifg_centroids(coords):
    backbone_centroids = np.zeros((coords.shape[0], 3))
    ifg_centroids = np.zeros((coords.shape[0], 3))
    for i in range(coords.shape[0]):
        backbone_centroids[i] = compute_centroid(coords[i][:3])
        ifg_centroids[i] = compute_centroid(coords[i][3:])
    # center the ifg centroids relative to their corresponding backbone centroids
    # which implicitly centers all the ifg centroids around (0, 0, 0) as the origin 
    ifg_centroids_centered = ifg_centroids - backbone_centroids
    return ifg_centroids_centered


# sph_coords: np.array(N)
def compute_bins(arr, num_bins=5):
    bins = np.linspace(arr.min(), arr.max(), num_bins)
    return np.digitize(arr, bins)


if __name__ == "__main__":
    dir_prefix = "/storage/bear/projects/ab-ag-binding/data/sabdab/ag-protein_res-3.2/chothia_selected/"
    contact_file = dir_prefix + "raw_contacts/Hchain-filtered-coords/all_coords.pkl"
    vdm_dir = dir_prefix + "vdms/Hchain-filtered/"
    compute_vdms(contact_file, vdm_dir)