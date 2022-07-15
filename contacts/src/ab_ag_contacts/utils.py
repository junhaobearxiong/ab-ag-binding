import numpy as np
from functional_groups import SMARTS_ATOM_MAP


''' global variables '''

# amino acids
AA_NAMES = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", 
    "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_ABV = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
AA_NAMES_TO_ABV = dict(zip(AA_NAMES, AA_ABV))
AA_NAMES_TO_IDX = dict(zip(AA_NAMES, np.arange(20)))
AA_ABV_TO_IDX = dict(zip(AA_ABV, np.arange(20)))
AA_IDX_TO_ABV = dict(zip(np.arange(20), AA_ABV))


# vdms
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OXT', 'H', 'H2', 'HA2', 'HXT'}
SYMMETRIC_IFGS = {'phenol': [[2,6], [3,5]], 'val_side': [[0,2]], 'guanidine': [[0,3]], \
    'secondary_amine': [[0,2]], 'isopropyl': [[0,2,3]], 'phenyl': [[0,1,2,3,4,5]], \
    'carboxylate': [[2,3]], 'thioether': [[0,2]], 'pro_ring': [[1,4], [2,3]]}

VDM_TYPES = [(aa, ifg, side_chain_bonding, order) for aa in AA_NAMES \
    for ifg in SMARTS_ATOM_MAP.keys() \
    for side_chain_bonding in (True, False) \
    for order in ("ab_ag", "ag_ab")
]

AA_ATOMS_ORDER = {
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ALA': ['N', 'CA', 'C', 'O', 'CB'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'GLY': ['N', 'CA', 'C', 'O'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG']
}


# antibody
# max residue number in the chothia numbering system
MAX_RESNUM_HEAVY = 113
MAX_RESNUM_LIGHT = 107

# cdr labels
# CDR-L1:L24-34; CDR-L2:L50-56; CDR-L3:L89-97; CDR-H1:H26-32; CDR-H2:H52-56; CDR-H3:H95-102
RES_BY_CDR = {
    'L1': np.arange(24, 35),
    'L2': np.arange(50, 57),
    'L3': np.arange(89, 98),
    'H1': np.arange(26, 33),
    'H2': np.arange(52, 57),
    'H3': np.arange(95, 103),
}
RES_BY_CDR_ARR = np.full(113, 'FR')
RES_BY_CDR_ARR[RES_BY_CDR["H1"]] = "H1"
RES_BY_CDR_ARR[RES_BY_CDR["H2"]] = "H2" 
RES_BY_CDR_ARR[RES_BY_CDR["H3"]] = "H3"