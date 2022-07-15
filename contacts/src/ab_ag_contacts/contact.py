from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities

from prody import Contacts, parsePDB

import pandas as pd
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import numpy as np
from collections import defaultdict
from operator import methodcaller
from itertools import chain

from process_sabdab import subset_summary, get_ids_from_fasta, get_all_ids_chain_type
from utils import AA_ATOMS_ORDER



''' get residues that are in contacts'''
# NOTE: should only run on structrues that has been processed by `SabdabSelect` in `process_sabdab.py`
# find all residues between a pair of ab and ag chains 
# with any heavy atom < 5A away from each other
# return List[Tuple(Tuple(resname, resnum, icode), Set(contacts))]
def get_contacts_ab_ag(pdb_file, pdb_id, chain_id_ab, chain_id_ag, radius=5):
    parser = PDBParser(PERMISSIVE=1)
    struct = parser.get_structure(pdb_id, pdb_file)
    assert len(struct) == 1
    chain_ab = struct[0][chain_id_ab]

    contacts = []   # all contacts of a single ab chain
    # `chain_id_ag` is taken from `summary`, which might contains multiple chains
    for chain_id_ag_single in chain_id_ag.split(" | "):
        chain_ag = struct[0][chain_id_ag_single]
        contacts_ag = set()    # all contacts of an ab chain with a single ag chain 
        for res_ab in chain_ab:
            res_ab_resname = res_ab.get_resname()
            res_ab_resnum = res_ab.get_id()[1]
            res_ab_icode = res_ab.get_id()[2]
            res_ab_id = (res_ab_resname, res_ab_resnum, res_ab_icode)
            for atom_ab in res_ab:
                atoms_ag = unfold_entities(chain_ag, 'A')
                ns = NeighborSearch(atoms_ag)
                res_contact_ag = ns.search(center=atom_ab.get_coord(), level="R", radius=radius)
                res_contact_ag_ids = [(
                    res_ag.get_resname(),
                    res_ag.get_id()[1],
                    res_ag.get_id()[2]
                ) for res_ag in res_contact_ag]
                contacts_ag.update([(res_ab_id, res_ag_id) for res_ag_id in res_contact_ag_ids])
        contacts += [((pdb_id, chain_id_ab, chain_id_ag_single), contacts_ag)]
    return contacts


# NOTE: should only run on structrues that has been processed by `SabdabSelect` in `process_sabdab.py`
# return: List[(contact_id, Set(contacts_ag))]
# `contact_id`: Tuple(pdb_id, chain_id_ab, chain_id_ag_single)
# `contacts_ag`: Tuple((res_ab_id), (res_ag_id))
# `res_ab_id`, `res_ag_id`: Tuple((resname, resnum, icode, relative resindex))
def get_contacts_ab_ag_prody(pdb_file, pdb_id, chain_id_ab, chain_id_ag, radius=5):
    struct = parsePDB(pdb_file)
    chain_ab = struct.getHierView()[chain_id_ab]
    chain_ab_resindices = struct.select(f"chain {chain_id_ab}").select("protein").select("calpha").getResindices()

    contacts = []   # all contacts of a single ab chain
    # `chain_id_ag` is taken from `summary`, which might contains multiple chains
    for chain_id_ag_single in chain_id_ag.split(" | "):
        chain_ag = struct.getHierView()[chain_id_ag_single]
        chain_ag_resindices = struct.select(f"chain {chain_id_ag_single}").select("protein").select("calpha").getResindices()
        contacts_ag = set()    # all contacts of an ab chain with a single ag chain 
        for res_ab in chain_ab:
            for atom_ab in res_ab:
                atoms_ag = chain_ag.select("protein")
                ct = Contacts(atoms_ag)
                atoms_contact_ag = ct.select(center=atom_ab.getCoords(), radius=radius)
                if atoms_contact_ag is not None:
                    contact_ag_resnames = atoms_contact_ag.getResnames()
                    contact_ag_resnums = atoms_contact_ag.getResnums()
                    contact_ag_icodes = atoms_contact_ag.getIcodes()
                    contact_ag_resindices = [int] * len(atoms_contact_ag)
                    for i, residx in enumerate(atoms_contact_ag.getResindices()):
                        # get the relative index of each residue
                        contact_ag_resindices[i] = np.where(chain_ag_resindices == residx)[0][0]
                    res_ab_id = (res_ab.getResname(), res_ab.getResnum(), res_ab.getIcode(), np.where(chain_ab_resindices == res_ab.getResindex())[0][0])
                    for i in range(len(contact_ag_resnames)):
                        res_ag_id = (contact_ag_resnames[i], contact_ag_resnums[i], contact_ag_icodes[i], contact_ag_resindices[i])
                        contacts_ag.add((res_ab_id, res_ag_id))
        contact_id = (pdb_id, chain_id_ab, chain_id_ag_single)
        contacts += [(contact_id, contacts_ag)]
    return contacts



''' get coordinates of atoms of contacted residues'''
# return (atoms, coords): (List[num atoms], np.array((num_atoms, 3)))
def get_atom_coords(res):
    atoms_in_res = []
    coords = np.zeros((len(res.get_list()), 3))
    # use the ordering of atoms in AA_ATOMS_ORDER
    curr_idx = 0
    for atom in AA_ATOMS_ORDER[res.get_resname()]:
        try:
            coord = res[atom].get_coord()
        except KeyError:
            continue
        atoms_in_res.append(atom)
        coords[curr_idx, :] = coord
        curr_idx += 1
    return atoms_in_res, coords


# get every pair of atoms that are in contact between res_ab and res_ag
def get_contact_atoms(res_ab, res_ag, radius=5):
    atoms_in_contact = []
    for atom_ab in res_ab.get_unpacked_list():
        # make sure only searching over atoms in AA_ATOMS_ORDER
        if atom_ab.get_name() not in AA_ATOMS_ORDER[res_ab.get_resname()]:
            continue
        atoms_ag = [atom for atom in res_ag.get_unpacked_list() if atom.get_name() in AA_ATOMS_ORDER[res_ag.get_resname()]]
        ns = NeighborSearch(atoms_ag)
        atoms_contact_ag = ns.search(center=atom_ab.get_coord(), level="A", radius=radius)
        atoms_in_contact.extend([(atom_ab.get_name(), atom_ag.get_name()) for atom_ag in atoms_contact_ag])
    return atoms_in_contact


# interface: a set of contacts of a pair of ab, ag chain
# return contact_coords: Tuple(aa1_data, aa2_data, atoms_in_contact)
def get_interface_coords(pdb_file, pdb_id, chain_id_ab, chain_id_ag, interface, output_file=None):
    contact_coords = defaultdict(list)
    struct = PDBParser().get_structure(pdb_id, pdb_file)
    ab_chain = struct[0][chain_id_ab]
    ag_chain = struct[0][chain_id_ag]
    for contact in interface:
        res_id_ab = (" ", contact[0][1], contact[0][2])
        res_id_ag = (" ", contact[1][1], contact[1][2])
        res_ab = ab_chain[res_id_ab]
        res_ag = ag_chain[res_id_ag]
        resname_ab = contact[0][0]
        resname_ag = contact[1][0]
        atoms_ab, coords_ab = get_atom_coords(res_ab)
        atoms_ag, coords_ag = get_atom_coords(res_ag)
        contact_coords[(resname_ab, resname_ag)].append((
            (resname_ab, atoms_ab, coords_ab),
            (resname_ag, atoms_ag, coords_ag),
            get_contact_atoms(res_ab, res_ag)
        ))
    if output_file is not None:
        with open(output_file, "wb") as f:
            pickle.dump(contact_coords, f)
    return contact_coords


# combine a list of interface_coords as returned from `get_interface_coords`
def combine_interface_coords(output_file, all_interface_coords):
    combined_data = defaultdict(list)
    dict_items = map(methodcaller("items"), all_interface_coords)
    for k, v in tqdm(chain.from_iterable(dict_items)):
            combined_data[k].extend(v)
    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)



''' other helper functions '''
# get list of pdb ids and chain ids for all ab and ag chains in extracted contacts
# `all_contacts` is a list of output from `get_contacts_ab_ag`
def get_ids_from_contacts(all_contacts, protein):
    all_ids = [None] * len(all_contacts)
    for i, ids in enumerate(all_contacts.keys()):
        pdb_id = ids[0]
        if protein == "ab":
            chain_id = ids[1]
        elif protein == "ag":
            chain_id = ids[2]
        else:
            raise ValueError(f"`protein` should be 'ab' or 'ag', not {protein}")
        all_ids[i] = (pdb_id, chain_id)
    pdb_ids, chain_ids = list(zip(*set(all_ids)))
    return pdb_ids, chain_ids


# filter contacts by ab ids
def filter_contacts_by_ab(ab_ids, contacts):
    contacts_filtered = {}
    for key, interface in contacts.items():
        contact_ab_id = key[0], key[1]
        if contact_ab_id in ab_ids:
            contacts_filtered[key] = interface
    return contacts_filtered


if __name__ == "__main__":
    # TODO: make these command line arguments
    dir_prefix = "/storage/bear/projects/ab-ag-binding/data/sabdab/ag-protein_res-3.2/"
    summary_file_path = dir_prefix + "summary.tsv"
    summary = subset_summary(summary_file_path)
    dir_prefix += "chothia_selected/"    

    ''' get set of ab ids after redundancy filtering '''
    '''
    # get a list of ab ids after mmseqs clustering
    ab_ids = get_ids_from_fasta(dir_prefix + "clusters/chain_heavy_clu_rep.fasta")
    # also filter out scfv chains
    scfv_ids = get_all_ids_chain_type(summary, "scfv")
    ab_ids = [ab_id for ab_id in ab_ids if ab_id not in scfv_ids]
    '''


    ''' getting set of contacts for ab chains in ab_ids '''
    '''
    summary = summary[pd.Series(zip(summary["pdb"], summary["Hchain"])).isin(ab_ids)].reset_index(drop=True)
    outputs = Parallel(n_jobs=-1)(delayed(get_contacts_ab_ag)(
            pdb_file=dir_prefix + f"pdb/{summary.loc[i, 'pdb']}.pdb",
            pdb_id=summary.loc[i, "pdb"],
            chain_id_ab=summary.loc[i, "Hchain"],
            chain_id_ag=summary.loc[i, "antigen_chain"]
        )
        for i in tqdm(range(summary.shape[0]))
    )
    print("finish processing all structures")

    # need to unpack nested list
    contacts_dict = dict([contacts_tuple for contacts_list in outputs for contacts_tuple in contacts_list])
    with open (dir_prefix + "contacts/contacts_filtered_ab-Hchain-sabdab-chothia_selected.pkl", "wb") as f:
        pickle.dump(contacts_dict, f)
    '''


    ''' filter contacts based on ab ids (not useful now, but can add more contact dependent filtering here) '''
    '''
    # load initial set of contacts
    with open(dir_prefix + "contacts/contacts-Hchain-sabdab-chothia_selected.pkl", "rb") as f:
        contacts = pickle.load(f)

    # filter contacts
    contacts_filtered = filter_contacts_by_ab(ab_ids, contacts)

    with open(dir_prefix + "contacts/contacts_filtered-Hchain-sabdab-chothia_selected.pkl", "wb") as f:
        pickle.dump(contacts_filtered, f)
    '''


    ''' get coordinates of contacted residues'''
    with open(dir_prefix + f"raw_contacts/Hchain-filtered.pkl", "rb") as f:
        all_contacts = pickle.load(f)
    
    outputs = Parallel(n_jobs=-1)(delayed(get_interface_coords)(
        pdb_file=dir_prefix + f"pdb/{key[0]}.pdb",
        pdb_id=key[0],
        chain_id_ab=key[1],
        chain_id_ag=key[2],
        interface=interface,
        output_file=dir_prefix + f"raw_contacts/Hchain-filtered-coords/{key[0]}_{key[1]}_{key[2]}.pkl"
    ) for key, interface in tqdm(all_contacts.items()))

    combine_interface_coords(dir_prefix + f"raw_contacts/Hchain-filtered-coords/all_coords.pkl", outputs)
    

