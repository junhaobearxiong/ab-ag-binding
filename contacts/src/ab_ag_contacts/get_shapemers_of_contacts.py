import argparse
import pickle
from tqdm import tqdm
from geometricus import SplitType

from utils.contact import get_ids_from_contacts
from utils.shapemer import get_shapemers


parser = argparse.ArgumentParser()
parser.add_argument("protein", metavar="PROTEIN", choices=["ab", "ag"])
parser.add_argument("-sr", "--split_size_radius", metavar="SIZE", type=int, default=10)
parser.add_argument("-sk", "--split_size_kmer", metavar="SIZE", type=int, default=16)
parser.add_argument("-r", "--resolution", metavar="RES", type=float, default=2)
parser.add_argument("-sb", "--subset", action="store_true")
args = parser.parse_args()

data_folder_name = "ag-protein_res-3.2"
input_dir = f"/storage/bear/sabdab/{data_folder_name}/chothia/"
output_dir = f"/storage/bear/ab-ag-binding/outputs/sabdab/{data_folder_name}/"

with open(output_dir + "contacts_Hchain_prody.pkl", "rb") as f:
    all_contacts = pickle.load(f)

# get ids for all ab/ag chains in with contacts
pdb_ids, chain_ids = get_ids_from_contacts(all_contacts, protein=args.protein)
pdb_files = [input_dir + f"{pdb_id}.pdb" for pdb_id in pdb_ids]

# for each shapemer, get the indices of residues in contact
# indices are 0-index 
if args.subset:
    if args.protein == "ab":
        chain_idx = 0
    else:
        chain_idx = 1
    residues_in_contact = [None] * len(pdb_files)
    for i in tqdm(range(len(residues_in_contact))):
        residues = set()
        # the same (pdb_id, chain_id) might appear in multiple contacts
        for contact_id in all_contacts.keys():
            if contact_id[0] == pdb_ids[i] and contact_id[chain_idx + 1] == chain_ids[i]:
                contacts = all_contacts[contact_id]
                residues.update([c[chain_idx][-1] for c in contacts])
        residues_in_contact[i] = sorted(list(residues))
else:
    residues_in_contact = None

embedder_radius = get_shapemers(
    pdb_files=pdb_files, 
    chain_ids=chain_ids, 
    split_type=SplitType.RADIUS, 
    split_size=args.split_size_radius, 
    resolution=args.resolution,
    subset_indices=residues_in_contact
)
with open(output_dir + f"shapemers-radius_{args.split_size_radius}-resolution_{args.resolution}-{args.protein}-subset_{args.subset}.pkl", "wb") as f:
    pickle.dump(embedder_radius, f)
print(f"finish {args.protein} type=radius, size={args.split_size_radius}, res={args.resolution}")

embedder_kmer = get_shapemers(
    pdb_files=pdb_files, 
    chain_ids=chain_ids, 
    split_type=SplitType.KMER, 
    split_size=args.split_size_kmer, 
    resolution=args.resolution,
    subset_indices=residues_in_contact
)
with open(output_dir + f"shapemers-kmer_{args.split_size_kmer}-resolution_{args.resolution}-{args.protein}-subset_{args.subset}.pkl", "wb") as f:
    pickle.dump(embedder_kmer, f)
print(f"finish {args.protein} type=kmer, size={args.split_size_kmer}, res={args.resolution}")