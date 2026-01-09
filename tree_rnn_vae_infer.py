import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import random
import re
from tree_rnn_vae_model import TreeEncoder, LatentHead, TreeVAE, TreeDecoder

def decode_fragments_to_smiles(batch_indices, vocab):
    smiles_list = []
    for seq in batch_indices:
        fragments = [vocab[idx] for idx in seq.tolist()]
        
        try:
            mol = Chem.MolFromSmiles(fragments)
            if mol is not None:
                smiles_list.append(Chem.MolToSmiles(mol))
            else:
                smiles_list.append(fragments)  # fallback
        except:
            smiles_list.append(fragments)
    return smiles_list

def normalize_attachment_points(smiles):
    return re.sub(r'\[\d+\*\]', '[*]', smiles)

def deduplicate_fragments(fragments):
    seen = set()
    unique = []
    
    for f in fragments:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique

def fragments_to_mols(fragments):
    mols = []
    for f in fragments:
        mol = Chem.MolFromSmiles(f)
        if mol is None:
            continue  # skip invalid fragment
        mols.append(mol)
    return mols

def get_dummy_atom_indices(mol):
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']

def get_linear_connection_pairs(mols):
    """Connect last dummy atom of mol i to first dummy atom of mol i+1"""
    pairs = []
    for i in range(len(mols)-1):
        idx_last_i = get_dummy_atom_indices(mols[i])[-1]
        idx_first_next = get_dummy_atom_indices(mols[i+1])[0]
        pairs.append((i, idx_last_i, i+1, idx_first_next))
    return pairs

def combine_fragments_and_connect(fragments, connection_pairs):
    mols = fragments_to_mols(fragments)
    if not mols:
        return None

    # Combine molecules sequentially
    combined = Chem.CombineMols(mols[0], mols[1]) if len(mols) > 1 else mols[0]
    offsets = [0, mols[0].GetNumAtoms()]
    for i in range(2, len(mols)):
        combined = Chem.CombineMols(combined, mols[i])
        offsets.append(sum([m.GetNumAtoms() for m in mols[:i]]))

    rw_mol = Chem.RWMol(combined)

    # Add bonds for connection pairs
    for f1, a1, f2, a2 in connection_pairs:
        rw_mol.AddBond(offsets[f1] + a1, offsets[f2] + a2, Chem.BondType.SINGLE)

    # Sanitize
    try:
        Chem.SanitizeMol(rw_mol)
        return Chem.MolToSmiles(rw_mol)
    except:
        return None


def process_one_molecule(raw_fragments):
    # Normalize & deduplicate
    frags = [normalize_attachment_points(f) for f in raw_fragments]
    frags = deduplicate_fragments(frags)

    if len(frags) == 0:
        return None

    # Convert to mols
    mols = fragments_to_mols(frags)
    if len(mols) < 1:
        return None

    # Build connections (linear for now)
    connection_pairs = get_linear_connection_pairs(mols)

    # Combine & sanitize
    smiles = combine_fragments_and_connect(frags, connection_pairs)
    return smiles

def replace_dummy_with_carbon(smiles):
    return smiles.replace('*', 'C')
    

#main
def generate_candidate_mol(num_samples=6, max_len=20):
    '''This function generates candidate molecules using an in-house-designed Tree-RNN VAE model. It returns a list of SMILES strings representing the generated molecules as well as displaying their molecular structure.'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx2frag = torch.load("dict_idx2frag.pt")      # {index: fragment_string}
    vocab_size = len(idx2frag)

    model = torch.load("tree_rnn-vae.pt", weights_only=False, map_location=device)

    model.eval()
    z = torch.randn(num_samples, model.decoder.z_to_hidden.in_features, device=device)
    sampled_indices = model.sample_from_z(z, sos_idx=None, max_len=max_len)  

    smiles_out = decode_fragments_to_smiles(sampled_indices, idx2frag)

    smiles_list = []
    for mol in smiles_out:
        smiles = process_one_molecule(mol)
        smiles_list.append(smiles)

    smiles_list = [replace_dummy_with_carbon(s) for s in smiles_list]
    
    '''mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(600,600), legends=smiles_list)
    img.show()'''

    return smiles_list

