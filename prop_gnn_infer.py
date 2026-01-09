import pandas as pd
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Dataset, DataLoader
from prop_gnn_model import MoleculeGINE
import joblib

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    #node
    node_features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic())
        ]
        node_features.append(feat)
    x = torch.tensor(node_features, dtype=torch.float)

    #edge
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        btype = bond.GetBondType()
        edge_feat = [
            int(btype == Chem.rdchem.BondType.SINGLE),
            int(btype == Chem.rdchem.BondType.DOUBLE),
            int(btype == Chem.rdchem.BondType.TRIPLE),
            int(btype == Chem.rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        edge_attr.append(edge_feat)
        edge_attr.append(edge_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return x, edge_index, edge_attr

def predict_properties(smiles, model, scaler, device="cuda"):
    model.eval()

    x, edge_index, edge_attr = mol_to_graph(smiles)

    data = Data(
        x=x.to(device),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device),
        batch=torch.zeros(x.size(0), dtype=torch.long).to(device)
    )

    with torch.no_grad():
        pred_scaled = model(data.x, data.edge_index, data.edge_attr, data.batch)

    pred = scaler.inverse_transform(pred_scaled.cpu().numpy())[0]

    return {
        "logP": float(pred[0]),
        "MW": float(pred[1]),
        "HBD": round(float(pred[2])),
        "HBA": round(float(pred[3]))
    }

#main
def predict_mol(test_smiles=["O=C1N=C2SCCN2C(=O)C1Cc1ccc(Cl)cc1", "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(C)C(=O)CO"]):
    '''This function takes a list of SMILES strings as input and returns a dictionary with predicted properties for each molecule via a in-house-designed Multi-Head GINE model.'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load("prop_gnn.pt", weights_only=False, map_location=device)
    scaler = joblib.load("scaler_for_gnn.pkl")
    prop_rec = []

    for s in test_smiles:
        prop = predict_properties(s, model, scaler)
        prop_rec.append(prop)
        '''print(f"SMILES: {s}")
        print(f"""Properties Predicted: \n logP = {prop["logP"]} \n Molecular Weight = {prop["MW"]} \n Hydrogen Bond Donor = {prop["HBD"]} \n Hydrogen Bond Acceptor = {prop["HBA"]}""")
        '''
    ret_dict = {s: p for s, p in zip(test_smiles, prop_rec)}
    return ret_dict


