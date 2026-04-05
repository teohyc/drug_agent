from prop_gnn_infer import predict_mol
from prop_gnn_model import MoleculeGINE

print(predict_mol(test_smiles=["O=C1N=C2SCCN2C(=O)C1Cc1ccc(Cl)cc1", "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(C)C(=O)CO"]))





