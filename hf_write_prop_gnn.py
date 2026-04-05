from huggingface_hub import HfApi, create_repo, upload_folder
import os

# ================= CONFIG =================
HF_USERNAME = "teohyc"   # change this
REPO_NAME = "Molecular_Property_Prediction-Multihead_GINE"   

TEMP_DIR = "hf_upload_gnn"  

#prepare folder
os.makedirs(TEMP_DIR, exist_ok=True)

#inference and model code
import shutil
shutil.copy("prop_gnn_infer.py", os.path.join(TEMP_DIR, "prop_gnn_infer.py"))
shutil.copy("prop_gnn_model.py", os.path.join(TEMP_DIR, "prop_gnn_model.py"))

#model weights and scaler
shutil.copy("prop_gnn.pt", os.path.join(TEMP_DIR, "prop_gnn.pt"))
shutil.copy("scaler_for_gnn.pkl", os.path.join(TEMP_DIR, "scaler_for_gnn.pkl"))


#readme
readme = f"""
# Multi-Head Graph Isomorphism Network (GINE) for Molecular Property Prediction

This is a Multi-Head Graph Isomorphism Network (GINE) model designed for predicting molecular properties such as lipophilicity, molecular weight, hydrogen bond donor count, and hydrogen bond acceptor count from SMILES strings. The model takes a SMILES string as input, converts it into a graph representation, and outputs the predicted properties.
Training data from ChemBL library

Full project file at https://github.com/teohyc/drug_agent


## Usage 

```python
from prop_gnn_infer import predict_mol
from prop_gnn_model import MoleculeGINE

#change to your test SMILES strings
print(predict_mol(test_smiles=["O=C1N=C2SCCN2C(=O)C1Cc1ccc(Cl)cc1", "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(C)C(=O)CO"]))
```"""

with open(os.path.join(TEMP_DIR, "README.md"), "w") as f:
    f.write(readme)

#create repo and upload

api = HfApi()

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

create_repo(repo_id, exist_ok=True)

print(f"Uploading to {repo_id}...")

upload_folder(
folder_path=TEMP_DIR,
repo_id=repo_id,
repo_type="model"
)

print("\nUpload complete!")
print(f"https://huggingface.co/{repo_id}")