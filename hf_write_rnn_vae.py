from huggingface_hub import HfApi, create_repo, upload_folder
import os

# ================= CONFIG =================
HF_USERNAME = "teohyc"   # change this
REPO_NAME = "DeNovoDrugGenerator-RNN-VAE"   

TEMP_DIR = "hf_upload_rnn_vae"  

#prepare folder
os.makedirs(TEMP_DIR, exist_ok=True)

#inference and model code
import shutil
shutil.copy("tree_rnn_vae_infer.py", os.path.join(TEMP_DIR, "tree_rnn_vae_infer.py"))
shutil.copy("tree_rnn_vae_model.py", os.path.join(TEMP_DIR, "tree_rnn_vae_model.py"))

#model weights and index to fragment dict
shutil.copy("tree_rnn-vae.pt", os.path.join(TEMP_DIR, "tree_rnn-vae.pt"))
shutil.copy("dict_idx2frag.pt", os.path.join(TEMP_DIR, "dict_idx2frag.pt"))


#readme
readme = """
# De Novo Drug Generator - RNN-VAE

De Novo Drug Generator - RNN-VAE is a deep learning model designed for generating novel drug molecules.
Training data from ChemBL library

Full project file at https://github.com/teohyc/drug_agent


## Usage 

```python
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from tree_rnn_vae_infer import generate_candidate_mol
from tree_rnn_vae_model import TreeEncoder, LatentHead, TreeVAE, TreeDecoder


def compute_molecule_props(mol):
    return {
        "MW": Descriptors.MolWt(mol),
        "logP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
    }


# display molecule
def render_molecule_grid(selected):
    if not selected:
        return

    mols, legends = [], []

    if isinstance(selected, dict):
        iterable = selected.items()
    else:
        iterable = enumerate(selected, 1)

    for i, item in iterable:
        if isinstance(selected, dict):
            smi, props = i, item
        else:
            smi, props = item, None

        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            if props is None:
                props = compute_molecule_props(mol)
            legends.append(
                f"M{i}\nMW={props['MW']:.0f}, logP={props['logP']:.2f}, "
                f"HBD={props['HBD']}, HBA={props['HBA']}"
            )

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(400, 400),
        legends=legends,
        useSVG=False,
    )
    return img


molecules = generate_candidate_mol(num_samples=6, max_len=20) #change to your desired molecule size and number
img = render_molecule_grid(molecules)
img.show()
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