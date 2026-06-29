from fastmcp import FastMCP
from typing import List, Dict
import re
import base64
import requests
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import Draw

# models
from tree_rnn_vae_model import TreeEncoder, LatentHead, TreeVAE, TreeDecoder 
from prop_gnn_model import MoleculeGINE
from tree_rnn_vae_infer import generate_candidate_mol
from prop_gnn_infer import predict_mol


# ----------------------------
# Initialize MCP Server
# ----------------------------

mcp = FastMCP(
    name="Drug Discovery MCP")
'''
# ----------------------------
# SMILES Extraction
# ----------------------------

SMILES_REGEX = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/\.]+$")

@mcp.tool()
def extract_smiles(text: str) -> List[str]:
    """Extract valid SMILES strings from text using RDKit."""
    tokens = re.split(r"[,\s]+", text)
    smiles_list = []

    for t in tokens:
        t = t.strip()
        if len(t) < 3:
            continue
        if not SMILES_REGEX.fullmatch(t):
            continue
        if Chem.MolFromSmiles(t):
            smiles_list.append(t)

    return smiles_list
'''

# Molecule Generator (Tree-RNN VAE)
@mcp.tool()
def generate_molecules(num_samples: int = 2, max_len: int = 6) -> List[str]:
    """Generate candidate molecules using Tree-RNN VAE."""
    return generate_candidate_mol(num_samples=num_samples, max_len=max_len)


# Property Predictor (GINE)
@mcp.tool()
def predict_properties(smiles: List[str]) -> Dict[str, Dict]:
    """Predict molecular properties using Multi-Head GINE."""
    if not smiles:
        return {}
    return predict_mol(smiles)


# Semantic Scholar Search
@mcp.tool()
def search_literature(query: str) -> str:
    """Search drug-related academic literature using Semantic Scholar."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 3,
        "fields": "title,year,abstract,url"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])

        if not data:
            return "No relevant papers found."

        summaries = []
        for p in data:
            abstract = p.get("abstract", "")
            if abstract and len(abstract) > 400:
                abstract = abstract[:400] + "..."
            summaries.append(
                f"{p.get('title')} ({p.get('year')})\n{abstract}\n{p.get('url')}\n---"
            )

        return "\n".join(summaries)

    except Exception as e:
        return f"Literature search error: {e}"

# Molecule Memory Formatter
@mcp.tool()
def format_molecule_memory(mols: Dict[str, Dict]) -> str:
    """Format molecules and properties into readable memory."""
    if not mols:
        return ""

    lines = ["Previously discussed molecules:"]
    for i, (smi, p) in enumerate(mols.items(), 1):
        lines.append(
            f"{i}. {smi}\n"
            f"   MW={p['MW']:.2f}, logP={p['logP']:.2f}, "
            f"HBD={p['HBD']}, HBA={p['HBA']}"
        )
    return "\n".join(lines)

# Molecule Image Renderer
@mcp.tool()
def render_molecule_images(molecules: Dict[str, Dict]) -> List[str]:
    """Render RDKit grid image of molecules. Returns base64 PNG."""
    if not molecules:
        return []

    mols = []
    legends = []

    for i, (smi, p) in enumerate(molecules.items(), 1):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            legends.append(
                f"M{i}\nMW={p['MW']:.0f}, logP={p['logP']:.2f}, "
                f"HBD={p['HBD']}, HBA={p['HBA']}"
            )

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(400, 400),
        legends=legends,
        useSVG=False
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return [base64.b64encode(buf.read()).decode("utf-8")]


# Run MCP Server
if __name__ == "__main__":
    mcp.run(transport="stdio")