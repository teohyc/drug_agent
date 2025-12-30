from typing import Annotated, Sequence, TypedDict, Literal, List, Dict, Optional
import json
import re
import requests
from PIL import Image

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from tree_rnn_vae_model import TreeEncoder, LatentHead, TreeVAE, TreeDecoder 
from prop_gnn_model import MoleculeGINE
from tree_rnn_vae_infer import generate_candidate_mol
from prop_gnn_infer import predict_mol

RDLogger.DisableLog('rdApp.*')

#state
class DrugState(TypedDict):
    # user
    user_query: str
    intent: Literal["knowledge_only", "predict_only", "explore"]

    # smiles
    user_smiles: Optional[List[str]]
    active_smiles: Optional[List[str]]
    generated_smiles: Optional[List[str]]

    # literature (ReAct only)
    literature_messages: Annotated[Sequence[BaseMessage], add_messages]

    # pipeline
    predictions: Optional[Dict[str, Dict]]
    selected: Optional[Dict[str, Dict]]

    # control
    iteration: int

    # output
    final_answer: Optional[str]


#define llm
def load_router_llm():
    return ChatOllama(model="granite4:latest", temperature=0.0)

def load_tool_llm():
    return ChatOllama(model="granite4:latest", temperature=0.0)

def load_explainer_llm():
    return ChatOllama(model="gemma3:latest", temperature=0.3)


#SMILES detection
SMILES_REGEX = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/\.]+$")

def extract_smiles(text: str) -> List[str]:

    tokens = re.split(r"[,\s]+", text)
    smiles_list = []

    for t in tokens:
        t = t.strip()
        if len(t) < 3:
            continue
        # Quick regex filter: skip tokens with letters outside typical SMILES chars
        if not SMILES_REGEX.fullmatch(t):
            continue
        # Validate with RDKit
        if Chem.MolFromSmiles(t):
            smiles_list.append(t)

    return smiles_list

#intent router node
def intent_router(state: DrugState) -> dict:
    router = load_router_llm()
    smiles = extract_smiles(state["user_query"])

    prompt = f"""
Return JSON ONLY.

Rules:
- If ANY valid SMILES present → "predict_only"
- Else if user asks to generate molecules → "explore"
- Else → "knowledge_only"

User query:
{state["user_query"]}

Format:
{{ "intent": "knowledge_only | predict_only | explore" }}
"""

    try:
        response = router.invoke(prompt).content
        intent = json.loads(response)["intent"]
        print(f"[Router] Intent detected: {intent}")
        print(f"[Router] Extracted SMILES: {smiles}")
    except Exception:
        intent = "knowledge_only"

    return {
        "intent": intent,
        "user_smiles": smiles or None,
        "active_smiles": smiles or None,
        "generated_smiles": None,
        "predictions": None,
        "selected": None,
        "iteration": 0,
        "literature_messages": [],
    }

@tool
def search_literature(query: str) -> str:
    """Search drug-related academic literature."""
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
                f"{p.get('title')} ({p.get('year')})\n"
                f"{abstract}\n{p.get('url')}\n---"
            )

        return "\n".join(summaries)

    except Exception as e:
        return f"Literature search error: {e}"

#literature node
def literature_node(state: DrugState) -> dict:
    llm = load_tool_llm().bind_tools([search_literature])

    system = SystemMessage(content=f"""
You are a biochemistry literature reviewer.
Answer the user's question using academic literature.
Search when needed, summarize when sufficient.
Your tool is 'search_literature'.
tool input argument is a search query string.
""")

    response = llm.invoke(
        [system] + state["literature_messages"]
    )

    return {"literature_messages": [response]}


def route_from_literature(state: DrugState) -> str:
    last = state["literature_messages"][-1]
    return "tools" if last.tool_calls else "final"

#generation node
def generator_node(state: DrugState) -> dict:
    smiles = generate_candidate_mol(num_samples=6, max_len=6)
    return {
        "generated_smiles": smiles,
        "active_smiles": smiles,
    }

#prediction node
def predictor_node(state: DrugState) -> dict:
    smiles = state["active_smiles"]
    if not smiles:
        return {"predictions": {}}

    return {"predictions": predict_mol(test_smiles=smiles)}

def route_from_predictor(state: DrugState) -> str:
    return "selection" if state["intent"] == "explore" else "final"

#selection node
def score_molecule(p):
    score = 0
    score += max(0, 1 - abs(p["logP"] - 2.5))
    score += max(0, 1 - abs(p["MW"] - 350) / 150)
    score += 1 if p["HBD"] <= 3 else 0
    score += 1 if p["HBA"] <= 6 else 0
    return score

def selection_node(state: DrugState) -> dict:
    scored = []
    for smi, props in (state["predictions"] or {}).items():
        scored.append((smi, props, score_molecule(props)))

    scored.sort(key=lambda x: x[2], reverse=True)
    selected = {smi: p for smi, p, _ in scored[:6]}

    return {
        "selected": selected,
        "iteration": state["iteration"] + 1,
    }

def route_from_selection(state: DrugState) -> str:
    return "generator" if state["iteration"] < 3 else "final"


#final explainer node
def final_explainer(state: DrugState) -> dict:
    llm = load_explainer_llm()

    literature = (
        state["literature_messages"][-1].content
        if state["literature_messages"]
        else "No literature reviewed."
    )

    prompt = f"""
If the user query suggests generating or predicting molecules, talk about the properties of the selected molecules only and maybe some extras (not long).
If the user query has SMILES content, in it explain the properties of those molecules with the properties predicted in the Prediction section.
It the user query suggest literature review or general questions, focus on summarizing the literature with citations added. 
Explain everything professionally and scientifically and NEVER HALLUCINATE.
Users may continue to ask about the properties of the molecules you generated or predicted, so be ready to answer those questions based on the Prediction section only using the literature you have reviewed.
Always provide citations when you mention literature.

User Query:
{state["user_query"]}

Intent:
{state["intent"]}

Literature:
{literature}

Selected Molecules:
{state.get("selected")}

Prediction:
{state.get("predictions")}

Explain clearly and scientifically and dont hallucinate. Provide the final answer in a concise manner.
Note: The molecule generation is performed by an in-house-designed Tree-RNN VAE model and the property prediction is performed by an in-house-designed Multi-Head GINE model.
"""

    answer = llm.invoke(prompt).content
    return {"final_answer": answer}

### GRAPH ###
graph = StateGraph(DrugState)

graph.add_node("router", intent_router)
graph.add_node("literature", literature_node)
graph.add_node("tools", ToolNode([search_literature]))
graph.add_node("generator", generator_node)
graph.add_node("predictor", predictor_node)
graph.add_node("selection", selection_node)
graph.add_node("final", final_explainer)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda s: s["intent"],
    {
        "knowledge_only": "literature",
        "predict_only": "predictor",
        "explore": "generator",
    },
)

graph.add_conditional_edges(
    "literature",
    route_from_literature,
    {
        "tools": "tools",
        "final": "final",
    },
)

graph.add_edge("tools", "literature")
graph.add_edge("generator", "predictor")

graph.add_conditional_edges(
    "predictor",
    route_from_predictor,
    {
        "selection": "selection",
        "final": "final",
    },
)

graph.add_conditional_edges(
    "selection",
    route_from_selection,
    {
        "generator": "generator",
        "final": "final",
    },
)

graph.add_edge("final", END)

drug_agent = graph.compile()

'''
from io import BytesIO 
png = drug_agent.get_graph().draw_mermaid_png() 
img = Image.open(BytesIO(png)) 
img.show() 
img.save("drug_agent_diagram.png")
'''

#visualise 
def display_molecule_grid(state: DrugState):
    selected = state.get("selected")
    if not selected:
        return

    mols, legends = [], []
    for i, (smi, p) in enumerate(selected.items(), 1):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            legends.append(f"M{i}\nMW={p['MW']:.0f}, logP={p['logP']:.2f}")

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(800, 800),
        legends=legends,
    )
    img.show()

    print("\nSMILES Mapping:")
    for i, smi in enumerate(selected.keys(), 1):
        print(f"M{i}: {smi}")

state: DrugState = {
    "user_query": "",
    "intent": "knowledge_only",
    "user_smiles": None,
    "active_smiles": None,
    "generated_smiles": None,
    "literature_messages": [],
    "predictions": None,
    "selected": None,
    "iteration": 0,
    "final_answer": None,
    }

while True:
    state["user_query"] = ""
    state["final_answer"] = None
    state["intent"] = "knowledge_only"
    
    user_input = input("\nUser: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    state["user_query"] = user_input
    state = drug_agent.invoke(state)

    print("\nAssistant:\n")
    print(state["final_answer"])

    if state["selected"]:
        display_molecule_grid(state)
        