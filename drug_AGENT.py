from typing import Annotated, Sequence, TypedDict, Literal, List, Dict, Optional
import json
import re
import requests
from PIL import Image

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
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

    #memory
    molecule_memory: Optional[str]

    # control
    iteration: int #selection loop
    recursion: int #literature loop

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

        #debugging 
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
        "recursion": 0,
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
    existing_msg = state.get("literature_messages", [])

    recursion = state.get("recursion", 0) + 1

    #debugging
    print(f"[Literature] recursion = {recursion}")
    print(state["molecule_memory"])

    system_msg = SystemMessage(content=f"""
        "You are a biochemistry literature reviewer.
        Answer the user's question using academic literature.
        Search when needed, summarize when sufficient.
        Your tool is 'search_literature'.
        You may use the tool a few times to gather relevant information.
        Tool input argument is a search query string.
        Respond concisely and scientifically. Never hallucinate.
        User may want you to find about previously predicted molecules so be ready to search using the info about the previously predicted or generated molecules and their specific infos below if any:
        {state["molecule_memory"]}
        ONLY search about the previously predicted molecules if asked by the user, if the user does not specifies previous molecule or any questions suggesting the previous molecule do not answer with any refernces or mentioning of them.
            """)

    if not existing_msg:
        response = llm.invoke([system_msg] + [HumanMessage(content=state["user_query"])])
        return {"literature_messages": [response],
                "recursion": recursion}
    
    else:
        response = llm.invoke([system_msg] + existing_msg )
        return {"literature_messages": [response],
                "recursion": recursion}


def route_from_literature(state: DrugState) -> str:
    last = state["literature_messages"][-1]

    if state["recursion"] >= 11:
        return "final"
    
    return "tools" if getattr(last, "tool_calls", None) else "final"

#generation node
def generator_node(state: DrugState) -> dict:
    smiles = generate_candidate_mol(num_samples=2, max_len=6)
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

    #debugging
    print(f"""[Selection] iteration = {state["iteration"]}""")
          
    return {
        "selected": selected,
        "iteration": state["iteration"] + 1,
    }

def route_from_selection(state: DrugState) -> str:
    return "generator" if state["iteration"] < 7 else "final"

#utility function for molecule memory
def format_molecule_memory(mols: Dict[str, Dict]) -> str:
    lines = ["Previously discussed molecules:"]
    for i, (smi, p) in enumerate(mols.items(), 1):
        lines.append(
            f"{i}. {smi}\n"
            f"   MW={p['MW']:.2f}, logP={p['logP']:.2f}, "
            f"HBD={p['HBD']}, HBA={p['HBA']}"
        )
    return "\n".join(lines)

#final explainer node
def final_explainer(state: DrugState) -> dict:
    llm = load_explainer_llm()

    literature = (
        state["literature_messages"][-1].content
        if state["literature_messages"]
        else "No literature reviewed."
    )

    #update molecule memory
    memory_update = None

    if state.get("selected"):
        memory_update = format_molecule_memory(state["selected"])

    elif state.get("predictions") and state["intent"] == "predict_only":
        memory_update = format_molecule_memory(state["predictions"])

    # append to existing memory if present
    if memory_update:
        if state.get("molecule_memory"):
            molecule_memory = state["molecule_memory"] + "\n\n" + memory_update
        else:
            molecule_memory = memory_update
    else:
        molecule_memory = state.get("molecule_memory")

    prompt = f"""
If the user query suggests generating or predicting molecules, talk about the predicted properties of all the respective selected molecules only and maybe some extras (not long).
If the user query has SMILES content, in it explain the properties of those molecules with the properties predicted in the Prediction section.
It the user query suggest literature review or general questions, focus on summarizing the literature with citations added and refernces at the end, write it profesionally not too long or too short, NEVER HALLUCINATE. 
Users may continue to ask about the properties of the molecules you previously generated or predicted, so be ready to answer those questions based on the Prediction section only using the literature you have reviewed.
Always provide citations when you mention literature.
If the user query are trivial like asking so far how may molecules have been generated or predicted, just give a short direct answer.
If the intent is to suggest generating or predicting molecules, you do not need to use the literature to explain the properties of the molecules, just use the predicted properties only, never hallucinate.
If intent is predict_only or explore, present the findings in a proper table and then explain the result.

User Query:
{state["user_query"]}

Intent:
{state["intent"]}

Previously known molecules (do not mention them unless user query specifies them strongly):
{molecule_memory or "None"}

Literature:
{literature}

Selected Molecules:
{state.get("selected")}

Prediction:
{state.get("predictions")}

Explain clearly and scientifically and dont hallucinate. always put up references and citation if knowledge_only is the intent.
Finally, ONLY answer the user's query and explain it, dont bring in unrelated information about the query all contents of response must be STRONGLY related to the user's query, do not incorporate redundant informations.
Note: The molecule generation is performed by an in-house-designed Tree-RNN VAE model and the property prediction is performed by an in-house-designed Multi-Head GINE model.
NEVER MAKE UP REFERENCES AND INFOS OR HALLUCINATE.

FINAL WARNING: DO NOT CITE unless literature was used.
"""

    answer = llm.invoke(prompt).content
    return {"final_answer": answer,
            "molecule_memory": molecule_memory}

### GRAPH ###
graph = StateGraph(DrugState)

graph.add_node("router", intent_router)
graph.add_node("literature", literature_node)
graph.add_node("tools", ToolNode([search_literature], messages_key="literature_messages"))
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


#for streamlit ui
from io import BytesIO
import streamlit as st

def render_molecule_grid(selected: Dict[str, Dict]):
    if not selected:
        return

    mols, legends = [], []

    for i, (smi, p) in enumerate(selected.items(), 1):
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
    st.image(buf.getvalue(), caption="Selected / Predicted Molecules")

    with st.expander("SMILES Mapping"):
        for i, smi in enumerate(selected.keys(), 1):
            st.code(f"M{i}: {smi}")

'''
from io import BytesIO 
png = drug_agent.get_graph().draw_mermaid_png() 
img = Image.open(BytesIO(png)) 
img.show() 
img.save("drug_agent_diagram.png")
'''

#visualise 
'''def display_molecule_grid(state: DrugState):
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
        print(f"M{i}: {smi}")'''

state: DrugState = {
    "user_query": "",
    "intent": "knowledge_only",
    "user_smiles": None,
    "active_smiles": None,
    "generated_smiles": None,
    "literature_messages": [],
    "predictions": None,
    "selected": None,
    "molecule_memory": None,
    "iteration": 0,
    "recursion": 0,
    "final_answer": None,
    }
'''
while True:
    
    user_input = input("\nUser: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    state["user_query"] = user_input
    state = drug_agent.invoke(state)

    print("\nAssistant:\n")
    print(state["final_answer"])

    if state["selected"]:
        display_molecule_grid(state)'''
        