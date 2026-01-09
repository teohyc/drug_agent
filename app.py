import streamlit as st

from drug_AGENT import drug_agent, render_molecule_grid, DrugState
from tree_rnn_vae_model import TreeVAE, TreeDecoder, TreeEncoder, LatentHead
from prop_gnn_model import MoleculeGINE

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Drug Research Agent",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 AI Drug Research Agent")
st.caption(
    "Tree-RNN VAE molecule generation + Multi-Head GINE property prediction "
    "with literature-aware reasoning"
)

# ---------------------------
# Session State Initialization
# ---------------------------
if "state" not in st.session_state:
    st.session_state.state = DrugState(
        user_query="",
        intent="knowledge_only",
        user_smiles=None,
        active_smiles=None,
        generated_smiles=None,
        literature_messages=[],
        predictions=None,
        selected=None,
        molecule_memory=None,
        iteration=0,
        recursion=0,
        final_answer=None,
    )

if "chat" not in st.session_state:
    st.session_state.chat = []

if "display_molecules" not in st.session_state:
    st.session_state.display_molecules = None

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown(
        """
        **Supported modes**
        - 📚 Knowledge-based questions
        - 🧬 SMILES property prediction
        - 🧪 De novo molecule exploration
        """
    )

    if st.button("🧹 Reset Session"):
        st.session_state.state = DrugState(
            user_query="",
            intent="knowledge_only",
            user_smiles=None,
            active_smiles=None,
            generated_smiles=None,
            literature_messages=[],
            predictions=None,
            selected=None,
            molecule_memory=None,
            iteration=0,
            recursion=0,
            final_answer=None,
        )
        st.session_state.chat = []
        st.session_state.display_molecules = None
        st.experimental_rerun()

# ---------------------------
# Render Chat History
# ---------------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Chat Input
# ---------------------------
user_input = st.chat_input(
    "Ask about drugs, paste a SMILES, or explore molecules..."
)

# ---------------------------
# Agent Execution
# ---------------------------
if user_input:
    # User message
    st.session_state.chat.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent
    st.session_state.state["user_query"] = user_input

    with st.spinner("Thinking like a computational chemist..."):
        st.session_state.state = drug_agent.invoke(st.session_state.state)

    assistant_reply = st.session_state.state.get("final_answer", "")

    # Assistant message
    st.session_state.chat.append({
        "role": "assistant",
        "content": assistant_reply,
    })

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    # Update persistent molecule canvas
    if st.session_state.state.get("selected"):
        st.session_state.display_molecules = st.session_state.state["selected"]

    elif st.session_state.state.get("predictions"):
        st.session_state.display_molecules = st.session_state.state["predictions"]

# ---------------------------
# Persistent Molecule Display
# ---------------------------
if st.session_state.display_molecules:
    st.divider()
    st.subheader("🧬 Molecular Structures")
    render_molecule_grid(st.session_state.display_molecules)

