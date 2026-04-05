DEMO VIDEO: https://youtu.be/4b9fYZC5avE
Drug generator model:https://huggingface.co/teohyc/DeNovoDrugGenerator-RNN-VAE
Molecule property predictor model:https://huggingface.co/teohyc/Molecular_Property_Prediction-Multihead_GINE

A drug research agent system involving a generative RNN-VAE de novo drug generator trained on 10k ChemBL molecule and a Multihead GINE molecule properties (lipophilicity, molecular weight, hydrogen bond donor, hydrogen bond acceptor) trained with the same 10k ChemBL data.
Literature agent run on semantic scholar with ReAct mechanism allowing it to use the research tool continuously.
Streamlit UI.
Agentic system poweredd by IBM Granite4 (router and literature node) and Google Gemma3 (explainer) ran locally.


