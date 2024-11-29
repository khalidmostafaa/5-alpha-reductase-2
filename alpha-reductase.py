import requests
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the trained model from GitHub
model_url = 'https://github.com/khalidmostafaa/5-alpha-reductase-2/raw/main/5-alpha_reductase2.pkl'
response = requests.get(model_url)
if response.status_code == 200:
    model = pickle.load(BytesIO(response.content))
else:
    st.error("Error loading model from GitHub")

# Function to generate PubChem-like fingerprints
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return list(fingerprint)

# Set the overall style
st.markdown(
    """
    <style>
    .intro-text { 
        font-size: 14px; 
        color: #2c3e50;
        background-color: #ecf0f1; 
        padding: 15px;
        border-radius: 8px; 
    }
    .disclaimer {
        font-size: 14px;
        color: #e74c3c;
        margin-top: 20px;
        padding: 10px;
        border-top: 1px solid #bdc3c7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app interface
st.title(" 5-alpha-reductase type 2 Inhibitor Predictor")

# Intro section
st.markdown(
    """
    <div class="intro-text">
       5-alpha-reductase type 2 converts testosterone into dihydrotestosterone (DHT), a potent androgen linked to male pattern baldness. 
       DHT causes hair follicle miniaturization, leading to thinner, shorter hair and eventual hair loss in genetically predisposed individuals.
       Drugs inhibit this enzyme, reducing DHT levels and slowing hair loss.

    </div>
    """,
    unsafe_allow_html=True
)

# SMILES input
smiles_input = st.text_input("Enter the canonical SMILES:", placeholder="Example: CC[C@H](C)[C@H](N)C(=O)N1CCCC1")



          if st.button("Predict pIC50"):
    if smiles_input:
        # Preprocess the SMILES string and predict
        try:
            # Assuming `featurize_smiles` is your function to generate descriptors from SMILES
            features = featurize_smiles(smiles)
            features_df = pd.DataFrame([features])
            prediction = model.predict(features_df)[0]
            
            # Categorize the prediction
            if prediction > 7:
                category = "Active"
                color = "green"
            elif 5 <= prediction <= 7:
                category = "Intermediate"
                color = "orange"
            else:
                category = "Inactive"
                color = "red"

            # Display the result with color
            st.markdown(
                f"<h3 style='color:{color};'>Predicted pIC50: {prediction:.2f} ({category})</h3>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid SMILES string.")

# Disclaimer section
st.markdown(
    """
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This model is intended for research purposes only and should not be used for medical treatments or diagnoses.
    </div>
    """,
    unsafe_allow_html=True
)
