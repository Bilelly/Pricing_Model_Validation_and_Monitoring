# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# === Configuration ===
st.set_page_config(page_title="Insurance Pricing & Drift Monitor", layout="wide")
st.title("💰 Insurance Pricing")
st.markdown("Prédiction de prime d'assurance")

# === Chargement des ressources ===
@st.cache_resource
def load_pricing_model():
    return joblib.load(Path("models") / "best_xgb_regressor_model.pkl")

@st.cache_data
def load_reference_data():
    return pd.read_csv(Path("data") / "X_train.csv")

@st.cache_data
def load_production_data():
    return pd.read_csv(Path("data") / "X_drift.csv")

# === Import de ta fonction de drift ===
try:
    from src.DriftComputePSI import calculate_psi
except ImportError as e:
    st.error(f"❌ Erreur d'import : {e}")
    st.stop()

# === Onglets ===
tab1, tab2 = st.tabs(["💰 Prédiction Prime", "📉 Détection de Drift"])

# ───────────────────────────────────────
# TAB 1 : PRÉDICTION DE LA PRIME
# ───────────────────────────────────────
with tab1:
    st.subheader("Prédire la prime d'assurance")
    
    # Charger le modèle
    try:
        model = load_pricing_model()
        # Extraire les noms des features attendues
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        else:
            # Si pipeline scikit-learn, essayer autre méthode
            ref_data = load_reference_data()
            expected_features = list(ref_data.columns)
    except Exception as e:
        st.error(f"Impossible de charger le modèle : {e}")
        st.stop()
        expected_features = []

    input_method = st.radio("Méthode d'entrée", ["Formulaire", "Upload CSV"], horizontal=True)

    if input_method == "Formulaire":
        st.info(f"Modèle attend {len(expected_features)} features. Exemples : {expected_features[:5]}...")
        
        # Créer un dictionnaire pour stocker les inputs
        input_dict = {}
        cols = st.columns(2)
        for i, col_name in enumerate(expected_features):
            with cols[i % 2]:
                dtype = load_reference_data()[col_name].dtype
                if np.issubdtype(dtype, np.number):
                    min_val = float(load_reference_data()[col_name].min())
                    max_val = float(load_reference_data()[col_name].max())
                    default = float(load_reference_data()[col_name].median())
                    input_dict[col_name] = st.number_input(
                        f"{col_name}", min_val, max_val, default
                    )
                else:
                    unique_vals = sorted(load_reference_data()[col_name].dropna().unique())
                    input_dict[col_name] = st.selectbox(f"{col_name}", unique_vals)

        if st.button("🚀 Prédire la prime", type="primary"):
            input_df = pd.DataFrame([input_dict])
            try:
                pred = model.predict(input_df)[0]
                st.success(f"✅ Prime estimée : **{pred:,.2f} €**")
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")
                st.write("Colonnes reçues :", list(input_df.columns))
                st.write("Colonnes attendues :", expected_features)

    else:
        uploaded_file = st.file_uploader("📤 Uploader un fichier CSV avec les mêmes colonnes que X_train.csv")
        if uploaded_file:
            df_input = pd.read_csv(uploaded_file)
            st.write("Aperçu des données :", df_input.head())
            missing_cols = set(expected_features) - set(df_input.columns)
            if missing_cols:
                st.warning(f"Colonnes manquantes : {missing_cols}")
            else:
                if st.button("Lancer les prédictions"):
                    try:
                        preds = model.predict(df_input[expected_features])
                        df_input["Predicted_Premium"] = preds
                        st.dataframe(df_input)
                        st.download_button(
                            "📥 Télécharger les prédictions",
                            df_input.to_csv(index=False),
                            "insurance_predictions.csv",
                            "text/csv"
                        )
                    except Exception as e:
                        st.error(f"Erreur : {e}")


# ───────────────────────────────────────
# FOOTER
# ───────────────────────────────────────
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Bilal Sayoud")
with col2:
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/bilalsayoud/) | "
        "[GitHub](https://github.com/Bilelly)"
    )