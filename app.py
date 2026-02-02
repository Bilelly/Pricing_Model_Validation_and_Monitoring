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
# TAB 2 : DÉTECTION DE DRIFT
# ───────────────────────────────────────
with tab2:
    st.subheader("📉 Détection de Drift ")

    try:
        ref_df = load_reference_data()      # X_train.csv
        prod_df = load_production_data()    # X_drift.csv
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        st.stop()

    st.info(f"🔹 Données de référence : {ref_df.shape[0]:,} lignes\n"
            f"🔹 Données de production : {prod_df.shape[0]:,} lignes")

    if st.button("🔍 Calculer le drift (PSI)", type="primary"):
        with st.spinner("Analyse du drift en cours..."):

            # Aligner les colonnes
            common_cols = ref_df.columns.intersection(prod_df.columns)
            if len(common_cols) == 0:
                st.error("❌ Aucune colonne commune entre les deux jeux de données.")
                st.stop()

            ref_aligned = ref_df[common_cols]
            prod_aligned = prod_df[common_cols]

            # Appel de ta fonction
            try:
                psi_summary = calculate_psi(ref_aligned, prod_aligned)
            except Exception as e:
                st.error(f"Erreur lors du calcul du PSI : {e}")
                st.exception(e)
                st.stop()

            # Trier par PSI décroissant
            psi_summary = psi_summary.sort_values("PSI", ascending=False).reset_index(drop=True)

            # Ajouter un emoji pour la lisibilité
            def add_emoji(row):
                if row["Drift"] == "Significant drift":
                    return "🔴 " + row["Variable"]
                elif row["Drift"] == "Moderate drift":
                    return "🟠 " + row["Variable"]
                else:
                    return "🟢 " + row["Variable"]

            psi_summary["Variable (état)"] = psi_summary.apply(add_emoji, axis=1)

            # Afficher le tableau principal
            st.markdown("### 📊 Résultats du PSI")
            st.dataframe(
                psi_summary[["Variable (état)", "PSI", "Drift"]],
                column_config={
                    "PSI": st.column_config.NumberColumn(format="%.4f"),
                },
                hide_index=True,
                use_container_width=True
            )

            # --- Métriques globales ---
            n_total = len(psi_summary)
            n_stable = len(psi_summary[psi_summary["Drift"] == "No drift"])
            n_moderate = len(psi_summary[psi_summary["Drift"] == "Moderate drift"])
            n_significant = len(psi_summary[psi_summary["Drift"] == "Significant drift"])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Variables", n_total)
            col2.metric("🟢 Stables", n_stable)
            col3.metric("🟠 Modérées", n_moderate)
            col4.metric("🔴 Significatives", n_significant)

            # --- Graphique natif : PSI par variable ---
            st.markdown("### 📈 PSI par variable (trié)")
            chart_data = psi_summary.set_index("Variable (état)")["PSI"]
            st.bar_chart(chart_data, color="#FF6B6B", height=400)

            # --- Alertes ---
            if n_significant > 0:
                st.error(f"⚠️ **{n_significant} variable(s)** avec **drift significatif** (PSI ≥ 0.25) — nécessite une investigation.")
            elif n_moderate > 0:
                st.warning(f"ℹ️ **{n_moderate} variable(s)** avec drift modéré (0.1 ≤ PSI < 0.25) — à surveiller.")
            else:
                st.success("✅ Aucun drift détecté. Les données sont stables.")

            # --- Optionnel : upload personnalisé ---
            st.markdown("### 📤 Tester avec vos propres données")
            uploaded_custom = st.file_uploader("Uploader un CSV de production (même format que X_train)", type="csv")
            if uploaded_custom:
                try:
                    custom_df = pd.read_csv(uploaded_custom)
                    missing_cols = set(common_cols) - set(custom_df.columns)
                    if missing_cols:
                        st.warning(f"Colonnes manquantes : {missing_cols}")
                    else:
                        psi_custom = calculate_psi(ref_aligned, custom_df[common_cols])
                        st.markdown("#### Résultats sur vos données")
                        st.dataframe(psi_custom, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur : {e}")

    else:
        st.write("Cliquez sur **Calculer le drift (PSI)** pour lancer l'analyse.")

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