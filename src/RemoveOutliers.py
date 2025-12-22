import pandas as pd
import numpy as np

def remove_outliers(df, features):
    """
    Supprime les lignes contenant des outliers (méthode IQR) sur au moins une des colonnes spécifiées.
    
    Parameters:
        df (pd.DataFrame): DataFrame d'entrée
        features (list): Liste des colonnes numériques à traiter
    
    Returns:
        pd.DataFrame: DataFrame sans les lignes contenant des outliers
    """
    df_cleaned = df.copy()
    mask = pd.Series([True] * len(df_cleaned), index=df_cleaned.index)  # Commence par garder toutes les lignes

    for feature in features:
        Q1 = df_cleaned[feature].quantile(0.25)
        Q3 = df_cleaned[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Met à jour le masque : garde uniquement les lignes valides POUR CETTE colonne
        mask &= (df_cleaned[feature] >= lower_bound) & (df_cleaned[feature] <= upper_bound)

    return df_cleaned[mask]