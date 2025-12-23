import pandas as pd
import numpy as np

def remove_outliers(df, features, factor=1.5):
    """
    Cap les outliers par la méthode IQR (winsorisation)
    """
    df_capped = df.copy()

    for feature in features:
        Q1 = df_capped[feature].quantile(0.25)
        Q3 = df_capped[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df_capped[feature] = df_capped[feature].clip(lower, upper)

    return df_capped
