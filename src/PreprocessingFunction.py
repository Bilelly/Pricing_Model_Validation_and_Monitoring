from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(X):
    # Identify numeric and categorical columns
    var_num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    var_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), var_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), var_cat)
        ]
    )
    
    return preprocessor