import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Optional: if you want to use feature importance or selection
# from feature_selection import get_feature_importance

def load_csv(csv_file):
    """Load a CSV file from path or file-like object."""
    try:
        df = pd.read_csv(csv_file)
        print(f"[INFO] Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

def analyze_columns(df, target_col):
    """Separate features into numerical and categorical, ignoring the target."""
    feature_cols = [col for col in df.columns if col != target_col]
    num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    return num_cols, cat_cols

def preprocess_data(df, target_col, test_size=0.2, random_state=42):
    """
    Cleans and prepares *any* tabular dataset for modeling.
    - Handles numerics (impute mean, scale)
    - Handles categoricals (impute mode, one-hot encode)
    - Drops target col, splits train/test
    - Returns processed train/test arrays, targets, and the fitted preprocessor.
    """
    num_cols, cat_cols = analyze_columns(df, target_col)

    # Split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 20 else None
    )

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)


    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def get_feature_names(preprocessor):
    """
    Returns the output feature names from the ColumnTransformer (for feature importance).
    """
    num_features = []
    cat_features = []
    # Handle numeric features
    if 'num' in preprocessor.named_transformers_ and hasattr(preprocessor.named_transformers_['num'], 'named_steps'):
        scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        # This will just repeat original column names (scaler doesn't add names)
        num_features = preprocessor.transformers_[0][2]
    # Handle categorical features
    if 'cat' in preprocessor.named_transformers_ and hasattr(preprocessor.named_transformers_['cat'], 'named_steps'):
        encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cats = encoder.get_feature_names_out(preprocessor.transformers_[1][2])
        cat_features = cats
    return np.concatenate([num_features, cat_features])

# Example usage for *any* CSV:
# df = load_csv('your_data.csv')
# X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_col='YourTargetCol')
# feature_names = get_feature_names(preprocessor)
