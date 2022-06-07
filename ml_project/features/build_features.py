import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from ml_project.data.read_params import FeatureParams


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("default", "passthrough", params.default_features),
            ("min_max", MinMaxScaler(), params.float_features),
            ("ohe", OneHotEncoder(sparse=False), params.integer_features),
        ]
    )


def extract_target(df: pd.DataFrame, params: FeatureParams) -> np.ndarray:
    return df[params.target_col].to_numpy()


def extract_features(df: pd.DataFrame, transformer: ColumnTransformer) -> np.ndarray:
    return transformer.transform(df)
