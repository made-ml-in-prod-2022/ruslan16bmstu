from typing import Tuple, NoReturn
import pandas as pd
from sklearn.model_selection import train_test_split
from data.read_params import SplittingParams, FeatureParams, DownloadParams

def download_data(params: DownloadParams):
    df = pd.read_csv(params.url)
    df.to_csv(f"{params.output_folder}/{params.filename}")

def read_data(params: DownloadParams) -> pd.DataFrame:
    df = pd.read_csv(f"{params.output_folder}/{params.filename}")
    return df

def split_data(df: pd.DataFrame, params: SplittingParams, feature: FeatureParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(
        df, test_size=params.val_size, stratify=df[feature.target_col] ,random_state=params.random_state
    )
    return df_train, df_test
