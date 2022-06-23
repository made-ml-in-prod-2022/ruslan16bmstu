from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.data.read_params import SplittingParams, DownloadParams


def download_data(params: DownloadParams):
    df = pd.read_csv(params.url)
    df.to_csv(params.file_path)


def read_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def split_data(
    df: pd.DataFrame,
    params: SplittingParams,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(
        df,
        test_size=params.val_size,
        stratify=df[target_col],
        random_state=params.random_state,
    )
    return df_train, df_test
