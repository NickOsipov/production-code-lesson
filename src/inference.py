"""
Module for making inferences with the model
"""
from typing import Optional

import joblib
import pandas as pd
from sklearn.base import BaseEstimator


def load_model(path: str) -> Optional[BaseEstimator]:
    """_summary_

    Parameters
    ----------
    path : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    try:
        return joblib.load(path)
    except Exception as e:
        print(f'Error loading model from {path}: {e}')
        return None


def predict(model: BaseEstimator, df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    model : BaseEstimator
        _description_
    df : pd.DataFrame
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    return model.predict(df)
