"""
Module: train.py
Description: This module contains functions for training a machine learning model.
"""

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

def train_model(model: BaseEstimator, model_params: dict, df_train: pd.DataFrame):
    """
    Train a machine learning model.

    Parameters
    ----------
    model : object
        The machine learning model to be trained.
    model_params : dict
        The parameters for the model.
    df_train : DataFrame
        The training data.

    Returns
    -------
    object
        The trained model.
    """

    X_train, y_train = df_train.drop('target', axis=1), df_train['target']

    model_instance = model(**model_params)
    model_instance.fit(X_train, y_train)
    
    return model_instance


def save_model(model: BaseEstimator, model_path: str):
    """
    Save the trained model to a file.

    Parameters
    ----------
    model : object
        The trained model to be saved.
    """
    
    joblib.dump(model, model_path)


def train_pipeline(df_train, model_path, model=RandomForestClassifier, model_params=None):
    model = train_model(model, model_params, df_train)
    save_model(model, model_path)

    return model