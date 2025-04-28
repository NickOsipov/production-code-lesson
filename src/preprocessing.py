"""
Module: preprocessing
Description: This module contains functions for preprocessing text data.
"""

import os

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load data from a CSV file.
    Returns:
        DataFrame: A pandas DataFrame containing the loaded data.
    """
    # Load the iris dataset as an example
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data = data.rename(columns={
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width'
    })
    data['target'] = iris.target
    return data


def split_data(
    df: pd.DataFrame, 
    test_size: float=0.2, 
    random_state: float=42
    ) -> tuple:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        input data
    test_size : float, optional
        Test size, by default 0.2
    random_state : float, optional
        Random state, by default 42

    Returns
    -------
    tuple
        Tuple result
    """

    train, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state
    )
    return train, test


def save_data(df: pd.DataFrame, file_path: str):
    """
    Save the DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str
        Path to save the CSV file
    """
    df.to_csv(file_path, index=False)


def preprocessing_pipeline(file_path: str):
    """
    Run the preprocessing pipeline.

    Parameters
    ----------
    file_path : str
        Path to save the CSV file
    """
    # Load data
    df = load_data()

    df_train_val, df_test = split_data(df)
    df_train, df_val = split_data(df_train_val)

    for df, name in zip(
        [df_train, df_val, df_test], 
        ['train', 'val', 'test']
    ):
        save_data(df, os.path.join(f"{file_path}", f"{name}.csv"))

    return df_train, df_val, df_test