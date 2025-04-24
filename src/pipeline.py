"""
Main file for the project
"""
import os

from loguru import logger

from preprocessing import preprocessing_pipeline
from train import train_pipeline
from inference import load_model
from evaluate import evaluate_model


def main():
    logger.info('Starting the pipeline')
    data_path = 'data'
    model_path = os.path.join("models", "model.joblib")

    logger.info('Preprocessing data')
    df_train, df_val, df_test = preprocessing_pipeline(data_path)

    logger.info('Training model')
    model = train_pipeline(df_train, model_path, model_params={'n_estimators': 100})

    logger.info('Evaluating model on validation set')
    metric = evaluate_model(model, df_val)
    logger.info(f'Validation metric: {metric}')

    logger.info('Loading model')
    model_loaded = load_model(model_path)

    logger.info('Evaluating model on test set')
    metric_test = evaluate_model(model_loaded, df_test)
    logger.info(f'Test metric: {metric_test}')


    logger.info('Pipeline finished')


if __name__ == "__main__":
    main()