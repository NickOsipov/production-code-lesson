"""
Module: app.py
Description: 
    This module contains the main application logic for the Flask application.
"""

from flask import Flask, jsonify, request
from loguru import logger
import pandas as pd

from src.inference import load_model, predict
from config.variables import MODEL_PATH

app = Flask(__name__)

logger.info("Starting the Flask application...")
logger.info("Loading the model...")
MODEL = load_model(MODEL_PATH)
logger.info("Model loaded successfully.")


@app.route('/healthcheck', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    Returns a simple message indicating that the service is running.
    """
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def prediction():
    """
    Prediction endpoint.
    Accepts a POST request with input data and returns the model's prediction.
    """
    try:
        logger.info("Received a prediction request.")
        data = request.json
        logger.info("Data received for prediction")

        df = pd.DataFrame(data, index=[0])
        df = df.reset_index(drop=True)
        logger.info("Data converted to DataFrame")
        logger.debug(f"DataFrame: {df.shape}")

        prediction = predict(MODEL, df)
        logger.info("Prediction made successfully.")

        classes = ["Setosa", "Versicolor", "Virginica"]
        prediction = classes[prediction[0]]
        logger.debug(f"Prediction: {prediction}")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
    return jsonify({"prediction": prediction})
