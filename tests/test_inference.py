import pandas as pd

from src.inference import predict


def test_predict():
    # Create a mock model
    class MockModel:
        def predict(self, X):
            return [1] * len(X)

    # Create a mock DataFrame
    df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

    # Create an instance of the mock model
    model = MockModel()

    # Call the predict function
    predictions = predict(model, df)

    # Check if the predictions are as expected
    assert predictions == [1, 1, 1], "Predictions do not match expected output"