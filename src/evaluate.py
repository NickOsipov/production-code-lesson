"""
Module for evaluating the performance of the model
"""

from sklearn.metrics import accuracy_score

from inference import predict

def evaluate_model(model, df):
    X_test, y_test = df.drop('target', axis=1), df['target']
    y_pred = predict(model, X_test)
    
    # print('Confusion Matrix:')
    # print(confusion_matrix(y_test, y_pred))
    # print('\nClassification Report:')
    # print(classification_report(y_test, y_pred))
    # print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    return accuracy_score(y_test, y_pred)
