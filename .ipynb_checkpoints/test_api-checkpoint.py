import pytest
import mlflow
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from elasticsearch import Elasticsearch
from fastapi.testclient import TestClient
from app import app  # Assuming app.py is your FastAPI application file
from model_pipeline import train_model, evaluate_model, save_model, load_model
from logger import log_to_mlflow, log_to_elasticsearch


# --- 1. Setup Elasticsearch Mock ---
@pytest.fixture
def es_mock():
    # Mocking Elasticsearch client
    es = Elasticsearch()
    es.index = lambda index, body: body  # Mock index method
    return es


# --- 2. Test for train_model function ---
def test_train_model():
    X_train = np.random.rand(10, 5)  # 10 samples, 5 features
    y_train = np.random.randint(0, 2, size=10)  # Binary target
    model = train_model(X_train, y_train)

    # Check if the model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, 'predict')  # Ensure the model has a predict method


# --- 3. Test for evaluate_model function ---
def test_evaluate_model():
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, size=10)
    model = train_model(X_train, y_train)
    X_test = np.random.rand(5, 5)
    y_test = np.random.randint(0, 2, size=5)

    accuracy = evaluate_model(model, X_test, y_test)

    # Check that the accuracy is between 0 and 1
    assert 0 <= accuracy <= 1


# --- 4. Test for save_model and load_model functions ---
def test_save_and_load_model():
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, size=10)
    model = train_model(X_train, y_train)

    save_model(model)

    # Check if the model file is saved
    assert joblib.load('model.pkl') is not None

    # Load the saved model and check its type
    loaded_model = load_model()
    assert isinstance(loaded_model, RandomForestClassifier)

    # Clean up the saved model file
    os.remove('model.pkl')


# --- 5. Test for log_to_mlflow function ---
def test_log_to_mlflow():
    model = RandomForestClassifier()
    model.n_features_in_ = 5  # Mock the feature count
    accuracy = 0.85
    with mlflow.start_run():
        log_to_mlflow(model, accuracy)

    # Check that the mlflow methods are called
    mlflow.log_param.assert_called_with("n_estimators", 100)
    mlflow.log_metric.assert_called_with("accuracy", accuracy)


# --- 6. Test for log_to_elasticsearch function ---
def test_log_to_elasticsearch(es_mock):
    run_id = 'run_123'
    accuracy = 0.85
    log_to_elasticsearch(es_mock, run_id, accuracy)

    # Check if Elasticsearch is called with the correct index and body
    es_mock.index.assert_called_with(
        index="mlflow-metrics",
        body={'run_id': run_id, 'accuracy': accuracy}
    )


# --- 7. Test for the FastAPI /predict endpoint ---
@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_predict(client):
    # Mock data for prediction
    data = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
    
    # Send a POST request to /predict
    response = client.post("/predict", json=data)

    # Check if the response is 200 OK
    assert response.status_code == 200

    # Check if the response contains the prediction
    assert "prediction" in response.json()


# --- 8. Test for the FastAPI /retrain endpoint ---
def test_retrain(client):
    data = {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
    
    # Send a POST request to /retrain
    response = client.post("/retrain", json=data)

    # Check if the response is 200 OK
    assert response.status_code == 200

    # Check if the retraining message is returned
    assert "Model retrained and logged to MLflow" in response.json().get("message")


# --- 9. Test for /feature-stats endpoint ---
def test_feature_stats(client):
    # Send a GET request to /feature-stats
    response = client.get("/feature-stats")

    # Check if the response is 200 OK
    assert response.status_code == 200

    # Check if the response contains the expected feature stats
    assert "mean_features" in response.json()


# --- 10. Test for / endpoint (Root API check) ---
def test_root(client):
    response = client.get("/")

    # Check if the response is 200 OK
    assert response.status_code == 200

    # Check if the response message is correct
    assert response.json() == {"message": "Welcome to the ML Model API with MLflow Tracking!"}
