from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import requests  # Used for sending logs to Elasticsearch

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API with MLflow Tracking",
    description="API for managing machine learning models and tracking experiments with MLflow",
    version="1.0.0",
)

# MLflow Config
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Assuming your MLflow container is at this URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Elasticsearch URL for logging
ELASTICSEARCH_URL = "http://elasticsearch:9200/logs-ml/_doc"

# Define paths
TRAIN_DATA_PATH = "churn-bigml-80.csv"
MODEL_PATH = "model.pkl"

# Function to set up Elasticsearch client
def setup_elasticsearch():
    es = Elasticsearch([{'scheme': 'http', 'host': 'localhost', 'port': 9200}])
    return es

# Function to load dataset and check feature count
def load_data():
    if not os.path.exists(TRAIN_DATA_PATH):
        raise RuntimeError(f"❌ Dataset missing at {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    NUMERIC_FEATURES = train_df.select_dtypes(include=[np.number]).columns.tolist()
    FEATURE_COUNT = len(NUMERIC_FEATURES) - 1
    if FEATURE_COUNT < 10:
        raise RuntimeError(f"❌ Dataset only has {FEATURE_COUNT} features, too few for ML.")
    return train_df, FEATURE_COUNT

# Load model if available
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model = joblib.load(f)
                print(f"✅ Model loaded with {model.n_features_in_} features")
                return model
        else:
            print("⚠ Model not found. Please retrain.")
            return None
    except Exception as e:
        raise RuntimeError(f"❌ Error loading model: {e}")

# Define request body for prediction
class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=10)

@app.post("/predict")
async def predict(data: PredictionInput):
    """Make predictions using the trained model."""
    model = load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="❌ Model not loaded. Train first.")
    
    try:
        input_array = np.array(data.features).reshape(1, -1)
        expected_features = model.n_features_in_

        if input_array.shape[1] != expected_features:
            raise HTTPException(status_code=400, detail=f"❌ Expected {expected_features} features, got {input_array.shape[1]}")

        prediction = model.predict(input_array)

        # Log prediction to Elasticsearch
        log_data = {"prediction": prediction.tolist(), "status": "success"}
        try:
            requests.post(ELASTICSEARCH_URL, json=log_data)
        except Exception as e:
            print(f"⚠️ Failed to send logs to Elasticsearch: {str(e)}")

        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"❌ Prediction error: {str(e)}")

# Define hyperparameters for retraining
class HyperParams(BaseModel):
    n_estimators: Optional[int] = Field(100, ge=1)
    max_depth: Optional[int] = Field(None, ge=1)
    min_samples_split: Optional[int] = Field(2, ge=2)
    min_samples_leaf: Optional[int] = Field(1, ge=1)

@app.post("/retrain")
async def retrain(params: HyperParams):
    """Retrain model and log to MLflow."""
    try:
        train_df, FEATURE_COUNT = load_data()
        df = train_df.select_dtypes(include=[np.number])

        X = df.iloc[:, :FEATURE_COUNT].values  
        y = df.iloc[:, -1].values  

        max_depth = params.max_depth if params.max_depth is None or params.max_depth >= 1 else None

        # Start MLflow Experiment
        with mlflow.start_run():
            # Train Model
            new_model = RandomForestClassifier(
                n_estimators=params.n_estimators,
                max_depth=max_depth,
                min_samples_split=params.min_samples_split,
                min_samples_leaf=params.min_samples_leaf,
                random_state=42
            )
            new_model.fit(X, y)

            # Log hyperparameters
            mlflow.log_param("n_estimators", params.n_estimators)
            mlflow.log_param("max_depth", params.max_depth)
            mlflow.log_param("min_samples_split", params.min_samples_split)
            mlflow.log_param("min_samples_leaf", params.min_samples_leaf)

            # Log Model
            mlflow.sklearn.log_model(new_model, "random_forest_model")

        # Save trained model
        joblib.dump(new_model, MODEL_PATH)
        
        # Log retraining to Elasticsearch
        es = setup_elasticsearch()
        log_data = {"message": "Model retrained and logged to MLflow", "status": "success"}
        try:
            requests.post(ELASTICSEARCH_URL, json=log_data)
        except Exception as e:
            print(f"⚠️ Failed to send retraining logs to Elasticsearch: {str(e)}")

        return {"message": "✅ Model retrained and logged to MLflow"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"❌ Retraining error: {str(e)}")

@app.get("/feature-stats")
async def feature_stats():
    """Return feature mean values for testing."""
    try:
        train_df, FEATURE_COUNT = load_data()
        df = train_df.select_dtypes(include=[np.number])
        mean_values = df.iloc[:, :FEATURE_COUNT].mean().tolist()
        return {"mean_features": mean_values}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error computing feature stats: {str(e)}")

@app.get("/")
def root():
    return {"message": "Welcome to the ML Model API with MLflow Tracking!"}
