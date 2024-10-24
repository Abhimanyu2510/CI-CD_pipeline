import pytest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from app import train_model, predict

def test_train_model():
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    
    model, scaler, scaled_data = train_model(data, n_clusters=3)
    
    # Test if model is trained
    assert isinstance(model, KMeans)
    assert model.n_clusters == 3
    
    # Test if data is scaled
    assert scaled_data.shape == data.shape
    
def test_predict():
    # Generate sample data
    np.random.seed(42)
    train_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    
    # Train model
    model, scaler, _ = train_model(train_data, n_clusters=3)
    
    # Generate test data
    test_data = pd.DataFrame({
        'feature1': np.random.randn(10),
        'feature2': np.random.randn(10)
    })
    
    # Test predictions
    predictions = predict(test_data, model, scaler)
    assert len(predictions) == len(test_data)
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
