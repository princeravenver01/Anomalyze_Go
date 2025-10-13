"""
Anomalyze API Server for Render Deployment
Python 3.12 Compatible
Hosts ML models and provides prediction API endpoints
"""

from __future__ import annotations
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
import io
from pathlib import Path
from utils.preprocessing import load_and_preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for Vercel frontend

# Configure models folder
MODELS_FOLDER = 'models'

# Global variables
ensemble_models = None
scaler = None
data_columns = None
optimal_threshold = None
model_scores = None

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'ensemble_models.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')
MODEL_SCORES_PATH = os.path.join(MODELS_FOLDER, 'model_scores.joblib')


def ensemble_predict(models: list, data: np.ndarray, threshold: float) -> np.ndarray:
    """Use ensemble of models for predictions with majority voting."""
    predictions = []
    
    for model in models:
        distances = model.transform(data).min(axis=1)
        pred = (distances > threshold).astype(int)
        predictions.append(pred)
    
    # Use majority voting
    ensemble_pred = np.array(predictions).mean(axis=0)
    return (ensemble_pred > 0.5).astype(int)


def calculate_anomaly_severity(distances: np.ndarray, threshold: float) -> list[str]:
    """Calculate severity levels for detected anomalies."""
    severity_levels = []
    
    # Ensure threshold is float
    threshold = float(threshold)
    
    for distance in distances:
        # Ensure distance is float
        distance = float(distance)
        
        if distance <= threshold:
            severity_levels.append('Normal')
        elif distance <= threshold * 1.5:
            severity_levels.append('Low')
        elif distance <= threshold * 2.0:
            severity_levels.append('Medium')
        elif distance <= threshold * 3.0:
            severity_levels.append('High')
        else:
            severity_levels.append('Critical')
    
    return severity_levels


def load_model() -> bool:
    """Loads the ensemble models and threshold."""
    global ensemble_models, scaler, data_columns, optimal_threshold, model_scores
    
    try:
        if not os.path.exists(MODEL_PATH):
            print("ERROR: Model files not found!")
            return False
            
        print("Loading ensemble models from disk...")
        ensemble_models = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        data_columns = joblib.load(COLUMNS_PATH)
        optimal_threshold = joblib.load(THRESHOLD_PATH)
        
        # Try to load model scores if they exist
        if os.path.exists(MODEL_SCORES_PATH):
            model_scores = joblib.load(MODEL_SCORES_PATH)
            print("Model scores loaded successfully.")
        else:
            print("Model scores not found, will use equal weighting.")
            model_scores = None
            
        print("Ensemble models loaded successfully.")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': ensemble_models is not None,
        'timestamp': time.time()
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded models."""
    if ensemble_models is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    return jsonify({
        'num_models': len(ensemble_models),
        'num_features': len(data_columns),
        'feature_names': list(data_columns),
        'optimal_threshold': float(optimal_threshold),
        'model_type': 'K-Means Ensemble'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    start_time = time.time()
    
    if ensemble_models is None or optimal_threshold is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read and preprocess data
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        stream.seek(0)
        df_test_original = pd.read_csv(stream, header=None)
        
        stream.seek(0)
        df_test = load_and_preprocess_data(stream)
        
        has_labels = 'label' in df_test.columns
        metrics = {}
        
        if has_labels:
            true_labels = df_test['label'].copy()
            true_labels_binary = (true_labels != 'normal').astype(int)
            df_test = df_test.drop('label', axis=1)
        
        # Ensure columns match and scale
        df_test = df_test.reindex(columns=data_columns, fill_value=0)
        
        # Ensure all columns are numeric before scaling
        for col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0).astype(float)
        
        df_test_scaled = scaler.transform(df_test)
        
        # Use ensemble prediction
        anomalies_mask = ensemble_predict(ensemble_models, df_test_scaled, optimal_threshold)
        
        # Calculate confidence scores and distances
        distances_list = []
        for model in ensemble_models:
            distances = model.transform(df_test_scaled).min(axis=1)
            distances_list.append(distances)
        distances = np.mean(distances_list, axis=0)
        confidence_scores = 1 / (1 + distances)
        
        # Calculate severity levels
        severity_levels = calculate_anomaly_severity(distances, optimal_threshold)
        
        # Build results
        anomalies_indices = np.where(anomalies_mask)[0]
        anomalies_data = []
        
        for idx in anomalies_indices:
            anomaly_record = df_test_original.iloc[idx].to_dict()
            anomaly_record['confidence'] = float(confidence_scores[idx])
            anomaly_record['severity'] = severity_levels[idx]
            anomaly_record['distance'] = float(distances[idx])
            anomaly_record['index'] = int(idx)
            anomalies_data.append(anomaly_record)
        
        # Calculate metrics if labels are available
        if has_labels:
            accuracy = accuracy_score(true_labels_binary, anomalies_mask)
            precision = precision_score(true_labels_binary, anomalies_mask, zero_division=0)
            recall = recall_score(true_labels_binary, anomalies_mask, zero_division=0)
            f1 = f1_score(true_labels_binary, anomalies_mask, zero_division=0)
            
            metrics = {
                'accuracy': float(accuracy * 100),
                'precision': float(precision * 100),
                'recall': float(recall * 100),
                'f1_score': float(f1 * 100),
                'total_samples': int(len(df_test_original)),
                'anomalies_detected': int(len(anomalies_data)),
                'processing_time': float(time.time() - start_time),
                'avg_confidence': float(np.mean(confidence_scores) * 100)
            }
        else:
            metrics = {
                'total_samples': int(len(df_test_original)),
                'anomalies_detected': int(len(anomalies_data)),
                'processing_time': float(time.time() - start_time),
                'avg_confidence': float(np.mean(confidence_scores) * 100)
            }
        
        return jsonify({
            'success': True,
            'anomalies': anomalies_data,
            'metrics': metrics,
            'has_labels': has_labels
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict-json', methods=['POST'])
def predict_json():
    """Prediction endpoint accepting JSON data."""
    start_time = time.time()
    
    if ensemble_models is None or optimal_threshold is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert JSON data to DataFrame
        df_test = pd.DataFrame(data['data'])
        
        has_labels = 'label' in df_test.columns
        if has_labels:
            true_labels = df_test['label'].copy()
            true_labels_binary = (true_labels != 'normal').astype(int)
            df_test = df_test.drop('label', axis=1)
        
        # Ensure columns match and scale
        df_test = df_test.reindex(columns=data_columns, fill_value=0)
        df_test_scaled = scaler.transform(df_test)
        
        # Use ensemble prediction
        anomalies_mask = ensemble_predict(ensemble_models, df_test_scaled, optimal_threshold)
        
        # Calculate confidence scores and distances
        distances_list = []
        for model in ensemble_models:
            distances = model.transform(df_test_scaled).min(axis=1)
            distances_list.append(distances)
        distances = np.mean(distances_list, axis=0)
        confidence_scores = 1 / (1 + distances)
        
        # Calculate severity levels
        severity_levels = calculate_anomaly_severity(distances, optimal_threshold)
        
        # Build results
        results = []
        for i, is_anomaly in enumerate(anomalies_mask):
            results.append({
                'index': i,
                'is_anomaly': bool(is_anomaly),
                'confidence': float(confidence_scores[i]),
                'severity': severity_levels[i],
                'distance': float(distances[i])
            })
        
        metrics = {
            'total_samples': len(df_test),
            'anomalies_detected': int(anomalies_mask.sum()),
            'processing_time': float(time.time() - start_time),
            'avg_confidence': float(np.mean(confidence_scores) * 100)
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Load models on startup
if __name__ == '__main__':
    print("Starting Anomalyze API Server...")
    if load_model():
        print("Models loaded successfully. Starting server...")
        port = int(os.environ.get('PORT', 10000))
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Failed to load models. Exiting...")
        exit(1)
else:
    # For gunicorn
    load_model()
