"""
Anomalyze API Server for Render Deployment
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
UPLOADED_LOGS_FOLDER = 'data/uploaded_logs'

# Create folders if they don't exist
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(UPLOADED_LOGS_FOLDER, exist_ok=True)

# Global variables
ensemble_models = None
scaler = None
data_columns = None
optimal_threshold = None
model_scores = None
upload_counter = 0  # Track number of uploads since last retrain
RETRAIN_THRESHOLD = 10  # Retrain after N uploads

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'ensemble_models.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')
MODEL_SCORES_PATH = os.path.join(MODELS_FOLDER, 'model_scores.joblib')
UPLOAD_COUNTER_PATH = os.path.join(MODELS_FOLDER, 'upload_counter.txt')


def ensemble_predict(models: list, data: np.ndarray, threshold: float) -> np.ndarray:
    """Use ensemble of models for predictions with majority voting - OPTIMIZED."""
    predictions = []
    threshold = float(threshold)
    
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


def save_uploaded_file(file_content: str, filename: str) -> tuple[str, bool]:
    """
    Save uploaded file to the uploaded_logs folder for future training.
    Returns tuple of (file_path, is_duplicate)
    """
    from datetime import datetime
    import hashlib
    import glob
    
    # Calculate hash of uploaded file
    file_hash = hashlib.sha256(file_content.encode()).hexdigest()
    
    # Check if this file already exists (compare hashes)
    existing_files = glob.glob(os.path.join(UPLOADED_LOGS_FOLDER, '*.txt'))
    
    for existing_file in existing_files:
        try:
            with open(existing_file, 'r') as f:
                existing_content = f.read()
                existing_hash = hashlib.sha256(existing_content.encode()).hexdigest()
                
                if file_hash == existing_hash:
                    print(f"⚠ Duplicate file detected! Matches: {os.path.basename(existing_file)}")
                    print(f"   File hash: {file_hash[:16]}...")
                    return existing_file, True  # Return existing file path, mark as duplicate
        except Exception as e:
            # Skip files that can't be read
            continue
    
    # Not a duplicate - save the file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(UPLOADED_LOGS_FOLDER, safe_filename)
    
    with open(file_path, 'w') as f:
        f.write(file_content)
    
    print(f"✓ Saved new uploaded file: {file_path}")
    print(f"   File hash: {file_hash[:16]}...")
    return file_path, False  # Return new file path, not a duplicate


def increment_upload_counter() -> int:
    """Increment and return the upload counter."""
    global upload_counter
    
    # Load counter from file if exists
    if os.path.exists(UPLOAD_COUNTER_PATH):
        with open(UPLOAD_COUNTER_PATH, 'r') as f:
            upload_counter = int(f.read().strip())
    
    # Increment
    upload_counter += 1
    
    # Save counter
    with open(UPLOAD_COUNTER_PATH, 'w') as f:
        f.write(str(upload_counter))
    
    return upload_counter


def should_retrain() -> bool:
    """Check if model should be retrained based on upload count."""
    return upload_counter >= RETRAIN_THRESHOLD


def trigger_retraining():
    """Trigger model retraining with all available data (async)."""
    global upload_counter
    
    print(f"\n{'='*60}")
    print(f"🔄 RETRAINING TRIGGERED - {upload_counter} uploads accumulated")
    print(f"{'='*60}\n")
    
    try:
        # Import training function
        import subprocess
        import sys
        
        # Run retrain script in background
        print("Starting retraining process...")
        result = subprocess.Popen(
            [sys.executable, 'retrain_model.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"✓ Retraining initiated (PID: {result.pid})")
        
        # Reset counter
        upload_counter = 0
        with open(UPLOAD_COUNTER_PATH, 'w') as f:
            f.write('0')
        
        return True
        
    except Exception as e:
        print(f"✗ Error triggering retraining: {e}")
        return False


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
    """Main prediction endpoint - OPTIMIZED for speed."""
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
        file_content = file.stream.read().decode("UTF8")
        stream = io.StringIO(file_content, newline=None)
        stream.seek(0)
        df_test_original = pd.read_csv(stream, header=None)
        
        # ASYNC: Save uploaded file in background (non-blocking)
        # This doesn't delay the response
        import threading
        def save_async():
            try:
                saved_path, is_duplicate = save_uploaded_file(file_content, file.filename or 'upload.txt')
                if not is_duplicate:
                    increment_upload_counter()
                    if should_retrain():
                        trigger_retraining()
            except:
                pass  # Don't let background task crash main request
        
        threading.Thread(target=save_async, daemon=True).start()
        
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
        
        # OPTIMIZED: Calculate distances once and reuse
        # Instead of calculating twice (once for prediction, once for confidence)
        distances_list = []
        for model in ensemble_models:
            distances = model.transform(df_test_scaled).min(axis=1)
            distances_list.append(distances)
        
        distances = np.mean(distances_list, axis=0)
        
        # Use distances for anomaly detection
        threshold = float(optimal_threshold)
        anomalies_mask = (distances > threshold).astype(int)
        
        confidence_scores = 1 / (1 + distances)
        
        # Calculate severity levels
        severity_levels = calculate_anomaly_severity(distances, optimal_threshold)
        
        # Build results - OPTIMIZED: Only process anomalies
        anomalies_indices = np.where(anomalies_mask)[0]
        
        # No limit - showing all anomalies
        total_anomalies = len(anomalies_indices)
        
        anomalies_data = []
        
        for idx in anomalies_indices:
            # Get the record and convert all values to JSON-serializable types
            anomaly_record = {}
            row_data = df_test_original.iloc[idx]
            for col_idx, value in enumerate(row_data):
                # Convert all values to appropriate JSON types
                if pd.isna(value):
                    anomaly_record[str(col_idx)] = None
                elif isinstance(value, (np.integer, int)):
                    anomaly_record[str(col_idx)] = int(value)
                elif isinstance(value, (np.floating, float)):
                    anomaly_record[str(col_idx)] = float(value)
                else:
                    anomaly_record[str(col_idx)] = str(value)
            
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
                'anomalies_detected': int(total_anomalies),  # Total anomalies found
                'anomalies_displayed': int(len(anomalies_data)),  # Shown in UI
                'processing_time': float(time.time() - start_time),
                'avg_confidence': float(np.mean(confidence_scores) * 100)
            }
        else:
            metrics = {
                'total_samples': int(len(df_test_original)),
                'anomalies_detected': int(total_anomalies),  # Total anomalies found
                'anomalies_displayed': int(len(anomalies_data)),  # Shown in UI
                'processing_time': float(time.time() - start_time),
                'avg_confidence': float(np.mean(confidence_scores) * 100)
            }
        
        display_msg = f"Processed {len(df_test_original)} samples in {time.time() - start_time:.2f}s - {total_anomalies} anomalies detected (showing all)"
        print(display_msg)
        
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
    load_model()
