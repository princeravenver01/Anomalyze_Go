from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Configure models folder
MODELS_FOLDER = 'models'

# Global variables - Initialize them properly
kmeans_model = None
scaler = None
data_columns = None
threshold = None

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'threshold.joblib')

def load_model():
    """Loads the pre-trained model and the pre-calculated threshold."""
    global kmeans_model, scaler, data_columns, threshold
    
    try:
        # Check if all files exist
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH, THRESHOLD_PATH]):
            raise FileNotFoundError("Model files not found. Please ensure all .joblib files are in the 'models' directory.")

        print("Loading saved model and threshold from disk...")
        kmeans_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        data_columns = joblib.load(COLUMNS_PATH)
        threshold = joblib.load(THRESHOLD_PATH)
        print(f"Model and threshold ({threshold}) loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    # Ensure model is loaded
    if kmeans_model is None:
        if not load_model():
            return "Error: Could not load model files. Please check server logs."
    return render_template('index.html', results=None, metrics=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    global kmeans_model, scaler, data_columns, threshold
    
    # Ensure model is loaded
    if kmeans_model is None or threshold is None:
        if not load_model():
            return "Error: Could not load model files. Please ensure all model files are present."
    
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        try:
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            stream.seek(0)
            df_test_original = pd.read_csv(stream, header=None)
            
            stream.seek(0)
            df_test = load_and_preprocess_data(stream)

            # Check if the file has labels (for accuracy calculation)
            has_labels = 'label' in df_test.columns
            true_labels = None
            
            if has_labels:
                # Extract true labels before removing them
                true_labels = df_test['label'].copy()
                # Convert to binary: 'normal' = 0, anything else = 1 (anomaly)
                true_labels_binary = (true_labels != 'normal').astype(int)
                df_test = df_test.drop('label', axis=1)

            df_test = df_test.reindex(columns=data_columns, fill_value=0)
            df_test_scaled = scaler.transform(df_test)
            distances = kmeans_model.transform(df_test_scaled).min(axis=1)

            # Use the pre-loaded threshold
            anomalies_mask = distances > threshold
            anomalies = df_test_original[anomalies_mask]
            
            # Calculate accuracy metrics if labels are available
            accuracy_metrics = None
            if has_labels:
                # Our predictions: 1 = anomaly, 0 = normal
                predicted_labels = anomalies_mask.astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(true_labels_binary, predicted_labels)
                precision = precision_score(true_labels_binary, predicted_labels, zero_division=0)
                recall = recall_score(true_labels_binary, predicted_labels, zero_division=0)
                f1 = f1_score(true_labels_binary, predicted_labels, zero_division=0)
                
                # Count statistics
                total_samples = len(true_labels_binary)
                actual_anomalies = sum(true_labels_binary)
                predicted_anomalies = sum(predicted_labels)
                
                accuracy_metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'total_samples': total_samples,
                    'actual_anomalies': actual_anomalies,
                    'predicted_anomalies': predicted_anomalies,
                    'true_positives': sum((true_labels_binary == 1) & (predicted_labels == 1)),
                    'false_positives': sum((true_labels_binary == 0) & (predicted_labels == 1)),
                    'true_negatives': sum((true_labels_binary == 0) & (predicted_labels == 0)),
                    'false_negatives': sum((true_labels_binary == 1) & (predicted_labels == 0))
                }

            return render_template('index.html', results=anomalies, metrics=accuracy_metrics)
            
        except Exception as e:
            return f"Error processing file: {str(e)}"

    return "File upload failed"

# Try to load the model when the module is imported
if __name__ == '__main__':
    print("Starting application...")
    if load_model():
        print("Model loaded successfully. Starting server...")
        app.run(debug=True, use_reloader=False)
    else:
        print("Failed to load model. Please check that all .joblib files exist in the 'models' folder.")
else:
    # For Vercel deployment
    load_model()