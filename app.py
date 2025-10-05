from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib
import io
from sklearn.metrics import accuracy_score
import requests

app = Flask(__name__)

# Configure models folder
MODELS_FOLDER = 'models'

# Global variables
kmeans_model = None
scaler = None
data_columns = None
optimal_threshold = None

# Render data server URL
RENDER_DATA_URL = "https://dataset-host.onrender.com"  # To fetch the data through cloud service sa render

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')

def download_training_data():
    """Download training data from Render server"""
    try:
        print("Downloading training data from Render...")
        response = requests.get(f"{RENDER_DATA_URL}/data/KDDTrain+.txt", timeout=30)
        response.raise_for_status()
        
        # Create a file-like object from the downloaded content
        return io.StringIO(response.text)
    except Exception as e:
        print(f"Error downloading training data: {e}")
        return None

def find_optimal_threshold():
    """Find the threshold that maximizes accuracy on the training data."""
    print("Finding optimal threshold using remote training data...")
    
    # Download training data from Render
    train_data_stream = download_training_data()
    if train_data_stream is None:
        print("Could not download training data. Using fallback threshold.")
        return 15.0  # Fallback threshold
    
    try:
        df_train = load_and_preprocess_data(train_data_stream)
        
        # Separate normal and anomaly data
        df_train_features = df_train.drop('label', axis=1)
        df_train_features = df_train_features.reindex(columns=data_columns, fill_value=0)
        true_labels = (df_train['label'] != 'normal').astype(int)
        
        # Scale the data
        df_train_scaled = scaler.transform(df_train_features)
        
        # Calculate distances for all training data
        distances = kmeans_model.transform(df_train_scaled).min(axis=1)
        
        # Try different thresholds and find the one with best accuracy
        best_threshold = None
        best_accuracy = 0
        
        # Test thresholds from 90th to 99.9th percentile
        percentiles = np.arange(90, 99.9, 0.5)
        
        for percentile in percentiles:
            threshold = np.percentile(distances, percentile)
            predicted_labels = (distances > threshold).astype(int)
            accuracy = accuracy_score(true_labels, predicted_labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold} (accuracy: {best_accuracy:.4f})")
        return best_threshold
        
    except Exception as e:
        print(f"Error calculating optimal threshold: {e}")
        return 15.0  # Fallback threshold

def load_model():
    """Loads the pre-trained model and finds the optimal threshold."""
    global kmeans_model, scaler, data_columns, optimal_threshold
    
    try:
        # Check if basic model files exist
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]):
            raise FileNotFoundError("Basic model files not found.")

        print("Loading saved model from disk...")
        kmeans_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        data_columns = joblib.load(COLUMNS_PATH)
        
        # Check if optimal threshold exists, if not, calculate it using remote data
        if os.path.exists(THRESHOLD_PATH):
            optimal_threshold = joblib.load(THRESHOLD_PATH)
            print(f"Loaded optimal threshold: {optimal_threshold}")
        else:
            optimal_threshold = find_optimal_threshold()
            # Note: We can't save to disk on Vercel, so we calculate it each time
            print(f"Calculated optimal threshold: {optimal_threshold}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    if kmeans_model is None:
        if not load_model():
            return "Error: Could not load model files."
    return render_template('index.html', results=None, accuracy=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    global kmeans_model, scaler, data_columns, optimal_threshold
    
    if kmeans_model is None or optimal_threshold is None:
        if not load_model():
            return "Error: Could not load model files."
    
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

            has_labels = 'label' in df_test.columns
            accuracy_percentage = None
            
            if has_labels:
                true_labels = df_test['label'].copy()
                true_labels_binary = (true_labels != 'normal').astype(int)
                df_test = df_test.drop('label', axis=1)

            df_test = df_test.reindex(columns=data_columns, fill_value=0)
            df_test_scaled = scaler.transform(df_test)
            distances = kmeans_model.transform(df_test_scaled).min(axis=1)

            # Use the optimal threshold
            anomalies_mask = distances > optimal_threshold
            anomalies = df_test_original[anomalies_mask]
            
            if has_labels:
                predicted_labels = anomalies_mask.astype(int)
                accuracy = accuracy_score(true_labels_binary, predicted_labels)
                accuracy_percentage = accuracy * 100

            return render_template('index.html', results=anomalies, accuracy=accuracy_percentage)
            
        except Exception as e:
            return f"Error processing file: {str(e)}"

    return "File upload failed"

# Load the model when the application starts
load_model()