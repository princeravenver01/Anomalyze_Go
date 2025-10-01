from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib
import io

app = Flask(__name__)

# Configure models folder
MODELS_FOLDER = 'models'

# Global variables
kmeans_model = None
scaler = None
data_columns = None
threshold = None # To hold the pre-calculated threshold

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'threshold.joblib') # Path to the threshold file

def load_model():
    """Loads the pre-trained model and the pre-calculated threshold."""
    global kmeans_model, scaler, data_columns, threshold
    
    # All files must exist in the deployment
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH, THRESHOLD_PATH]):
        raise FileNotFoundError("Model files not found. Please ensure the model is trained and all .joblib files are in the 'models' directory.")

    print("Loading saved model and threshold from disk...")
    kmeans_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    data_columns = joblib.load(COLUMNS_PATH)
    threshold = joblib.load(THRESHOLD_PATH) # Load the threshold
    print(f"Model and threshold ({threshold}) loaded successfully.")


@app.route('/')
def index():
    return render_template('index.html', results=None)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        stream.seek(0)
        df_test_original = pd.read_csv(stream, header=None)
        
        stream.seek(0)
        df_test = load_and_preprocess_data(stream)

        df_test = df_test.reindex(columns=data_columns, fill_value=0)
        df_test_scaled = scaler.transform(df_test)
        distances = kmeans_model.transform(df_test_scaled).min(axis=1)

        # Use the pre-loaded threshold
        anomalies_mask = distances > threshold
        anomalies = df_test_original[anomalies_mask]

        return render_template('index.html', results=anomalies)

    return "File upload failed"

# This file is imported by api/index.py, which Vercel runs.
# This function call will execute when the serverless function starts.
load_model()