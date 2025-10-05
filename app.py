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

app = Flask(__name__)

# Configure models folder
MODELS_FOLDER = 'models'

# Global variables
kmeans_model = None
scaler = None
data_columns = None
optimal_threshold = None

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')

def find_optimal_threshold():
    """Find the threshold that maximizes accuracy on the training data."""
    print("Finding optimal threshold...")
    
    train_data_path = 'data/KDDTrain+.txt'
    if not os.path.exists(train_data_path):
        print("Training data not found, using fallback threshold")
        return 15.0
    
    try:
        df_train = load_and_preprocess_data(train_data_path)
        df_train_features = df_train.drop('label', axis=1)
        df_train_features = df_train_features.reindex(columns=data_columns, fill_value=0)
        true_labels = (df_train['label'] != 'normal').astype(int)
        df_train_scaled = scaler.transform(df_train_features)
        distances = kmeans_model.transform(df_train_scaled).min(axis=1)
        
        best_threshold = None
        best_accuracy = 0
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
        print(f"Error calculating threshold: {e}")
        return 15.0

def load_model():
    """Loads the pre-trained model and finds the optimal threshold."""
    global kmeans_model, scaler, data_columns, optimal_threshold
    
    try:
        if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]):
            raise FileNotFoundError("Model files not found.")

        print("Loading saved model from disk...")
        kmeans_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        data_columns = joblib.load(COLUMNS_PATH)
        
        if os.path.exists(THRESHOLD_PATH):
            optimal_threshold = joblib.load(THRESHOLD_PATH)
            print(f"Loaded optimal threshold: {optimal_threshold}")
        else:
            optimal_threshold = find_optimal_threshold()
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    load_model()