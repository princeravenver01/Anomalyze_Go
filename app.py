from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

app = Flask(__name__)

#Configure folders
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
for folder in [UPLOAD_FOLDER, MODELS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

#Global variables for the model and data features
kmeans_model = None
scaler = None
data_columns = None

#Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')

def load_or_train_model():
    """
    Loads a pre-trained K-means model from disk if available,
    otherwise trains a new one and saves it.
    """
    global kmeans_model, scaler, data_columns

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLUMNS_PATH):
        print("Loading saved model from disk...")
        kmeans_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        data_columns = joblib.load(COLUMNS_PATH)
        print("Model loaded successfully.")
    else:
        print("No saved model found. Training a new model...")
        train_data_path = 'data/KDDTrain+.txt'
        if not os.path.exists(train_data_path):
            print(f"Training data not found at {train_data_path}. Please add the file to proceed.")
            return

        #Load and preprocess the training data
        df_train = load_and_preprocess_data(train_data_path)
        
        #Separate normal traffic for training
        df_normal = df_train[df_train['label'] == 'normal']
        
        #Drop the label column as it's not a feature
        df_normal = df_normal.drop('label', axis=1)
        
        data_columns = df_normal.columns
        
        #Scale the features
        scaler = StandardScaler()
        df_normal_scaled = scaler.fit_transform(df_normal)
        
        #Train the K-means model
        kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans_model.fit(df_normal_scaled)
        
        #Save the trained model, scaler, and columns to disk
        joblib.dump(kmeans_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(data_columns, COLUMNS_PATH)
        print("Model trained and saved to disk.")

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

    return "File upload failed"

if __name__ == '__main__':
    load_or_train_model() #Load or train the model on startup
    app.run(debug=True, use_reloader=False)