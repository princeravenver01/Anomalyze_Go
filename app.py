from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from werkzeug.utils import secure_filename
from utils.preprocessing import load_and_preprocess_data
import joblib

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
for folder in [UPLOAD_FOLDER, MODELS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Global variables for the model and data features
kmeans_model = None
scaler = None
data_columns = None

# Paths for saved model files
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
            print(f"FATAL: Training data not found at {train_data_path}. Please add the file to proceed.")
            return

        # Load and preprocess the training data
        df_train = load_and_preprocess_data(train_data_path)
        
        # Separate normal traffic for training
        df_normal = df_train[df_train['label'] == 'normal']
        
        # Drop the label column as it's not a feature
        df_normal = df_normal.drop('label', axis=1)
        
        data_columns = df_normal.columns
        
        # Scale the features
        scaler = StandardScaler()
        df_normal_scaled = scaler.fit_transform(df_normal)
        
        # Train the K-means model
        kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans_model.fit(df_normal_scaled)
        
        # Save the trained model, scaler, and columns to disk
        joblib.dump(kmeans_model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(data_columns, COLUMNS_PATH)
        print("Model trained and saved to disk.")

@app.route('/')
def index():
    return render_template('index.html', results=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    global kmeans_model, scaler, data_columns

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Ensure the model is loaded or trained
        if kmeans_model is None:
            load_or_train_model()
            if kmeans_model is None:
                return "Error: Model could not be trained. Check if KDDTrain+.txt is in the 'data' folder.", 500

        # Load and preprocess the uploaded data
        df_test_original = pd.read_csv(filepath, header=None, names=[
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'label', 'difficulty'
        ])
        
        df_test = load_and_preprocess_data(filepath)
        
        # Align columns with the training data
        df_test = df_test.reindex(columns=data_columns, fill_value=0)

        # Scale the test data
        df_test_scaled = scaler.transform(df_test)

        # Calculate distances to the nearest cluster centroid
        distances = kmeans_model.transform(df_test_scaled).min(axis=1)
        
        # Set a threshold for anomaly detection
        # A simple approach: anomalies are points further than 99% of the training points
        # We calculate the threshold from the training data distances
        # Note: This part is simplified. For a real system, threshold tuning is critical.
        if 'df_normal_scaled' in locals():
             train_distances = kmeans_model.transform(df_normal_scaled).min(axis=1)
             threshold = np.percentile(train_distances, 99)
        else:
            # Fallback if we loaded the model and don't have train_distances in memory
            # A pre-computed or estimated threshold would be better here.
            # For this example, we'll set a fixed, potentially less accurate threshold.
            threshold = 15 
        
        # Identify anomalies
        anomalies_mask = distances > threshold
        anomalies = df_test_original[anomalies_mask]

        return render_template('index.html', results=anomalies)

if __name__ == '__main__':
    load_or_train_model() # Load or train the model on startup
    app.run(debug=True, use_reloader=False)