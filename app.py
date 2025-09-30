from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from werkzeug.utils import secure_filename
from utils.preprocessing import load_and_preprocess_data

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables for the model and data features
kmeans_model = None
scaler = None
data_columns = None

def train_model():
    """
    Trains the K-means model on the NSL-KDD training data.
    """
    global kmeans_model, scaler, data_columns
    
    train_data_path = 'data/KDDTrain+.txt'
    if not os.path.exists(train_data_path):
        # If the training data is not available, we can't proceed.
        # In a real application, you might want to handle this more gracefully.
        print("Training data not found. Please add KDDTrain+.txt to the 'data' folder.")
        return

    # Load and preprocess the training data
    df_train = load_and_preprocess_data(train_data_path)
    
    # Separate normal traffic for training
    df_normal = df_train[df_train['label'] == 'normal']
    
    # Drop the label column as it's not a feature
    df_normal = df_normal.drop('label', axis=1)
    
    # Align columns for consistency
    data_columns = df_normal.columns
    
    # Scale the features
    scaler = StandardScaler()
    df_normal_scaled = scaler.fit_transform(df_normal)
    
    # Train the K-means model
    # We choose 2 clusters as a starting point: one for "normal" and one for potential "anomalies"
    # In a real-world scenario, you might use the elbow method to find the optimal K.
    kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_model.fit(df_normal_scaled)
    print("Model trained successfully.")

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

        # If the model is not trained, train it first
        if kmeans_model is None:
            train_model()
            if kmeans_model is None:
                # Handle case where training failed (e.g., data not found)
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
        
        # Set a threshold for anomaly detection (e.g., 95th percentile of distances from training)
        # For simplicity, we'll use a pre-calculated value or a simple statistical measure.
        # A more robust approach would be to calculate this from the training data distances.
        threshold = np.percentile(kmeans_model.transform(scaler.transform(pd.DataFrame(columns=data_columns).reindex(columns=data_columns, fill_value=0))).min(axis=1), 95) if 'df_normal_scaled' in locals() else 10 # Fallback threshold
        
        # Identify anomalies
        anomalies_mask = distances > threshold
        anomalies = df_test_original[anomalies_mask]

        return render_template('index.html', results=anomalies)

if __name__ == '__main__':
    train_model() # Train the model on startup
    app.run(debug=True, use_reloader=False)