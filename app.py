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
ensemble_models = None
scaler = None
data_columns = None
optimal_threshold = None

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'ensemble_models.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')

def create_ensemble_models():
    """Create multiple K-means models with different configurations."""
    print("Creating ensemble models...")
    
    train_data_path = 'data/KDDTrain+.txt'
    df_train = load_and_preprocess_data(train_data_path)
    df_normal = df_train[df_train['label'] == 'normal']
    df_normal = df_normal.drop('label', axis=1)
    
    data_columns = df_normal.columns
    scaler = StandardScaler()
    df_normal_scaled = scaler.fit_transform(df_normal)
    
    # Create ensemble of models with different parameters
    models = []
    
    # Different numbers of clusters
    cluster_configs = [
        {'n_clusters': 5, 'random_state': 42},
        {'n_clusters': 8, 'random_state': 42},
        {'n_clusters': 10, 'random_state': 42},
        {'n_clusters': 8, 'random_state': 123},  # Same clusters, different seed
        {'n_clusters': 8, 'random_state': 456}   # Another different seed
    ]
    
    for i, config in enumerate(cluster_configs):
        print(f"Training ensemble model {i+1}/{len(cluster_configs)}...")
        model = KMeans(n_init=10, max_iter=500, **config)
        model.fit(df_normal_scaled)
        models.append(model)
    
    return models, scaler, data_columns

def ensemble_predict(models, data, threshold):
    """Use ensemble of models for more robust predictions."""
    predictions = []
    
    for model in models:
        distances = model.transform(data).min(axis=1)
        pred = (distances > threshold).astype(int)
        predictions.append(pred)
    
    # Use majority voting
    ensemble_pred = np.array(predictions).mean(axis=0)
    return (ensemble_pred > 0.5).astype(int)  # Majority vote

def load_model():
    """Loads the ensemble models and finds the optimal threshold."""
    global ensemble_models, scaler, data_columns, optimal_threshold
    
    try:
        # Check if ensemble model file exists
        if os.path.exists(MODEL_PATH):
            print("Loading ensemble models from disk...")
            ensemble_models = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            data_columns = joblib.load(COLUMNS_PATH)
            optimal_threshold = joblib.load(THRESHOLD_PATH)
            print("Ensemble models loaded successfully.")
        else:
            print("Creating new ensemble models...")
            ensemble_models, scaler, data_columns = create_ensemble_models()
            
            # Find optimal threshold using the first model in ensemble
            train_data_path = 'data/KDDTrain+.txt'
            df_train = load_and_preprocess_data(train_data_path)
            df_train_features = df_train.drop('label', axis=1)
            true_labels = (df_train['label'] != 'normal').astype(int)
            df_train_scaled = scaler.transform(df_train_features)
            
            # Use ensemble prediction for threshold optimization
            best_threshold = None
            best_accuracy = 0
            percentiles = np.arange(90, 99.9, 0.5)
            
            for percentile in percentiles:
                # Use first model's distances for threshold calculation
                distances = ensemble_models[0].transform(df_train_scaled).min(axis=1)
                threshold = np.percentile(distances, percentile)
                predicted_labels = ensemble_predict(ensemble_models, df_train_scaled, threshold)
                accuracy = accuracy_score(true_labels, predicted_labels)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
            
            optimal_threshold = best_threshold
            
            # Save ensemble models
            joblib.dump(ensemble_models, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            joblib.dump(data_columns, COLUMNS_PATH)
            joblib.dump(optimal_threshold, THRESHOLD_PATH)
            print("Ensemble models created and saved.")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    if ensemble_models is None:
        if not load_model():
            return "Error: Could not load model files."
    return render_template('index.html', results=None, accuracy=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    global ensemble_models, scaler, data_columns, optimal_threshold
    
    if ensemble_models is None or optimal_threshold is None:
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

            # Use ensemble prediction instead of single model
            anomalies_mask = ensemble_predict(ensemble_models, df_test_scaled, optimal_threshold)
            anomalies = df_test_original[anomalies_mask.astype(bool)]
            
            if has_labels:
                accuracy = accuracy_score(true_labels_binary, anomalies_mask)
                accuracy_percentage = accuracy * 100

            return render_template('index.html', results=anomalies, accuracy=accuracy_percentage)
            
        except Exception as e:
            return f"Error processing file: {str(e)}"

    return "File upload failed"

if __name__ == '__main__':
    print("Starting application with ensemble models...")
    if load_model():
        print("Ensemble models loaded successfully. Starting server...")
        app.run(debug=True, use_reloader=False)
    else:
        print("Failed to load ensemble models.")
else:
    load_model()