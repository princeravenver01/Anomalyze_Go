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
model_scores = None

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

def enhanced_ensemble_predict(models, model_scores, data, threshold):
    """Enhanced ensemble prediction with weighted voting based on model quality"""
    predictions = []
    distances_list = []
    weights = []
    
    # Calculate weights based on model quality (silhouette score)
    total_silhouette = sum(score['silhouette'] for score in model_scores)
    
    for i, model in enumerate(models):
        distances = model.transform(data).min(axis=1)
        distances_list.append(distances)
        
        # Weight based on silhouette score (higher is better)
        weight = model_scores[i]['silhouette'] / total_silhouette
        weights.append(weight)
        
        pred = (distances > threshold).astype(int)
        predictions.append(pred * weight)  # Apply weight
    
    # Weighted ensemble prediction
    ensemble_pred = np.sum(predictions, axis=0)
    
    # Also calculate confidence scores
    weighted_avg_distances = np.average(distances_list, axis=0, weights=weights)
    confidence_scores = 1 / (1 + weighted_avg_distances)  # Higher distance = lower confidence
    
    # Threshold for final prediction (can be tuned)
    final_predictions = (ensemble_pred > 0.5).astype(int)
    
    return final_predictions, confidence_scores, weighted_avg_distances

def calculate_anomaly_severity(distances, threshold):
    """Calculate severity levels for detected anomalies"""
    severity_levels = []
    
    for distance in distances:
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
    start_time = time.time()
    global ensemble_models, scaler, data_columns, optimal_threshold, model_scores
    
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
            metrics = {}
            
            if has_labels:
                true_labels = df_test['label'].copy()
                true_labels_binary = (true_labels != 'normal').astype(int)
                df_test = df_test.drop('label', axis=1)

            # Apply enhanced preprocessing
            from utils.preprocessing import enhanced_preprocessing_for_kmeans
            df_test = enhanced_preprocessing_for_kmeans(df_test)
            df_test = df_test.reindex(columns=data_columns, fill_value=0)
            df_test_scaled = scaler.transform(df_test)

            # Use enhanced ensemble prediction
            anomalies_mask, confidence_scores, distances = enhanced_ensemble_predict(
                ensemble_models, model_scores, df_test_scaled, optimal_threshold
            )
            
            # Calculate severity levels
            severity_levels = calculate_anomaly_severity(distances, optimal_threshold)
            
            # Add confidence and severity to results
            anomalies_indices = np.where(anomalies_mask)[0]
            anomalies = df_test_original.iloc[anomalies_indices].copy()
            
            if len(anomalies) > 0:
                anomalies['confidence'] = confidence_scores[anomalies_indices]
                anomalies['severity'] = [severity_levels[i] for i in anomalies_indices]
                anomalies['distance'] = distances[anomalies_indices]
            
            if has_labels:
                from sklearn.metrics import precision_score, recall_score, f1_score
                accuracy = accuracy_score(true_labels_binary, anomalies_mask)
                precision = precision_score(true_labels_binary, anomalies_mask, zero_division=0)
                recall = recall_score(true_labels_binary, anomalies_mask, zero_division=0)
                f1 = f1_score(true_labels_binary, anomalies_mask, zero_division=0)
                
                metrics = {
                    'accuracy': accuracy * 100,
                    'precision': precision * 100,
                    'recall': recall * 100,
                    'f1_score': f1 * 100,
                    'total_samples': len(df_test_original),
                    'anomalies_detected': len(anomalies),
                    'processing_time': time.time() - start_time,
                    'avg_confidence': np.mean(confidence_scores) * 100
                }

            return render_template('index.html', 
                                 results=anomalies, 
                                 metrics=metrics,
                                 accuracy=metrics.get('accuracy'))
            
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