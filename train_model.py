# LOCAL TRAINING SCRIPT FOR IMPROVEMENT

from flask import Flask
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib
from sklearn.metrics import accuracy_score

# Configure folders
MODELS_FOLDER = 'models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')

def find_optimal_clusters_and_threshold():
    """Find the best number of clusters and threshold combination."""
    print("Finding optimal clusters and threshold...")
    
    train_data_path = 'data/KDDTrain+.txt'
    df_train = load_and_preprocess_data(train_data_path)
    df_normal = df_train[df_train['label'] == 'normal']
    df_normal = df_normal.drop('label', axis=1)
    
    data_columns = df_normal.columns
    scaler = StandardScaler()
    df_normal_scaled = scaler.fit_transform(df_normal)
    
    # Test different numbers of clusters - IMPROVEMENT 3
    cluster_options = [5, 8, 10, 12, 15]
    best_accuracy = 0
    best_model = None
    best_threshold = None
    best_clusters = None
    
    # Load full training data for threshold optimization
    df_train_features = df_train.drop('label', axis=1)
    df_train_features = df_train_features.reindex(columns=data_columns, fill_value=0)
    true_labels = (df_train['label'] != 'normal').astype(int)
    df_train_scaled = scaler.transform(df_train_features)
    
    for n_clusters in cluster_options:
        print(f"Testing {n_clusters} clusters...")
        
        # Train model with this number of clusters
        kmeans_model = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10, 
            max_iter=500  # More iterations for better convergence
        )
        kmeans_model.fit(df_normal_scaled)
        
        # Calculate distances for threshold optimization
        distances = kmeans_model.transform(df_train_scaled).min(axis=1)
        
        # Find best threshold for this cluster configuration
        percentiles = np.arange(90, 99.9, 0.5)
        
        for percentile in percentiles:
            threshold = np.percentile(distances, percentile)
            predicted_labels = (distances > threshold).astype(int)
            accuracy = accuracy_score(true_labels, predicted_labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = kmeans_model
                best_threshold = threshold
                best_clusters = n_clusters
    
    print(f"Best configuration: {best_clusters} clusters, threshold: {best_threshold:.4f}, accuracy: {best_accuracy:.4f}")
    
    return best_model, scaler, data_columns, best_threshold

def create_optimized_model():
    """Create and save the optimized model."""
    print("Creating optimized model with multiple cluster options...")
    
    train_data_path = 'data/KDDTrain+.txt'
    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data not found at {train_data_path}")
        return False

    # Find optimal configuration
    kmeans_model, scaler, data_columns, optimal_threshold = find_optimal_clusters_and_threshold()
    
    # Save all components
    joblib.dump(kmeans_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(data_columns, COLUMNS_PATH)
    joblib.dump(optimal_threshold, THRESHOLD_PATH)
    
    print(f"Optimized model saved:")
    print(f"- Model: {MODEL_PATH}")
    print(f"- Scaler: {SCALER_PATH}")
    print(f"- Columns: {COLUMNS_PATH}")
    print(f"- Threshold: {THRESHOLD_PATH}")
    
    return True

if __name__ == '__main__':
    print("=== OPTIMIZED MODEL TRAINING ===")
    print("This will test different numbers of clusters and find the best configuration.")
    print()
    
    if create_optimized_model():
        print("\n✅ SUCCESS: Optimized model created!")
        print("You can now run your main app.py to use the improved model.")
    else:
        print("\n❌ FAILED: Could not create optimized model.")