# Model Training Script

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib
from sklearn.metrics import accuracy_score
import argparse

# Configure folders
MODELS_FOLDER = 'models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')

def find_optimal_configuration(optimize_clusters=True, optimize_threshold=True):
    """Find the best configuration for the model."""
    print("Loading and preprocessing training data...")
    
    train_data_path = 'data/KDDTrain+.txt'
    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data not found at {train_data_path}")
        return None
    
    df_train = load_and_preprocess_data(train_data_path)
    df_normal = df_train[df_train['label'] == 'normal']
    df_normal = df_normal.drop('label', axis=1)
    
    data_columns = df_normal.columns
    scaler = StandardScaler()
    df_normal_scaled = scaler.fit_transform(df_normal)
    
    # Determine cluster options
    if optimize_clusters:
        cluster_options = [5, 8, 10, 12, 15]
        print("Testing multiple cluster configurations...")
    else:
        cluster_options = [8]  # Fixed for faster deployment
        print("Using fixed 8 clusters for quick deployment...")
    
    best_accuracy = 0
    best_model = None
    best_threshold = None
    best_clusters = None
    
    # Prepare data for threshold optimization if needed
    if optimize_threshold:
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
            max_iter=500
        )
        kmeans_model.fit(df_normal_scaled)
        
        if optimize_threshold:
            # Calculate distances for threshold optimization
            distances = kmeans_model.transform(df_train_scaled).min(axis=1)
            
            # Find best threshold for this cluster configuration
            percentiles = np.arange(90, 99.9, 0.5)
            best_threshold_for_clusters = None
            best_accuracy_for_clusters = 0
            
            for percentile in percentiles:
                threshold = np.percentile(distances, percentile)
                predicted_labels = (distances > threshold).astype(int)
                accuracy = accuracy_score(true_labels, predicted_labels)
                
                if accuracy > best_accuracy_for_clusters:
                    best_accuracy_for_clusters = accuracy
                    best_threshold_for_clusters = threshold
            
            if best_accuracy_for_clusters > best_accuracy:
                best_accuracy = best_accuracy_for_clusters
                best_model = kmeans_model
                best_threshold = best_threshold_for_clusters
                best_clusters = n_clusters
        else:
            # Just use the first (and only) model
            best_model = kmeans_model
            best_clusters = n_clusters
            best_threshold = None  # Will be calculated dynamically
    
    if optimize_threshold:
        print(f"Best configuration: {best_clusters} clusters, threshold: {best_threshold:.4f}, accuracy: {best_accuracy:.4f}")
    else:
        print(f"Created model with {best_clusters} clusters (threshold will be calculated dynamically)")
    
    return best_model, scaler, data_columns, best_threshold

def create_model(mode="local"):
    """Create model based on deployment mode."""
    print(f"=== CREATING MODEL FOR {mode.upper()} DEPLOYMENT ===")
    
    if mode == "local":
        # Full optimization for local use
        model, scaler, data_columns, threshold = find_optimal_configuration(
            optimize_clusters=True, 
            optimize_threshold=True
        )
        
        # Save all components including threshold
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(data_columns, COLUMNS_PATH)
        if threshold:
            joblib.dump(threshold, THRESHOLD_PATH)
        
        print(f"\n✅ LOCAL MODEL CREATED:")
        print(f"- Model: {MODEL_PATH}")
        print(f"- Scaler: {SCALER_PATH}")
        print(f"- Columns: {COLUMNS_PATH}")
        if threshold:
            print(f"- Threshold: {THRESHOLD_PATH}")
        
    elif mode == "vercel":
        # Quick model for Vercel (threshold calculated dynamically)
        model, scaler, data_columns, _ = find_optimal_configuration(
            optimize_clusters=False,  # Use fixed 8 clusters for speed
            optimize_threshold=False  # Threshold calculated on Vercel
        )
        
        # Save only basic components (no threshold)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(data_columns, COLUMNS_PATH)
        
        # Remove threshold file if it exists (force dynamic calculation)
        if os.path.exists(THRESHOLD_PATH):
            os.remove(THRESHOLD_PATH)
        
        print(f"\n✅ VERCEL MODEL CREATED:")
        print(f"- Model: {MODEL_PATH}")
        print(f"- Scaler: {SCALER_PATH}")
        print(f"- Columns: {COLUMNS_PATH}")
        print(f"- Threshold: Will be calculated dynamically using Render data")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Anomalyze model')
    parser.add_argument('--mode', choices=['local', 'vercel'], default='local',
                       help='Deployment mode: local (full optimization) or vercel (quick deployment)')
    
    args = parser.parse_args()
    
    # Delete existing model files
    print("Cleaning existing model files...")
    for file_path in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH, THRESHOLD_PATH]:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Create new model
    if create_model(args.mode):
        if args.mode == "local":
            print("\n🎉 Ready for local development!")
            print("Run: python app.py")
        else:
            print("\n🚀 Ready for Vercel deployment!")
            print("1. Update RENDER_DATA_URL in app.py")
            print("2. Commit and deploy to Vercel")
    else:
        print("\n❌ Model creation failed!")