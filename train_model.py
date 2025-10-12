"""
Anomalyze Model Training Script
Optimized for high performance and accuracy

"""

from __future__ import annotations
import sys
from pathlib import Path  # Modern path handling
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, accuracy_score
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib

# Configure folders
MODELS_FOLDER = 'models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'ensemble_models.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')

def create_simple_ensemble():
    """Create a simple but effective ensemble model"""
    print("Creating simple ensemble models...")
    
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

def find_optimal_threshold_simple(models, scaler, data_columns):
    """Find optimal threshold using simple but effective method"""
    print("Finding optimal threshold...")
    
    train_data_path = 'data/KDDTrain+.txt'
    df_train = load_and_preprocess_data(train_data_path)
    
    true_labels = (df_train['label'] != 'normal').astype(int)
    df_train_features = df_train.drop('label', axis=1)
    df_train_features = df_train_features.reindex(columns=data_columns, fill_value=0)
    df_train_scaled = scaler.transform(df_train_features)
    
    # Calculate ensemble distances
    all_distances = []
    for model in models:
        distances = model.transform(df_train_scaled).min(axis=1)
        all_distances.append(distances)
    
    # Use average distance across ensemble
    avg_distances = np.mean(all_distances, axis=0)
    
    best_threshold = None
    best_f1_score = 0
    
    # Test different threshold percentiles
    percentiles = np.arange(85, 99.5, 1.0)  # Less granular for speed
    
    for percentile in percentiles:
        threshold = np.percentile(avg_distances, percentile)
        predicted_labels = (avg_distances > threshold).astype(int)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.6f} with F1 score: {best_f1_score:.4f}")
    return best_threshold

def find_optimal_threshold_advanced(models, scaler, data_columns):
    """Advanced threshold optimization using multiple metrics"""
    print("Finding optimal threshold with advanced metrics...")
    
    train_data_path = 'data/KDDTrain+.txt'
    df_train = load_and_preprocess_data(train_data_path)
    
    # Apply same enhanced preprocessing
    from utils.preprocessing import enhanced_preprocessing_for_kmeans
    df_train_processed = enhanced_preprocessing_for_kmeans(df_train.copy())
    
    true_labels = (df_train['label'] != 'normal').astype(int)
    df_train_features = df_train_processed.drop('label', axis=1)
    df_train_features = df_train_features.reindex(columns=data_columns, fill_value=0)
    df_train_scaled = scaler.transform(df_train_features)
    
    best_threshold = None
    best_f1_score = 0
    best_metrics = None
    
    # Test different threshold percentiles
    percentiles = np.arange(85, 99.5, 0.5)
    
    for percentile in percentiles:
        # Calculate ensemble distances
        all_distances = []
        for model in models:
            distances = model.transform(df_train_scaled).min(axis=1)
            all_distances.append(distances)
        
        # Use average distance across ensemble
        avg_distances = np.mean(all_distances, axis=0)
        threshold = np.percentile(avg_distances, percentile)
        
        # Make predictions
        predicted_labels = (avg_distances > threshold).astype(int)
        
        # Calculate comprehensive metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Use F1 score as primary metric (balances precision and recall)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
            best_metrics = {
                'percentile': percentile,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
    
    print(f"\nBest threshold configuration:")
    print(f"  Percentile: {best_metrics['percentile']}")
    print(f"  Threshold: {best_threshold:.6f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1 Score: {best_metrics['f1_score']:.4f}")
    
    return best_threshold, best_metrics

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

def create_optimized_ensemble_model():
    """Create and save the optimized ensemble model."""
    print("Creating optimized ensemble model...")
    
    train_data_path = 'data/KDDTrain+.txt'
    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data not found at {train_data_path}")
        return False

    # Create simple ensemble
    ensemble_models, scaler, data_columns = create_simple_ensemble()
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold_simple(ensemble_models, scaler, data_columns)
    
    # Save all components
    joblib.dump(ensemble_models, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(data_columns, COLUMNS_PATH)
    joblib.dump(optimal_threshold, THRESHOLD_PATH)
    
    print(f"Optimized ensemble model saved:")
    print(f"- Models: {MODEL_PATH}")
    print(f"- Scaler: {SCALER_PATH}")
    print(f"- Columns: {COLUMNS_PATH}")
    print(f"- Threshold: {THRESHOLD_PATH}")
    
    return True

if __name__ == '__main__':
    print("=== OPTIMIZED K-MEANS ENSEMBLE TRAINING ===")
    print("This will create a fast and accurate K-means ensemble.")
    print()
    
    if create_optimized_ensemble_model():
        print("\n✅ SUCCESS: Optimized ensemble model created!")
        print("You can now run your main app.py to use the improved model.")
    else:
        print("\n❌ FAILED: Could not create optimized ensemble model.")