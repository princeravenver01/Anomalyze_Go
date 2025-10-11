# LOCAL TRAINING SCRIPT FOR IMPROVEMENT

from flask import Flask
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np
import os
from utils.preprocessing import load_and_preprocess_data
import joblib
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score

# Configure folders
MODELS_FOLDER = 'models'
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

# Paths for saved model files
MODEL_PATH = os.path.join(MODELS_FOLDER, 'kmeans_model.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')

def create_advanced_kmeans_ensemble():
    """Create an advanced K-means ensemble with different configurations"""
    print("Creating advanced K-means ensemble...")
    
    train_data_path = 'data/KDDTrain+.txt'
    df_train = load_and_preprocess_data(train_data_path)
    df_normal = df_train[df_train['label'] == 'normal']
    df_normal = df_normal.drop('label', axis=1)
    
    # Apply enhanced preprocessing
    from utils.preprocessing import enhanced_preprocessing_for_kmeans
    df_normal = enhanced_preprocessing_for_kmeans(df_normal)
    
    data_columns = df_normal.columns
    
    # Use RobustScaler for better K-means performance
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    df_normal_scaled = scaler.fit_transform(df_normal)
    
    # Enhanced K-means configurations
    kmeans_configs = [
        # Different cluster numbers with optimized parameters
        {'n_clusters': 8, 'init': 'k-means++', 'n_init': 20, 'max_iter': 500, 'random_state': 42},
        {'n_clusters': 10, 'init': 'k-means++', 'n_init': 20, 'max_iter': 500, 'random_state': 42},
        {'n_clusters': 12, 'init': 'k-means++', 'n_init': 20, 'max_iter': 500, 'random_state': 42},
        
        # Same clusters but different random states for diversity
        {'n_clusters': 10, 'init': 'k-means++', 'n_init': 20, 'max_iter': 500, 'random_state': 123},
        {'n_clusters': 10, 'init': 'k-means++', 'n_init': 20, 'max_iter': 500, 'random_state': 456},
        
        # MiniBatch K-means for different perspective
        {'model_type': 'minibatch', 'n_clusters': 10, 'random_state': 42, 'batch_size': 1000},
    ]
    
    models = []
    model_scores = []
    
    for i, config in enumerate(kmeans_configs):
        print(f"Training K-means model {i+1}/{len(kmeans_configs)}...")
        
        if config.get('model_type') == 'minibatch':
            config_copy = config.copy()
            del config_copy['model_type']
            model = MiniBatchKMeans(**config_copy)
        else:
            model = KMeans(**config)
        
        model.fit(df_normal_scaled)
        
        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(df_normal_scaled, model.labels_)
        calinski_score = calinski_harabasz_score(df_normal_scaled, model.labels_)
        
        models.append(model)
        model_scores.append({
            'silhouette': silhouette_avg,
            'calinski': calinski_score,
            'clusters': config['n_clusters']
        })
        
        print(f"  Silhouette Score: {silhouette_avg:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski_score:.3f}")
    
    # Print model quality summary
    print("\n=== Model Quality Summary ===")
    for i, score in enumerate(model_scores):
        print(f"Model {i+1}: Clusters={score['clusters']}, "
              f"Silhouette={score['silhouette']:.3f}, "
              f"Calinski-Harabasz={score['calinski']:.3f}")
    
    return models, scaler, data_columns, model_scores

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

def create_optimized_kmeans_model():
    """Create and save the optimized K-means ensemble model."""
    print("Creating optimized K-means ensemble model...")
    
    train_data_path = 'data/KDDTrain+.txt'
    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data not found at {train_data_path}")
        return False

    # Create advanced ensemble
    ensemble_models, scaler, data_columns, model_scores = create_advanced_kmeans_ensemble()
    
    # Find optimal threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold_advanced(
        ensemble_models, scaler, data_columns
    )
    
    # Save all components
    joblib.dump(ensemble_models, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(data_columns, COLUMNS_PATH)
    joblib.dump(optimal_threshold, THRESHOLD_PATH)
    
    # Save model scores for weighted prediction
    MODEL_SCORES_PATH = os.path.join(MODELS_FOLDER, 'model_scores.joblib')
    joblib.dump(model_scores, MODEL_SCORES_PATH)
    
    # Save threshold metrics for analysis
    THRESHOLD_METRICS_PATH = os.path.join(MODELS_FOLDER, 'threshold_metrics.joblib')
    joblib.dump(threshold_metrics, THRESHOLD_METRICS_PATH)
    
    print(f"Enhanced K-means ensemble saved:")
    print(f"- Models: {MODEL_PATH}")
    print(f"- Scaler: {SCALER_PATH}")
    print(f"- Columns: {COLUMNS_PATH}")
    print(f"- Threshold: {THRESHOLD_PATH}")
    print(f"- Model Scores: {MODEL_SCORES_PATH}")
    print(f"- Threshold Metrics: {THRESHOLD_METRICS_PATH}")
    
    return True

if __name__ == '__main__':
    print("=== ENHANCED K-MEANS ENSEMBLE TRAINING ===")
    print("This will create an advanced K-means ensemble with optimized features.")
    print()
    
    if create_optimized_kmeans_model():
        print("\n✅ SUCCESS: Enhanced K-means ensemble created!")
        print("You can now run your main app.py to use the improved model.")
    else:
        print("\n❌ FAILED: Could not create enhanced K-means ensemble.")