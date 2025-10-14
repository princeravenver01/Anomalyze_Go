"""
Anomalyze Fast Model Training Script
Optimized for SPEED with minimal accuracy trade-off

Creates 3 models instead of 5 for 40% faster predictions
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
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

def create_fast_ensemble():
    """Create a fast but effective ensemble with only 3 models."""
    print("Creating FAST ensemble models (3 models)...")
    
    train_data_path = 'data/KDDTrain+.txt'
    df_train = load_and_preprocess_data(train_data_path)
    df_normal = df_train[df_train['label'] == 'normal']
    df_normal = df_normal.drop('label', axis=1)
    
    data_columns = df_normal.columns
    scaler = StandardScaler()
    df_normal_scaled = scaler.fit_transform(df_normal)
    
    # Create ONLY 3 models for faster prediction
    models = []
    cluster_configs = [
        {'n_clusters': 8, 'random_state': 42},
        {'n_clusters': 8, 'random_state': 123},
        {'n_clusters': 10, 'random_state': 42}
    ]
    
    for i, config in enumerate(cluster_configs):
        print(f"Training fast model {i+1}/{len(cluster_configs)}...")
        model = KMeans(n_init=10, max_iter=300, **config)  # Reduced iterations
        model.fit(df_normal_scaled)
        models.append(model)
    
    return models, scaler, data_columns

def find_optimal_threshold_fast(models, scaler, data_columns):
    """Find optimal threshold quickly."""
    print("Finding optimal threshold (fast mode)...")
    
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
    
    avg_distances = np.mean(all_distances, axis=0)
    
    best_threshold = None
    best_f1_score = 0
    
    # Coarser search for speed
    percentiles = np.arange(88, 98, 2.0)  # Wider steps
    
    for percentile in percentiles:
        threshold = np.percentile(avg_distances, percentile)
        predicted_labels = (avg_distances > threshold).astype(int)
        
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.6f} (F1-Score: {best_f1_score:.4f})")
    return best_threshold

def main():
    print("=" * 60)
    print("Anomalyze FAST Model Training")
    print("3 Models for 40% faster predictions")
    print("=" * 60)
    
    # Create fast ensemble
    models, scaler, data_columns = create_fast_ensemble()
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold_fast(models, scaler, data_columns)
    
    # Save all components
    print("\nSaving models...")
    joblib.dump(models, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(data_columns, COLUMNS_PATH)
    joblib.dump(optimal_threshold, THRESHOLD_PATH)
    
    print("\n" + "=" * 60)
    print("FAST Training Complete!")
    print(f"Models saved to: {MODELS_FOLDER}")
    print(f"Number of models: {len(models)} (reduced from 5)")
    print(f"Optimal threshold: {optimal_threshold:.6f}")
    print("Expected speedup: ~40% faster predictions")
    print("=" * 60)

if __name__ == '__main__':
    main()
