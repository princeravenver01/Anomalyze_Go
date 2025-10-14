"""
Anomalyze Automatic Retraining Script
Retrains the model with original training data + uploaded logs
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from utils.preprocessing import load_and_preprocess_data
import glob
from datetime import datetime

# Configure folders
MODELS_FOLDER = 'models'
DATA_FOLDER = 'data'
UPLOADED_LOGS_FOLDER = 'data/uploaded_logs'
ORIGINAL_TRAINING_DATA = 'data/KDDTrain+.txt'

# Paths
MODEL_PATH = os.path.join(MODELS_FOLDER, 'ensemble_models.joblib')
SCALER_PATH = os.path.join(MODELS_FOLDER, 'scaler.joblib')
COLUMNS_PATH = os.path.join(MODELS_FOLDER, 'data_columns.joblib')
THRESHOLD_PATH = os.path.join(MODELS_FOLDER, 'optimal_threshold.joblib')
MODEL_SCORES_PATH = os.path.join(MODELS_FOLDER, 'model_scores.joblib')


def combine_training_data():
    """Combine original training data with uploaded logs."""
    print("="*70)
    print("STEP 1: COMBINING TRAINING DATA")
    print("="*70)
    
    # Load original training data
    print(f"\n📂 Loading original training data: {ORIGINAL_TRAINING_DATA}")
    df_original = load_and_preprocess_data(ORIGINAL_TRAINING_DATA)
    print(f"   ✓ Original data shape: {df_original.shape}")
    
    # Find all uploaded log files
    uploaded_files = glob.glob(os.path.join(UPLOADED_LOGS_FOLDER, '*.txt'))
    print(f"\n📂 Found {len(uploaded_files)} uploaded log files")
    
    if len(uploaded_files) == 0:
        print("   ⚠ No new uploads found - using original data only")
        return df_original
    
    # Load and combine uploaded files
    uploaded_dfs = []
    for i, file_path in enumerate(uploaded_files, 1):
        try:
            print(f"   {i}. Loading {os.path.basename(file_path)}...")
            df_upload = load_and_preprocess_data(file_path)
            uploaded_dfs.append(df_upload)
            print(f"      ✓ Shape: {df_upload.shape}")
        except Exception as e:
            print(f"      ✗ Error loading {file_path}: {e}")
    
    if not uploaded_dfs:
        print("\n   ⚠ No valid uploaded files - using original data only")
        return df_original
    
    # Combine all dataframes
    print(f"\n🔗 Combining {len(uploaded_dfs) + 1} datasets...")
    df_combined = pd.concat([df_original] + uploaded_dfs, ignore_index=True)
    print(f"   ✓ Combined data shape: {df_combined.shape}")
    print(f"   ✓ New samples added: {df_combined.shape[0] - df_original.shape[0]}")
    
    return df_combined


def retrain_ensemble_models(df_train):
    """Retrain ensemble K-means models."""
    print("\n" + "="*70)
    print("STEP 2: RETRAINING ENSEMBLE MODELS")
    print("="*70)
    
    # Extract normal traffic for clustering
    print(f"\n🔍 Filtering normal traffic...")
    df_normal = df_train[df_train['label'] == 'normal']
    df_normal = df_normal.drop('label', axis=1)
    print(f"   ✓ Normal samples: {df_normal.shape[0]}")
    
    # Save column names
    data_columns = df_normal.columns
    print(f"   ✓ Features: {len(data_columns)}")
    
    # Scale the data
    print(f"\n📊 Scaling data...")
    scaler = StandardScaler()
    df_normal_scaled = scaler.fit_transform(df_normal)
    print(f"   ✓ Scaled data shape: {df_normal_scaled.shape}")
    
    # Create ensemble of models
    print(f"\n🤖 Training ensemble models...")
    models = []
    model_scores = []
    
    cluster_configs = [
        {'n_clusters': 5, 'random_state': 42},
        {'n_clusters': 8, 'random_state': 42},
        {'n_clusters': 10, 'random_state': 42},
        {'n_clusters': 8, 'random_state': 123},
        {'n_clusters': 8, 'random_state': 456}
    ]
    
    for i, config in enumerate(cluster_configs, 1):
        print(f"   Model {i}/{len(cluster_configs)}: {config['n_clusters']} clusters (seed={config['random_state']})")
        
        model = KMeans(**config, max_iter=300, n_init=10, algorithm='lloyd')
        model.fit(df_normal_scaled)
        
        # Calculate quality metrics
        silhouette = silhouette_score(df_normal_scaled, model.labels_)
        inertia = model.inertia_
        
        models.append(model)
        model_scores.append({
            'silhouette': silhouette,
            'inertia': inertia,
            'n_clusters': config['n_clusters']
        })
        
        print(f"      ✓ Silhouette: {silhouette:.4f}, Inertia: {inertia:.2f}")
    
    return models, scaler, data_columns, model_scores


def find_optimal_threshold(models, scaler, data_columns, df_train):
    """Find optimal detection threshold."""
    print("\n" + "="*70)
    print("STEP 3: OPTIMIZING DETECTION THRESHOLD")
    print("="*70)
    
    true_labels = (df_train['label'] != 'normal').astype(int)
    df_train_features = df_train.drop('label', axis=1)
    df_train_features = df_train_features.reindex(columns=data_columns, fill_value=0)
    df_train_scaled = scaler.transform(df_train_features)
    
    # Calculate ensemble distances
    print(f"\n📏 Calculating distances for all samples...")
    all_distances = []
    for i, model in enumerate(models, 1):
        distances = model.transform(df_train_scaled).min(axis=1)
        all_distances.append(distances)
        print(f"   Model {i}: distances calculated")
    
    avg_distances = np.mean(all_distances, axis=0)
    
    # Find best threshold
    print(f"\n🎯 Finding optimal threshold...")
    from sklearn.metrics import f1_score
    
    best_threshold = None
    best_f1 = 0
    
    percentiles = [70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98]
    
    for percentile in percentiles:
        threshold = np.percentile(avg_distances, percentile)
        predictions = (avg_distances > threshold).astype(int)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"   ✓ Optimal threshold: {best_threshold:.6f}")
    print(f"   ✓ F1 Score: {best_f1:.4f}")
    
    return best_threshold


def save_models(models, scaler, data_columns, threshold, model_scores):
    """Save retrained models."""
    print("\n" + "="*70)
    print("STEP 4: SAVING MODELS")
    print("="*70)
    
    print(f"\n💾 Saving models to {MODELS_FOLDER}/")
    
    joblib.dump(models, MODEL_PATH)
    print(f"   ✓ Ensemble models saved")
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"   ✓ Scaler saved")
    
    joblib.dump(data_columns, COLUMNS_PATH)
    print(f"   ✓ Data columns saved")
    
    joblib.dump(threshold, THRESHOLD_PATH)
    print(f"   ✓ Optimal threshold saved")
    
    joblib.dump(model_scores, MODEL_SCORES_PATH)
    print(f"   ✓ Model scores saved")


def archive_uploaded_logs():
    """Archive uploaded logs after training."""
    print("\n" + "="*70)
    print("STEP 5: ARCHIVING UPLOADED LOGS")
    print("="*70)
    
    archive_folder = os.path.join(UPLOADED_LOGS_FOLDER, 'archived')
    os.makedirs(archive_folder, exist_ok=True)
    
    uploaded_files = glob.glob(os.path.join(UPLOADED_LOGS_FOLDER, '*.txt'))
    
    if not uploaded_files:
        print("   ℹ No files to archive")
        return
    
    print(f"\n📦 Archiving {len(uploaded_files)} files...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for file_path in uploaded_files:
        filename = os.path.basename(file_path)
        archive_path = os.path.join(archive_folder, f"{timestamp}_{filename}")
        os.rename(file_path, archive_path)
    
    print(f"   ✓ Files archived to: {archive_folder}")


def main():
    """Main retraining workflow."""
    print("\n" + "="*70)
    print("🔄 ANOMALYZE MODEL RETRAINING")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Combine data
        df_combined = combine_training_data()
        
        # Step 2: Retrain models
        models, scaler, data_columns, model_scores = retrain_ensemble_models(df_combined)
        
        # Step 3: Optimize threshold
        threshold = find_optimal_threshold(models, scaler, data_columns, df_combined)
        
        # Step 4: Save models
        save_models(models, scaler, data_columns, threshold, model_scores)
        
        # Step 5: Archive uploaded logs
        archive_uploaded_logs()
        
        print("\n" + "="*70)
        print("✅ RETRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ RETRAINING FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
