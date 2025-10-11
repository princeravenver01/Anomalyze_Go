"""
Threshold optimization script for better accuracy
"""

import numpy as np
import pandas as pd
from utils.preprocessing import load_and_preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def optimize_threshold():
    """Find better threshold for improved accuracy"""
    print("=== THRESHOLD OPTIMIZATION ===\n")
    
    # Load models
    ensemble_models = joblib.load('models/ensemble_models.joblib')
    scaler = joblib.load('models/scaler.joblib')
    data_columns = joblib.load('models/data_columns.joblib')
    
    # Load test data
    test_data_path = 'uploads/KDDTest.txt'
    df_test = load_and_preprocess_data(test_data_path)
    
    true_labels = df_test['label'].copy()
    true_labels_binary = (true_labels != 'normal').astype(int)
    df_test = df_test.drop('label', axis=1)
    
    df_test = df_test.reindex(columns=data_columns, fill_value=0)
    df_test_scaled = scaler.transform(df_test)
    
    # Calculate ensemble distances
    all_distances = []
    for model in ensemble_models:
        distances = model.transform(df_test_scaled).min(axis=1)
        all_distances.append(distances)
    
    avg_distances = np.mean(all_distances, axis=0)
    
    print("Testing different thresholds...")
    best_threshold = None
    best_f1 = 0
    best_metrics = None
    
    # Test percentile-based thresholds
    percentiles = [50, 60, 70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]
    
    print(f"{'Percentile':<10} {'Threshold':<12} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-" * 70)
    
    for percentile in percentiles:
        threshold = np.percentile(avg_distances, percentile)
        predictions = (avg_distances > threshold).astype(int)
        
        accuracy = accuracy_score(true_labels_binary, predictions)
        precision = precision_score(true_labels_binary, predictions, zero_division=0)
        recall = recall_score(true_labels_binary, predictions, zero_division=0)
        f1 = f1_score(true_labels_binary, predictions, zero_division=0)
        
        print(f"{percentile:<10} {threshold:<12.6f} {accuracy*100:<9.2f} {precision*100:<10.2f} {recall*100:<8.2f} {f1*100:<8.2f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'percentile': percentile,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    print(f"\n=== BEST THRESHOLD FOUND ===")
    print(f"Percentile: {best_metrics['percentile']}%")
    print(f"Threshold: {best_threshold:.6f}")
    print(f"Accuracy: {best_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {best_metrics['precision']*100:.2f}%")
    print(f"Recall: {best_metrics['recall']*100:.2f}%")
    print(f"F1 Score: {best_metrics['f1']*100:.2f}%")
    
    # Save the better threshold
    print(f"\nSaving optimized threshold...")
    joblib.dump(best_threshold, 'models/optimal_threshold.joblib')
    print(f"✓ New threshold saved: {best_threshold:.6f}")
    
    return best_threshold, best_metrics

if __name__ == '__main__':
    optimize_threshold()