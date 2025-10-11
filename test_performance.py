"""
Performance test script for Anomalyze
Tests the speed and accuracy of the model
"""

import time
import pandas as pd
import numpy as np
from utils.preprocessing import load_and_preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def test_model_performance():
    """Test the performance of the trained model"""
    print("=== ANOMALYZE PERFORMANCE TEST ===\n")
    
    # Load the trained models
    print("Loading models...")
    start_time = time.time()
    
    ensemble_models = joblib.load('models/ensemble_models.joblib')
    scaler = joblib.load('models/scaler.joblib')
    data_columns = joblib.load('models/data_columns.joblib')
    optimal_threshold = joblib.load('models/optimal_threshold.joblib')
    
    load_time = time.time() - start_time
    print(f"✓ Models loaded in {load_time:.3f} seconds")
    
    # Test with sample data
    test_data_path = 'uploads/KDDTest.txt'
    print(f"\nTesting with: {test_data_path}")
    
    # Preprocessing test
    preprocess_start = time.time()
    df_test = load_and_preprocess_data(test_data_path)
    
    has_labels = 'label' in df_test.columns
    if has_labels:
        true_labels = df_test['label'].copy()
        true_labels_binary = (true_labels != 'normal').astype(int)
        df_test = df_test.drop('label', axis=1)
    
    df_test = df_test.reindex(columns=data_columns, fill_value=0)
    df_test_scaled = scaler.transform(df_test)
    
    preprocess_time = time.time() - preprocess_start
    print(f"✓ Preprocessing completed in {preprocess_time:.3f} seconds")
    print(f"✓ Dataset shape: {df_test_scaled.shape}")
    
    # Prediction test
    predict_start = time.time()
    
    # Use ensemble prediction
    predictions = []
    distances_list = []
    
    for model in ensemble_models:
        distances = model.transform(df_test_scaled).min(axis=1)
        distances_list.append(distances)
        pred = (distances > optimal_threshold).astype(int)
        predictions.append(pred)
    
    # Majority voting
    ensemble_pred = np.array(predictions).mean(axis=0)
    final_predictions = (ensemble_pred > 0.5).astype(int)
    
    # Calculate confidence and distances
    avg_distances = np.mean(distances_list, axis=0)
    confidence_scores = 1 / (1 + avg_distances)
    
    predict_time = time.time() - predict_start
    print(f"✓ Prediction completed in {predict_time:.3f} seconds")
    
    # Results summary
    total_time = load_time + preprocess_time + predict_time
    samples_per_second = len(df_test_scaled) / total_time
    
    print(f"\n=== PERFORMANCE SUMMARY ===")
    print(f"Total processing time: {total_time:.3f} seconds")
    print(f"Samples processed: {len(df_test_scaled):,}")
    print(f"Processing speed: {samples_per_second:.1f} samples/second")
    print(f"Anomalies detected: {final_predictions.sum():,}")
    print(f"Detection rate: {(final_predictions.sum() / len(final_predictions)) * 100:.2f}%")
    print(f"Average confidence: {np.mean(confidence_scores):.3f}")
    
    if has_labels:
        print(f"\n=== ACCURACY METRICS ===")
        accuracy = accuracy_score(true_labels_binary, final_predictions)
        precision = precision_score(true_labels_binary, final_predictions, zero_division=0)
        recall = recall_score(true_labels_binary, final_predictions, zero_division=0)
        f1 = f1_score(true_labels_binary, final_predictions, zero_division=0)
        
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1 Score:  {f1 * 100:.2f}%")
        
        # Performance assessment
        print(f"\n=== PERFORMANCE ASSESSMENT ===")
        if total_time < 5.0:
            print("✓ FAST: Processing time is excellent")
        elif total_time < 10.0:
            print("✓ GOOD: Processing time is acceptable")
        else:
            print("⚠ SLOW: Processing time could be improved")
            
        if f1 > 0.7:
            print("✓ EXCELLENT: Model accuracy is very good")
        elif f1 > 0.5:
            print("✓ GOOD: Model accuracy is acceptable")
        elif f1 > 0.3:
            print("⚠ FAIR: Model accuracy could be improved")
        else:
            print("❌ POOR: Model accuracy needs improvement")
    
    return {
        'total_time': total_time,
        'samples_per_second': samples_per_second,
        'accuracy': accuracy if has_labels else None,
        'f1_score': f1 if has_labels else None
    }

if __name__ == '__main__':
    try:
        results = test_model_performance()
        print(f"\n🎯 Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()