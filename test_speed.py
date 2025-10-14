"""
Quick performance test script
Tests the optimized API endpoint
"""

import requests
import time

# API endpoint
API_URL = "http://localhost:10000/api/predict"

# Test file
TEST_FILE = "uploads/KDDTest.txt"

def test_prediction_speed():
    print("=" * 60)
    print("Testing Optimized Anomalyze Performance")
    print("=" * 60)
    
    print(f"\n📁 Test file: {TEST_FILE}")
    
    # Open and send file
    with open(TEST_FILE, 'rb') as f:
        files = {'file': ('KDDTest.txt', f, 'text/plain')}
        
        print("🚀 Uploading file and starting analysis...")
        start_time = time.time()
        
        try:
            response = requests.post(API_URL, files=files, timeout=300)
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print("\n✅ Analysis Complete!")
                print("=" * 60)
                print(f"⏱️  Total Response Time: {elapsed_time:.2f} seconds")
                
                if 'metrics' in result:
                    metrics = result['metrics']
                    print(f"\n📊 Performance Metrics:")
                    print(f"   - Server Processing Time: {metrics.get('processing_time', 0):.2f}s")
                    print(f"   - Total Samples: {metrics.get('total_samples', 0):,}")
                    print(f"   - Anomalies Detected: {metrics.get('anomalies_detected', 0):,}")
                    print(f"   - Samples per Second: {metrics.get('total_samples', 0) / metrics.get('processing_time', 1):,.0f}")
                    
                    if 'accuracy' in metrics:
                        print(f"\n🎯 Accuracy Metrics:")
                        print(f"   - Accuracy: {metrics.get('accuracy', 0):.2f}%")
                        print(f"   - Precision: {metrics.get('precision', 0):.2f}%")
                        print(f"   - Recall: {metrics.get('recall', 0):.2f}%")
                        print(f"   - F1-Score: {metrics.get('f1_score', 0):.2f}%")
                    
                    print(f"\n💡 Average Confidence: {metrics.get('avg_confidence', 0):.2f}%")
                
                print("\n" + "=" * 60)
                print("✨ Optimization working successfully!")
                print("=" * 60)
                
            else:
                print(f"\n❌ Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.Timeout:
            print("\n❌ Request timed out after 5 minutes")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_prediction_speed()
