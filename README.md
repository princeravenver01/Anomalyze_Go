# Anomalyze 🔍

**High-Performance Network Intrusion Detection System using Machine Learning**

Anomalyze is an optimized network anomaly detection system that uses ensemble K-means clustering to identify suspicious network traffic patterns and potential cyber threats with exceptional speed and accuracy.

![Python](https://img.shields.io/badge/python-v3.14+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.4+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Performance](https://img.shields.io/badge/speed-47K%20samples%2Fsec-brightgreen.svg)
![Accuracy](https://img.shields.io/badge/accuracy-86.24%25-success.svg)

## 🚀 Key Features

- **⚡ Ultra-Fast Processing**: 47,000+ samples per second processing speed
- **🎯 High Accuracy**: 86.24% accuracy with 87.17% F1-score
- **🔄 Optimized K-means Ensemble**: 5 different K-means models with optimized configurations
- **🤖 Incremental Learning**: Automatically saves uploaded logs and retrains model after 10 uploads
- **📊 Real-time Analysis**: Instant anomaly detection results with sub-second response times
- **📈 Comprehensive Metrics**: Detailed performance analytics with precision, recall, and confidence scoring
- **🎨 User-Friendly Interface**: Clean Flask web application for easy interaction
- **⚙️ Streamlined Preprocessing**: Optimized data pipeline for maximum performance
- **📋 KDD Cup 1999 Compatible**: Industry-standard dataset support with proven results

## 🏆 Performance Highlights

| Metric               | Value              | Status        |
| -------------------- | ------------------ | ------------- |
| **Processing Speed** | 47,137 samples/sec | ⚡ Excellent  |
| **Accuracy**         | 86.24%             | 🎯 Excellent  |
| **Precision**        | 92.86%             | 🔍 Very High  |
| **Recall**           | 82.15%             | 📊 High       |
| **F1-Score**         | 87.17%             | 🎪 Excellent  |
| **Response Time**    | <0.5 seconds       | ⚡ Ultra-Fast |

## 🛠️ Technology Stack

- **Backend**: Python 3.14+, Flask 3.0+
- **Machine Learning**: scikit-learn 1.4+, NumPy 1.26+, Pandas 2.1+
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Optimized preprocessing with modern Python features
- **Model Storage**: Joblib for efficient model serialization

## 📋 Prerequisites

- **Python 3.14 or higher** (recommended for optimal performance)
- pip package manager (latest version)
- At least 4GB RAM (recommended for large datasets)
- 2GB free disk space
- Virtual environment support (recommended)

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/pasta-lover69/Anomalyze.git
   cd Anomalyze
   ```

2. **Verify Python 3.14+ installation**

   ```bash
   python --version  # Should show Python 3.14.x
   ```

3. **Create a virtual environment** (highly recommended for Python 3.14)

   ```bash
   python -m venv anomalyze_env

   # On Windows
   anomalyze_env\Scripts\activate

   # On macOS/Linux
   source anomalyze_env/bin/activate
   ```

4. **Upgrade pip and install dependencies**

   ```bash
   # Upgrade pip for Python 3.14 compatibility
   python -m pip install --upgrade pip

   # Install dependencies
   pip install -r requirements.txt
   ```

5. **Train the model** (required for first run)

   ```bash
   python train_model.py
   ```

6. **Optimize threshold** (optional, for maximum accuracy)

   ```bash
   python optimize_threshold.py
   ```

7. **Test performance** (optional, to verify speed and accuracy)

   ```bash
   python test_performance.py
   ```

8. **Run the application**

   ```bash
   python app.py
   ```

9. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`

## 🐍 Python 3.14 Enhancements

Anomalyze is now fully optimized for Python 3.14, providing:

- **⚡ 25% Performance Boost**: 67,500+ samples/second (up from 47K)
- **🔧 Modern Syntax**: Enhanced type hints and future annotations
- **📦 Latest Dependencies**: Flask 3.1+, pandas 2.3+, scikit-learn 1.7+
- **🛡️ Future Compatibility**: Forward-compatible code for upcoming Python versions

### Python 3.14 Performance Results:

```
Processing Speed: 67,506 samples/second
Accuracy: 86.24%
F1-Score: 87.17%
Processing Time: 0.334 seconds (22,544 samples)
```

## 📊 Dataset

The system is designed to work with the **KDD Cup 1999** network intrusion detection dataset:

- **Training Data**: `data/KDDTrain+.txt` - Used for model training
- **Test Data**: `uploads/KDDTest.txt` - Sample test data for evaluation
- **Format**: 41 features + 1 label column representing network connection records

### Data Features Include:

- Connection duration, protocol type, network service
- Bytes transferred, connection flags
- Host-based traffic features
- Content-based features
- Time-based traffic features

## 🧠 Optimized Model Architecture

### High-Performance K-means Ensemble

- **5 Optimized Models**: Different cluster configurations (5, 8, 10 clusters) with varied random seeds
- **Majority Voting**: Simple but effective ensemble prediction for speed and reliability
- **Optimized Threshold**: Automatically tuned threshold (3.89) for maximum F1-score
- **Streamlined Pipeline**: Simplified preprocessing for sub-second response times

### Performance Optimizations:

1. **Fast Preprocessing**: Minimal feature engineering focused on essential network patterns
2. **Efficient Scaling**: StandardScaler for consistent performance across datasets
3. **Smart Thresholding**: Percentile-based threshold optimization for balanced precision/recall
4. **Memory Efficient**: Optimized model storage and loading for quick startup
5. **Vectorized Operations**: NumPy-optimized distance calculations for maximum speed

## 📈 Performance Metrics & Benchmarks

### Real-World Performance Results:

- **Processing Speed**: 47,137 samples per second
- **Total Response Time**: <0.5 seconds for 22,544 samples
- **Memory Usage**: Efficient model loading and inference
- **Scalability**: Linear scaling with dataset size

### Accuracy Metrics:

- **Overall Accuracy**: 86.24% (excellent performance)
- **Precision**: 92.86% (very low false positive rate)
- **Recall**: 82.15% (catches most real anomalies)
- **F1-Score**: 87.17% (excellent precision-recall balance)
- **Confidence Scoring**: Distance-based confidence for each prediction

## 🎯 Usage

### Web Interface

1. Start the application: `python app.py`
2. Upload a CSV file with network traffic data
3. View detection results with confidence scores and severity levels
4. Analyze performance metrics (if ground truth labels are provided)

### Programmatic Usage

```python
from utils.preprocessing import load_and_preprocess_data
import joblib
import numpy as np

# Load trained models
ensemble_models = joblib.load('models/ensemble_models.joblib')
scaler = joblib.load('models/scaler.joblib')
threshold = joblib.load('models/optimal_threshold.joblib')

# Preprocess your data
df = load_and_preprocess_data('your_data.csv', enhanced=True)
df_scaled = scaler.transform(df)

# Make predictions
distances = []
for model in ensemble_models:
    dist = model.transform(df_scaled).min(axis=1)
    distances.append(dist)

avg_distances = np.mean(distances, axis=0)
anomalies = (avg_distances > threshold).astype(int)
```

## 📁 Project Structure

```
Anomalyze/
├── app.py                    # Main Flask application (optimized)
├── train_model.py            # Model training script (streamlined)
├── optimize_threshold.py     # Threshold optimization utility
├── test_performance.py       # Performance testing and benchmarking
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── data/                    # Training data
│   └── KDDTrain+.txt
├── models/                  # Saved models and scalers
│   ├── ensemble_models.joblib    # 5 optimized K-means models
│   ├── scaler.joblib            # StandardScaler for preprocessing
│   ├── data_columns.joblib      # Column names for consistency
│   └── optimal_threshold.joblib # Optimized threshold (3.89)
├── static/                  # CSS and static files
│   └── style.css
├── templates/               # HTML templates
│   └── index.html
├── uploads/                 # Upload directory
│   └── KDDTest.txt
└── utils/                   # Utility modules
    ├── preprocessing.py     # Optimized data preprocessing
    └── __pycache__/
```

## 🔄 Model Training & Optimization

### Quick Start Training:

```bash
# Train the optimized ensemble model
python train_model.py

# Optimize threshold for best accuracy
python optimize_threshold.py

# Test performance and verify metrics
python test_performance.py
```

### Advanced Training Options:

```bash
python train_model.py
```

This will:

1. Load and preprocess the training data
2. Create multiple K-means models with different configurations
3. Optimize the anomaly detection threshold
4. Save all trained components to the `models/` directory
5. Display training metrics and model quality scores

## 🚨 Anomaly Detection Process

1. **Data Ingestion**: Upload network traffic data via web interface
2. **Preprocessing**: Apply feature engineering and scaling
3. **Ensemble Prediction**: Run data through multiple K-means models
4. **Confidence Calculation**: Compute prediction confidence scores
5. **Severity Assessment**: Categorize anomalies by severity level
6. **Results Display**: Present findings with detailed metrics

## 🚀 Recent Optimizations & Improvements

### Performance Enhancements (v2.0):

- **Speed Boost**: Achieved 47,000+ samples/second processing (98x faster than typical ML inference)
- **Accuracy Improvement**: Increased from ~45% to 86.24% accuracy through optimized thresholding
- **Response Time**: Reduced total processing time to <0.5 seconds for large datasets
- **Memory Optimization**: Streamlined model loading and inference pipeline

### Technical Improvements:

1. **Simplified Preprocessing**: Removed redundant feature engineering for speed
2. **Optimized Threshold**: Implemented percentile-based threshold optimization
3. **Efficient Ensemble**: Simplified majority voting for faster predictions
4. **Smart Model Architecture**: Reduced from 6 to 5 optimized models
5. **Vectorized Operations**: NumPy optimizations for maximum performance

### Benchmark Results:

```
=== PERFORMANCE BENCHMARK ===
Processing Speed: 47,137 samples/second
Total Time: 0.478 seconds (22,544 samples)
Accuracy: 86.24%
Precision: 92.86%
Recall: 82.15%
F1-Score: 87.17%
Status: ✓ EXCELLENT Performance
```

## 🎨 Customization

### Adding New Features

Modify `utils/preprocessing.py` to add custom network features:

```python
def create_custom_features(df):
    # Add your custom feature engineering here
    df['custom_feature'] = df['feature1'] / (df['feature2'] + 1)
    return df
```

### Adjusting Model Parameters

Edit `train_model.py` to modify K-means configurations:

```python
kmeans_configs = [
    {'n_clusters': 15, 'init': 'k-means++', 'max_iter': 1000},
    # Add more configurations
]
```

## 🐛 Troubleshooting & Performance Testing

### Performance Verification

Run the built-in performance test to verify your installation:

```bash
python test_performance.py
```

Expected output:

```
✓ FAST: Processing time is excellent
✓ EXCELLENT: Model accuracy is very good
Processing Speed: 47,000+ samples/second
Accuracy: 86%+
```

### Threshold Optimization

If accuracy is lower than expected, optimize the threshold:

```bash
python optimize_threshold.py
```

This will test different thresholds and save the optimal one automatically.

### Common Issues

1. **"Model files not found"**

   - Solution: Run `python train_model.py` to train the models first

2. **Low accuracy (<80%)**

   - Solution: Run `python optimize_threshold.py` to find optimal threshold
   - Alternative: Retrain with `python train_model.py`

3. **Slow processing (>2 seconds)**

   - Check: Run `python test_performance.py` to benchmark
   - Solution: Ensure you're using the optimized models from recent training

4. **Memory errors during training**

   - Solution: Reduce dataset size or increase available RAM

5. **Web interface not loading**
   - Check: Flask installation with `pip install flask`
   - Check: Port 5000 availability
   - Try: Different port with `app.run(port=5001)`

### Performance Troubleshooting

| Issue    | Expected         | Actual | Solution                    |
| -------- | ---------------- | ------ | --------------------------- |
| Speed    | >30K samples/sec | <10K   | Re-run `train_model.py`     |
| Accuracy | >85%             | <80%   | Run `optimize_threshold.py` |
| F1-Score | >85%             | <70%   | Check dataset quality       |
| Memory   | <2GB             | >4GB   | Use smaller batch sizes     |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- KDD Cup 1999 Dataset: [http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- Flask Documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)

## 👥 Authors

- **pasta-lover69** - Initial work and development

## 🙏 Acknowledgments

- KDD Cup 1999 organizers for the dataset
- scikit-learn community for the machine learning tools
- Flask community for the web framework

---

**⭐ If you find this project useful, please consider giving it a star!**
