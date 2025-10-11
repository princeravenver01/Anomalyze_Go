# Anomalyze 🔍

**Network Intrusion Detection System using Machine Learning**

Anomalyze is an advanced network anomaly detection system that uses ensemble K-means clustering to identify suspicious network traffic patterns and potential cyber threats in real-time.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

- **Advanced K-means Ensemble**: Multiple K-means models with different configurations for robust detection
- **Real-time Analysis**: Upload network traffic data and get instant anomaly detection results
- **Comprehensive Metrics**: Detailed performance metrics including accuracy, precision, recall, and F1-score
- **Confidence Scoring**: Each detection comes with a confidence score and severity level
- **Web Interface**: User-friendly Flask web application for easy interaction
- **Enhanced Preprocessing**: Advanced feature engineering optimized for network security data
- **KDD Cup 1999 Compatible**: Trained and tested on the industry-standard KDD Cup 1999 dataset

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, NumPy, Pandas
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Robust preprocessing with feature engineering
- **Model Storage**: Joblib for efficient model serialization

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (recommended)
- 2GB free disk space

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/pasta-lover69/Anomalyze.git
   cd Anomalyze
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv anomalyze_env

   # On Windows
   anomalyze_env\Scripts\activate

   # On macOS/Linux
   source anomalyze_env/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already trained)

   ```bash
   python train_model.py
   ```

5. **Run the application**

   ```bash
   python app.py
   ```

6. **Access the web interface**
   - Open your browser and navigate to `http://localhost:5000`

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

## 🧠 Model Architecture

### Ensemble K-means Clustering

- **Multiple Models**: 6 different K-means configurations
- **Cluster Variations**: Different cluster numbers (8, 10, 12) and random states
- **MiniBatch K-means**: Included for computational efficiency
- **Weighted Voting**: Models weighted by clustering quality metrics

### Key Components:

1. **Feature Engineering**: Advanced network-specific feature creation
2. **Robust Preprocessing**: Outlier handling and robust scaling
3. **Ensemble Prediction**: Quality-weighted voting system
4. **Threshold Optimization**: Automated threshold selection using F1-score
5. **Confidence Scoring**: Distance-based confidence calculation

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confidence Score**: Average confidence in predictions
- **Processing Time**: Time taken for detection

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
├── app.py                 # Main Flask application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/                 # Training data
│   └── KDDTrain+.txt
├── models/               # Saved models and scalers
│   ├── ensemble_models.joblib
│   ├── scaler.joblib
│   ├── data_columns.joblib
│   ├── optimal_threshold.joblib
│   └── model_scores.joblib
├── static/               # CSS and static files
│   └── style.css
├── templates/            # HTML templates
│   └── index.html
├── uploads/              # Upload directory
│   └── KDDTest.txt
└── utils/                # Utility modules
    ├── preprocessing.py  # Data preprocessing functions
    └── __pycache__/
```

## 🔄 Model Training

To retrain the model with new data or improved parameters:

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

## 🐛 Troubleshooting

### Common Issues

1. **"Model files not found"**

   - Run `python train_model.py` to train the models first

2. **Memory errors during training**

   - Reduce the dataset size or use MiniBatch K-means exclusively

3. **Poor detection accuracy**

   - Retrain with more representative data
   - Adjust threshold parameters in `train_model.py`

4. **Web interface not loading**
   - Check if Flask is installed: `pip install flask`
   - Ensure port 5000 is not in use

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
