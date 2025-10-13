# Anomalyze 🔍

**High-Performance Network Intrusion Detection System using Machine Learning**

Anomalyze is an optimized network anomaly detection system that uses ensemble K-means clustering to identify suspicious network traffic patterns and potential cyber threats with exceptional speed and accuracy.

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Deployment](https://img.shields.io/badge/deployment-Render%20%2B%20Vercel-blueviolet.svg)

## 🏗️ Architecture

Anomalyze uses a modern split architecture for optimal performance and scalability:

```
User → Vercel (Frontend) → Render (API + ML Models) → Results
```

- **Vercel**: Hosts the web interface and handles user interactions
- **Render**: Hosts ML models, training data, and provides prediction API
- **Benefits**: Independent scaling, cost-effective, easy maintenance

## 🚀 Key Features

- **⚡ Ultra-Fast Processing**: 47,000+ samples per second processing speed
- **🎯 High Accuracy**: 86.24% accuracy with 87.17% F1-score
- **🔄 Optimized K-means Ensemble**: 5 different K-means models with optimized configurations
- **📊 Real-time Analysis**: Instant anomaly detection results via REST API
- **📈 Comprehensive Metrics**: Detailed performance analytics with precision, recall, and confidence scoring
- **🎨 User-Friendly Interface**: Clean web application deployed on Vercel
- **⚙️ Streamlined Preprocessing**: Optimized data pipeline for maximum performance
- **📋 KDD Cup 1999 Compatible**: Industry-standard dataset support
- **☁️ Cloud-Native**: Deployed on Render and Vercel for reliability

## 🏆 Performance Highlights

| Metric               | Value              | Status       |
| -------------------- | ------------------ | ------------ |
| **Processing Speed** | 47,137 samples/sec | ⚡ Excellent |
| **Accuracy**         | 86.24%             | 🎯 Excellent |
| **Precision**        | 92.86%             | 🔍 Very High |
| **Recall**           | 82.15%             | 📊 High      |
| **F1-Score**         | 87.17%             | 🎪 Excellent |
| **API Response**     | <1 second          | ⚡ Fast      |

## 🛠️ Technology Stack

### Backend (Render)

- **Runtime**: Python 3.12
- **Framework**: Flask 3.0+ with Flask-CORS
- **ML Libraries**: scikit-learn 1.5+, NumPy 1.26+, Pandas 2.2+
- **Server**: Gunicorn
- **Storage**: Models and training data included

### Frontend (Vercel)

- **Runtime**: Python 3.12
- **Framework**: Flask 3.0+
- **HTTP Client**: Requests
- **Deployment**: Serverless functions

## 📁 Project Structure

```
Anomalyze/
├── api_server.py              # Render API backend
├── app_vercel.py              # Vercel frontend
├── app.py                     # Original standalone app
├── requirements-render.txt    # API dependencies
├── requirements-vercel.txt    # Frontend dependencies
├── requirements.txt           # All dependencies
├── render.yaml                # Render configuration
├── vercel.json                # Vercel configuration
├── runtime.txt                # Python version
├── models/                    # ML models (for Render)
│   ├── ensemble_models.joblib
│   ├── scaler.joblib
│   ├── data_columns.joblib
│   └── optimal_threshold.joblib
├── data/                      # Training data (for Render)
│   └── KDDTrain+.txt
├── utils/                     # Shared utilities
│   └── preprocessing.py
├── templates/                 # HTML templates
│   └── index.html
├── static/                    # CSS/JS files
│   └── style.css
└── docs/                      # Documentation
    ├── DEPLOYMENT.md
    ├── QUICKSTART.md
    ├── ARCHITECTURE.md
    └── PRE_DEPLOYMENT_CHECKLIST.md
```

## 🚀 Quick Start

### Option 1: Deploy to Cloud (Recommended)

**Prerequisites:**

- GitHub account
- Render account (free tier available)
- Vercel account (free tier available)

**Steps:**

1. Clone this repository
2. Follow **[QUICKSTART.md](QUICKSTART.md)** for 3-step deployment
3. Your app will be live in ~15 minutes!

### Option 2: Run Locally

#### Run API Server

```bash
# Install dependencies
pip install -r requirements-render.txt

# Start API server
python api_server.py
```

API available at `http://localhost:10000`

#### Run Frontend

```bash
# Install dependencies
pip install -r requirements-vercel.txt

# Set API URL (Windows PowerShell)
$env:ANOMALYZE_API_URL="http://localhost:10000"

# Set API URL (Linux/Mac)
export ANOMALYZE_API_URL="http://localhost:10000"

# Start frontend
python app_vercel.py
```

Frontend available at `http://localhost:5000`

#### Run Standalone (Original)

```bash
pip install -r requirements.txt
python app.py
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick 3-step deployment guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Comprehensive deployment instructions
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture diagrams
- **[DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md)** - Deployment overview
- **[PRE_DEPLOYMENT_CHECKLIST.md](PRE_DEPLOYMENT_CHECKLIST.md)** - Pre-deployment checklist

## 🔌 API Endpoints

### Render API

```
GET  /health
     └─► Health check and status

GET  /api/model-info
     └─► Get model information

POST /api/predict
     └─► File upload prediction (multipart/form-data)

POST /api/predict-json
     └─► JSON data prediction
```

### Example API Usage

```bash
# Health check
curl https://your-api.onrender.com/health

# Predict from file
curl -X POST https://your-api.onrender.com/api/predict \
  -F "file=@network_data.txt"

# Predict from JSON
curl -X POST https://your-api.onrender.com/api/predict-json \
  -H "Content-Type: application/json" \
  -d '{"data": [{"duration": 0, "protocol_type": 1, ...}]}'
```

## 📊 Dataset

Uses the **NSL-KDD dataset** (improved version of KDD Cup 1999):

- Training: KDDTrain+.txt
- Testing: KDDTest.txt
- Features: 41 network traffic features
- Labels: Normal and various attack types

## 🎯 Model Details

### Ensemble Configuration

- **5 K-Means models** with different configurations
- Cluster counts: 5, 8, 10 (with different random seeds)
- **Majority voting** for final predictions
- **Weighted confidence scoring**
- **Severity level classification**

### Features

- 20 most important features selected
- StandardScaler normalization
- Log transformation for skewed features
- Advanced network-specific features

## 🔒 Security & Best Practices

- ✅ HTTPS enforced on both platforms
- ✅ CORS configured for secure cross-origin requests
- ✅ Environment variables for sensitive data
- ✅ Input validation on all endpoints
- ✅ Error handling without exposing internals
- ✅ Rate limiting recommended for production

## 💰 Cost

### Free Tier (Available)

- **Render**: 750 hours/month, cold starts after 15min inactivity
- **Vercel**: 100GB bandwidth/month, 100 hours serverless execution
- **Total**: $0/month for moderate usage

### Paid Options

- **Render**: From $7/month (no cold starts, better performance)
- **Vercel**: From $20/month (more bandwidth, team features)

## 🐛 Troubleshooting

**API not responding?**

- Check Render logs in dashboard
- Verify models folder exists in repository
- Free tier may have cold start delay (~30s)

**Frontend can't connect?**

- Verify `ANOMALYZE_API_URL` environment variable
- Check CORS settings in api_server.py
- Test API health endpoint directly

**Build failures?**

- Ensure Python 3.12 is specified
- Check all dependencies are compatible
- Review build logs for specific errors

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for detailed troubleshooting.

## 📈 Performance Optimization

- Ensemble models provide robust predictions
- Optimized feature selection (20 features)
- Efficient preprocessing pipeline
- Fast StandardScaler transformation
- Minimal API overhead

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (local and deployed)
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NSL-KDD dataset creators
- scikit-learn community
- Flask framework
- Render and Vercel platforms

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: See `/docs` folder
- **Deployment Help**: See DEPLOYMENT.md

## 🗺️ Roadmap

- [ ] Add authentication for production use
- [ ] Implement rate limiting
- [ ] Add more visualization options
- [ ] Support for real-time streaming data
- [ ] Model retraining pipeline
- [ ] Extended dataset support

---

**Built with ❤️ using Python 3.12, Flask, and Machine Learning**

**Ready to deploy?** → Start with [QUICKSTART.md](QUICKSTART.md) 🚀
