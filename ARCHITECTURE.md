# Anomalyze Architecture Diagrams

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        USER BROWSER                           │
│                  (Accesses web interface)                     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            │ HTTPS
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    VERCEL (Frontend)                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ app_vercel.py                                          │  │
│  │ - Renders HTML templates                              │  │
│  │ - Handles file uploads                                │  │
│  │ - Forwards requests to Render API                     │  │
│  │ - Displays results                                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  Files: app_vercel.py, templates/, static/                   │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            │ HTTP/REST API
                            │ POST /api/predict
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     RENDER (API Backend)                      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ api_server.py                                          │  │
│  │ - Loads ML models                                      │  │
│  │ - Preprocesses data                                    │  │
│  │ - Makes predictions                                    │  │
│  │ - Returns JSON responses                               │  │
│  └────────────────────────────────────────────────────────┘  │
│           │                       │                           │
│           ▼                       ▼                           │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ models/         │    │ data/           │                 │
│  │ - ensemble      │    │ - KDDTrain+.txt │                 │
│  │ - scaler        │    │                 │                 │
│  │ - threshold     │    │                 │                 │
│  │ - columns       │    │                 │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                                               │
│  Files: api_server.py, utils/, models/, data/                │
└──────────────────────────────────────────────────────────────┘
```

## Request Flow

```
1. User Action
   ┌─────────────────┐
   │ User uploads    │
   │ CSV/TXT file    │
   └────────┬────────┘
            │
            ▼
2. Frontend Processing
   ┌─────────────────────────┐
   │ Vercel receives file    │
   │ Validates request       │
   │ Forwards to Render API  │
   └────────┬────────────────┘
            │
            ▼
3. API Processing
   ┌─────────────────────────┐
   │ Render API receives     │
   │ - Loads models          │
   │ - Preprocesses data     │
   │ - Runs predictions      │
   │ - Calculates metrics    │
   └────────┬────────────────┘
            │
            ▼
4. Response
   ┌─────────────────────────┐
   │ Returns JSON:           │
   │ - Anomalies detected    │
   │ - Confidence scores     │
   │ - Severity levels       │
   │ - Performance metrics   │
   └────────┬────────────────┘
            │
            ▼
5. Display
   ┌─────────────────────────┐
   │ Vercel renders results  │
   │ Shows to user           │
   └─────────────────────────┘
```

## Data Flow

```
CSV File → Vercel → HTTP POST → Render API
                                     ↓
                              Preprocessing
                                     ↓
                              Feature Engineering
                                     ↓
                              Scaling (StandardScaler)
                                     ↓
                              Ensemble Prediction
                              (5 K-Means models)
                                     ↓
                              Anomaly Detection
                                     ↓
                              Confidence & Severity
                                     ↓
                              JSON Response → Vercel → User
```

## Component Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│                    VERCEL (Frontend)                         │
├─────────────────────────────────────────────────────────────┤
│ ✓ User Interface (HTML/CSS)                                 │
│ ✓ File Upload Handling                                      │
│ ✓ Request Routing                                           │
│ ✓ Result Display                                            │
│ ✓ Error Handling                                            │
│                                                              │
│ ✗ No ML Models                                              │
│ ✗ No Data Processing                                        │
│ ✗ No Training Data                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    RENDER (Backend API)                      │
├─────────────────────────────────────────────────────────────┤
│ ✓ ML Model Hosting                                          │
│ ✓ Data Preprocessing                                        │
│ ✓ Prediction Logic                                          │
│ ✓ Feature Engineering                                       │
│ ✓ Training Data Access                                      │
│ ✓ API Endpoints                                             │
│                                                              │
│ ✗ No UI Rendering                                           │
│ ✗ No Frontend Assets                                        │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Flow

```
┌──────────────────┐
│ Developer Pushes │
│ Code to GitHub   │
└────────┬─────────┘
         │
         ├─────────────────────┬─────────────────────┐
         ▼                     ▼                     │
┌──────────────────┐  ┌──────────────────┐         │
│ Render Detects   │  │ Vercel Detects   │         │
│ Changes          │  │ Changes          │         │
└────────┬─────────┘  └────────┬─────────┘         │
         │                     │                     │
         ▼                     ▼                     │
┌──────────────────┐  ┌──────────────────┐         │
│ Build Process:   │  │ Build Process:   │         │
│ - Install deps   │  │ - Install deps   │         │
│ - Copy models/   │  │ - Copy templates │         │
│ - Copy data/     │  │ - Copy static/   │         │
└────────┬─────────┘  └────────┬─────────┘         │
         │                     │                     │
         ▼                     ▼                     │
┌──────────────────┐  ┌──────────────────┐         │
│ Start API Server │  │ Deploy Frontend  │         │
│ (gunicorn)       │  │ (serverless)     │         │
└────────┬─────────┘  └────────┬─────────┘         │
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Both Services    │
                   │ Live & Connected │
                   └──────────────────┘
```

## File Distribution

```
GitHub Repository
├── Render Deployment
│   ├── api_server.py ────────────► API endpoints
│   ├── requirements-render.txt ──► Dependencies
│   ├── render.yaml ──────────────► Configuration
│   ├── utils/
│   │   └── preprocessing.py ─────► Data processing
│   ├── models/
│   │   ├── ensemble_models.joblib
│   │   ├── scaler.joblib
│   │   ├── data_columns.joblib
│   │   └── optimal_threshold.joblib
│   └── data/
│       └── KDDTrain+.txt
│
└── Vercel Deployment
    ├── app_vercel.py ────────────► Frontend app
    ├── requirements-vercel.txt ──► Dependencies
    ├── vercel.json ──────────────► Configuration
    ├── runtime.txt ──────────────► Python version
    ├── templates/
    │   └── index.html ───────────► UI template
    └── static/
        └── style.css ────────────► Styling
```

## API Endpoints

```
RENDER API (https://your-app.onrender.com)

GET  /health
     └─► Health check
         Response: {"status": "healthy", "models_loaded": true}

GET  /api/model-info
     └─► Model information
         Response: {"num_models": 5, "num_features": 20, ...}

POST /api/predict
     └─► File upload prediction
         Input: multipart/form-data with file
         Response: {"success": true, "anomalies": [...], "metrics": {...}}

POST /api/predict-json
     └─► JSON data prediction
         Input: {"data": [{...}, {...}]}
         Response: {"success": true, "results": [...], "metrics": {...}}
```

```
VERCEL FRONTEND (https://your-app.vercel.app)

GET  /
     └─► Main page (upload interface)

POST /upload
     └─► Upload handler (forwards to Render API)

GET  /api/health
     └─► Frontend and API health check

GET  /api/model-info
     └─► Get model info (proxies to Render API)
```

## Security Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Security Layers                       │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. Transport Layer                                       │
│     ├─ HTTPS (Vercel) ─────────────► Encrypted           │
│     └─ HTTPS (Render) ─────────────► Encrypted           │
│                                                           │
│  2. CORS (Cross-Origin)                                   │
│     └─ Enabled in api_server.py ───► Allows Vercel       │
│                                                           │
│  3. Environment Variables                                 │
│     ├─ API URL stored securely ────► Vercel env          │
│     └─ No hardcoded secrets ───────► Best practice       │
│                                                           │
│  4. Input Validation                                      │
│     ├─ File type checking ─────────► Both layers         │
│     └─ Size limits ────────────────► Platform enforced   │
│                                                           │
│  5. Error Handling                                        │
│     ├─ No sensitive data in errors ► Sanitized           │
│     └─ Proper status codes ────────► HTTP standards      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## Scaling Architecture

```
Load Increases
     │
     ▼
┌─────────────────────────────────────┐
│         Vercel Auto-Scales          │
│  (Serverless functions replicate)   │
└─────────────────┬───────────────────┘
                  │
                  │ More requests to API
                  ▼
┌─────────────────────────────────────┐
│         Render Scales               │
│  Free tier: Limited                 │
│  Paid tier: Horizontal scaling      │
└─────────────────────────────────────┘
```

---

**Visual representations to help understand the Anomalyze deployment architecture!**
