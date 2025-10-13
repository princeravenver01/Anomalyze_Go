# Anomalyze Quick Start Guide

## 🚀 Quick Deployment Checklist

### Before You Deploy

- [ ] Python 3.12 installed locally for testing
- [ ] GitHub account with repository access
- [ ] Render account created
- [ ] Vercel account created
- [ ] Models trained and saved in `models/` folder
- [ ] Training data in `data/` folder

### Files You Should Have

**For Render (API Backend):**

- ✅ `api_server.py`
- ✅ `requirements-render.txt`
- ✅ `render.yaml`
- ✅ `utils/preprocessing.py`
- ✅ `models/` folder with `.joblib` files
- ✅ `data/` folder with training data

**For Vercel (Frontend):**

- ✅ `app_vercel.py`
- ✅ `requirements-vercel.txt`
- ✅ `vercel.json`
- ✅ `runtime.txt`
- ✅ `templates/` folder
- ✅ `static/` folder

### 3-Step Deployment

#### Step 1: Deploy API to Render (10 minutes)

```bash
# 1. Push code to GitHub
git add .
git commit -m "Ready for deployment"
git push origin main

# 2. Go to render.com
# 3. Create New Web Service
# 4. Connect GitHub repo
# 5. Use these settings:
#    - Build Command: pip install -r requirements-render.txt
#    - Start Command: gunicorn api_server:app
#    - Add env var: PYTHON_VERSION = 3.12.0

# 6. Wait for deployment
# 7. Copy your API URL (e.g., https://anomalyze-api.onrender.com)
```

#### Step 2: Deploy Frontend to Vercel (5 minutes)

```bash
# Option A: Vercel Dashboard
# 1. Go to vercel.com
# 2. Import GitHub repo
# 3. Add environment variable:
#    ANOMALYZE_API_URL = [Your Render URL from Step 1]
# 4. Deploy

# Option B: Vercel CLI
vercel login
vercel
# Follow prompts and add ANOMALYZE_API_URL when asked
```

#### Step 3: Test (2 minutes)

```bash
# Test API
curl https://your-render-url.onrender.com/health

# Test Frontend
# Visit: https://your-vercel-url.vercel.app
# Upload a test file
```

### Common Issues & Quick Fixes

**Issue: API returns 500 error**

- Check Render logs: Dashboard → Your Service → Logs
- Verify models folder exists in repo

**Issue: Frontend can't connect to API**

- Verify ANOMALYZE_API_URL is set in Vercel
- Check CORS is enabled in api_server.py (it is by default)

**Issue: Build fails**

- Check Python version is 3.12
- Verify requirements files are correct

### Local Testing

```bash
# Terminal 1: Run API
pip install -r requirements-render.txt
python api_server.py
# API runs on http://localhost:10000

# Terminal 2: Run Frontend
pip install -r requirements-vercel.txt
# Windows:
$env:ANOMALYZE_API_URL="http://localhost:10000"
# Linux/Mac:
export ANOMALYZE_API_URL="http://localhost:10000"

python app_vercel.py
# Frontend runs on http://localhost:5000
```

### Need More Help?

See detailed guide: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 📊 What Each File Does

| File                      | Purpose               | Used By |
| ------------------------- | --------------------- | ------- |
| `api_server.py`           | ML prediction API     | Render  |
| `app_vercel.py`           | Web frontend          | Vercel  |
| `requirements-render.txt` | API dependencies      | Render  |
| `requirements-vercel.txt` | Frontend dependencies | Vercel  |
| `render.yaml`             | Render configuration  | Render  |
| `vercel.json`             | Vercel configuration  | Vercel  |
| `runtime.txt`             | Python version        | Vercel  |
| `utils/preprocessing.py`  | Data preprocessing    | Both    |

---

**That's it! Your app should be live in ~15 minutes total! 🎉**
