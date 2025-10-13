# Pre-Deployment Checklist

## ✅ Before You Deploy - Complete This Checklist

### 1. Code Preparation

- [ ] All Python files use Python 3.12 compatible syntax
- [ ] Type hints added where appropriate
- [ ] No syntax errors (run files locally to verify)
- [ ] All imports resolve correctly

### 2. Dependencies

- [ ] `requirements.txt` updated with Python 3.12 versions
- [ ] `requirements-render.txt` created with API dependencies
- [ ] `requirements-vercel.txt` created with frontend dependencies
- [ ] All version numbers are compatible

### 3. Configuration Files

- [ ] `api_server.py` exists and works locally
- [ ] `app_vercel.py` exists and works locally
- [ ] `render.yaml` is configured correctly
- [ ] `vercel.json` is configured correctly
- [ ] `runtime.txt` specifies Python 3.12
- [ ] `.gitignore` is updated
- [ ] `.renderignore` created
- [ ] `.vercelignore` created

### 4. Models and Data

- [ ] Models folder (`models/`) contains all `.joblib` files:
  - [ ] `ensemble_models.joblib`
  - [ ] `scaler.joblib`
  - [ ] `data_columns.joblib`
  - [ ] `optimal_threshold.joblib`
- [ ] Data folder (`data/`) contains training data:
  - [ ] `KDDTrain+.txt`
- [ ] Model files are not corrupted (test locally)

### 5. Templates and Static Files

- [ ] `templates/` folder exists with:
  - [ ] `index.html`
- [ ] `static/` folder exists with:
  - [ ] `style.css`
- [ ] HTML template references are correct in code

### 6. Local Testing

#### Test API Server (Render)

```bash
pip install -r requirements-render.txt
python api_server.py
```

- [ ] Server starts without errors
- [ ] Visit `http://localhost:10000/health`
- [ ] Response: `{"status": "healthy", "models_loaded": true}`
- [ ] Test prediction endpoint with curl or Postman

#### Test Frontend (Vercel)

```bash
pip install -r requirements-vercel.txt
# Windows PowerShell:
$env:ANOMALYZE_API_URL="http://localhost:10000"
# Linux/Mac:
export ANOMALYZE_API_URL="http://localhost:10000"
python app_vercel.py
```

- [ ] Server starts without errors
- [ ] Visit `http://localhost:5000`
- [ ] Upload a test file
- [ ] Results display correctly
- [ ] No console errors

### 7. Git Repository

- [ ] All files committed to Git
- [ ] Repository pushed to GitHub
- [ ] Repository is public or accessible by Render/Vercel
- [ ] No sensitive data in repository
- [ ] `.gitignore` properly excludes unnecessary files

### 8. Account Setup

- [ ] Render account created at https://render.com
- [ ] Vercel account created at https://vercel.com
- [ ] GitHub account connected to both services
- [ ] Credit card added if using paid features (optional)

### 9. Environment Variables Prepared

- [ ] Know your GitHub repository URL
- [ ] Ready to note Render API URL after deployment
- [ ] Understand where to set `ANOMALYZE_API_URL` in Vercel

### 10. Documentation Review

- [ ] Read `QUICKSTART.md`
- [ ] Read `DEPLOYMENT.md`
- [ ] Understand the architecture from `ARCHITECTURE.md`
- [ ] Know where to find logs in both platforms

---

## 🚀 Deployment Order

Once all items are checked above, deploy in this order:

### Step 1: Deploy API to Render (FIRST)

1. Go to render.com
2. Create new Web Service
3. Connect GitHub repository
4. Configure build and start commands
5. Wait for deployment
6. **Save the API URL** (e.g., `https://anomalyze-api.onrender.com`)
7. Test: `curl https://your-api-url.onrender.com/health`

### Step 2: Deploy Frontend to Vercel (SECOND)

1. Go to vercel.com
2. Import GitHub repository
3. Add environment variable: `ANOMALYZE_API_URL` = [Your Render URL]
4. Deploy
5. Visit your Vercel URL
6. Test by uploading a file

---

## 🧪 Post-Deployment Testing

After both services are deployed:

### Test API Directly

```bash
# Health check
curl https://your-render-url.onrender.com/health

# Model info
curl https://your-render-url.onrender.com/api/model-info

# Prediction (with file)
curl -X POST https://your-render-url.onrender.com/api/predict \
  -F "file=@path/to/test.txt"
```

### Test Frontend

- [ ] Visit Vercel URL in browser
- [ ] UI loads correctly
- [ ] Upload test file (e.g., `uploads/KDDTest.txt`)
- [ ] Results display within reasonable time
- [ ] Metrics show correctly
- [ ] No errors in browser console

### Test Integration

- [ ] Frontend successfully calls Render API
- [ ] Check `/api/health` on frontend shows both services healthy
- [ ] Upload various file sizes
- [ ] Test with files that have/don't have labels
- [ ] Verify accuracy metrics display when labels present

---

## 🐛 Common Issues to Check

### Before Deployment

- [ ] Models folder is in Git repository
- [ ] Data folder is in Git repository
- [ ] No typos in configuration files
- [ ] Python version is 3.12 in all configs
- [ ] File paths use forward slashes (/) not backslashes (\)

### After Deployment

- [ ] Check Render logs if API fails
- [ ] Check Vercel logs if frontend fails
- [ ] Verify environment variables are set
- [ ] Ensure CORS is enabled in API
- [ ] Test with small files first

---

## 📊 Success Criteria

Your deployment is successful when:

✅ Render API responds to health checks  
✅ Vercel frontend loads in browser  
✅ File upload works end-to-end  
✅ Predictions return correctly  
✅ Metrics display properly  
✅ No errors in logs  
✅ Response time is acceptable

---

## 🆘 If Something Goes Wrong

1. **Check logs first**

   - Render: Dashboard → Your Service → Logs
   - Vercel: Dashboard → Your Project → Deployment → View Logs

2. **Verify environment variables**

   - Render: Dashboard → Your Service → Environment
   - Vercel: Dashboard → Your Project → Settings → Environment Variables

3. **Test locally again**

   - If it works locally but not deployed, it's likely a config issue
   - Compare local setup vs. deployed setup

4. **Review documentation**

   - `DEPLOYMENT.md` has detailed troubleshooting
   - `QUICKSTART.md` has quick fixes

5. **Common fixes**
   - Redeploy after fixing issues
   - Clear build cache if necessary
   - Restart services if stuck

---

## 📝 Notes

- Keep this checklist handy during deployment
- Check off items as you complete them
- Don't skip steps - they're all important!
- First deployment takes longer; subsequent ones are faster
- Free tier cold starts are normal (~30 seconds)

---

**Ready to deploy? Start with Step 1! 🚀**

**Need help? See `DEPLOYMENT.md` for detailed instructions.**
