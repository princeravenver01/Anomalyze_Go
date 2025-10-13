# Anomalyze Deployment Guide

This guide explains how to deploy the Anomalyze application with a split architecture:

- **Render**: Hosts ML models, training data, and provides API
- **Vercel**: Hosts the web frontend that calls the Render API

## Architecture Overview

```
User Browser
    ↓
Vercel (Frontend) → Render (API + Models)
                       ↓
                    Models & Data
```

## Prerequisites

- Git repository with your code
- Render account (free tier available)
- Vercel account (free tier available)
- Python 3.12 environment for local testing

---

## Part 1: Deploy API to Render

### Step 1: Prepare Render Deployment

1. **Ensure you have these files in your repository:**

   - `api_server.py` - API server code
   - `requirements-render.txt` - Python dependencies for Render
   - `render.yaml` - Render configuration
   - `utils/preprocessing.py` - Preprocessing utilities
   - `models/` folder with trained models
   - `data/` folder with training data

2. **Commit and push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

### Step 2: Create Render Web Service

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:

   - **Name**: `anomalyze-api`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements-render.txt`
   - **Start Command**: `gunicorn api_server:app`
   - **Plan**: Free (or choose based on needs)

5. **Add Environment Variables:**

   - `PYTHON_VERSION`: `3.12.0`
   - `PORT`: `10000`

6. Click **"Create Web Service"**

### Step 3: Wait for Deployment

- Render will build and deploy your service (takes 5-10 minutes)
- Once deployed, you'll get a URL like: `https://anomalyze-api.onrender.com`
- **Save this URL** - you'll need it for Vercel configuration

### Step 4: Test the API

Test your API endpoints:

```bash
# Health check
curl https://your-render-url.onrender.com/health

# Model info
curl https://your-render-url.onrender.com/api/model-info
```

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Prepare Vercel Deployment

1. **Ensure you have these files:**
   - `app_vercel.py` - Frontend application
   - `requirements-vercel.txt` - Python dependencies for Vercel
   - `vercel.json` - Vercel configuration
   - `templates/` folder with HTML templates
   - `static/` folder with CSS/JS files

### Step 2: Install Vercel CLI (Optional)

```bash
npm install -g vercel
```

Or use the Vercel dashboard for deployment.

### Step 3: Deploy to Vercel

#### Option A: Using Vercel Dashboard

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New..."** → **"Project"**
3. Import your GitHub repository
4. Configure:

   - **Framework Preset**: Other
   - **Root Directory**: `./` (leave as is)
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)

5. **Add Environment Variable:**

   - **Name**: `ANOMALYZE_API_URL`
   - **Value**: Your Render API URL (e.g., `https://anomalyze-api.onrender.com`)

6. Click **"Deploy"**

#### Option B: Using Vercel CLI

```bash
# Login to Vercel
vercel login

# Deploy
vercel

# Follow the prompts and add environment variable when asked:
# ANOMALYZE_API_URL=https://your-render-url.onrender.com

# For production deployment:
vercel --prod
```

### Step 4: Configure Environment Variables

In Vercel Dashboard:

1. Go to your project
2. Settings → Environment Variables
3. Add: `ANOMALYZE_API_URL` = `https://your-render-url.onrender.com`
4. Redeploy if necessary

---

## Part 3: Verify Deployment

### Test Frontend

1. Visit your Vercel URL (e.g., `https://anomalyze.vercel.app`)
2. Upload a test file
3. Verify anomaly detection results are displayed

### Test API Connection

```bash
# From frontend health endpoint
curl https://your-vercel-url.vercel.app/api/health
```

---

## Troubleshooting

### Render Issues

**Problem: Models not loading**

- Ensure `models/` folder is in repository
- Check logs: Render Dashboard → Your Service → Logs
- Verify file paths in `api_server.py`

**Problem: Build fails**

- Check Python version is 3.12
- Verify all dependencies in `requirements-render.txt`
- Check build logs for specific errors

**Problem: API timeout**

- Free tier may have cold start delays (first request takes longer)
- Consider upgrading to paid tier for better performance

### Vercel Issues

**Problem: Can't connect to API**

- Verify `ANOMALYZE_API_URL` environment variable is set correctly
- Check Render API is running
- Verify CORS is enabled in `api_server.py`

**Problem: File upload fails**

- Check file size limits (Vercel has 4.5MB limit for serverless functions)
- Consider uploading directly to Render API for large files

**Problem: Deployment fails**

- Ensure Python version compatibility
- Check `vercel.json` configuration
- Verify all dependencies in `requirements-vercel.txt`

---

## Local Development

### Running API Server Locally

```bash
# Install dependencies
pip install -r requirements-render.txt

# Run API server
python api_server.py
```

API will be available at `http://localhost:10000`

### Running Frontend Locally

```bash
# Install dependencies
pip install -r requirements-vercel.txt

# Set environment variable
# Windows PowerShell:
$env:ANOMALYZE_API_URL="http://localhost:10000"

# Linux/Mac:
export ANOMALYZE_API_URL="http://localhost:10000"

# Run frontend
python app_vercel.py
```

Frontend will be available at `http://localhost:5000`

---

## Updating Deployments

### Update API (Render)

```bash
git add .
git commit -m "Update API"
git push origin main
```

Render auto-deploys on push (if enabled).

### Update Frontend (Vercel)

```bash
git add .
git commit -m "Update frontend"
git push origin main
```

Vercel auto-deploys on push.

---

## Cost Considerations

### Free Tier Limits

**Render (Free)**:

- 750 hours/month
- Sleeps after 15 minutes of inactivity
- Cold start delay (~30 seconds)

**Vercel (Free)**:

- 100GB bandwidth/month
- 100 hours serverless function execution
- Fair use policy

### Recommendations

- Use free tiers for development/testing
- Upgrade Render for production (eliminates sleep/cold starts)
- Monitor usage in both dashboards

---

## Security Best Practices

1. **Use Environment Variables** for sensitive data
2. **Enable HTTPS** (both platforms provide it)
3. **Add Authentication** for production use
4. **Rate Limiting** to prevent abuse
5. **Input Validation** on both frontend and API

---

## Monitoring

### Render Monitoring

- Dashboard → Your Service → Metrics
- View logs for errors and performance

### Vercel Monitoring

- Dashboard → Your Project → Analytics
- Monitor function execution and bandwidth

---

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

## Support

For issues:

1. Check logs in Render/Vercel dashboards
2. Review error messages
3. Consult troubleshooting section above
4. Check GitHub repository for updates

---

## Architecture Benefits

✅ **Separation of Concerns**: Frontend and ML models are decoupled  
✅ **Scalability**: Each component can scale independently  
✅ **Cost-Effective**: Use free tiers for both services  
✅ **Maintainability**: Easy to update models without redeploying frontend  
✅ **Performance**: ML processing on dedicated backend

---

**Happy Deploying! 🚀**
