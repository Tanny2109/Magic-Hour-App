# Magic Hour AI - Deployment Guide

This guide covers multiple ways to deploy Magic Hour AI for web sharing.

## Prerequisites

Before deploying, ensure you have:
- FAL_KEY (from fal.ai)
- HF_TOKEN (from Hugging Face)
- Git repository (recommended)

---

## Option 1: Render (Recommended - Easiest) ⭐

Render offers free tier hosting with automatic deploys.

### Steps:

1. **Push code to GitHub** (if not already)
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin master
   ```

2. **Sign up at [Render.com](https://render.com)**

3. **Create a New Web Service**
   - Connect your GitHub repository
   - Select "Python" as runtime
   - Configure:
     - **Name**: `magichour-ai`
     - **Build Command**:
       ```bash
       cd mh_agentic_workflow && pip install -r requirements.txt && cd ../magicHourApp && pip install -r requirements.txt && npm install && npm run build
       ```
     - **Start Command**:
       ```bash
       cd magicHourApp && python api_server.py
       ```

4. **Set Environment Variables** (in Render dashboard):
   - `FAL_KEY`: your fal.ai API key
   - `HF_TOKEN`: your Hugging Face token
   - `PRODUCTION`: `true`
   - `PORT`: `8000`

5. **Deploy!** - Render will build and deploy automatically

**Cost**: Free tier (limited hours/month) or $7/month for always-on

---

## Option 2: Railway (Fast & Easy)

Railway offers $5 free credit/month.

### Steps:

1. **Install Railway CLI** (optional):
   ```bash
   npm install -g @railway/cli
   ```

2. **Deploy via web** at [railway.app](https://railway.app):
   - Connect GitHub repo
   - Railway auto-detects the app
   - Add environment variables:
     - `FAL_KEY`
     - `HF_TOKEN`
     - `PRODUCTION=true`

3. **Or deploy via CLI**:
   ```bash
   railway login
   railway init
   railway up
   railway variables set FAL_KEY=your_key
   railway variables set HF_TOKEN=your_token
   railway variables set PRODUCTION=true
   ```

**Cost**: $5 free credit/month, then pay-as-you-go

---

## Option 3: Docker + Any Cloud (AWS, GCP, Azure, DigitalOcean)

Use the included `Dockerfile` to deploy anywhere.

### Build and Test Locally:

```bash
# Build the image
docker build -t magichour-ai .

# Run locally
docker run -p 8000:8000 \
  -e FAL_KEY=your_key \
  -e HF_TOKEN=your_token \
  -e PRODUCTION=true \
  magichour-ai
```

### Deploy to Cloud:

#### **AWS Elastic Beanstalk**:
```bash
eb init -p docker magichour-ai
eb create magichour-env
eb setenv FAL_KEY=xxx HF_TOKEN=xxx PRODUCTION=true
eb deploy
```

#### **Google Cloud Run**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/magichour
gcloud run deploy --image gcr.io/YOUR_PROJECT/magichour \
  --set-env-vars FAL_KEY=xxx,HF_TOKEN=xxx,PRODUCTION=true \
  --allow-unauthenticated
```

#### **DigitalOcean App Platform**:
- Upload Dockerfile via web interface
- Set environment variables
- Deploy

**Cost**: Varies by provider (~$5-20/month)

---

## Option 4: Vercel + Railway (Separate Frontend/Backend)

Split frontend and backend for better scaling.

### Backend (Railway):
1. Deploy only the backend to Railway:
   ```bash
   railway init
   railway up
   # Set env vars
   ```

2. Note the Railway backend URL (e.g., `https://magichour-api.railway.app`)

### Frontend (Vercel):
1. Build frontend with backend URL:
   ```bash
   cd magicHourApp
   echo "VITE_API_URL=https://magichour-api.railway.app" > .env.production
   npm run build
   ```

2. Deploy to Vercel:
   ```bash
   npm install -g vercel
   vercel --prod
   ```

**Cost**: Both have free tiers

---

## Option 5: Fly.io (Global Edge Network)

Fly.io deploys apps globally with low latency.

### Steps:

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login and launch**:
   ```bash
   fly auth login
   fly launch
   ```

3. **Set secrets**:
   ```bash
   fly secrets set FAL_KEY=your_key HF_TOKEN=your_token PRODUCTION=true
   ```

4. **Deploy**:
   ```bash
   fly deploy
   ```

**Cost**: Free tier includes 3 VMs

---

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `FAL_KEY` | ✅ Yes | Fal.ai API key | `xxxxxxx-xxxx-xxxx` |
| `HF_TOKEN` | ✅ Yes | Hugging Face token | `hf_xxxxxxx` |
| `PRODUCTION` | ✅ Yes | Enable production mode | `true` |
| `PORT` | ⚠️ Auto | Port (auto-set by most platforms) | `8000` |
| `VITE_API_URL` | ❌ No | Override API URL (frontend) | `https://api.example.com` |
| `FRONTEND_URL` | ❌ No | CORS allowed domain | `https://myapp.com` |

---

## Production Checklist

Before deploying:

- [ ] Set `PRODUCTION=true` environment variable
- [ ] Add `FAL_KEY` and `HF_TOKEN` secrets
- [ ] Build frontend: `cd magicHourApp && npm run build`
- [ ] Test locally with production mode:
  ```bash
  PRODUCTION=true python magicHourApp/api_server.py
  ```
- [ ] Ensure `.env` files are in `.gitignore` (never commit secrets!)
- [ ] Update CORS origins if using custom domain

---

## Monitoring & Troubleshooting

### Check Health:
```bash
curl https://your-app-url.com/api/health
```

### Common Issues:

1. **"Frontend not built" error**:
   - Run `npm run build` in `magicHourApp/`
   - Ensure `dist/` folder exists

2. **CORS errors**:
   - Set `FRONTEND_URL` environment variable
   - Or allow all origins with `FRONTEND_URL=*` (not recommended for production)

3. **API key errors**:
   - Verify `FAL_KEY` and `HF_TOKEN` are set correctly
   - Check platform-specific secrets/env var sections

4. **Build failures**:
   - Ensure Node.js 18+ and Python 3.11+ are specified
   - Check platform build logs for specific errors

---

## Cost Comparison

| Platform | Free Tier | Paid Starting | Best For |
|----------|-----------|---------------|----------|
| **Render** | 750 hrs/month | $7/month | Beginners |
| **Railway** | $5 credit/month | Pay-as-you-go | Fast deploys |
| **Vercel** | Unlimited | $20/month | Frontend |
| **Fly.io** | 3 VMs free | $5/month | Global edge |
| **DigitalOcean** | $200 credit (60 days) | $5/month | Control |

---

## Security Best Practices

1. **Never commit secrets** to Git:
   ```bash
   echo ".env" >> .gitignore
   echo "*.key" >> .gitignore
   ```

2. **Use environment variables** for all sensitive data

3. **Enable HTTPS** (most platforms do this automatically)

4. **Restrict CORS** to specific domains in production

5. **Monitor API usage** to prevent unexpected charges

---

## Support

For deployment issues:
- Check platform-specific documentation
- Review application logs via platform dashboard
- Test locally with `PRODUCTION=true` first

For API/code issues:
- Check `/api/health` endpoint
- Review server logs for errors
- Ensure all dependencies are installed
