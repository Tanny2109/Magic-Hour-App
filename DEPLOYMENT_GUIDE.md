# Magic Hour AI - Render Deployment Guide (LangGraph)

## 🚀 Quick Deploy

This guide deploys the **LangGraph implementation** (NOT smolagents) to Render.

### Prerequisites

1. Render account (free tier works)
2. GitHub repo with your code
3. FAL API key

---

## Option 1: Docker Deployment (Recommended)

Uses the provided `Dockerfile` for consistent builds.

### Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy LangGraph to Render"
   git push origin main
   ```

2. **Create New Web Service on Render**
   - Go to https://render.com/dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repo

3. **Configure Service**
   - **Name**: `magichour-langgraph`
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Context**: `.`
   - **Instance Type**: Free

4. **Set Environment Variables**
   ```
   FAL_KEY=your_fal_api_key_here
   FAL_MODEL_NAME=google/gemini-2.5-flash
   LLM_TEMPERATURE=0.7
   LLM_MAX_TOKENS=4096
   PRODUCTION=true
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for build
   - Your app will be at: `https://magichour-langgraph.onrender.com`

### Using render.yaml (Blueprint)

Alternatively, use the provided `render.yaml`:

```bash
# Render will auto-detect render.yaml and use it
git push origin main
```

Then on Render Dashboard:
- Click "New +" → "Blueprint"
- Connect repo
- Select `render.yaml`
- Add `FAL_KEY` secret

---

## Option 2: Native Build (Faster, Less RAM)

Uses Python + Node directly without Docker.

### Steps:

1. **Rename config file**
   ```bash
   mv render.yaml render.docker.yaml
   mv render.native.yaml render.yaml
   ```

2. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Use native build for Render"
   git push origin main
   ```

3. **Follow steps 2-5 from Option 1**

---

## Verify Deployment

### Check Health
```bash
curl https://your-app.onrender.com/api/health
```

Expected response:
```json
{
  "status": "ok",
  "agent_ready": true,
  "backend": "langgraph"
}
```

### Test Chat
```bash
curl -X POST https://your-app.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "generate an image of a cat",
    "settings": {"mode": "fast"},
    "history": []
  }'
```

---

## Frontend Access

The frontend is served by the backend at the root URL:
```
https://your-app.onrender.com/
```

Render automatically serves:
- `/` → `index.html` (React app)
- `/api/*` → Backend API
- `/api/media` → Generated images/videos

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FAL_KEY` | **Yes** | - | Your fal.ai API key |
| `FAL_MODEL_NAME` | No | `google/gemini-2.5-flash` | LLM model to use |
| `LLM_TEMPERATURE` | No | `0.7` | Temperature for LLM (0-1) |
| `LLM_MAX_TOKENS` | No | `4096` | Max tokens for LLM response |
| `PRODUCTION` | No | `false` | Set to `true` for production |
| `PORT` | No | `8000` | Server port (Render sets this) |

---

## Troubleshooting

### Build Fails

**Issue**: Dockerfile fails with "No such file or directory"
- **Fix**: Make sure you're pushing from the `Magic Hour ML role` directory
- Check `.dockerignore` isn't excluding required files

**Issue**: npm install fails in Docker
- **Fix**: Check `magicHourApp/package.json` exists
- Verify Node version (using 18)

### Deployment Fails

**Issue**: "Port already in use"
- **Fix**: Render sets `PORT` env var automatically, no action needed

**Issue**: "Agent not initialized"
- **Fix**: Check `FAL_KEY` is set in environment variables
- Verify `mh_langgraph_workflow/` is in the repo

### Runtime Issues

**Issue**: Images not displaying
- **Fix**: Check CORS settings in `api_server.py`
- Verify `/api/media` endpoint works

**Issue**: Slow visual analysis
- **Fix**: This is expected for first run after sleep (free tier)
- Consider upgrading to paid tier for always-on

---

## Free Tier Limitations

Render free tier has:
- ✅ 750 hours/month
- ⚠️ Spins down after 15 min of inactivity
- ⚠️ Cold starts take 30-60 seconds
- ⚠️ 512 MB RAM (might need upgrade for large images)

**Recommendation**: Upgrade to Starter ($7/mo) for:
- Always-on (no cold starts)
- 2 GB RAM
- Better performance

---

## Architecture

```
User → Render (HTTPS) → FastAPI Server → LangGraph Agent
                           ↓
                      fal.ai API (images/video)
                           ↓
                      Response with media URLs
```

**Key Components:**
- Frontend: React (Vite) - Built into Docker image
- Backend: FastAPI + LangGraph
- Agent: `ContentAgent` from `mh_langgraph_workflow`
- Tools: `generate_images`, `edit_images`, `generate_video`
- LLM: fal.ai OpenRouter with Gemini 2.5 Flash

---

## Monitoring

### Logs
View logs on Render dashboard:
- Go to your service
- Click "Logs" tab
- Look for:
  - `✅ Agent initialized`
  - `DEBUG -` statements
  - API requests

### Performance
Monitor in Render dashboard:
- Response times
- Memory usage
- Error rates

---

## Updating Deployment

### Code Changes
```bash
git add .
git commit -m "Update: description"
git push origin main
```

Render auto-deploys on push to main.

### Environment Variables
- Go to Render dashboard
- Select your service
- "Environment" tab
- Update values
- Click "Save Changes" (triggers redeploy)

---

## Rollback

If deployment fails:
1. Go to Render dashboard
2. Click "Events" tab
3. Find last successful deploy
4. Click "Redeploy"

---

## Cost Optimization

### Tips for Free Tier
1. Use `mode: "fast"` for image generation (Prodia)
2. Set `LLM_TEMPERATURE` lower (0.3-0.5) for cheaper LLM calls
3. Limit visual analysis (agent does this automatically)

### When to Upgrade
- If you hit 750 hours/month
- Need instant responses (no cold start)
- Generating many high-res images (need more RAM)

---

## Support

- **Render Issues**: https://render.com/docs
- **fal.ai Issues**: https://fal.ai/docs
- **LangGraph Issues**: https://langchain-ai.github.io/langgraph/

---

## What's Different from Smolagents?

| Feature | Smolagents | LangGraph |
|---------|-----------|-----------|
| Framework | HuggingFace Smolagents | LangChain/LangGraph |
| Reasoning | Built-in CoT | Custom `ReasoningMessage` |
| Context | Limited | Full conversation history |
| Visual Analysis | Manual | Automatic (on-demand) |
| UI Integration | Basic | Collapsible dropdowns |
| Performance | Slower (always analyzes) | Faster (smart detection) |
| Deployment | ✅ Already deployed | 🆕 New deployment |

---

## Success Checklist

- [ ] Pushed code to GitHub
- [ ] Created Render web service
- [ ] Set `FAL_KEY` environment variable
- [ ] Deployment succeeded (check logs)
- [ ] Health endpoint returns `{"status": "ok"}`
- [ ] Frontend loads at root URL
- [ ] Can generate images via chat
- [ ] Reasoning displays in collapsible UI
- [ ] Visual analysis works for edit requests

🎉 **You're live!**
