# Quick Deploy Guide - Magic Hour AI

The fastest ways to get your app online and share it with others.

---

## 🚀 5-Minute Deploy (Render)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push
   ```

2. **Go to [render.com](https://render.com)** and sign up

3. **Click "New +" → "Web Service"**

4. **Connect your GitHub repo**

5. **Fill in these fields**:
   - **Name**: `magichour`
   - **Runtime**: Python 3
   - **Build Command**:
     ```
     cd mh_agentic_workflow && pip install -r requirements.txt && cd ../magicHourApp && pip install -r requirements.txt && npm install && npm run build
     ```
   - **Start Command**:
     ```
     cd magicHourApp && python api_server.py
     ```

6. **Add Environment Variables**:
   - `FAL_KEY` = your fal.ai key
   - `HF_TOKEN` = your hugging face token
   - `PRODUCTION` = `true`

7. **Click "Create Web Service"** ✨

Your app will be live at `https://magichour.onrender.com` in ~5 minutes!

---

## 🎯 Alternative: Railway (Even Faster)

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Railway auto-detects everything!
5. Add environment variables:
   - `FAL_KEY`
   - `HF_TOKEN`
   - `PRODUCTION=true`
6. Done! 🎉

---

## 🐳 Docker (Self-Hosted)

If you have a server (VPS, AWS, etc.):

```bash
# Build
docker build -t magichour .

# Run
docker run -d -p 8000:8000 \
  -e FAL_KEY=your_key \
  -e HF_TOKEN=your_token \
  -e PRODUCTION=true \
  --name magichour \
  magichour

# Your app is now at http://your-server-ip:8000
```

---

## 📝 What You Need

Before deploying anywhere, make sure you have:

✅ **FAL_KEY** - Get it from [fal.ai/dashboard](https://fal.ai/dashboard)
✅ **HF_TOKEN** - Get it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
✅ **GitHub account** (for Render/Railway)

---

## 💡 Tips

- **Free Tier**: Render gives 750 hours/month free (sleeps after 15min inactivity)
- **Always On**: Upgrade to Render's $7/month plan for 24/7 uptime
- **Custom Domain**: Add your own domain in the platform's settings
- **Monitoring**: Check `/api/health` to see if your app is running

---

## 🔧 Troubleshooting

**Build failed?**
- Check you have both `requirements.txt` files
- Ensure `package.json` is in `magicHourApp/`

**App not loading?**
- Verify environment variables are set
- Check the build logs in your platform's dashboard
- Try the health endpoint: `https://your-app.com/api/health`

**Need help?**
- See full [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed guides
- Check your platform's documentation
- Review application logs

---

## 🎉 Share Your App!

Once deployed, share the URL with anyone:
- `https://your-app-name.onrender.com` (Render)
- `https://your-app-name.up.railway.app` (Railway)

They can use it immediately - no installation needed!
