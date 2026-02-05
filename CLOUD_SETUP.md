# Setting Up the Basketball Action Recognition Server on the Cloud

This guide covers deploying the FastAPI inference server (`ai_server.py`) so your Hoops Clips backend can call it via `AI_WORKER_URL`.

## Prerequisites

1. **Model checkpoint**  
   The server expects the R(2+1)D checkpoint at  
   `model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt`  
   (or set `CHECKPOINT` to the path/URL of your `.pt` file). The checkpoint is not in the repo; you must add it to the image or mount it at runtime.

2. **Docker**  
   Install [Docker](https://docs.docker.com/get-docker/) so you can build and run the container locally and push to a registry.

---

## Option 1: Google Cloud Run

Good for serverless scaling; CPU-only is fine (GPU is optional and more expensive).

### 1. Build with checkpoint in image (recommended)

```bash
cd /Users/hanfei/Basketball_action_recoginition_sever

# Ensure checkpoint exists locally
ls model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt

# Uncomment the COPY line in Dockerfile for model_checkpoints, then:
docker build -t gcr.io/YOUR_PROJECT_ID/basketball-ai .
docker push gcr.io/YOUR_PROJECT_ID/basketball-ai
```

Replace `YOUR_PROJECT_ID` with your Google Cloud project ID.

### 2. Deploy to Cloud Run

```bash
gcloud run deploy basketball-ai \
  --image gcr.io/YOUR_PROJECT_ID/basketball-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars "CHECKPOINT=/app/model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt"
```

If the checkpoint is in the image at that path, you can omit the env var. Increase `--memory` if the model fails to load (e.g. 4Gi). `--timeout 300` allows long-running inference.

### 3. Get the URL

After deploy, Cloud Run prints the service URL, e.g.  
`https://basketball-ai-xxxxx-uc.a.run.app`.  
Your Hoops Clips API base is:

```text
AI_WORKER_URL=https://basketball-ai-xxxxx-uc.a.run.app/api/ai
AI_WORKER_VERSION=basketball-r2plus1d-13
```

---

## Option 2: Railway / Render / Fly.io

Same Docker image works on any container host.

### Railway

1. Connect your GitHub repo (or push the image to a registry Railway supports).
2. Set root directory to this repo (or use a Dockerfile path).
3. Add the checkpoint: either include it in the repo in `model_checkpoints/` (and uncomment the COPY in the Dockerfile) or use a volume/build secret if the platform supports it.
4. Set env: `CHECKPOINT=/app/model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt` if you baked it into the image.
5. Expose port 8787 (or set `PORT` to what Railway assigns).
6. Copy the public URL and set in Hoops Clips:  
   `AI_WORKER_URL=https://your-app.railway.app/api/ai`

### Render

1. New → Web Service → Connect repo.
2. Build: `docker build -t basketball-ai .` (or use Render’s Dockerfile detection).
3. Start: same as Dockerfile `CMD` (uvicorn on `PORT`).
4. Add env var `CHECKPOINT` if needed. Ensure the checkpoint is in the image or on a persistent disk Render provides.
5. Use the generated URL: `AI_WORKER_URL=https://your-service.onrender.com/api/ai`

### Fly.io

```bash
cd /Users/hanfei/Basketball_action_recoginition_sever
fly launch
# Follow prompts; then scale memory if needed:
fly scale memory 2048
fly deploy
```

Set `CHECKPOINT` in `fly.toml` or via `fly secrets set CHECKPOINT=/app/model_checkpoints/...`.  
Then: `AI_WORKER_URL=https://your-app.fly.dev/api/ai`.

---

## Option 3: VM (GCP, AWS, or any VPS)

1. **Build and run with Docker on the VM**

   On the VM (or from your machine if you push to a registry):

   ```bash
   cd /Users/hanfei/Basketball_action_recoginition_sever
   # Copy checkpoint onto the VM first, e.g. into ./model_checkpoints/...
   docker build -t basketball-ai .
   docker run -d -p 8787:8787 \
     -v /path/on/host/model_checkpoints:/app/model_checkpoints \
     -e CHECKPOINT=/app/model_checkpoints/r2plus1d_clean/r2plus1d_multiclass_13_0.0001.pt \
     --name basketball-ai basketball-ai
   ```

2. **Or run without Docker**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements-cloud.txt
   export CHECKPOINT=/path/to/r2plus1d_multiclass_13_0.0001.pt
   uvicorn ai_server:app --host 0.0.0.0 --port 8787
   ```

3. Open port 8787 in the firewall and (if needed) put a reverse proxy (e.g. nginx) in front with HTTPS.  
   Then set:  
   `AI_WORKER_URL=https://your-vm-domain-or-ip:8787/api/ai`

---

## Wiring the Hoops Clips Backend

In the project that uses this server (e.g. `rork-hoops-clips-cloud-clone-clone-clone-132`):

1. Copy `env.backend.example` to `env.backend` (if you haven’t already).
2. Set:

   ```bash
   AI_WORKER_URL=https://YOUR_DEPLOYED_SERVER_URL/api/ai
   AI_WORKER_VERSION=basketball-r2plus1d-13
   ```

   Use the full base URL of the basketball server **without** a trailing slash; the backend appends `/analyze` and `/result/:jobId`.  
   Example: `AI_WORKER_URL=https://basketball-ai-xxxxx-uc.a.run.app/api/ai`.

3. Restart your backend. It will send analyze requests to the cloud worker and poll for results.

---

## Checklist

- [ ] Checkpoint file is available in the image or on a mounted volume, and `CHECKPOINT` points to it.
- [ ] `dataset/labels_dict.json` is in the image (included by the Dockerfile).
- [ ] Port 8787 (or `PORT`) is exposed and reachable (and HTTPS in front if required).
- [ ] Memory ≥ 2GB for the container (PyTorch + model); increase if you see OOM.
- [ ] Request timeout on the Hoops Clips backend is sufficient (e.g. `AI_WORKER_TIMEOUT_MS=120000`).
- [ ] If the basketball server is public, consider adding auth or keeping it in a private VPC and only allowing your backend to call it.
