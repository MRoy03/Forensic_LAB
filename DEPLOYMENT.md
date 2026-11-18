# Deploying ForensicAI Lab

## Why not GitHub Pages or Vercel

Both were asked about, so this is worth being direct about: **neither can run this app**, regardless of configuration.

- **GitHub Pages** only serves static files (HTML/CSS/JS). It cannot run a Python process at all -- no FastAPI backend, no Streamlit frontend.
- **Vercel** runs short-lived serverless functions, not a persistent server. Streamlit needs a long-running process holding an open WebSocket connection to the browser, and the FastAPI backend needs to stay alive to keep the ONNX model loaded in memory. Neither fits a serverless function's stateless, time-limited execution model.

## Why not Hugging Face Spaces (as of mid-2026)

This was the original recommendation here, but Hugging Face changed its policy in **July 2026**: creating a **Docker** Space (or a CPU **Gradio** Space) now requires a **PRO subscription ($9/month)**. Free accounts are limited to **Static** Spaces (no Python at all) or up to 2 Gradio Spaces on a shared, GPU-oriented **ZeroGPU** pool -- neither fits a persistent FastAPI+Streamlit app. If you ever do get a PRO account for other reasons, Spaces' Docker SDK would still work fine with the files below; just don't expect it free.

## Option A (recommended): Render, Docker-based, one free host for everything

Render's free web-service tier **does** build straight from a `Dockerfile` and is genuinely free (no card required) -- 750 instance-hours/month (enough for one always-on service), 512MB RAM, 0.1 CPU, sleeps after 15 minutes of inactivity with about a 1-minute cold-start wake. Runs the FastAPI backend and Streamlit frontend together in one container, using the `Dockerfile`/`start.sh` already in this repo.

**Manual steps:**
1. Create a free Render account at render.com (GitHub sign-in works, no card needed for the free tier).
2. **New** -> **Web Service** -> connect this GitHub repo.
3. Render auto-detects the `Dockerfile` -- confirm "Docker" is selected as the environment/runtime (it should be automatic).
4. Choose the **Free** instance type.
5. **Important**: `models/image_model.onnx` is gitignored in this dev repo (it's a large trained artifact, deliberately not committed here) -- but the deployed build needs it present to serve predictions. Either:
   - Temporarily commit it to a **separate deployment branch** Render builds from (`git checkout -b deploy && git add -f models/image_model.onnx && git commit -m "Add model for deployment" && git push origin deploy`, then point Render at the `deploy` branch), or
   - Use Render's **persistent disk** feature (paid add-on) or a startup download step in `start.sh` that fetches the model from a URL you control (e.g. a GitHub Release asset) if you'd rather not commit it anywhere.
6. Deploy. Render builds the image and starts `start.sh`, which launches the backend, waits for `/health`, then starts Streamlit bound to the `$PORT` Render assigns (not a fixed port -- `start.sh` already reads `$PORT` dynamically).
7. Your app is live at the `https://<your-service-name>.onrender.com` URL Render gives you.

Free, no card, single deployment -- the closest equivalent to the original HF Spaces plan.

## Option B: split hosts -- Streamlit Community Cloud (frontend) + Render (backend only)

Use this if you'd rather use Streamlit's own official free hosting for the frontend specifically. Note Streamlit Community Cloud's free tier allows **one private app at a time** and sleeps after **12 hours idle**.

**Manual steps:**
1. **Backend**: same Render signup as Option A, but create the Web Service pointing only at the backend -- either reuse the same `Dockerfile` (it still works, Streamlit just won't receive any traffic if you don't call it) or simplify by letting Render auto-detect Python and run `uvicorn app.main:app --host 0.0.0.0 --port $PORT` directly as the start command, skipping Docker entirely. Same model-file note as Option A applies. Note the backend's public URL once deployed (e.g. `https://forensicai-backend.onrender.com`).
2. **Frontend**: go to share.streamlit.io, sign in with GitHub, and deploy `frontend/streamlit_app.py` from this repo directly.
3. In the Streamlit Cloud app's settings, set a **secret/environment variable** `API_BASE_URL` to the backend's public URL from step 1 (the frontend code already reads this via `os.environ.get("API_BASE_URL", ...)`, falling back to `config/settings.yaml`'s local default otherwise).

Two moving pieces instead of one, each on the platform built for it -- worth it mainly if you specifically want Streamlit's own branded hosting for the frontend.

## Local testing before deploying

Build and run the combined Docker image locally first, to catch anything before pushing:

```bash
docker build -t forensicai-lab .
docker run -p 7860:7860 forensicai-lab
```

Then open `http://localhost:7860` (no `$PORT` env var set locally, so `start.sh` falls back to 7860). If the model fails to load, check the container logs for the same `ERROR:services.detection.loader` message you'd see locally -- it means `models/image_model.onnx` wasn't present in the build context (the `.dockerignore` does *not* exclude `models/*.onnx`, so if the file exists on disk when you run `docker build`, it gets included).
