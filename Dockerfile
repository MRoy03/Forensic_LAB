# Combined FastAPI + Streamlit image for a single free host -- Render's
# free tier builds straight from this Dockerfile (Hugging Face Spaces'
# Docker SDK now requires a paid PRO plan as of mid-2026, so that's no
# longer a free option; Render's free web-service tier still is). Runs
# the backend on 8000 internally, exposes Streamlit on whatever $PORT the
# host assigns at runtime (see start.sh) -- the frontend talks to the
# backend over localhost inside this one container, via API_BASE_URL set
# below. See DEPLOYMENT.md for the split-host alternative (separate free
# backend + Streamlit Community Cloud) and why GitHub Pages/Vercel don't
# fit this app's architecture.

FROM python:3.11-slim

# opencv-python-headless still needs these system libs on a slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL=http://localhost:8000
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

RUN chmod +x start.sh
CMD ["./start.sh"]
