#!/usr/bin/env bash
# Entrypoint for the combined Docker image (see Dockerfile/DEPLOYMENT.md).
# Starts the FastAPI backend in the background, waits for it to answer
# /health, then runs Streamlit in the foreground bound to 0.0.0.0:$PORT --
# binding to localhost only would be unreachable from outside the container.
#
# $PORT is read dynamically, not hardcoded: Render (and most host-assigns-
# the-port platforms) injects its own $PORT at runtime and routes external
# traffic there -- a fixed port would silently not receive any traffic.
# Falls back to 7860 (Hugging Face Spaces' Docker SDK convention) if $PORT
# isn't set, e.g. for local `docker run` testing.
set -e

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend to become healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is up."
        break
    fi
    sleep 1
done

streamlit run frontend/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-7860}" \
    --server.headless true

kill "$BACKEND_PID" 2>/dev/null || true
