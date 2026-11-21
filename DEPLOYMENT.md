# Deploying ForensicAI Lab

## Why not GitHub Pages or Vercel

Both were asked about, so this is worth being direct about: **neither can run this app**, regardless of configuration.

- **GitHub Pages** only serves static files (HTML/CSS/JS). It cannot run a Python process at all -- no FastAPI backend, no Streamlit frontend.
- **Vercel** runs short-lived serverless functions, not a persistent server. Streamlit needs a long-running process holding an open WebSocket connection to the browser, and the FastAPI backend needs to stay alive to keep the ONNX model loaded in memory. Neither fits a serverless function's stateless, time-limited execution model.

## Why not Hugging Face Spaces (as of mid-2026)

This was the original recommendation here, but Hugging Face changed its policy in **July 2026**: creating a **Docker** Space (or a CPU **Gradio** Space) now requires a **PRO subscription ($9/month)**. Free accounts are limited to **Static** Spaces (no Python at all) or up to 2 Gradio Spaces on a shared, GPU-oriented **ZeroGPU** pool -- neither fits a persistent FastAPI+Streamlit app. If you ever do get a PRO account for other reasons, Spaces' Docker SDK would still work fine with the files below; just don't expect it free.

## Memory budget -- read this before picking an option below

Phase 1 added three more models (image ensemble, audio deepfake detector, RAG embedder) on top of the original custom model. Measured directly on real hardware, not estimated:

| Model | On disk |
|---|---|
| Custom image/video model | 15.0 MB |
| Ensemble second opinion (detectra-v1, fp16) | 41.7 MB |
| RAG embedder (MiniLM) | 86.2 MB |
| Audio deepfake detector (wav2vec2, **int8-quantized** -- see `scripts/quantize_audio_model.py`) | 116.4 MB (was 361.0 MB fp32) |

With all four loaded in one process, the backend alone measured **~466MB working set / ~622MB private bytes** -- before Streamlit (which runs in the *same container* in the combined `Dockerfile`/`start.sh`) adds its own ~100-150MB on top. Render's free tier hard-caps at **512MB RAM**. Quantizing the audio model (by far the largest piece) cut its size 3x and reduced committed memory substantially, but barely moved the *resident* working-set number that most affects an OOM kill -- so **the single combined container is not reliably safe on Render's free tier**, even quantized. This isn't a theoretical concern; it's a real measurement on this project's actual model files.

Three ways to deal with that, in the order it's worth trying them:

## Option A: Oracle Cloud Always Free VM (recommended if you have this account)

An Always Free Ampere A1 VM gives you a real, persistent VM -- not a request-scoped container -- with (as of a 2026 Oracle policy change you should know about: the old "4 OCPU / 24GB forever" figure is **stale**, it's now **2 OCPU / 12GB**) still roughly **24x** the RAM Render's free tier gives you. At that size, none of the splitting below is necessary -- the existing combined `Dockerfile` runs everything, unmodified, with huge headroom. The tradeoffs are entirely on the setup side: Oracle requires a card for identity verification at signup (not charged for Always Free usage), and claiming an A1 instance is well known to sometimes hit "out of host capacity" errors in popular regions -- if that happens, retrying in a different region or at a different time of day is the usual workaround, not a sign anything is misconfigured.

**The pool, concretely**: the whole Always Free Ampere A1 allocation is 2 OCPUs / 12GB *total per tenancy*, not per instance -- Oracle's own docs give the example "one instance with 2 OCPUs" **or** "two instances with 1 OCPU each" (the sum across every A1 instance you run can't exceed 2/12). If another VM already claims the full pool, there's nothing left for a second A1 instance until that one is resized down. Decide this *before* starting the wizard below.

**Manual steps** (since I can't act on your cloud console myself):

1. **Create Compute Instance**: OCI Console -> hamburger menu -> Compute -> Instances -> Create Instance.
2. **Name** it (e.g. `forensicai-lab`), pick a Compartment.
3. **Image and shape**: click "Edit" next to Image, choose **Canonical Ubuntu** 22.04 or 24.04 (the "Always Free Eligible" / aarch64 build). Click "Edit" next to Shape, select **Ampere** -> **VM.Standard.A1.Flex**, then set the **OCPU count** and **Memory (GB)** fields explicitly -- this is where you make the pool decision above real: enter `2` / `12` if this VM gets the whole pool, or e.g. `1` / `6` if you're splitting it with another instance.
4. **Networking**: if you don't have a VCN yet, leave "Create new virtual cloud network" selected -- it auto-creates a VCN with a public subnet, an internet gateway, and a default security list together, which is what gets you a working public IP with no extra setup. Under the subnet section, confirm **"Assign a public IPv4 address"** is checked -- it's on by default for a public subnet, but worth confirming since without it the instance isn't reachable from outside OCI at all.
5. **SSH keys**: choose **"Generate a key pair for me"** and click **"Save private key"** immediately -- Oracle does not let you download it again later. Move it somewhere safe and lock down its permissions (`chmod 400 your-key.pem` on macOS/Linux/WSL; on native Windows, `icacls` or just use it from WSL/Git Bash).
6. **Boot volume**: the default 50GB is plenty for this app; Always Free gives you 200GB total boot+block storage across your whole tenancy, so a 50GB boot volume here still leaves room for other instances.
7. Click **Create**. If you get an "Out of host capacity" error for the A1 shape, that's a known, common occurrence for this specific free shape in busy regions -- retrying shortly after, or trying a different Availability Domain if your region has more than one, is the normal workaround, not a sign of misconfiguration.
8. Once it's **Running**, note the **Public IP** shown on the instance's detail page.
9. **Open the port in OCI's firewall**: go to the instance's attached **Subnet** -> its **Security List** -> **Add Ingress Rules**: Source CIDR `0.0.0.0/0`, IP Protocol `TCP`, Destination Port Range `8501` (or whichever port you plan to expose).
10. **SSH in**: `ssh -i /path/to/your-key.pem ubuntu@<public_ip>` (Ubuntu images on OCI use the `ubuntu` username).
11. **Open the port in the VM's own firewall too -- this is the step people most often miss.** Oracle's official Ubuntu images ship with iptables rules that block everything except SSH by default, *independently* of the Security List rule you just added in step 9 -- both have to allow the port, or it's still unreachable even though the OCI console shows it open:
    ```bash
    sudo iptables -L INPUT -n --line-numbers   # find the line number of the final REJECT rule
    sudo iptables -I INPUT <that-line-number> -m state --state NEW -p tcp --dport 8501 -j ACCEPT
    sudo netfilter-persistent save             # persist across reboots
    ```
12. Install Docker:
    ```bash
    sudo apt-get update && sudo apt-get install -y docker.io git
    sudo systemctl enable --now docker
    sudo usermod -aG docker $USER   # log out/in once for this to take effect
    ```
13. Clone this repo (code only -- the large model files are gitignored on purpose):
    ```bash
    git clone <your-repo-url> forensicai-lab && cd forensicai-lab
    ```
14. Copy the model files up from your local machine (they're not in git). From your **local** machine, not the VM:
    ```bash
    scp -i /path/to/your-key.pem -r models/image_model.onnx models/ensemble models/audio models/rag ubuntu@<public_ip>:~/forensicai-lab/models/
    ```
    (`models/pending/` -- Organika/sdxl-detector's unconverted weights -- doesn't need to go; it's not wired up yet.)
15. Build and run, reusing the existing combined `Dockerfile` unmodified -- no `ENABLE_AUDIO`/`ENABLE_CORE` split needed here, there's room for everything:
    ```bash
    docker build -t forensicai-lab .
    docker run -d --restart unless-stopped -p 8501:7860 --name forensicai forensicai-lab
    ```
16. Visit `http://<public_ip>:8501`. `--restart unless-stopped` plus Docker's own systemd service (already enabled in step 12) means it comes back up automatically after a VM reboot.
17. Optional, once this is working: put Caddy or nginx in front for a real domain + free HTTPS (Let's Encrypt) instead of hitting the bare IP:port -- skip this for a personal project unless you want it.

No sleep, no cold starts, no RAM juggling -- this is the only option here that makes the memory budget above a non-issue.

## Option B: Render, split into two services

Use this if Oracle sign-up/capacity doesn't work out, or you'd rather stay on a PaaS than manage a VM. Isolates the audio model (the one dominating the memory budget) into its own free service so neither competes with the other for one container's RAM.

I added `app/main.py`/`app/routes.py` support for this already: two env vars, `ENABLE_CORE` and `ENABLE_AUDIO` (both default `true`), control which routers a given process actually registers -- and I verified both split instances independently via live local runs (`/health` correctly reports only its own enabled subsystems, and the disabled routes 404 rather than erroring).

**Manual steps:**
1. **Core service** (image + video + ensemble + RAG + Streamlit): same as the old single-service setup -- new Render Web Service, Docker environment, root `Dockerfile`, Free instance type. In its **Environment** settings, add `ENABLE_AUDIO=false`. Force-add `models/image_model.onnx`, `models/ensemble/`, and `models/rag/` (not `models/audio/`) to a deploy branch Render builds from -- same gitignore caveat as before.
2. **Audio service**: a *second*, separate Render Web Service, same repo, but pointed at `Dockerfile.audio` instead of the root `Dockerfile` (Render's dashboard lets you set a custom Dockerfile path under Settings -> Build). Force-add only `models/audio/` to the deploy branch. No env vars needed -- `Dockerfile.audio` already sets `ENABLE_CORE=false`/`ENABLE_AUDIO=true`.
3. Note the audio service's URL once deployed (e.g. `https://forensicai-audio.onrender.com`). In the **core** service's environment, add `AUDIO_API_BASE_URL=<that URL>` so the frontend's Audio Detection page calls the right place (it already reads this var, falling back to `API_BASE_URL` when unset -- see `frontend/streamlit_app.py`).

Measured (two separate local processes, same models, same machine): core-only landed at ~343MB working set / ~498MB private, audio-only at ~313MB/~468MB -- both comfortably clear of 512MB **on their own**. The core service still needs Streamlit added on top in the real deployed container though, which this measurement doesn't include, so it's an improvement with a real margin, not an ironclad guarantee the way Option A is.

Tradeoff: Render's 750 free instance-hours/month is **shared per account, not per service** (confirmed from Render's own docs) -- running two services with a keep-alive ping on both will burn that shared budget roughly twice as fast as one service did.

## Option C: split hosts -- Streamlit Community Cloud (frontend) + Render (backend only)

Use this if you'd rather use Streamlit's own official free hosting for the frontend specifically. Note Streamlit Community Cloud's free tier allows **one private app at a time** and sleeps after **12 hours idle**.

**Manual steps:**
1. **Backend**: same Render signup as Option B, but create the Web Service pointing only at the backend -- either reuse the same `Dockerfile` (it still works, Streamlit just won't receive any traffic if you don't call it) or simplify by letting Render auto-detect Python and run `uvicorn app.main:app --host 0.0.0.0 --port $PORT` directly as the start command, skipping Docker entirely. Same model-file gitignore caveat applies. Note the backend's public URL once deployed (e.g. `https://forensicai-backend.onrender.com`).
2. **Frontend**: go to share.streamlit.io, sign in with GitHub, and deploy `frontend/streamlit_app.py` from this repo directly.
3. In the Streamlit Cloud app's settings, set a **secret/environment variable** `API_BASE_URL` to the backend's public URL from step 1 (the frontend code already reads this via `os.environ.get("API_BASE_URL", ...)`, falling back to `config/settings.yaml`'s local default otherwise).

Two moving pieces instead of one, each on the platform built for it -- worth it mainly if you specifically want Streamlit's own branded hosting for the frontend.

## Local testing before deploying

Build and run the combined image locally first, to catch anything before pushing (this is what Option A's Oracle VM runs as-is):

```bash
docker build -t forensicai-lab .
docker run -p 7860:7860 forensicai-lab
```

Then open `http://localhost:7860` (no `$PORT` env var set locally, so `start.sh` falls back to 7860). If a model fails to load, check the container logs for an `ERROR:services.detection.loader`-style message -- it means that model file wasn't present in the build context (the `.dockerignore` does *not* exclude any `models/**/*.onnx`/`*.safetensors`, so if the files exist on disk when you run `docker build`, they get included).

If you're testing Option B's split instead, build/run the audio-only image the same way, pointing at the other Dockerfile:

```bash
docker build -f Dockerfile.audio -t forensicai-audio .
docker run -p 7861:7860 forensicai-audio
```

I verified the underlying env-var behavior directly (two local `uvicorn` processes, one with `ENABLE_AUDIO=false`, one with `ENABLE_CORE=false`) rather than through Docker itself -- Docker wasn't available in the environment I built this in, so do a quick sanity check here before trusting `Dockerfile.audio` in production.
