# Free Deployment Guide: RAG Chatbot & Multi-Agent System

This guide explains how to deploy your RAG Document Chatbot and Multi-Task Agent system completely free of charge across GitHub Pages, Hugging Face Spaces, Render, or Google Cloud (Cloud Run & Vertex AI).

---

## Architecture Summary
- **Frontend UI (`index.html`)**: Self-contained client-side web application with built-in hybrid vector indexing and multi-agent simulation.
- **Backend API (`main.py`)**: FastAPI server managing ChromaDB vector storage and LangGraph-style agent orchestration.

---

## 1. Deploy Frontend Free to GitHub Pages (100% Free Forever)

Because `index.html` has a complete client-side vector search and autonomous multi-agent simulation built in, you can host it on GitHub Pages immediately with zero server costs:

### Step-by-Step:
1. **Create a GitHub Repository**:
   - Go to [GitHub](https://github.com) and click **New Repository** (e.g., `rag-agent-chatbot`).
   - Keep it **Public** (required for free GitHub Pages).
2. **Push the Files**:
   - Initialize git in this directory and push:
     ```bash
     git init
     git add .
     git commit -m "Initial commit of RAG Agent Chatbot"
     git branch -M main
     git remote add origin https://github.com/<YOUR-USERNAME>/rag-agent-chatbot.git
     git push -u origin main
     ```
3. **Enable GitHub Pages**:
   - In your repository, click **Settings** → **Pages** (under Code and automation).
   - Under **Build and deployment** → **Source**, select **GitHub Actions** (the included `.github/workflows/deploy.yml` will automatically build and deploy your site on push), or select **Deploy from a branch** → choose `main` and root `/`.
4. **Access Your Live App**:
   - Your chatbot will be live at: `https://<YOUR-USERNAME>.github.io/rag-agent-chatbot/`

---

## 2. Deploy Backend Free to Hugging Face Spaces (100% Free Docker Hosting)

Hugging Face Spaces provides permanent free 2 vCPU / 16 GB RAM Docker environments:

### Step-by-Step:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your space (e.g., `rag-agent-api`), select **Docker** as the SDK, and choose the **Blank / Custom** template.
3. Under Hardware, choose **CPU basic • 2 vCPU • 16GB • Free**.
4. Clone the space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/<YOUR-USERNAME>/rag-agent-api
   ```
5. Copy `Dockerfile`, `main.py`, `rag_engine.py`, `agents.py`, `models.py`, and `requirements.txt` into the space folder.
6. Commit and push:
   ```bash
   git add .
   git commit -m "Deploy FastAPI Multi-Agent RAG service"
   git push
   ```
7. Hugging Face builds the Docker container and exposes a public HTTPS URL (e.g., `https://<YOUR-USERNAME>-rag-agent-api.hf.space`).
8. Open your frontend `index.html` (or via GitHub Pages), go to the **Settings** tab, and enter your Hugging Face Space URL in the **Custom API Backend Endpoint** field!

---

## 3. Deploy to Google Cloud (Cloud Run & Vertex AI Free Tier)

Google Cloud offers an ongoing **Always Free Tier** for Cloud Run (2 million requests/month, 360,000 GB-seconds memory free) and free credits for Vertex AI / Gemini API:

### Step-by-Step:
1. **Install Google Cloud SDK** (`gcloud`) and log in:
   ```bash
   gcloud auth login
   gcloud config set project <YOUR-GCP-PROJECT-ID>
   ```
2. **Enable Cloud Run and Vertex AI APIs**:
   ```bash
   gcloud services enable run.googleapis.com aiplatform.googleapis.com artifactregistry.googleapis.com
   ```
3. **Deploy with 1-Command Source Deploy**:
   ```bash
   gcloud run deploy rag-agent-backend \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars="PORT=8080"
   ```
4. Cloud Run will build your container with Cloud Build and output a secure service URL:
   `https://rag-agent-backend-xxxx-uc.a.run.app`
5. **Connect Vertex AI**:
   - Set up application credentials or pass `GOOGLE_APPLICATION_CREDENTIALS` to query Gemini models on Vertex AI via `google-cloud-aiplatform`.

---

## 4. Alternative: Deploy to Render Free Tier

1. Sign up at [Render](https://render.com).
2. Click **New +** → **Web Service** and connect your GitHub repository.
3. Render automatically detects `render.yaml` or you can configure:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
4. Click **Deploy**. Your service will be live on `https://<your-app-name>.onrender.com`.

---

## Summary Recommendation
- For zero hassle and instantaneous access, use **GitHub Pages** for the UI.
- If you need persistent server-side vector storage and live LLM model calls, pair it with **Hugging Face Spaces** (100% free Docker container) or **Google Cloud Run** (free tier).
