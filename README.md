# FraudShield — Real-Time Payment Fraud Detection System

![CI Status](https://github.com/your-username/fraudshield/actions/workflows/ci.yml/badge.svg)

FraudShield is an end-to-end, production-grade payment fraud detection engine built with Gradient Boosted Trees (XGBoost), Deep Autoencoders (PyTorch), SHAP explainability, and a high-performance FastAPI inference microservice.

---

## Live Demo URLs
- **Web Dashboard**: `https://fraudshield-frontend.vercel.app` *(Placeholder — update after Vercel deployment)*
- **API Documentation**: `https://fraudshield-api.onrender.com/docs` *(Placeholder — update after Render deployment)*
- **API Health Check**: `https://fraudshield-api.onrender.com/health`

---

## Run with Docker (Recommended)

Run the entire stack (PostgreSQL + FastAPI + React Frontend + Nginx) with a single command from a clean clone.

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/fraudshield.git
cd fraudshield

# Copy safe default environment variables
cp .env.example .env
```

### 2. Build & Launch Full Stack
```bash
docker compose up --build
```

Services started:
- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)
- **PostgreSQL Database**: `localhost:5432` (persistent data stored in named volume `pgdata`)

### 3. Run the Demo Simulator in Docker
Stream simulated transactions from `data/sample/demo_transactions.csv` to the running API:

```bash
docker compose --profile demo up simulator
```

To stop all containers:
```bash
docker compose down
```

---

## Deployment Guide

### Architecture & Platform Recommendations
- **Backend API & PostgreSQL: [Render](https://render.com)**
  - *Why Render*: Free native PostgreSQL managed database and Docker web services defined declaratively via [`render.yaml`](render.yaml).
  - *Trade-off*: Free instances spin down after 15 minutes of inactivity (causing ~50s cold start on wake-up; `/health` polling keeps it warm).
  - *Alternative (Railway)*: Zero-cold-start performance, but uses an expiring credit model rather than a perpetual free tier.
- **Frontend Dashboard: [Vercel](https://vercel.com)**
  - *Why Vercel*: Global Edge CDN, instant Vite builds, zero-downtime deploys, and native SPA routing configured via [`frontend/vercel.json`](frontend/vercel.json).

### Beginner Step-by-Step Deployment Checklist

#### Step 1: Deploy Backend & Database on Render
1. Push your repository to GitHub.
2. Sign up / Log in to [Render](https://dashboard.render.com/) with GitHub.
3. Click **New +** -> **Blueprint**.
4. Connect your GitHub repository. Render will automatically detect [`render.yaml`](render.yaml).
5. Click **Apply Blueprint**. Render will provision:
   - A managed PostgreSQL instance (`fraudshield-postgres`).
   - The Dockerized API web service (`fraudshield-api`), automatically injecting `DATABASE_URL` and `PORT`.
6. Once deployed, copy your API URL (e.g., `https://fraudshield-api.onrender.com`).
7. Verify by opening `https://<YOUR-API-URL>/health` in your browser.

#### Step 2: Deploy Frontend on Vercel
1. Sign up / Log in to [Vercel](https://vercel.com/) with GitHub.
2. Click **Add New...** -> **Project** and select your GitHub repository.
3. In **Root Directory**, click edit and select `frontend`.
4. Under **Environment Variables**, add:
   - `VITE_API_URL`: `https://<YOUR-RENDER-API-URL>` (paste your Render API URL without trailing slash).
5. Click **Deploy**. Vercel will build and serve your React dashboard.
6. Once deployed, copy your Vercel URL (e.g., `https://fraudshield-frontend.vercel.app`).

#### Step 3: Link CORS in Render
1. Go back to your Render Dashboard -> `fraudshield-api` -> **Environment**.
2. Update the `CORS_ORIGINS` variable to include your Vercel URL:
   `https://fraudshield-frontend.vercel.app,http://localhost:3000,http://localhost:5173`
3. Click **Save Changes** (Render will automatically redeploy with CORS active).

---

## How to Run Locally (Without Docker)

### 1. Prerequisites and Installation
Ensure Python 3.11+ and Node 20+ are installed:

```bash
# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the API Server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Launch the Frontend
```bash
cd frontend
npm install
npm run dev
```
Dashboard: [http://localhost:5173](http://localhost:5173)

---

## Testing & Quality Assurance

```bash
# Python backend, ML & simulator tests
pytest -v

# Code quality and style linting
ruff check .

# Frontend test suite
cd frontend && npm run test && npm run lint && npm run build
```
