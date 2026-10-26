# FraudShield — Real-time Fraud Detection Dashboard

A high-performance React + TypeScript frontend dashboard for monitoring payment fraud predictions, real-time transaction streaming, and SHAP explainability insights.

---

## Features

- **Real-Time Stat Cards**: Live updates of total transactions scored, flagged count, flagged fraud rate (%), average latency, and 95th percentile latency (SLA monitoring).
- **Interactive Score Distribution**: Recharts score histogram bucketed from 0.0 to 1.0 with dynamic decision threshold reference line (`0.835`).
- **Live Transaction Stream Table**:
  - Live polling every 2 seconds with pause/resume toggle.
  - "All Transactions" vs "Flagged Only" instant filter.
  - Formatted currency amounts, timestamp, status badge, and click-to-copy transaction ID.
- **SHAP Explanation Drawer/Modal**: Click any transaction row to inspect top-5 SHAP feature contributions (positive values in red push toward fraud, negative in green push toward legitimate) and the complete 30-feature vector.
- **Interactive Transaction Tester**:
  - 1-click test buttons: "Load Legitimate Sample" and "Load Fraud Sample".
  - Instant inference through `POST /predict` with inline SHAP attribution chart and latency measurement.
- **Graceful Error Recovery**: Informative notice banner if the backend API goes offline with retry button.

---

## Tech Stack

- **React 19** + **TypeScript**
- **Vite** (Build tool & dev server)
- **Tailwind CSS v4** (Modern dark-mode design system)
- **Recharts** (SVG charts & data visualization)
- **Vitest** + **React Testing Library** (Unit & component testing)
- **ESLint** (Strict code quality)

---

## Getting Started

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Ensure `VITE_API_URL` points to your running FastAPI instance (defaults to `http://localhost:8000`).

### 3. Start Development Server

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Available Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start local Vite development server with Hot Module Replacement (HMR) |
| `npm run build` | TypeScript type-check and build production bundle into `dist/` |
| `npm run test` | Run all Vitest unit and component tests |
| `npm run lint` | Run ESLint across `src/` with zero-warning enforcement |
| `npm run preview` | Locally preview production build |

---

## Running with FastAPI Backend & Simulator

To experience live end-to-end streaming:

1. **Start FastAPI Backend**:
   ```bash
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. **Start Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
3. **Start Simulator Stream**:
   ```bash
   python -m simulator.stream --rate 5 --duration 60 --fraud-boost 0.2
   ```
Watch transactions stream live into the dashboard tables and charts in real time!
