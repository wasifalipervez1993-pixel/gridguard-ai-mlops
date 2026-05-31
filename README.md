# GridGuard AI — Industry-Grade MLOps Platform for Smart Energy Forecasting and Anomaly Detection

**Project theme:** Production MLOps system for electricity demand forecasting, energy anomaly detection, prediction logging, monitoring, dashboarding, and cloud/container deployment.

This is designed as an industry-grade end-to-end MLOps project using:

- GitHub + GitHub Actions
- Kaggle real-world dataset ingestion
- Full ML pipeline: ingestion, validation, preprocessing, feature engineering, training, evaluation, artifact versioning
- FastAPI model serving
- MongoDB Atlas prediction/feedback logging
- Docker and Docker Compose
- Hugging Face Spaces deployment option
- Kubernetes deployment manifests
- Prometheus metrics endpoint and Prometheus scraping
- Streamlit dashboard

---

## 1. Business Problem

Modern energy providers, smart-building operators, and grid managers need short-term electricity demand forecasts and anomaly alerts. A production system can help them:

1. Forecast near-term electricity load.
2. Detect abnormal consumption patterns.
3. Log predictions and real outcomes for future retraining.
4. Monitor API latency, request count, model errors, and prediction trends.
5. Provide an operational dashboard for business and engineering teams.

---

## 2. Dataset

Primary Kaggle dataset:

**Household Electric Power Consumption**  
Kaggle: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

The dataset contains measurements of electric power consumption in one household with a one-minute sampling rate over almost four years. This is sufficient for time-series forecasting and anomaly detection experiments.

---

## 3. Architecture

```text
Kaggle Dataset
     |
     v
Data Ingestion -> Validation -> Preprocessing -> Feature Engineering
     |                                      |
     v                                      v
Processed Parquet                    Train Forecast + Anomaly Models
                                            |
                                            v
                                     Versioned Artifacts
                                            |
                                            v
                         FastAPI Model Service + Prometheus Metrics
                                            |
                         +------------------+------------------+
                         |                                     |
                         v                                     v
                  MongoDB Atlas Logs                    Streamlit Dashboard
                                            |
                                            v
                                   Retraining Feedback Loop
```

---

## 4. Repository Structure

```text
gridguard-ai-mlops/
├── src/
│   ├── gridguard/              # Core ML pipeline code
│   ├── api/                    # FastAPI serving layer
│   └── dashboard/              # Streamlit dashboard
├── scripts/                    # CLI helper scripts
├── tests/                      # Unit tests
├── configs/                    # Configuration files
├── docker/prometheus/          # Prometheus config
├── k8s/                        # Kubernetes manifests
├── data/                       # Local data folder, ignored by git
├── artifacts/                  # Trained model artifacts, ignored by git
├── reports/                    # Evaluation reports, ignored by git
├── .github/workflows/          # CI workflow
├── Dockerfile                  # API Docker image
├── Dockerfile.dashboard        # Dashboard Docker image
├── docker-compose.yml          # Local multi-container stack
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 5. VS Code Setup

### Step 1 — Create and activate environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — Create environment file

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS/Linux
```

Edit `.env` and add MongoDB Atlas URI when available:

```bash
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-url>/gridguard?retryWrites=true&w=majority
MONGODB_DB=gridguard
```

If `MONGODB_URI` is not provided, the app logs predictions to `logs/prediction_logs.jsonl` so the project still runs locally.

---

## 6. Kaggle Dataset Download

### Step 1 — Install Kaggle CLI

```bash
pip install kaggle
```

### Step 2 — Add Kaggle API token

Download `kaggle.json` from your Kaggle account settings and place it here:

Windows:

```text
C:\Users\<YOUR_USER>\.kaggle\kaggle.json
```

macOS/Linux:

```text
~/.kaggle/kaggle.json
```

### Step 3 — Download dataset

```bash
python scripts/download_data.py
```

Expected raw file:

```text
data/raw/household_power_consumption.txt
```

---

## 7. Run the Complete ML Pipeline

### Step 1 — Preprocess and create features

```bash
python scripts/make_dataset.py
```

Output:

```text
data/processed/hourly_features.parquet
```

### Step 2 — Train models

For fast local demo:

```bash
python scripts/train_model.py --max-rows 250000
```

For full training:

```bash
python scripts/train_model.py
```

Outputs:

```text
artifacts/model.joblib
artifacts/anomaly_model.joblib
artifacts/feature_columns.json
artifacts/metrics.json
artifacts/model_metadata.json
reports/training_report.json
```

### Step 3 — Run tests

```bash
pytest -q
flake8 src tests scripts
```

---

## 8. Run FastAPI Locally

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

Example request:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"timestamp\":\"2010-11-26T18:00:00\",\"global_active_power\":1.5,\"global_reactive_power\":0.1,\"voltage\":240,\"global_intensity\":6,\"sub_metering_1\":0,\"sub_metering_2\":1,\"sub_metering_3\":18,\"lag_1h\":1.4,\"lag_2h\":1.3,\"lag_3h\":1.2,\"lag_24h\":1.1,\"lag_168h\":1.0,\"rolling_mean_24h\":1.2,\"rolling_std_24h\":0.2}" 
```

---

## 9. Run Dashboard Locally

In another terminal:

```bash
streamlit run src/dashboard/app.py
```

Open:

```text
http://localhost:8501
```

---

## 10. Docker Workflow

### Build API image

```bash
docker build -t gridguard-api:latest .
```

### Run API container

```bash
docker run --env-file .env -p 8000:8000 gridguard-api:latest
```

### Run full local stack

```bash
docker compose up --build
```

Services:

- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Grafana default login:

```text
username: admin
password: admin
```

---

## 11. MongoDB Atlas Setup

1. Create a free MongoDB Atlas cluster.
2. Create database user.
3. Allow your IP address or use `0.0.0.0/0` only for testing.
4. Copy the connection string.
5. Put it in `.env` as `MONGODB_URI`.
6. Start the API and send predictions.
7. Check collections:
   - `prediction_logs`
   - `feedback_logs`

---

## 12. Prometheus Monitoring

The API exposes `/metrics` using `prometheus-client`.

Important metrics:

- `gridguard_prediction_requests_total`
- `gridguard_prediction_errors_total`
- `gridguard_prediction_latency_seconds`
- `gridguard_predicted_power_value`

Prometheus config is located at:

```text
docker/prometheus/prometheus.yml
```

---

## 13. Kubernetes Deployment

### Step 1 — Build and push Docker images

Replace `<dockerhub_user>` with your Docker Hub username.

```bash
docker build -t <dockerhub_user>/gridguard-api:latest .
docker build -f Dockerfile.dashboard -t <dockerhub_user>/gridguard-dashboard:latest .

docker push <dockerhub_user>/gridguard-api:latest
docker push <dockerhub_user>/gridguard-dashboard:latest
```

### Step 2 — Update image names

Edit:

```text
k8s/api-deployment.yaml
k8s/dashboard-deployment.yaml
```

### Step 3 — Create secret for MongoDB Atlas

```bash
kubectl create namespace gridguard
kubectl -n gridguard create secret generic gridguard-secrets ^
  --from-literal=MONGODB_URI="mongodb+srv://<username>:<password>@<cluster-url>/gridguard?retryWrites=true&w=majority"
```

macOS/Linux:

```bash
kubectl create namespace gridguard
kubectl -n gridguard create secret generic gridguard-secrets \
  --from-literal=MONGODB_URI="mongodb+srv://<username>:<password>@<cluster-url>/gridguard?retryWrites=true&w=majority"
```

### Step 4 — Apply manifests

```bash
kubectl apply -f k8s/
```

### Step 5 — Check pods

```bash
kubectl -n gridguard get pods
kubectl -n gridguard get svc
```

### Step 6 — Port forward

```bash
kubectl -n gridguard port-forward svc/gridguard-api-service 8000:8000
kubectl -n gridguard port-forward svc/gridguard-dashboard-service 8501:8501
```

---

## 14. Hugging Face Spaces Deployment

Use the dashboard as the public-facing demo.

1. Create a new Hugging Face Space.
2. Select **Docker** as SDK.
3. Add secrets in Space Settings:
   - `MONGODB_URI`
   - `MONGODB_DB`
4. Push this repository or the required files to the Space.
5. In the Space repo, use `Dockerfile.hf` as `Dockerfile`.
6. Ensure trained `artifacts/` are included or hosted in a separate Hugging Face model repo and downloaded during startup.

Minimal Hugging Face Space metadata is provided in:

```text
hf_space/README.md
```

---

## 15. Suggested GitHub Workflow

Use conventional commits:

```bash
git init
git add .
git commit -m "chore: initialize gridguard mlops project"

git checkout -b feat/data-pipeline
git add src/gridguard scripts
git commit -m "feat(data): add kaggle ingestion and preprocessing pipeline"
git push origin feat/data-pipeline
```

Create a pull request to `main`, merge it, and repeat for:

- `feat/model-training`
- `feat/api-serving`
- `feat/mongodb-logging`
- `feat/docker-deployment`
- `feat/kubernetes-manifests`
- `feat/monitoring-dashboard`

---

## 16. Defendable Novelty

This project is defendable because it is not limited to model training. It operationalizes smart-energy intelligence:

- Forecasting future electricity demand.
- Detecting abnormal consumption behavior.
- Logging real predictions and future actuals for retraining.
- Monitoring API and model behavior using Prometheus.
- Deploying the same model service through Docker, Hugging Face, and Kubernetes.
- Providing a dashboard for non-technical stakeholders.

This resembles an industry platform used by energy retailers, smart-building operators, IoT energy platforms, and sustainability teams.

---

## 17. Production Extensions

For an even stronger final submission, add:

1. Model drift detection using Evidently AI.
2. MLflow experiment tracking.
3. GitHub Actions deployment to Docker Hub or GHCR.
4. Grafana dashboard provisioning.
5. Scheduled retraining with GitHub Actions or Kubernetes CronJob.
6. Feature store using Feast.
7. Authentication for the FastAPI endpoints.
8. Canary deployment in Kubernetes.

