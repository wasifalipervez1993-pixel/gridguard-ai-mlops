# GridGuard AI Architecture

## 1. System overview

GridGuard AI is a registry-driven MLOps platform for next-hour electricity-load forecasting.

It includes:

- Data preprocessing pipeline.
- Global champion model.
- Regional specialist models.
- Model registry.
- FastAPI prediction API.
- Streamlit dashboard.
- Prometheus monitoring.
- MongoDB Atlas logging.
- Docker and Hugging Face deployment support.
- Automated professional tests.

---

## 2. Architecture diagram

```mermaid
flowchart TD
    A[Raw GloElecLoad data] --> B[Preprocessing pipeline]
    B --> C[Feature dataset]

    C --> D[Global champion training]
    C --> E[AEMO specialist training]
    C --> F[NYISO specialist training]

    D --> G[Global model artifacts]
    E --> H[AEMO specialist artifacts]
    F --> I[NYISO specialist artifacts]

    G --> J[Model registry builder]
    H --> J
    I --> J

    J --> K[artifacts/model_registry.json]

    K --> L[Registry-driven model router]

    L --> M[FastAPI /predict endpoint]
    M --> N[Prediction response]
    M --> O[MongoDB Atlas logs]
    M --> P[Prometheus metrics]

    N --> Q[Streamlit dashboard]
    P --> Q
    O --> Q